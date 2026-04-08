#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR Evaluation - V4
======================================================================

V4 设计理念：
- 训练：统一从纯 noise 开始（standard flow matching）
- 推理：从 noise 和 lr_lat 的插值开始（加速推理）

与 train_dual_control_v4.py 配套使用

Usage:
    python evaluate_dual_control_v4.py \
        --checkpoint checkpoints/dual_control_v4/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --start_t 0.7 --num_steps 20
"""

import os
import gc
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Note: lpips not installed. Run: pip install lpips")


# ============================================================================
# Pixel Feature Extractor (与训练代码完全一致)
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )
        
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    """Calculate PSNR between numpy arrays [0, 255]"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    """Calculate SSIM between numpy arrays"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


# ============================================================================
# Evaluator
# ============================================================================

class DualStreamEvaluator(nn.Module):
    """V4 Evaluator: 推理从 noise 和 lr_lat 插值开始"""
    
    def __init__(self, model_name, device, checkpoint_path, pixel_weight=1.0):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.pixel_weight = pixel_weight
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
    
    def load(self):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.eval()
        self.vae.enable_tiling()
        
        print("Loading Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.eval()
        
        print("Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=dtype
        ).to(self.device)
        self.controlnet.eval()
        
        print("Creating Pixel Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.eval()
        
        print("Loading checkpoint...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load pixel_extractor
        if 'pixel_extractor' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
            self.pixel_extractor.load_state_dict(state)
        
        # Load controlnet
        if 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            self.controlnet.load_state_dict(state)
        
        # Load pixel_weight from checkpoint if available
        if 'pixel_weight' in ckpt:
            self.pixel_weight = ckpt['pixel_weight']
        
        print(f"Checkpoint loaded: epoch={ckpt.get('epoch', '?')}, psnr={ckpt.get('psnr', '?'):.2f}")
        print(f"Pixel Weight: {self.pixel_weight}")
        
        # Cache text embeddings
        print("Encoding text embeddings...")
        self._cache_text_embeddings()
        
        # Memory optimization
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # 默认使用 PyTorch 2.0 SDPA
    
    def _cache_text_embeddings(self):
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        prompt = "high resolution, 4K, detailed, sharp"
        
        clip_tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        clip_model = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        clip_model.eval()
        
        clip_ids = clip_tok(prompt, max_length=77, padding="max_length",
                            truncation=True, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            pooled = clip_model(clip_ids).pooler_output
        
        del clip_model, clip_tok
        gc.collect()
        torch.cuda.empty_cache()
        
        t5_tok = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")
        t5_model = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        t5_model.eval()
        
        t5_ids = t5_tok(prompt, max_length=512, padding="max_length",
                        truncation=True, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            t5_out = t5_model(t5_ids).last_hidden_state
        
        del t5_model, t5_tok
        gc.collect()
        torch.cuda.empty_cache()
        
        self._cached_embeds = {
            'pooled': pooled.detach(),
            'prompt': t5_out.detach(),
            'text_ids': torch.zeros(512, 3, device=self.device, dtype=dtype)
        }
    
    def encode(self, img):
        """Encode image to latent (with shift_factor correction)"""
        raw = self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample()
        shift = getattr(self.vae.config, 'shift_factor', 0.0)
        return (raw - shift) * self.vae.config.scaling_factor
    
    def decode(self, lat):
        """Decode latent to image (with shift_factor correction)"""
        shift = getattr(self.vae.config, 'shift_factor', 0.0)
        lat_unscaled = lat / self.vae.config.scaling_factor + shift
        return self.vae.decode(lat_unscaled.to(self.vae.dtype)).sample
    
    def _pack(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // 2) * (W // 2), C * 4)
    
    def _unpack(self, x, H, W):
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)
    
    def _img_ids(self, H, W, device, dtype):
        h, w = H // 2, W // 2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h * w, 3)
    
    @torch.no_grad()
    def forward(self, noisy, lr_lat, lr_pixel, t, guidance=3.5):
        """Forward pass: predict velocity"""
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel)
        
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Fuse
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        # Pack
        noisy_packed = self._pack(noisy.to(dtype))
        fused_packed = self._pack(fused_cond)
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Text embeddings
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        text_ids = self._cached_embeds['text_ids']
        
        t_input = t.to(dtype)
        guidance_tensor = torch.full((B,), guidance, device=device, dtype=dtype)
        
        # ControlNet
        ctrl_out = self.controlnet(
            hidden_states=noisy_packed,
            controlnet_cond=fused_packed,
            timestep=t_input,
            guidance=guidance_tensor,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        
        # Transformer
        out = self.transformer(
            hidden_states=noisy_packed,
            timestep=t_input,
            guidance=guidance_tensor,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=text_ids,
            img_ids=img_ids,
            controlnet_block_samples=ctrl_out[0],
            controlnet_single_block_samples=ctrl_out[1],
            return_dict=False,
        )[0]
        
        return self._unpack(out, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, start_t=0.7):
        """
        V4 推理：从 noise 和 lr_lat 的插值开始
        
        Args:
            start_t: 起点的噪声比例 (0=lr_lat, 1=noise)
                     默认 0.7 表示 70% noise + 30% lr_lat
        """
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        noise = torch.randn_like(lr_lat)
        
        # 🌟 V4 核心：从插值点开始
        lat = start_t * noise + (1 - start_t) * lr_lat
        
        # 跳过前 (1 - start_t) 比例的步数
        start_step = round((1.0 - start_t) * num_steps)
        dt = 1.0 / num_steps
        
        # 从 start_step 开始积分
        for i in range(start_step, num_steps):
            t_val = 1.0 - i * dt
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.forward(lat, lr_lat, lr_pixel, t, guidance)
            lat = lat - dt * v
        
        return lat


# ============================================================================
# Tiled Inference
# ============================================================================

def run_sr_tiled(evaluator, lr_t, device, num_steps=20, guidance=3.5,
                 tile_size=512, overlap=64, blend_mode='linear', start_t=0.7):
    """Run SR with tiling for large images"""
    _, _, H, W = lr_t.shape
    
    # 如果图像足够小，直接处理
    if H <= tile_size and W <= tile_size:
        lr_lat = evaluator.encode(lr_t)
        sr_lat = evaluator.inference(lr_lat, lr_t, num_steps=num_steps, 
                                     guidance=guidance, start_t=start_t)
        return evaluator.decode(sr_lat)
    
    stride = tile_size - overlap
    out = torch.zeros_like(lr_t)
    weight = torch.zeros((1, 1, H, W), device=device)
    
    # 计算 tile 位置（处理边界情况）
    if H <= tile_size:
        y_positions = [0]
    else:
        y_positions = list(range(0, H - tile_size + 1, stride))
        if not y_positions:
            y_positions = [0]
        elif y_positions[-1] + tile_size < H:
            y_positions.append(H - tile_size)
    
    if W <= tile_size:
        x_positions = [0]
    else:
        x_positions = list(range(0, W - tile_size + 1, stride))
        if not x_positions:
            x_positions = [0]
        elif x_positions[-1] + tile_size < W:
            x_positions.append(W - tile_size)
    
    total_tiles = len(y_positions) * len(x_positions)
    
    with tqdm(total=total_tiles, desc="Tiled SR", leave=False) as pbar:
        for y in y_positions:
            for x in x_positions:
                # 计算实际 tile 边界
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                tile_h = y_end - y
                tile_w = x_end - x
                
                tile = lr_t[:, :, y:y_end, x:x_end]
                
                # 如果 tile 小于 tile_size，需要 padding
                if tile_h < tile_size or tile_w < tile_size:
                    padded = torch.zeros((1, 3, tile_size, tile_size), device=device, dtype=lr_t.dtype)
                    padded[:, :, :tile_h, :tile_w] = tile
                    tile = padded
                
                tile_lat = evaluator.encode(tile)
                sr_lat = evaluator.inference(tile_lat, tile, num_steps=num_steps, 
                                            guidance=guidance, start_t=start_t)
                sr_tile = evaluator.decode(sr_lat)
                
                # 只取有效部分
                sr_tile = sr_tile[:, :, :tile_h, :tile_w]
                
                # 创建 blend mask
                if tile_h == tile_size and tile_w == tile_size:
                    tile_blend = torch.ones((1, 1, tile_size, tile_size), device=device)
                    if blend_mode == 'linear':
                        for i in range(min(overlap, tile_size // 2)):
                            factor = i / overlap
                            tile_blend[:, :, i, :] *= factor
                            tile_blend[:, :, -i-1, :] *= factor
                            tile_blend[:, :, :, i] *= factor
                            tile_blend[:, :, :, -i-1] *= factor
                else:
                    tile_blend = torch.ones((1, 1, tile_h, tile_w), device=device)
                
                out[:, :, y:y_end, x:x_end] += sr_tile * tile_blend
                weight[:, :, y:y_end, x:x_end] += tile_blend
                
                pbar.update(1)
    
    return out / weight.clamp(min=1e-8)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Stream FLUX SR Evaluation V4')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--pixel_weight', type=float, default=None,
                        help='Override pixel_weight from checkpoint')
    
    # V4 核心参数
    parser.add_argument('--start_t', type=float, default=0.7,
                        help='推理起点的噪声比例 (0=lr_lat, 1=noise). 默认 0.7')
    
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--blend_mode', type=str, default='linear', choices=['linear', 'none'])
    
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', action='store_true', default=True)
    parser.add_argument('--save_comparisons', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower:
            args.dataset = 'Urban100'
        elif 'div2k' in hr_lower:
            args.dataset = 'DIV2K'
        elif 'set5' in hr_lower:
            args.dataset = 'Set5'
        elif 'set14' in hr_lower:
            args.dataset = 'Set14'
        elif 'bsd100' in hr_lower:
            args.dataset = 'BSD100'
        elif 'manga' in hr_lower:
            args.dataset = 'Manga109'
        elif 'realsr' in hr_lower:
            args.dataset = 'RealSR'
        elif 'drealsr' in hr_lower:
            args.dataset = 'DRealSR'
        else:
            args.dataset = 'Unknown'
    
    # Load model
    initial_pixel_weight = args.pixel_weight if args.pixel_weight is not None else 1.0
    evaluator = DualStreamEvaluator(args.model_name, device, args.checkpoint, initial_pixel_weight)
    evaluator.load()
    
    # Override pixel_weight if specified
    if args.pixel_weight is not None:
        evaluator.pixel_weight = args.pixel_weight
    
    # Experiment name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{ts}_v4_t{args.start_t}_{args.num_steps}step"
    
    # Output directory
    output_dir = os.path.join(args.output_base, args.dataset, 'DualV4', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("=" * 70)
    print("Dual-Stream FLUX SR Evaluation - V4")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Start T: {args.start_t} ({args.start_t*100:.0f}% noise + {(1-args.start_t)*100:.0f}% lr)")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}")
    print(f"Tile: {args.tile_size}, Overlap: {args.overlap}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load LPIPS
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        lpips_fn.eval()
    
    # Get files
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    # Metrics
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    filenames = []
    
    for hf in tqdm(hr_files, desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        
        # Find matching LR file (支持多种命名格式)
        lf = None
        # 格式1: 相同名称 (reorganized datasets)
        # 格式2: DIV2K 格式 (0001x4.png)
        # 格式3: RealSR 原始格式 (Canon_001_LR4.png)
        # 格式4: DRealSR 原始格式 (Canon_10_x1.png)
        for suffix in ['', 'x4', 'x2', '_x4', '_x2', '_LR4', '_LR2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = base_name + suffix + ext
                if candidate in lr_files:
                    lf = candidate
                    break
            if lf:
                break
        
        if lf is None:
            print(f"Warning: No LR file found for {hf}, skipping...")
            continue
        
        filenames.append(base_name)
        
        hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        
        hr_np = np.array(hr_img)
        H, W = hr_np.shape[0], hr_np.shape[1]
        
        lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        
        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device).to(torch.bfloat16)
        
        sr_t = run_sr_tiled(
            evaluator, lr_t, device,
            num_steps=args.num_steps,
            guidance=args.guidance,
            tile_size=args.tile_size,
            overlap=args.overlap,
            blend_mode=args.blend_mode,
            start_t=args.start_t
        )
        
        sr_np = ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Metrics - SR
        psnr_val = calculate_psnr(sr_np, hr_np)
        ssim_val = calculate_ssim(sr_np, hr_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        # Metrics - Bicubic
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        # LPIPS
        if lpips_fn:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            sr_t_lpips = torch.from_numpy(sr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lr_t_lpips = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            
            lpips_val = lpips_fn(sr_t_lpips.to(device), hr_t.to(device)).item()
            lpips_bic = lpips_fn(lr_t_lpips.to(device), hr_t.to(device)).item()
            lpips_list.append(lpips_val)
            lpips_bic_list.append(lpips_bic)
        
        # Save images
        if args.save_images:
            Image.fromarray(sr_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))
        
        if args.save_comparisons:
            comp = Image.new('RGB', (W * 4, H))
            lr_display = lr_img.resize((W, H), Image.NEAREST)
            comp.paste(lr_display, (0, 0))
            comp.paste(lr_bicubic, (W, 0))
            comp.paste(Image.fromarray(sr_np), (W * 2, 0))
            comp.paste(hr_img, (W * 3, 0))
            comp.save(os.path.join(output_dir, 'comparisons', f'{base_name}_compare.png'))
        
        torch.cuda.empty_cache()
    
    # Results
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    print(f"Start T: {args.start_t}")
    if LPIPS_AVAILABLE:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"Dual-V4:  PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"Δ:        {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
    else:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
        print(f"Dual-V4:  PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
    print("=" * 70)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("Dual-Stream FLUX SR Evaluation - V4\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Start T: {args.start_t} ({args.start_t*100:.0f}% noise + {(1-args.start_t)*100:.0f}% lr)\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}\n")
        f.write(f"Tile: {args.tile_size}, Overlap: {args.overlap}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary:\n")
        f.write("=" * 60 + "\n")
        if LPIPS_AVAILABLE:
            f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}\n")
            f.write(f"Dual-V4:  PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}\n")
            f.write(f"Δ:        {avg_psnr - avg_psnr_bic:+.4f}, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}\n")
        else:
            f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}\n")
            f.write(f"Dual-V4:  PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Per-image results:\n")
        f.write("=" * 60 + "\n")
        for i, fname in enumerate(filenames):
            delta_psnr = psnr_list[i] - psnr_bic_list[i]
            if LPIPS_AVAILABLE:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (Δ{delta_psnr:+.2f}), LPIPS={lpips_list[i]:.4f}\n")
            else:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (Δ{delta_psnr:+.2f})\n")
    
    print(f"\n✅ Results saved: {output_dir}")


if __name__ == '__main__':
    main()
