#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR Evaluation - 鐎靛綊缍堢€规ɑ鏌?Diffusers 濞翠胶鈻?
======================================================================

娑?train_dual_control.py 闁板秴顨滄担璺ㄦ暏

Usage:
    python evaluate_dual_control.py \
        --checkpoint checkpoints/dual_control/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --strength 0.7 --num_steps 20
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

from diffusers import FlowMatchEulerDiscreteScheduler

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Note: lpips not installed. Run: pip install lpips")


# ============================================================================
# Pixel Feature Extractor
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
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
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

def clear_memory(device):
    """Aggressively clear Python/CUDA memory."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================================
# Evaluator - 鐎靛綊缍堢€规ɑ鏌熷ù浣衡柤
# ============================================================================

class DualStreamEvaluator(nn.Module):
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
        self.scheduler = None
        self._cached_embeds = None
        self.strength = 0.7
        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0
        self.conditioning_scale = 1.0
    
    def load(self):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        # Load Scheduler
        print("Loading Scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name, subfolder="scheduler"
        )
        
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.enable_tiling()
        
        # Cache text embeddings
        print("Caching text embeddings...")
        self._cache_text_embeddings()

        print("Loading Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        self.transformer.eval()

        print("Loading ControlNet...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            "jasperai/Flux.1-dev-Controlnet-Upscaler", torch_dtype=dtype
        ).to(self.device)
        self.controlnet.requires_grad_(False)
        self.controlnet.eval()

        print("Creating Pixel Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.requires_grad_(False)
        self.pixel_extractor.eval()

        print("Loading checkpoint...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if 'pixel_extractor' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
            self.pixel_extractor.load_state_dict(state)

        if 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            self.controlnet.load_state_dict(state)

        if 'pixel_weight' in ckpt:
            self.pixel_weight = ckpt['pixel_weight']
        self.conditioning_scale = ckpt.get('conditioning_scale', 1.0)
        self.strength = ckpt.get('strength', 0.7)
        self.control_guidance_start = ckpt.get('control_guidance_start', 0.0)
        self.control_guidance_end = ckpt.get('control_guidance_end', 1.0)

        print(f"Checkpoint: epoch={ckpt.get('epoch', '?')}, psnr={ckpt.get('psnr', 0):.2f}")
        print(f"Pixel Weight: {self.pixel_weight}, Strength: {self.strength}")
        print(f"Conditioning Scale: {self.conditioning_scale}")
        print(f"Control Guidance Window: [{self.control_guidance_start}, {self.control_guidance_end}]")
        
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except:
            pass
    
    def _cache_text_embeddings(self):
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        text_enc = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        
        text_enc_2 = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        tok_2 = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")
        
        with torch.no_grad():
            clip_out = text_enc(tok([""], padding="max_length", max_length=77,
                                    truncation=True, return_tensors="pt").input_ids.to(self.device))
            t5_out = text_enc_2(tok_2([""], padding="max_length", max_length=512,
                                      truncation=True, return_tensors="pt").input_ids.to(self.device))
            self._cached_embeds = {
                'pooled': clip_out.pooler_output.to(dtype),
                'prompt': t5_out[0].to(dtype),
                'text_ids': torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }
        
        del text_enc, text_enc_2, tok, tok_2, clip_out, t5_out
        clear_memory(self.device)
    
    @torch.no_grad()
    def encode(self, img):
        lat = self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample()
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            lat = lat * self.vae.config.scaling_factor
        return lat
    
    @torch.no_grad()
    def decode(self, lat):
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            lat = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            lat = lat / self.vae.config.scaling_factor
        return self.vae.decode(lat.to(self.vae.dtype)).sample
    
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

    def _compute_scheduler_mu(self, latent):
        """Compute `mu` for FlowMatch dynamic shifting when required."""
        if not getattr(self.scheduler.config, "use_dynamic_shifting", False):
            return None

        base_shift = float(getattr(self.scheduler.config, "base_shift", 0.5))
        max_shift = float(getattr(self.scheduler.config, "max_shift", 1.15))
        base_seq = int(getattr(self.scheduler.config, "base_image_seq_len", 256))
        max_seq = int(getattr(self.scheduler.config, "max_image_seq_len", 4096))

        h, w = latent.shape[-2:]
        image_seq_len = int((h // 2) * (w // 2))

        if max_seq == base_seq:
            return base_shift

        slope = (max_shift - base_shift) / (max_seq - base_seq)
        mu = slope * image_seq_len + (base_shift - slope * base_seq)
        return float(mu)

    def _set_scheduler_timesteps(self, num_steps, device, latent_for_mu):
        """Set scheduler timesteps with backward-compatible `mu` handling."""
        mu = self._compute_scheduler_mu(latent_for_mu)
        if mu is not None:
            try:
                self.scheduler.set_timesteps(num_steps, device=device, mu=mu)
                return
            except TypeError:
                # Older diffusers may not accept `mu` in set_timesteps
                pass
        self.scheduler.set_timesteps(num_steps, device=device)
    
    @torch.no_grad()
    def forward(self, noisy, lr_lat, lr_pixel, timestep, guidance=3.5, controlnet_scale=1.0):
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        pixel_feat = self.pixel_extractor(lr_pixel)
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )
        
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        del pixel_feat
        
        noisy_packed = self._pack(noisy.to(dtype))
        fused_packed = self._pack(fused_cond)
        del fused_cond
        img_ids = self._img_ids(H, W, device, dtype)
        
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        text_ids = self._cached_embeds['text_ids']
        
        if isinstance(timestep, torch.Tensor):
            t_input = timestep.to(device=device, dtype=dtype)
            if t_input.ndim == 0:
                t_input = t_input.expand(B)
            elif t_input.ndim == 1 and t_input.shape[0] == 1 and B > 1:
                t_input = t_input.expand(B)
            elif t_input.ndim != 1:
                raise ValueError(f"Expected 1D timestep tensor, got shape {tuple(t_input.shape)}")
        else:
            t_input = torch.full((B,), float(timestep), device=device, dtype=dtype)
        controlnet_guidance = None
        transformer_guidance = None
        if getattr(self.controlnet.config, "guidance_embeds", False):
            controlnet_guidance = torch.full((B,), guidance, device=device, dtype=dtype)
        if getattr(self.transformer.config, "guidance_embeds", False):
            transformer_guidance = torch.full((B,), guidance, device=device, dtype=dtype)
        
        ctrl_out = self.controlnet(
            hidden_states=noisy_packed,
            controlnet_cond=fused_packed,
            conditioning_scale=float(self.conditioning_scale * controlnet_scale),
            timestep=t_input,
            guidance=controlnet_guidance,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )
        del fused_packed
        
        out = self.transformer(
            hidden_states=noisy_packed,
            timestep=t_input,
            guidance=transformer_guidance,
            pooled_projections=pooled,
            encoder_hidden_states=prompt,
            txt_ids=text_ids,
            img_ids=img_ids,
            controlnet_block_samples=ctrl_out[0],
            controlnet_single_block_samples=ctrl_out[1],
            return_dict=False,
        )[0]
        del ctrl_out, noisy_packed, img_ids
        
        return self._unpack(out, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, strength=0.7):
        """Inference using the official scheduler."""
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # Set timesteps (dynamic shifting may require mu)`r`n        self._set_scheduler_timesteps(num_steps, device, lr_lat)
        timesteps = self.scheduler.timesteps
        
        # 閺嶈宓?strength 鐠侊紕鐣荤挧宄邦潗閻?
        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        if len(timesteps) == 0:
            raise ValueError(
                f"No timesteps left after applying strength={strength}. "
                f"Please increase num_steps (current: {num_steps}) or strength."
            )

        # 閸滃苯鐣奸弬?img2img 鐎靛綊缍堥敍姘▔瀵繗顔曠純?begin_index閿涘苯鍟€鐠嬪啰鏁?scale_noise
        self.scheduler.set_begin_index(t_start)

        noise = torch.randn_like(lr_lat)

        # 娴ｈ法鏁ょ€规ɑ鏌?scale_noise
        timestep_batch = timesteps[:1].expand(B)
        latents = self.scheduler.scale_noise(lr_lat, timestep_batch, noise)
        del noise
        
        # 閸樿娅斿顏嗗箚
        total_steps = len(timesteps)
        for i, t in enumerate(timesteps):
            timestep_model = t / 1000.0
            if total_steps <= 1:
                step_ratio = 1.0
            else:
                step_ratio = i / float(total_steps - 1)
            keep = self.control_guidance_start <= step_ratio <= self.control_guidance_end
            controlnet_scale = 1.0 if keep else 0.0

            model_output = self.forward(
                latents, lr_lat, lr_pixel, timestep_model, guidance,
                controlnet_scale=controlnet_scale
            )
            latents = self.scheduler.step(model_output, t, latents, return_dict=False)[0]
            del model_output
        
        return latents


# ============================================================================
# Tiled Inference
# ============================================================================

@torch.no_grad()
def run_sr_tiled(evaluator, lr_t, device, num_steps=20, guidance=3.5,
                 tile_size=512, overlap=64, blend_mode='linear', strength=0.7):
    _, _, H, W = lr_t.shape
    
    if H <= tile_size and W <= tile_size:
        lr_lat = evaluator.encode(lr_t)
        sr_lat = evaluator.inference(lr_lat, lr_t, num_steps=num_steps, 
                                     guidance=guidance, strength=strength)
        result = evaluator.decode(sr_lat)
        del lr_lat, sr_lat
        return result
    
    stride = tile_size - overlap
    out = torch.zeros((1, 3, H, W), device='cpu', dtype=torch.float32)
    weight = torch.zeros((1, 1, H, W), device='cpu', dtype=torch.float32)
    
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
    full_blend = torch.ones((1, 1, tile_size, tile_size), dtype=torch.float32)
    if blend_mode == 'linear':
        for i in range(min(overlap, tile_size // 2)):
            factor = i / overlap
            full_blend[:, :, i, :] *= factor
            full_blend[:, :, -i-1, :] *= factor
            full_blend[:, :, :, i] *= factor
            full_blend[:, :, :, -i-1] *= factor
    
    with tqdm(total=total_tiles, desc="Tiled SR", leave=False) as pbar:
        for y in y_positions:
            for x in x_positions:
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                tile_h = y_end - y
                tile_w = x_end - x
                
                tile = lr_t[:, :, y:y_end, x:x_end]
                
                if tile_h < tile_size or tile_w < tile_size:
                    padded = torch.zeros((1, 3, tile_size, tile_size), device=device, dtype=lr_t.dtype)
                    padded[:, :, :tile_h, :tile_w] = tile
                    tile = padded
                
                tile_lat = evaluator.encode(tile)
                sr_lat = evaluator.inference(tile_lat, tile, num_steps=num_steps, 
                                            guidance=guidance, strength=strength)
                sr_tile = evaluator.decode(sr_lat)
                sr_tile_cpu = sr_tile[:, :, :tile_h, :tile_w].float().cpu()
                del tile_lat, sr_lat, sr_tile, tile
                clear_memory(device)
                
                if tile_h == tile_size and tile_w == tile_size:
                    tile_blend = full_blend
                else:
                    tile_blend = torch.ones((1, 1, tile_h, tile_w), dtype=torch.float32)
                
                out[:, :, y:y_end, x:x_end] += sr_tile_cpu * tile_blend
                weight[:, :, y:y_end, x:x_end] += tile_blend
                del sr_tile_cpu
                
                pbar.update(1)
    
    return out / weight.clamp(min=1e-8)


@torch.no_grad()
def run_sr_tiled_with_oom_retry(
    evaluator, lr_t, device, num_steps=20, guidance=3.5,
    tile_size=512, overlap=64, blend_mode='linear', strength=0.7,
    min_tile_size=256
):
    """
    OOM-safe tiled inference:
    - 鐏忔繆鐦ぐ鎾冲 tile_size
    - OOM 閺冭埖绔婚悶鍡樻▔鐎涙ê鑻熺亸?tile_size 閸戝繐宕愰柌宥堢槸閿涘瞼娲块崚?min_tile_size
    """
    current_tile = tile_size
    last_err = None
    while current_tile >= min_tile_size:
        try:
            clear_memory(device)
            return run_sr_tiled(
                evaluator, lr_t, device, num_steps=num_steps, guidance=guidance,
                tile_size=current_tile, overlap=min(overlap, max(0, current_tile // 4)),
                blend_mode=blend_mode, strength=strength
            )
        except Exception as e:
            is_oom = isinstance(e, torch.OutOfMemoryError)
            if (not is_oom) and isinstance(e, RuntimeError):
                msg = str(e).lower()
                is_oom = ("out of memory" in msg) or ("cuda error: out of memory" in msg)

            if not is_oom:
                raise

            last_err = e
            clear_memory(device)
            if current_tile == min_tile_size:
                break
            next_tile = max(min_tile_size, current_tile // 2)
            print(f"[OOM] tile_size={current_tile} failed, retry with tile_size={next_tile}")
            current_tile = next_tile
    raise last_err if last_err is not None else RuntimeError("OOM retry failed without exception details.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Stream FLUX SR Evaluation (Official Scheduler)')
    
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--pixel_weight', type=float, default=None)
    parser.add_argument('--strength', type=float, default=None,
                        help='閹恒劎鎮婄挧椋庡仯 (1.0=娴犲海鍑介崳顏勶紣閿?.7=鐠哄疇绻?0%)')
    parser.add_argument('--control_guidance_start', type=float, default=None)
    parser.add_argument('--control_guidance_end', type=float, default=None)
    
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--min_tile_size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--blend_mode', type=str, default='linear')
    parser.add_argument('--calc_lpips', action='store_true')
    parser.add_argument('--lpips_device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
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
    
    if args.pixel_weight is not None:
        evaluator.pixel_weight = args.pixel_weight
    
    strength = args.strength if args.strength is not None else evaluator.strength
    control_guidance_start = (
        args.control_guidance_start
        if args.control_guidance_start is not None
        else evaluator.control_guidance_start
    )
    control_guidance_end = (
        args.control_guidance_end
        if args.control_guidance_end is not None
        else evaluator.control_guidance_end
    )
    evaluator.control_guidance_start = control_guidance_start
    evaluator.control_guidance_end = control_guidance_end
    
    # Experiment name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.exp_name:
        exp_name = args.exp_name
    else:
        exp_name = f"{ts}_str{strength}_{args.num_steps}step"
    
    output_dir = os.path.join(args.output_base, args.dataset, 'DualOfficial', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    if args.save_comparisons:
        os.makedirs(os.path.join(output_dir, 'comparisons'), exist_ok=True)
    
    print("=" * 70)
    print("Dual-Stream FLUX SR Evaluation - Official Scheduler")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Strength: {strength}")
    print(f"Control Guidance Window: [{control_guidance_start}, {control_guidance_end}]")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    lpips_fn = None
    lpips_device = torch.device('cpu')
    if args.calc_lpips:
        if not LPIPS_AVAILABLE:
            print("Warning: --calc_lpips set but lpips package not available, skip LPIPS.")
        else:
            if args.lpips_device == 'cuda' and torch.cuda.is_available():
                lpips_device = torch.device('cuda')
            else:
                lpips_device = torch.device('cpu')
            lpips_fn = lpips.LPIPS(net='alex').to(lpips_device)
            lpips_fn.eval()
    
    hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nEvaluating {len(hr_files)} images...\n")
    
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    filenames = []
    
    for hf in tqdm(hr_files, desc="Evaluating"):
        base_name = os.path.splitext(hf)[0]
        
        lf = None
        for suffix in ['', 'x4', 'x2', '_x4', '_x2', '_LR4', '_LR2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = base_name + suffix + ext
                if candidate in lr_files:
                    lf = candidate
                    break
            if lf:
                break
        
        if lf is None:
            print(f"Warning: No LR file for {hf}, skipping...")
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
        
        sr_t = run_sr_tiled_with_oom_retry(
            evaluator, lr_t, device,
            num_steps=args.num_steps,
            guidance=args.guidance,
            tile_size=args.tile_size,
            min_tile_size=args.min_tile_size,
            overlap=args.overlap,
            blend_mode=args.blend_mode,
            strength=strength
        )
        
        sr_np = ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
        
        psnr_val = calculate_psnr(sr_np, hr_np)
        ssim_val = calculate_ssim(sr_np, hr_np)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
        ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
        psnr_bic_list.append(psnr_bic)
        ssim_bic_list.append(ssim_bic)
        
        if lpips_fn:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lr_t_lpips = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            
            hr_t = hr_t.to(lpips_device)
            lr_t_lpips = lr_t_lpips.to(lpips_device)
            lpips_val = lpips_fn(sr_t.float().clamp(-1, 1).to(lpips_device), hr_t).item()
            lpips_bic = lpips_fn(lr_t_lpips, hr_t).item()
            lpips_list.append(lpips_val)
            lpips_bic_list.append(lpips_bic)
        
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
        
        del lr_t, sr_t
        if lpips_fn:
            del hr_t, lr_t_lpips
        del hr_np, lr_bicubic_np, sr_np
        clear_memory(device)
    
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_psnr_bic = np.mean(psnr_bic_list)
    avg_ssim_bic = np.mean(ssim_bic_list)
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Strength: {strength}")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    if lpips_fn:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"SR:       PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"铻?        {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
    else:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
        print(f"SR:       PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
    print("=" * 70)
    
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("Dual-Stream FLUX SR - Official Scheduler\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Strength: {strength}\n")
        f.write(f"Control Guidance Window: [{control_guidance_start}, {control_guidance_end}]\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary:\n")
        if lpips_fn:
            f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}\n")
            f.write(f"SR:       PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}\n")
        else:
            f.write(f"Bicubic:  PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}\n")
            f.write(f"SR:       PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Per-image:\n")
        for i, fname in enumerate(filenames):
            delta = psnr_list[i] - psnr_bic_list[i]
            if lpips_fn:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (铻杮delta:+.2f}), LPIPS={lpips_list[i]:.4f}\n")
            else:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (铻杮delta:+.2f})\n")
    
    print(f"\n閴?Results saved: {output_dir}")


if __name__ == '__main__':
    main()
