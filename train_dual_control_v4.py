#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training - V4
======================================================================

V4 设计理念：
- 训练：统一从纯 noise 开始（standard flow matching）
- 推理：从 noise 和 lr_lat 的插值开始（加速推理）

这样可以利用预训练模型学到的完整去噪轨迹，同时在推理时跳过早期步骤。

基于 V2 的所有修复：
1. PixelExtractor 学习率 10x 放大
2. 梯度累积正确使用 accelerator.accumulate() 上下文
3. Scheduler 只在 sync_gradients 时 step
4. num_crops 支持
5. 验证集中心裁剪

Usage:
    accelerate launch --num_processes=8 \
        --gradient_accumulation_steps=8 \
        train_dual_control_v4.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --batch_size 4 --epochs 120 --lr 1e-5
"""

import os
import gc
import math
import argparse
import numpy as np
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，映射到 Latent 空间维度
    使用 Zero Conv 确保初始化时不破坏预训练 ControlNet
    
    Input: RGB image [B, 3, H, W] (H, W = 512)
    Output: Latent-space features [B, 16, H/8, W/8] (64x64)
    """
    def __init__(self, latent_channels=16):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            
            # 256 → 128
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # 128 → 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )
        
        # Zero Conv: 初始化权重为 0
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
    
    def forward(self, x):
        feat = self.encoder(x)
        return self.zero_conv(feat)


# ============================================================================
# Dataset
# ============================================================================

class SRDataset(Dataset):
    """
    Super-Resolution Dataset
    
    Features:
    - num_crops: 每张图像随机裁剪次数
    - is_val: 验证集使用中心裁剪
    - 自动匹配 LR 文件名
    """
    
    def __init__(self, hr_dir, lr_dir, resolution=512, num_crops=1, is_val=False):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.num_crops = num_crops
        self.scale = 4
        self.is_val = is_val
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.hr_files) * self.num_crops
    
    def _find_lr_file(self, hr_name):
        """自动查找对应的 LR 文件（适配 DIV2K 等数据集）"""
        base = os.path.splitext(hr_name)[0]
        
        # 尝试多种后缀
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base + suffix + ext)
                if os.path.exists(candidate):
                    return candidate
        
        # 回退：直接使用相同文件名
        return os.path.join(self.lr_dir, hr_name)
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_crops
        hr_name = self.hr_files[img_idx]
        
        hr_img = Image.open(os.path.join(self.hr_dir, hr_name)).convert('RGB')
        lr_img = Image.open(self._find_lr_file(hr_name)).convert('RGB')
        
        # 裁剪逻辑
        hr_w, hr_h = hr_img.size
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale
        
        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                # 验证集使用中心裁剪
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
            else:
                # 训练集使用随机裁剪
                x = np.random.randint(0, hr_w - crop_size + 1)
                y = np.random.randint(0, hr_h - crop_size + 1)
            
            hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
            
            # 对应 LR 区域
            lr_x, lr_y = x // self.scale, y // self.scale
            lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        else:
            # 图像太小，直接 resize
            hr_crop = hr_img.resize((crop_size, crop_size), Image.BICUBIC)
            lr_crop = lr_img.resize((lr_crop_size, lr_crop_size), Image.BICUBIC)
        
        # Bicubic upsample LR to match HR size
        lr_up = lr_crop.resize((crop_size, crop_size), Image.BICUBIC)
        
        # To tensor [-1, 1]
        hr_t = torch.from_numpy(np.array(hr_crop)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """
    Dual-Stream FLUX SR System
    
    V4: 训练统一从纯 noise 开始（standard flow matching）
    """
    
    def __init__(self, model_name, device, pretrained_controlnet=None, 
                 train_controlnet=True, pixel_weight=1.0):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)
    
    def _load_models(self, pretrained_controlnet):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        import time
        
        dtype = torch.bfloat16
        
        # 错峰加载
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank > 0:
            time.sleep(local_rank * 5)
        
        # Load VAE
        print(f"[Rank {local_rank}] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae.enable_tiling()
        
        # Load Transformer
        print(f"[Rank {local_rank}] Loading Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        
        # Load ControlNet
        print(f"[Rank {local_rank}] Loading ControlNet...")
        controlnet_path = pretrained_controlnet or "jasperai/Flux.1-dev-Controlnet-Upscaler"
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype
        ).to(self.device)
        
        if self.train_controlnet:
            self.controlnet.train()
            self.controlnet.requires_grad_(True)
        else:
            self.controlnet.eval()
            self.controlnet.requires_grad_(False)
        
        # Create Pixel Extractor
        print(f"[Rank {local_rank}] Creating Pixel Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.train()
        
        # Cache text embeddings
        print(f"[Rank {local_rank}] Encoding text embeddings...")
        self._cache_text_embeddings()
        
        # Memory optimization
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except:
            pass  # Fall back to PyTorch 2.0 SDPA
        
        print(f"[Rank {local_rank}] Models loaded!")
    
    def _cache_text_embeddings(self):
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        prompt = "high resolution, 4K, detailed, sharp"
        
        # CLIP
        clip_tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        clip_model = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        clip_model.eval()
        
        clip_ids = clip_tok(prompt, max_length=77, padding="max_length",
                            truncation=True, return_tensors="pt").input_ids.to(self.device)
        pooled = clip_model(clip_ids).pooler_output
        
        del clip_model, clip_tok
        gc.collect()
        torch.cuda.empty_cache()
        
        # T5
        t5_tok = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")
        t5_model = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        t5_model.eval()
        
        t5_ids = t5_tok(prompt, max_length=512, padding="max_length",
                        truncation=True, return_tensors="pt").input_ids.to(self.device)
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
        # FLUX VAE 需要 shift_factor 校正
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
        
        训练时从纯 noise 开始，但推理时可以从中间点开始加速。
        
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
        # lat = start_t * noise + (1 - start_t) * lr_lat
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
    
    def get_trainable_params(self):
        """Get trainable parameters"""
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return params


# ============================================================================
# Training Functions
# ============================================================================

def compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel):
    """
    V4: Standard Flow Matching Loss（统一从纯 noise 开始）
    
    轨迹：noise (t=1) → hr_lat (t=0)
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    t = torch.rand(B, device=device, dtype=dtype)
    noise = torch.randn_like(hr_lat)
    
    # 插值：x_t = t * noise + (1 - t) * hr_lat
    t_expand = t.view(B, 1, 1, 1)
    x_t = t_expand * noise + (1 - t_expand) * hr_lat
    
    # 目标速度：v = noise - hr_lat
    target_v = noise - hr_lat
    
    # 预测
    unwrapped = system.module if hasattr(system, 'module') else system
    v_pred = unwrapped.forward(x_t, lr_lat, lr_pixel, t)
    
    # Loss (float32 计算)
    loss = F.mse_loss(v_pred.float(), target_v.float())
    
    return loss


def calculate_psnr(pred, target):
    """Calculate PSNR between tensors in [-1, 1]"""
    pred = (pred.clamp(-1, 1) + 1) / 2
    target = (target + 1) / 2
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def validate(system, accelerator, val_loader, device, num_samples=5, num_steps=20, start_t=0.7):
    """
    V4 验证：推理时从插值点开始
    """
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.controlnet.eval()
    
    psnr_list = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device).to(torch.bfloat16)
        lr = batch['lr'].to(device).to(torch.bfloat16)
        
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        
        # 🌟 V4：推理从插值点开始
        sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps, start_t=start_t)
        sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
    
    # 恢复训练模式
    unwrapped.pixel_extractor.train()
    if unwrapped.train_controlnet:
        unwrapped.controlnet.train()
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, path):
    """Save checkpoint"""
    unwrapped = accelerator.unwrap_model(system)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'pixel_weight': pixel_weight,
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'controlnet': unwrapped.controlnet.state_dict(),
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Stream FLUX SR Training V4')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_crops', type=int, default=2,
                        help='Random crops per image per epoch')
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default=None)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--guidance', type=float, default=3.5)
    
    # V4: 推理起点参数
    parser.add_argument('--start_t', type=float, default=0.7,
                        help='推理起点的噪声比例 (0=lr_lat, 1=noise)')
    parser.add_argument('--val_num_steps', type=int, default=20,
                        help='验证时的推理步数')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/dual_control_v4')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.freeze_controlnet:
        args.train_controlnet = False
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
    )
    
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_dual_v4_res{args.resolution}_crop{args.num_crops}"
    save_dir = os.path.join(args.save_dir, exp_name)
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        # 打印配置
        print("\n" + "=" * 70)
        print("FLUX SR Training with Dual-Stream ControlNet - V4")
        print("=" * 70)
        print(f"\nHR Dir: {args.hr_dir}")
        print(f"LR Dir: {args.lr_dir}")
        print(f"Resolution: {args.resolution}")
        print(f"Num Crops: {args.num_crops}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Training: Standard Flow (noise → hr)")
        print(f"Inference: start_t={args.start_t} ({args.start_t*100:.0f}% noise + {(1-args.start_t)*100:.0f}% lr)")
        print(f"Learning Rate: {args.lr} (PixelExtractor: {args.lr * 10})")
        print(f"Warmup Epochs: {args.warmup_epochs}")
        print(f"Save Dir: {save_dir}")
        print("=" * 70 + "\n")
    
    # Create model
    train_controlnet = args.train_controlnet
    system = DualStreamFLUXSR(
        args.model_name, device, args.pretrained_controlnet,
        train_controlnet=train_controlnet, pixel_weight=args.pixel_weight
    )
    
    # Enable gradient checkpointing
    if hasattr(system.transformer, 'enable_gradient_checkpointing'):
        system.transformer.enable_gradient_checkpointing()
    if hasattr(system.controlnet, 'enable_gradient_checkpointing'):
        system.controlnet.enable_gradient_checkpointing()
    
    # Optimizer groups (PixelExtractor 10x LR)
    if train_controlnet:
        optimizer_grouped_parameters = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    
    trainable_params = system.get_trainable_params()
    
    if is_main:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Create datasets
    train_dataset = SRDataset(
        args.hr_dir, args.lr_dir, args.resolution,
        num_crops=args.num_crops, is_val=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = None
    if args.val_hr_dir and args.val_lr_dir:
        val_dataset = SRDataset(
            args.val_hr_dir, args.val_lr_dir, args.resolution,
            num_crops=1, is_val=True
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            print(f"[Data] Training: {len(train_dataset)}, Validation: {len(val_dataset)}")
    else:
        if is_main:
            print(f"[Data] Training: {len(train_dataset)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare with Accelerator
    system, optimizer, train_loader, scheduler = accelerator.prepare(
        system, optimizer, train_loader, scheduler
    )
    
    # Resume
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        
        unwrapped = accelerator.unwrap_model(system)
        
        if 'pixel_extractor' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
            unwrapped.pixel_extractor.load_state_dict(state)
        
        if train_controlnet and 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            unwrapped.controlnet.load_state_dict(state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # 初始化 log 文件
    log_path = os.path.join(save_dir, 'training_log.txt')
    if is_main:
        with open(log_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FLUX SR Training with Dual-Stream ControlNet - V4\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"HR Dir: {args.hr_dir}\n")
            f.write(f"LR Dir: {args.lr_dir}\n")
            f.write(f"Resolution: {args.resolution}\n")
            f.write(f"Num Crops: {args.num_crops}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            f.write(f"Training: Standard Flow (noise → hr)\n")
            f.write(f"Inference: start_t={args.start_t}\n")
            f.write(f"Learning Rate: {args.lr} (PixelExtractor: {args.lr * 10})\n")
            f.write(f"Warmup Epochs: {args.warmup_epochs}\n\n")
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Set train mode
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        if train_controlnet:
            unwrapped.controlnet.train()
        
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            hr = batch['hr'].to(device).to(torch.bfloat16)
            lr = batch['lr'].to(device).to(torch.bfloat16)
            
            with torch.no_grad():
                hr_lat = unwrapped.encode(hr)
                lr_lat = unwrapped.encode(lr)
            
            # V4: 只用 standard flow matching
            with accelerator.accumulate(system):
                loss = compute_flow_matching_loss(system, hr_lat, lr_lat, lr)
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # 只在梯度同步时更新学习率
            if accelerator.sync_gradients:
                scheduler.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        val_psnr = 0.0
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_psnr = validate(system, accelerator, val_loader, device, 
                               num_samples=5, num_steps=args.val_num_steps, start_t=args.start_t)
        
        # Logging and saving (main process)
        if is_main:
            lr_current = scheduler.get_last_lr()[0]
            
            # 写入 log 文件
            log_line = f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, LR={lr_current:.2e}\n"
            with open(log_path, 'a') as f:
                f.write(log_line)
            
            # 打印
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, lr={lr_current:.2e}")
            
            # Save best
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    system, accelerator, epoch, avg_loss, val_psnr, 
                    args.pixel_weight,
                    os.path.join(save_dir, 'best_model.pt')
                )
                print(f"  → New best PSNR: {best_psnr:.2f} dB")
            
            # Periodic save
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(
                    system, accelerator, epoch, avg_loss, val_psnr, 
                    args.pixel_weight,
                    os.path.join(save_dir, f'epoch{epoch+1}.pt')
                )
            
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
    
    # Final save
    if is_main:
        save_checkpoint(
            system, accelerator, args.epochs - 1, avg_loss, best_psnr, 
            args.pixel_weight,
            os.path.join(save_dir, 'final_model.pt')
        )
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
