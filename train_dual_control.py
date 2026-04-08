#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training - 对齐官方 Diffusers 流程
======================================================================

核心改动（对比之前版本）：
1. 使用 FlowMatchEulerDiscreteScheduler 而非自定义 flow matching
2. 训练时用 scheduler.sigmas 采样，timestep 传入 sigma * 1000
3. 推理时用 scheduler.scale_noise() 和 scheduler.step()
4. strength 参数控制推理起点（官方语义）

这样确保 frozen FLUX transformer 和预训练 ControlNet 接收的
timestep/scheduler 语义与官方一致。

Usage:
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \
        train_dual_control.py \
        --hr_dir Data/DIV2K/DIV2K_train_HR \
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --batch_size 4 --epochs 120 --lr 1e-5 \
        --strength 0.7
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

from diffusers import FlowMatchEulerDiscreteScheduler


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """
    从原始像素空间提取高频特征，映射到 Latent 空间维度
    使用 Zero Conv 确保初始化时不破坏预训练 ControlNet
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
        
        # Zero Conv
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
        base = os.path.splitext(hr_name)[0]
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base + suffix + ext)
                if os.path.exists(candidate):
                    return candidate
        return os.path.join(self.lr_dir, hr_name)
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_crops
        hr_name = self.hr_files[img_idx]
        
        hr_img = Image.open(os.path.join(self.hr_dir, hr_name)).convert('RGB')
        lr_img = Image.open(self._find_lr_file(hr_name)).convert('RGB')
        
        hr_w, hr_h = hr_img.size
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale
        
        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
            else:
                x = np.random.randint(0, hr_w - crop_size + 1)
                y = np.random.randint(0, hr_h - crop_size + 1)
            
            hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
            lr_x, lr_y = x // self.scale, y // self.scale
            lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        else:
            hr_crop = hr_img.resize((crop_size, crop_size), Image.BICUBIC)
            lr_crop = lr_img.resize((lr_crop_size, lr_crop_size), Image.BICUBIC)
        
        lr_up = lr_crop.resize((crop_size, crop_size), Image.BICUBIC)
        
        hr_t = torch.from_numpy(np.array(hr_crop)).float().permute(2, 0, 1) / 127.5 - 1
        lr_t = torch.from_numpy(np.array(lr_up)).float().permute(2, 0, 1) / 127.5 - 1
        
        return {'hr': hr_t, 'lr': lr_t}


# ============================================================================
# Dual-Stream FLUX SR System - 对齐官方流程
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """
    Dual-Stream FLUX SR System - 使用官方 Scheduler
    """
    
    def __init__(self, model_name, device, pretrained_controlnet=None, 
                 train_controlnet=True, pixel_weight=1.0, conditioning_scale=1.0):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.conditioning_scale = conditioning_scale  # 🌟 ControlNet 条件强度
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self.scheduler = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)
    
    def _load_models(self, pretrained_controlnet):
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FluxControlNetModel
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        import time
        
        dtype = torch.bfloat16
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank > 0:
            time.sleep(local_rank * 5)
        
        # Load Scheduler（关键！）
        print(f"[Rank {local_rank}] Loading Scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name, subfolder="scheduler"
        )
        
        # Load VAE
        print(f"[Rank {local_rank}] Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            self.model_name, subfolder="vae", torch_dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.enable_tiling()
        
        # Load Transformer (frozen)
        print(f"[Rank {local_rank}] Loading FLUX Transformer...")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.model_name, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        
        # Load ControlNet
        controlnet_path = pretrained_controlnet or "jasperai/Flux.1-dev-Controlnet-Upscaler"
        print(f"[Rank {local_rank}] Loading ControlNet from {controlnet_path}...")
        self.controlnet = FluxControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype
        ).to(self.device)
        
        if self.train_controlnet:
            self.controlnet.train()
            self.controlnet.requires_grad_(True)
        else:
            self.controlnet.eval()
            self.controlnet.requires_grad_(False)
        
        # Pixel Feature Extractor
        print(f"[Rank {local_rank}] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_extractor.train()
        
        # Cache text embeddings (空字符串)
        print(f"[Rank {local_rank}] Caching text embeddings...")
        self._cache_text_embeddings()
        
        # Enable Flash Attention
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
            if local_rank == 0:
                print("[Flash] ✓ Enabled xformers memory efficient attention")
        except Exception:
            if local_rank == 0:
                print("[Flash] Using PyTorch 2.0 SDPA")
        
        print(f"[Rank {local_rank}] ✓ All models loaded")
    
    def _cache_text_embeddings(self):
        """缓存空文本的 embeddings"""
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        
        dtype = torch.bfloat16
        
        # CLIP
        text_enc = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        
        # T5
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
        
        del text_enc, text_enc_2
        torch.cuda.empty_cache()
        gc.collect()
    
    def encode(self, img):
        """Encode image to latent (FLUX VAE with shift_factor)"""
        lat = self.vae.encode(img.to(self.vae.dtype)).latent_dist.sample()
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            lat = (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            lat = lat * self.vae.config.scaling_factor
        return lat
    
    def decode(self, lat):
        """Decode latent to image"""
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            lat = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            lat = lat / self.vae.config.scaling_factor
        return self.vae.decode(lat.to(self.vae.dtype)).sample
    
    def _pack(self, x):
        """Pack latent: [B, C, H, W] -> [B, H*W/4, C*4]"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // 2) * (W // 2), C * 4)
    
    def _unpack(self, x, H, W):
        """Unpack transformer output"""
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)
    
    def _img_ids(self, H, W, device, dtype):
        """Generate image position IDs"""
        h, w = H // 2, W // 2
        ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
        return ids.reshape(h * w, 3)
    
    def forward(self, noisy, lr_lat, lr_pixel, timestep, guidance=3.5, conditioning_scale=None):
        """
        Forward pass: predict velocity
        
        Args:
            noisy: 当前 noisy latent
            lr_lat: LR 图像的 latent
            lr_pixel: LR 图像的 pixel tensor
            timestep: 🌟 官方格式的 timestep（已经 / 1000）
            guidance: CFG guidance scale
            conditioning_scale: 🌟 ControlNet 条件强度（官方参数）
        """
        if conditioning_scale is None:
            conditioning_scale = self.conditioning_scale
            
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel)
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Fuse: lr_lat + pixel features
        fused_cond = (lr_lat + self.pixel_weight * pixel_feat).to(dtype)
        
        # Pack
        noisy_packed = self._pack(noisy.to(dtype))
        fused_packed = self._pack(fused_cond)
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Text embeddings
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        text_ids = self._cached_embeds['text_ids']
        
        # 🌟 timestep 已经是 / 1000 后的值，直接使用
        t_input = timestep.to(dtype) if isinstance(timestep, torch.Tensor) else torch.tensor([timestep], device=device, dtype=dtype).expand(B)
        guidance_tensor = torch.full((B,), guidance, device=device, dtype=dtype)
        
        # ControlNet（🌟 添加 conditioning_scale）
        ctrl_out = self.controlnet(
            hidden_states=noisy_packed,
            controlnet_cond=fused_packed,
            conditioning_scale=conditioning_scale,  # 🌟 官方参数
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
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, strength=0.7,
                  conditioning_scale=None, control_guidance_start=0.0, control_guidance_end=1.0):
        """
        使用官方 Scheduler 的推理
        
        Args:
            strength: 官方 img2img 语义 (1.0 = 从纯噪声, 0.7 = 跳过前 30%)
            conditioning_scale: ControlNet 条件强度
            control_guidance_start: ControlNet 开始生效的比例 (0.0 = 从头开始)
            control_guidance_end: ControlNet 结束生效的比例 (1.0 = 到最后)
        """
        if conditioning_scale is None:
            conditioning_scale = self.conditioning_scale
            
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # 🌟 设置 timesteps（严格照搬官方 get_timesteps 逻辑）
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 官方 get_timesteps 逻辑
        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = timesteps[t_start * self.scheduler.order:]
        num_inference_steps = len(timesteps)
        
        # 🌟 设置 scheduler 的 begin_index（如果支持）
        if hasattr(self.scheduler, 'set_begin_index'):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        
        # 生成噪声
        noise = torch.randn_like(lr_lat)
        
        # 🌟 使用官方 scale_noise 加噪
        latents = self.scheduler.scale_noise(lr_lat, timesteps[0], noise)
        
        # 🌟 计算 controlnet_keep（官方 control_guidance_start/end 逻辑）
        controlnet_keep = []
        for i, t in enumerate(timesteps):
            keeps = 1.0 - float(i / len(timesteps) < control_guidance_start or 
                                (i + 1) / len(timesteps) > control_guidance_end)
            controlnet_keep.append(keeps)
        
        # 去噪循环
        for i, t in enumerate(timesteps):
            # 🌟 当前步的 ControlNet 强度
            cond_scale = conditioning_scale * controlnet_keep[i]
            
            # 🌟 传给模型的 timestep 需要 / 1000
            timestep_model = t / 1000.0
            
            # 预测 velocity
            model_output = self.forward(
                latents, lr_lat, lr_pixel, timestep_model, 
                guidance=guidance, 
                conditioning_scale=cond_scale
            )
            
            # 🌟 使用官方 scheduler.step 更新
            latents = self.scheduler.step(model_output, t, latents, return_dict=False)[0]
        
        return latents
    
    def get_trainable_params(self):
        params = list(self.pixel_extractor.parameters())
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return params


# ============================================================================
# Training Functions - 对齐官方 Scheduler
# ============================================================================

def compute_flow_matching_loss(system, hr_lat, lr_lat, lr_pixel, guidance=3.5,
                                conditioning_scale=1.0, num_train_timesteps=1000):
    """
    Flow Matching Loss - 对齐官方 Scheduler 的 timestep 分布
    
    🌟 改进：从 scheduler.sigmas 采样，而不是均匀采样
    这样训练时 timestep 分布更接近推理时的分布
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    unwrapped = system.module if hasattr(system, 'module') else system
    
    # 🌟 从 scheduler.sigmas 采样（如果 scheduler 已初始化）
    if hasattr(unwrapped, 'scheduler') and unwrapped.scheduler is not None:
        # 设置一个合理的 num_steps 来获取 sigmas
        unwrapped.scheduler.set_timesteps(num_train_timesteps, device=device)
        sigmas = unwrapped.scheduler.sigmas[:-1]  # 排除最后的 0
        
        # 随机选择 B 个 sigma
        indices = torch.randint(0, len(sigmas), (B,), device=device)
        sigma = sigmas[indices].to(dtype)
    else:
        # 回退到均匀采样
        sigma = torch.rand(B, device=device, dtype=dtype)
    
    noise = torch.randn_like(hr_lat)
    
    # 🌟 官方加噪: noisy = sigma * noise + (1 - sigma) * sample
    sigma_expand = sigma.view(B, 1, 1, 1)
    noisy = sigma_expand * noise + (1 - sigma_expand) * hr_lat
    
    # 目标: v = noise - hr_lat（flow matching velocity）
    target_v = noise - hr_lat
    
    # 🌟 传给模型 sigma，同时传入 guidance 和 conditioning_scale
    v_pred = unwrapped.forward(
        noisy, lr_lat, lr_pixel, sigma,
        guidance=guidance,
        conditioning_scale=conditioning_scale
    )
    
    # Loss
    loss = F.mse_loss(v_pred.float(), target_v.float())
    
    return loss


def calculate_psnr(pred, target):
    pred = (pred.clamp(-1, 1) + 1) / 2
    target = (target + 1) / 2
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return (10 * torch.log10(1.0 / mse)).item()


@torch.no_grad()
def validate(system, accelerator, val_loader, device, num_samples=5, 
             num_steps=20, guidance=3.5, strength=0.7, conditioning_scale=1.0,
             control_guidance_start=0.0, control_guidance_end=1.0):
    """验证（使用官方 scheduler，传入所有参数）"""
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
        
        # 🌟 传入所有参数
        sr_lat = unwrapped.inference(
            lr_lat, lr, 
            num_steps=num_steps, 
            guidance=guidance,
            strength=strength,
            conditioning_scale=conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end
        )
        sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
    
    # 恢复训练模式
    unwrapped.pixel_extractor.train()
    if unwrapped.train_controlnet:
        unwrapped.controlnet.train()
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, 
                    conditioning_scale, strength, path):
    """保存 checkpoint（包含所有关键参数）"""
    unwrapped = accelerator.unwrap_model(system)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'pixel_weight': pixel_weight,
        'conditioning_scale': conditioning_scale,  # 🌟 新增
        'strength': strength,
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'controlnet': unwrapped.controlnet.state_dict(),
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Stream FLUX SR Training (Official Scheduler)')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_crops', type=int, default=2)
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default=None)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    
    # 🌟 ControlNet 参数（官方参数）
    parser.add_argument('--conditioning_scale', type=float, default=1.0,
                        help='ControlNet 条件强度')
    parser.add_argument('--control_guidance_start', type=float, default=0.0,
                        help='ControlNet 开始生效的比例 (0.0 = 从头开始)')
    parser.add_argument('--control_guidance_end', type=float, default=1.0,
                        help='ControlNet 结束生效的比例 (1.0 = 到最后)')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--guidance', type=float, default=3.5,
                        help='CFG guidance scale')
    
    # 🌟 Strength (官方 img2img 语义)
    parser.add_argument('--strength', type=float, default=0.7,
                        help='推理时的 strength (1.0=从纯噪声开始，0.7=跳过前30%步数)')
    parser.add_argument('--val_num_steps', type=int, default=20)
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/dual_control')
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
    
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(args.seed)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_official_res{args.resolution}_str{args.strength}_cond{args.conditioning_scale}"
    save_dir = os.path.join(args.save_dir, exp_name)
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("FLUX SR Training - 对齐官方 Diffusers 流程")
        print("=" * 70)
        print(f"\nHR Dir: {args.hr_dir}")
        print(f"LR Dir: {args.lr_dir}")
        print(f"Resolution: {args.resolution}")
        print(f"Num Crops: {args.num_crops}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Conditioning Scale: {args.conditioning_scale}")  # 🌟 新增
        print(f"Control Guidance: [{args.control_guidance_start}, {args.control_guidance_end}]")  # 🌟 新增
        print(f"Guidance: {args.guidance}")  # 🌟 新增
        print(f"Strength: {args.strength} (推理时跳过 {(1-args.strength)*100:.0f}% 步数)")
        print(f"Learning Rate: {args.lr} (PixelExtractor: {args.lr * 10})")
        print(f"Save Dir: {save_dir}")
        print("=" * 70 + "\n")
    
    # Create model（🌟 传入 conditioning_scale）
    system = DualStreamFLUXSR(
        args.model_name, device, args.pretrained_controlnet,
        train_controlnet=args.train_controlnet, 
        pixel_weight=args.pixel_weight,
        conditioning_scale=args.conditioning_scale  # 🌟 新增
    )
    
    # Enable gradient checkpointing
    if hasattr(system.transformer, 'enable_gradient_checkpointing'):
        system.transformer.enable_gradient_checkpointing()
    if hasattr(system.controlnet, 'enable_gradient_checkpointing'):
        system.controlnet.enable_gradient_checkpointing()
    
    # Optimizer groups (PixelExtractor 10x LR)
    if args.train_controlnet:
        optimizer_grouped_parameters = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": system.pixel_extractor.parameters(), "lr": args.lr * 10.0}
        ]
    
    if is_main:
        total_params = sum(p.numel() for p in system.get_trainable_params())
        print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Datasets
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
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    
    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Prepare
    system, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        system, optimizer, train_loader, lr_scheduler
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
        if args.train_controlnet and 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            unwrapped.controlnet.load_state_dict(state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Log file
    log_path = os.path.join(save_dir, 'training_log.txt')
    if is_main:
        with open(log_path, 'w') as f:
            f.write("FLUX SR Training - Official Scheduler\n")
            f.write("=" * 60 + "\n")
            f.write(f"Strength: {args.strength}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n\n")
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    
    for epoch in range(start_epoch, args.epochs):
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        if args.train_controlnet:
            unwrapped.controlnet.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        
        for batch in pbar:
            hr = batch['hr'].to(device).to(torch.bfloat16)
            lr = batch['lr'].to(device).to(torch.bfloat16)
            
            with torch.no_grad():
                hr_lat = unwrapped.encode(hr)
                lr_lat = unwrapped.encode(lr)
            
            with accelerator.accumulate(system):
                # 🌟 传入 guidance 和 conditioning_scale
                loss = compute_flow_matching_loss(
                    system, hr_lat, lr_lat, lr,
                    guidance=args.guidance,
                    conditioning_scale=args.conditioning_scale
                )
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                lr_scheduler.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'})
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation（🌟 传入所有参数）
        val_psnr = 0.0
        if val_loader and (epoch + 1) % args.val_interval == 0:
            val_psnr = validate(
                system, accelerator, val_loader, device,
                num_samples=5, 
                num_steps=args.val_num_steps, 
                guidance=args.guidance,
                strength=args.strength,
                conditioning_scale=args.conditioning_scale,
                control_guidance_start=args.control_guidance_start,
                control_guidance_end=args.control_guidance_end
            )
        
        if is_main:
            lr_current = lr_scheduler.get_last_lr()[0]
            log_line = f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, LR={lr_current:.2e}\n"
            with open(log_path, 'a') as f:
                f.write(log_line)
            
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, lr={lr_current:.2e}")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                # 🌟 传入 conditioning_scale
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.conditioning_scale, args.strength,
                               os.path.join(save_dir, 'best_model.pt'))
                print(f"  → New best PSNR: {best_psnr:.2f} dB")
            
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.conditioning_scale, args.strength,
                               os.path.join(save_dir, f'epoch{epoch+1}.pt'))
            
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
    
    # Final save
    if is_main:
        save_checkpoint(system, accelerator, args.epochs - 1, avg_loss, best_psnr,
                       args.pixel_weight, args.conditioning_scale, args.strength,
                       os.path.join(save_dir, 'final_model.pt'))
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
