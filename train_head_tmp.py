#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training (with Degradation Pipeline)
======================================================================

This is the NEW training entry.
- Supports `--degrade_mode {paired, bicubic, realesrgan}`
- `realesrgan` mode performs on-the-fly second-order degradation from HR patches
- Keeps scheduler semantics aligned with official FlowMatch usage

Typical commands:

1) Stage 1 (recommended): paired bicubic pretraining
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \
        train_dual_control.py \
        --hr_dir Data/Mix16K_HR \
        --lr_dir Data/Mix16K_LR_bicubic_X4 \
        --degrade_mode paired --scale 4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --batch_size 4 --epochs 40 --num_crops 2 --lr 1e-5 \
        --strength 1 --pixel_gate_init 4 \
        --lpips_weight 0.1 --lpips_resize 256 --lpips_apply_prob 0.1 \
        --empty_cache_steps 50

2) Stage 2 (recommended): realesrgan degradation fine-tuning
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \
        train_dual_control.py \
        --hr_dir Data/Mix16K_HR \
        --degrade_mode realesrgan --scale 4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --resume checkpoints/dual_control/<stage1_exp>/best_model.pt \
        --batch_size 4 --epochs 20 --num_crops 2 --lr 5e-6 \
        --warmup_epochs 2 \
        --strength 1 --pixel_gate_init 4 \
        --lpips_weight 0.1 --lpips_resize 256 --lpips_apply_prob 0.1 \
        --empty_cache_steps 50
"""


import os
import gc
import math
import io
import argparse
import numpy as np
from contextlib import nullcontext
from datetime import datetime
from PIL import Image, ImageFilter

# Reduce CUDA allocator fragmentation by default (can still be overridden by env).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
# Avoid DeepSpeed Triton autotune cache on NFS home dirs on Linux clusters.
if os.name == "posix":
    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        os.path.join("/tmp", os.environ.get("USER", "user"), "triton_autotune_cache"),
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedType
from tqdm import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False

if hasattr(Image, "Resampling"):
    PIL_RESAMPLING = Image.Resampling
else:
    PIL_RESAMPLING = Image


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
    def __init__(
        self,
        hr_dir,
        lr_dir=None,
        resolution=512,
        num_crops=1,
        is_val=False,
        degrade_mode='paired',
        scale=4,
        realesrgan_cfg=None,
    ):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.num_crops = num_crops
        self.scale = int(scale)
        self.is_val = is_val
        self.degrade_mode = degrade_mode
        self.realesrgan_cfg = dict(realesrgan_cfg or {})
        self._resample_map = {
            'area': PIL_RESAMPLING.BOX,
            'bilinear': PIL_RESAMPLING.BILINEAR,
            'bicubic': PIL_RESAMPLING.BICUBIC,
            'lanczos': PIL_RESAMPLING.LANCZOS,
        }
        self._resample_choices = ['area', 'bilinear', 'bicubic', 'lanczos']
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.hr_files) * self.num_crops

    @staticmethod
    def _normalize_triplet(prob_triplet):
        p = np.array(prob_triplet, dtype=np.float32)
        p = np.clip(p, 1e-6, None)
        p = p / p.sum()
        return p

    def _sample_resample(self):
        mode = np.random.choice(self._resample_choices)
        return self._resample_map[mode]
    
    def _find_lr_file(self, hr_name):
        if not self.lr_dir:
            return None
        base = os.path.splitext(hr_name)[0]
        for suffix in ['', 'x4', 'x2', '_x4', '_x2']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.lr_dir, base + suffix + ext)
                if os.path.exists(candidate):
                    return candidate
        fallback = os.path.join(self.lr_dir, hr_name)
        return fallback if os.path.exists(fallback) else None

    def _crop_hr_only(self, hr_img):
        hr_w, hr_h = hr_img.size
        crop_size = self.resolution

        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
                x = x - (x % self.scale)
                y = y - (y % self.scale)
            else:
                x = np.random.randint(0, hr_w - crop_size + 1)
                y = np.random.randint(0, hr_h - crop_size + 1)
            return hr_img.crop((x, y, x + crop_size, y + crop_size))

        return hr_img.resize((crop_size, crop_size), PIL_RESAMPLING.BICUBIC)

    def _paired_crop(self, hr_img, lr_img):
        hr_w, hr_h = hr_img.size
        lr_w, lr_h = lr_img.size
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale

        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
                x = x - (x % self.scale)
                y = y - (y % self.scale)
            else:
                lr_x = np.random.randint(0, max(1, lr_w - lr_crop_size + 1))
                lr_y = np.random.randint(0, max(1, lr_h - lr_crop_size + 1))
                x = lr_x * self.scale
                y = lr_y * self.scale

            hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
            lr_x, lr_y = x // self.scale, y // self.scale
            lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        else:
            hr_crop = hr_img.resize((crop_size, crop_size), PIL_RESAMPLING.BICUBIC)
            lr_crop = lr_img.resize((lr_crop_size, lr_crop_size), PIL_RESAMPLING.BICUBIC)

        lr_up = lr_crop.resize((crop_size, crop_size), PIL_RESAMPLING.BICUBIC)
        return hr_crop, lr_up

    def _bicubic_degrade(self, hr_crop):
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale
        lr_crop = hr_crop.resize((lr_crop_size, lr_crop_size), PIL_RESAMPLING.BICUBIC)
        return lr_crop.resize((crop_size, crop_size), PIL_RESAMPLING.BICUBIC)

    def _random_resize(self, img, resize_prob, resize_range, min_size):
        p = self._normalize_triplet(resize_prob)
        choice = int(np.random.choice(3, p=p))
        low = float(min(resize_range))
        high = float(max(resize_range))
        if choice == 0:  # up
            scale = np.random.uniform(1.0, max(1.0, high))
        elif choice == 1:  # down
            scale = np.random.uniform(min(low, 1.0), 1.0)
        else:  # keep
            scale = 1.0

        w, h = img.size
        out_w = max(int(min_size), int(round(w * scale)))
        out_h = max(int(min_size), int(round(h * scale)))
        return img.resize((out_w, out_h), self._sample_resample())

    @staticmethod
    def _clip_uint8(arr):
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)

    @staticmethod
    def _normalize_kernel(kernel):
        kernel = kernel.astype(np.float32)
        s = float(kernel.sum())
        if abs(s) < 1e-8:
            h, w = kernel.shape
            kernel[:] = 0.0
            kernel[h // 2, w // 2] = 1.0
            s = 1.0
        kernel /= s
        return kernel

    @staticmethod
    def _build_blur_kernel(kernel_size, sigma_x, sigma_y, theta, beta=2.0, plateau=False):
        """Build anisotropic blur kernels used in RealESRGAN-style degradation."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        x = np.arange(-half, half + 1, dtype=np.float32)
        xx, yy = np.meshgrid(x, x)

        ct = math.cos(float(theta))
        st = math.sin(float(theta))
        xr = xx * ct + yy * st
        yr = -xx * st + yy * ct

        sx = max(float(sigma_x), 1e-6)
        sy = max(float(sigma_y), 1e-6)
        b = max(float(beta), 1e-6)

        power = (np.abs(xr) / sx) ** b + (np.abs(yr) / sy) ** b
        if plateau:
            kernel = 1.0 / (1.0 + power)
        else:
            kernel = np.exp(-0.5 * power)
        return SRDataset._normalize_kernel(kernel)

    def _random_blur(self, img, sigma_range):
        cfg = self.realesrgan_cfg
        default_prob = np.array([0.45, 0.25, 0.15, 0.15], dtype=np.float32)  # iso/aniso/gen/plateau
        kernel_prob = np.array(cfg.get('blur_kernel_prob', default_prob), dtype=np.float32).reshape(-1)
        if kernel_prob.size != 4:
            kernel_prob = default_prob.copy()
        kernel_prob = np.clip(kernel_prob, 1e-6, None)
        kernel_prob = kernel_prob / kernel_prob.sum()
        kind = int(np.random.choice(4, p=kernel_prob))

        sigma_min = float(min(sigma_range))
        sigma_max = float(max(sigma_range))
        sigma_x = float(np.random.uniform(sigma_min, sigma_max))
        sigma_y = float(np.random.uniform(sigma_min, sigma_max))
        theta = float(np.random.uniform(0.0, np.pi))

        if kind == 0:
            sigma = float(np.random.uniform(sigma_min, sigma_max))
            return img.filter(ImageFilter.GaussianBlur(radius=max(0.0, sigma)))

        ksize_min, ksize_max = cfg.get('blur_kernel_size_range', [7, 21])
        odd_sizes = [k for k in range(int(ksize_min), int(ksize_max) + 1) if k % 2 == 1]
        if not odd_sizes:
            odd_sizes = [7, 9, 11, 13, 15, 17, 19, 21]
        kernel_size = int(np.random.choice(odd_sizes))

        if kind == 1:  # anisotropic gaussian
            kernel = self._build_blur_kernel(kernel_size, sigma_x, sigma_y, theta, beta=2.0, plateau=False)
        elif kind == 2:  # generalized gaussian
            beta_min, beta_max = cfg.get('blur_gen_beta_range', [0.5, 4.0])
            beta = float(np.random.uniform(float(beta_min), float(beta_max)))
            kernel = self._build_blur_kernel(kernel_size, sigma_x, sigma_y, theta, beta=beta, plateau=False)
        else:  # plateau-shaped
            beta_min, beta_max = cfg.get('blur_plateau_beta_range', [1.0, 2.0])
            beta = float(np.random.uniform(float(beta_min), float(beta_max)))
            kernel = self._build_blur_kernel(kernel_size, sigma_x, sigma_y, theta, beta=beta, plateau=True)
        return self._apply_kernel(img, kernel)

    @staticmethod
    def _build_sinc_kernel(cutoff, kernel_size):
        """Build circular low-pass sinc kernel (Bessel form when available)."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2
        x = np.arange(-half, half + 1, dtype=np.float32)
        xx, yy = np.meshgrid(x, x)
        rr = np.sqrt(xx ** 2 + yy ** 2).astype(np.float32)
        wc = float(cutoff)

        if hasattr(torch.special, "bessel_j1"):
            r_t = torch.from_numpy(rr)
            z = wc * r_t
            kernel_t = torch.empty_like(z, dtype=torch.float32)
            nz = z != 0
            kernel_t[nz] = (wc * torch.special.bessel_j1(z[nz])) / (2.0 * np.pi * z[nz])
            kernel_t[~nz] = (wc * wc) / (4.0 * np.pi)
            kernel = kernel_t.numpy()
        else:
            # Fallback circular sinc (radial); still isotropic unlike separable outer-product version.
            kernel = np.sinc((wc / np.pi) * rr).astype(np.float32)
            kernel[half, half] = 1.0

        return SRDataset._normalize_kernel(kernel)

    @staticmethod
    def _apply_kernel(img, kernel):
        arr = np.array(img).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        k = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        pad = int(kernel.shape[0] // 2)
        t = F.pad(t, (pad, pad, pad, pad), mode='reflect')
        k = k.repeat(3, 1, 1, 1)
        out = F.conv2d(t, k, groups=3).squeeze(0).permute(1, 2, 0).numpy()
        out = np.clip(out, 0.0, 1.0)
        return Image.fromarray(np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8), mode='RGB')

    def _random_noise(self, img, gaussian_prob, sigma_range, poisson_scale_range, gray_prob):
        arr = np.array(img).astype(np.float32) / 255.0
        use_gray = np.random.rand() < float(gray_prob)

        if np.random.rand() < float(gaussian_prob):
            sigma = float(np.random.uniform(float(sigma_range[0]), float(sigma_range[1]))) / 255.0
            if use_gray:
                noise = np.random.randn(arr.shape[0], arr.shape[1], 1).astype(np.float32) * sigma
                noise = np.repeat(noise, 3, axis=2)
            else:
                noise = np.random.randn(*arr.shape).astype(np.float32) * sigma
            arr = np.clip(arr + noise, 0.0, 1.0)
        else:
            scale = float(np.random.uniform(float(poisson_scale_range[0]), float(poisson_scale_range[1])))
            scale = max(scale, 1e-6)
            if use_gray:
                gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
                noisy_gray = np.random.poisson(np.clip(gray, 0.0, 1.0) * 255.0 * scale).astype(np.float32)
                noisy_gray = noisy_gray / (255.0 * scale)
                noise = (noisy_gray - gray).astype(np.float32)
                noise = np.repeat(noise[..., None], 3, axis=2)
                arr = arr + noise
            else:
                arr = np.random.poisson(np.clip(arr, 0.0, 1.0) * 255.0 * scale).astype(np.float32) / (255.0 * scale)
            arr = np.clip(arr, 0.0, 1.0)

        return Image.fromarray(self._clip_uint8(arr * 255.0), mode='RGB')

    @staticmethod
    def _jpeg_compress(img, jpeg_range):
        q_min = int(min(jpeg_range))
        q_max = int(max(jpeg_range))
        quality = int(np.random.randint(q_min, q_max + 1))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        out = Image.open(buffer).convert('RGB')
        out.load()
        buffer.close()
        return out

    def _degrade_realesrgan(self, hr_crop):
        cfg = self.realesrgan_cfg

        blur_prob = float(cfg.get('blur_prob', 0.8))
        blur_sigma = cfg.get('blur_sigma', [0.2, 3.0])
        resize_prob1 = cfg.get('resize_prob1', [0.2, 0.7, 0.1])
        resize_range1 = cfg.get('resize_range1', [0.15, 1.5])
        gaussian_noise_prob1 = float(cfg.get('gaussian_noise_prob1', 0.5))
        noise_sigma1 = cfg.get('noise_sigma1', [1.0, 30.0])
        poisson_scale1 = cfg.get('poisson_scale1', [0.05, 3.0])
        gray_noise_prob1 = float(cfg.get('gray_noise_prob1', 0.4))
        jpeg_range1 = cfg.get('jpeg_range1', [30, 95])

        second_blur_prob = float(cfg.get('second_blur_prob', 0.8))
        blur_sigma2 = cfg.get('blur_sigma2', [0.2, 1.5])
        resize_prob2 = cfg.get('resize_prob2', [0.3, 0.4, 0.3])
        resize_range2 = cfg.get('resize_range2', [0.3, 1.2])
        gaussian_noise_prob2 = float(cfg.get('gaussian_noise_prob2', 0.5))
        noise_sigma2 = cfg.get('noise_sigma2', [1.0, 25.0])
        poisson_scale2 = cfg.get('poisson_scale2', [0.05, 2.5])
        gray_noise_prob2 = float(cfg.get('gray_noise_prob2', 0.4))
        jpeg_range2 = cfg.get('jpeg_range2', [30, 95])
        final_sinc_prob = float(cfg.get('final_sinc_prob', 0.8))

        target_lr = self.resolution // self.scale
        lq = hr_crop

        if np.random.rand() < blur_prob:
            lq = self._random_blur(lq, blur_sigma)
        lq = self._random_resize(lq, resize_prob1, resize_range1, min_size=target_lr)
        lq = self._random_noise(lq, gaussian_noise_prob1, noise_sigma1, poisson_scale1, gray_noise_prob1)
        lq = self._jpeg_compress(lq, jpeg_range1)

        if np.random.rand() < second_blur_prob:
            lq = self._random_blur(lq, blur_sigma2)
        lq = self._random_resize(lq, resize_prob2, resize_range2, min_size=target_lr)
        lq = self._random_noise(lq, gaussian_noise_prob2, noise_sigma2, poisson_scale2, gray_noise_prob2)

        if np.random.rand() < 0.5:
            lq = lq.resize((target_lr, target_lr), self._sample_resample())
            if np.random.rand() < final_sinc_prob:
                k_size = int(np.random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
                cutoff = float(np.random.uniform(np.pi / 5.0, np.pi))
                lq = self._apply_kernel(lq, self._build_sinc_kernel(cutoff, k_size))
            lq = self._jpeg_compress(lq, jpeg_range2)
        else:
            lq = self._jpeg_compress(lq, jpeg_range2)
            lq = lq.resize((target_lr, target_lr), self._sample_resample())
            if np.random.rand() < final_sinc_prob:
                k_size = int(np.random.choice([7, 9, 11, 13, 15, 17, 19, 21]))
                cutoff = float(np.random.uniform(np.pi / 5.0, np.pi))
                lq = self._apply_kernel(lq, self._build_sinc_kernel(cutoff, k_size))

        lq = Image.fromarray(self._clip_uint8(np.array(lq).astype(np.float32)), mode='RGB')
        return lq.resize((self.resolution, self.resolution), PIL_RESAMPLING.BICUBIC)
    
    def __getitem__(self, idx):
        img_idx = idx // self.num_crops
        hr_name = self.hr_files[img_idx]
        
        hr_img = Image.open(os.path.join(self.hr_dir, hr_name)).convert('RGB')
        lr_up = None
        if self.is_val:
            lr_path = self._find_lr_file(hr_name)
            if lr_path:
                lr_img = Image.open(lr_path).convert('RGB')
                hr_crop, lr_up = self._paired_crop(hr_img, lr_img)
            else:
                hr_crop = self._crop_hr_only(hr_img)
                lr_up = self._bicubic_degrade(hr_crop)
        else:
            if self.degrade_mode == 'realesrgan':
                hr_crop = self._crop_hr_only(hr_img)
                lr_up = self._degrade_realesrgan(hr_crop)
            elif self.degrade_mode == 'bicubic':
                hr_crop = self._crop_hr_only(hr_img)
                lr_up = self._bicubic_degrade(hr_crop)
            else:
                lr_path = self._find_lr_file(hr_name)
                if lr_path:
                    lr_img = Image.open(lr_path).convert('RGB')
                    hr_crop, lr_up = self._paired_crop(hr_img, lr_img)
                else:
                    hr_crop = self._crop_hr_only(hr_img)
                    lr_up = self._bicubic_degrade(hr_crop)
        
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
                 train_controlnet=True, pixel_weight=1.0,
                 control_guidance_start=0.0, control_guidance_end=1.0,
                 conditioning_scale=1.0):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.control_guidance_start = control_guidance_start
        self.control_guidance_end = control_guidance_end
        self.conditioning_scale = conditioning_scale
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self.pixel_fuse_proj = None
        self.pixel_gate_logit = None
        self.scheduler = None
        self._cached_embeds = None
        
        self._load_models(pretrained_controlnet)

    @staticmethod
    def _from_pretrained_with_dtype(model_cls, *args, dtype=None, **kwargs):
        """Prefer `dtype`, but fallback to `torch_dtype` for older library versions."""
        if dtype is None:
            return model_cls.from_pretrained(*args, **kwargs)
        try:
            return model_cls.from_pretrained(*args, dtype=dtype, **kwargs)
        except TypeError:
            return model_cls.from_pretrained(*args, torch_dtype=dtype, **kwargs)
    
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
        self.vae = self._from_pretrained_with_dtype(
            AutoencoderKL, self.model_name, subfolder="vae", dtype=dtype
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        
        # Cache text embeddings (空字符串) - 提前做，避免初始化显存峰值
        print(f"[Rank {local_rank}] Caching text embeddings...")
        self._cache_text_embeddings()

        # Load Transformer (frozen)
        print(f"[Rank {local_rank}] Loading FLUX Transformer...")
        self.transformer = self._from_pretrained_with_dtype(
            FluxTransformer2DModel, self.model_name, subfolder="transformer", dtype=dtype
        ).to(self.device)
        self.transformer.requires_grad_(False)
        self.transformer.eval()

        # Load ControlNet
        controlnet_path = pretrained_controlnet or "jasperai/Flux.1-dev-Controlnet-Upscaler"
        print(f"[Rank {local_rank}] Loading ControlNet from {controlnet_path}...")
        self.controlnet = self._from_pretrained_with_dtype(
            FluxControlNetModel, controlnet_path, dtype=dtype
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
        self.pixel_fuse_proj = nn.Conv2d(16, 16, kernel_size=1).to(self.device).to(dtype)
        self._reset_pixel_fuse_proj_to_identity()
        self.pixel_gate_logit = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.pixel_extractor.train()
        self.pixel_fuse_proj.train()
        
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
        text_enc = self._from_pretrained_with_dtype(
            CLIPTextModel, self.model_name, subfolder="text_encoder", dtype=dtype
        ).to(self.device)
        tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")
        
        # T5
        text_enc_2 = self._from_pretrained_with_dtype(
            T5EncoderModel, self.model_name, subfolder="text_encoder_2", dtype=dtype
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
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    def _reset_pixel_fuse_proj_to_identity(self):
        """Initialize 1x1 projection as identity so channel statistics are preserved."""
        if self.pixel_fuse_proj is None:
            return
        with torch.no_grad():
            self.pixel_fuse_proj.weight.zero_()
            self.pixel_fuse_proj.bias.zero_()
            channels = min(self.pixel_fuse_proj.out_channels, self.pixel_fuse_proj.in_channels)
            idx = torch.arange(channels, device=self.pixel_fuse_proj.weight.device)
            self.pixel_fuse_proj.weight[idx, idx, 0, 0] = 1.0

    def get_pixel_branch_params(self):
        params = list(self.pixel_extractor.parameters())
        params += list(self.pixel_fuse_proj.parameters())
        params.append(self.pixel_gate_logit)
        return params
    
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
    
    def forward(self, noisy, lr_lat, lr_pixel, timestep, guidance=3.5, controlnet_scale=1.0):
        """
        Forward pass: predict velocity
        
        Args:
            noisy: 当前 noisy latent
            lr_lat: LR 图像的 latent
            lr_pixel: LR 图像的 pixel tensor
            timestep: 🌟 官方格式的 timestep（已经 / 1000）
            guidance: CFG guidance scale
        """
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16
        
        # Pixel features
        pixel_feat = self.pixel_extractor(lr_pixel)
        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )

        # Gated fusion keeps latent prior stable while allowing pixel branch to grow when useful.
        pixel_feat = self.pixel_fuse_proj(pixel_feat.to(dtype))
        pixel_gate = torch.sigmoid(self.pixel_gate_logit).to(dtype)
        fused_cond = (lr_lat + self.pixel_weight * pixel_gate * pixel_feat).to(dtype)
        
        # Pack
        noisy_packed = self._pack(noisy.to(dtype))
        fused_packed = self._pack(fused_cond)
        img_ids = self._img_ids(H, W, device, dtype)
        
        # Text embeddings
        pooled = self._cached_embeds['pooled'].expand(B, -1)
        prompt = self._cached_embeds['prompt'].expand(B, -1, -1)
        text_ids = self._cached_embeds['text_ids']
        
        # 🌟 timestep 已经是 / 1000 后的值，直接使用
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
        # Keep guidance behavior consistent with previous V2 training path.
        controlnet_guidance = torch.full((B,), guidance, device=device, dtype=dtype)
        transformer_guidance = torch.full((B,), guidance, device=device, dtype=dtype)
        
        # ControlNet
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
        
        # Transformer
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
        
        return self._unpack(out, H, W)
    
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, strength=0.7):
        """
        使用官方 Scheduler 的推理
        
        Args:
            strength: 官方 img2img 语义
                     1.0 = 从纯噪声开始（完整去噪）
                     0.7 = 跳过前 30% 步数
        """
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16
        
        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)
        
        # 设置 timesteps（dynamic shifting 时需要 mu）
        self._set_scheduler_timesteps(num_steps, device, lr_lat)
        timesteps = self.scheduler.timesteps
        
        # 🌟 根据 strength 计算起始点（官方 img2img 方式）
        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        if len(timesteps) == 0:
            raise ValueError(
                f"No timesteps left after applying strength={strength}. "
                f"Please increase num_steps (current: {num_steps}) or strength."
            )

        # 和官方 img2img 对齐：显式设置 begin_index，再调用 scale_noise
        self.scheduler.set_begin_index(t_start)

        # 生成噪声
        noise = torch.randn_like(lr_lat)

        # 🌟 使用官方 scale_noise 加噪
        # scale_noise: sample = sigma * noise + (1 - sigma) * sample
        timestep_batch = timesteps[:1].expand(B)
        latents = self.scheduler.scale_noise(lr_lat, timestep_batch, noise)
        
        # 去噪循环
        total_steps = len(timesteps)
        for i, t in enumerate(timesteps):
            # 🌟 传给模型的 timestep 需要 / 1000
            timestep_model = t / 1000.0

            if total_steps <= 1:
                step_ratio = 1.0
            else:
                step_ratio = i / float(total_steps - 1)
            keep = self.control_guidance_start <= step_ratio <= self.control_guidance_end
            controlnet_scale = 1.0 if keep else 0.0
            
            # 预测 velocity
            model_output = self.forward(
                latents, lr_lat, lr_pixel, timestep_model, guidance,
                controlnet_scale=controlnet_scale
            )
            
            # 🌟 使用官方 scheduler.step 更新
            latents = self.scheduler.step(model_output, t, latents, return_dict=False)[0]
            del model_output
        
        return latents
    
    def get_trainable_params(self):
        params = self.get_pixel_branch_params()
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        return params


# ============================================================================
# Training Functions - 对齐官方 Scheduler
# ============================================================================

def compute_flow_matching_loss(
    system,
    hr_lat,
    lr_lat,
    lr_pixel,
    hr_pixel=None,
    guidance=3.5,
    num_train_timesteps=1000,
    lpips_model=None,
    lpips_weight=0.0,
    lpips_resize=256,
    lpips_apply_prob=0.25,
):
    """
    Flow Matching Loss - 对齐官方 Scheduler 的 timestep 格式
    
    官方 FlowMatchEulerDiscreteScheduler:
    - sigmas 从 1.0 到 ~0
    - timestep = sigma * num_train_timesteps
    - 传给模型: timestep / 1000
    
    所以训练时:
    1. 采样 sigma ~ U[0, 1]
    2. noisy = sigma * noise + (1 - sigma) * hr_lat
    3. target_v = noise - hr_lat
    4. 传给模型: sigma（因为 sigma = timestep / 1000）
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16
    
    # 采样 sigma（对应官方 scheduler.sigmas，而不是手工 U[0,1]）
    # 这样能保留 shift / dynamic shifting 等配置语义
    unwrapped = system.module if hasattr(system, 'module') else system
    unwrapped._set_scheduler_timesteps(num_train_timesteps, device, hr_lat)
    sched_sigmas = unwrapped.scheduler.sigmas[:-1] if unwrapped.scheduler.sigmas.shape[0] > 1 else unwrapped.scheduler.sigmas
    sigma_idx = torch.randint(0, sched_sigmas.shape[0], (B,), device=device)
    sigma = sched_sigmas.to(device=device, dtype=dtype)[sigma_idx]
    noise = torch.randn_like(hr_lat)
    
    # 🌟 官方加噪: noisy = sigma * noise + (1 - sigma) * sample
    sigma_expand = sigma.view(B, 1, 1, 1)
    noisy = sigma_expand * noise + (1 - sigma_expand) * hr_lat
    
    # 目标: v = noise - hr_lat（flow matching velocity）
    target_v = noise - hr_lat
    
    # 🌟 传给模型的 timestep = sigma（因为官方是 timestep / 1000，而 timestep = sigma * 1000）
    # IMPORTANT: keep forward on wrapped `system` so DDP/DeepSpeed hooks remain active.
    v_pred = system(noisy, lr_lat, lr_pixel, sigma, guidance)
    
    # Base FM loss
    loss = F.mse_loss(v_pred.float(), target_v.float())

    # Optional perceptual regularizer in pixel space.
    do_lpips = (
        lpips_model is not None
        and lpips_weight > 0
        and hr_pixel is not None
        and (lpips_apply_prob >= 1.0 or torch.rand(1, device=device).item() < lpips_apply_prob)
    )
    if do_lpips:
        pred_hr_lat = noisy - sigma_expand * v_pred
        pred_hr = unwrapped.decode(pred_hr_lat)

        pred_lp = pred_hr.float().clamp(-1, 1)
        target_lp = hr_pixel.float().clamp(-1, 1)
        if lpips_resize is not None and lpips_resize > 0:
            size = (int(lpips_resize), int(lpips_resize))
            pred_lp = F.interpolate(pred_lp, size=size, mode='bilinear', align_corners=False)
            target_lp = F.interpolate(target_lp, size=size, mode='bilinear', align_corners=False)

        lpips_term = lpips_model(pred_lp, target_lp).mean().float()
        loss = loss + float(lpips_weight) * lpips_term
        del pred_hr_lat, pred_hr, pred_lp, target_lp, lpips_term

    del target_v, noise, sigma_expand, noisy, v_pred
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
             num_steps=20, guidance=3.5, strength=0.7):
    """验证（使用官方 scheduler）"""
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.pixel_fuse_proj.eval()
    unwrapped.controlnet.eval()
    
    psnr_list = []
    
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        
        hr = batch['hr'].to(device).to(torch.bfloat16)
        lr = batch['lr'].to(device).to(torch.bfloat16)
        
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        
        sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps, guidance=guidance, strength=strength)
        sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
        del hr, lr, hr_lat, lr_lat, sr_lat, sr
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 恢复训练模式
    unwrapped.pixel_extractor.train()
    unwrapped.pixel_fuse_proj.train()
    if unwrapped.train_controlnet:
        unwrapped.controlnet.train()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return np.mean(psnr_list) if psnr_list else 0.0


def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, strength,
                    control_guidance_start, control_guidance_end, path,
                    lpips_weight=0.0, lpips_apply_prob=0.25):
    unwrapped = accelerator.unwrap_model(system)
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'pixel_weight': pixel_weight,
        'conditioning_scale': unwrapped.conditioning_scale,
        'strength': strength,
        'lpips_weight': lpips_weight,
        'lpips_apply_prob': lpips_apply_prob,
        'control_guidance_start': control_guidance_start,
        'control_guidance_end': control_guidance_end,
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'pixel_fuse_proj': unwrapped.pixel_fuse_proj.state_dict(),
        'pixel_gate_logit': unwrapped.pixel_gate_logit.detach().cpu(),
        'controlnet': unwrapped.controlnet.state_dict(),
    }, path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Stream FLUX SR Training (Official Scheduler)')
    
    # Data
    parser.add_argument('--hr_dir', type=str, required=True)
    parser.add_argument('--lr_dir', type=str, default=None,
                        help='Paired LR dir. Required when --degrade_mode=paired.')
    parser.add_argument('--val_hr_dir', type=str, default=None)
    parser.add_argument('--val_lr_dir', type=str, default=None)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_crops', type=int, default=2)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--degrade_mode', type=str, default='paired', choices=['paired', 'bicubic', 'realesrgan'],
                        help='Train-time degradation mode. Validation still uses paired LR when val_lr_dir is set.')

    # Real-ESRGAN-style second-order degradation
    parser.add_argument('--realesrgan_blur_prob', type=float, default=0.8)
    parser.add_argument('--realesrgan_second_blur_prob', type=float, default=0.8)
    parser.add_argument('--realesrgan_gaussian_noise_prob1', type=float, default=0.5)
    parser.add_argument('--realesrgan_gaussian_noise_prob2', type=float, default=0.5)
    parser.add_argument('--realesrgan_gray_noise_prob1', type=float, default=0.4)
    parser.add_argument('--realesrgan_gray_noise_prob2', type=float, default=0.4)
    parser.add_argument('--realesrgan_final_sinc_prob', type=float, default=0.8)
    parser.add_argument('--realesrgan_blur_sigma1', type=float, nargs=2, default=[0.2, 3.0])
    parser.add_argument('--realesrgan_blur_sigma2', type=float, nargs=2, default=[0.2, 1.5])
    parser.add_argument('--realesrgan_resize_prob1', type=float, nargs=3, default=[0.2, 0.7, 0.1])
    parser.add_argument('--realesrgan_resize_prob2', type=float, nargs=3, default=[0.3, 0.4, 0.3])
    parser.add_argument('--realesrgan_resize_range1', type=float, nargs=2, default=[0.15, 1.5])
    parser.add_argument('--realesrgan_resize_range2', type=float, nargs=2, default=[0.3, 1.2])
    parser.add_argument('--realesrgan_noise_sigma1', type=float, nargs=2, default=[1.0, 30.0])
    parser.add_argument('--realesrgan_noise_sigma2', type=float, nargs=2, default=[1.0, 25.0])
    parser.add_argument('--realesrgan_poisson_scale1', type=float, nargs=2, default=[0.05, 3.0])
    parser.add_argument('--realesrgan_poisson_scale2', type=float, nargs=2, default=[0.05, 2.5])
    parser.add_argument('--realesrgan_jpeg_range1', type=int, nargs=2, default=[30, 95])
    parser.add_argument('--realesrgan_jpeg_range2', type=int, nargs=2, default=[30, 95])
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default=None)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--conditioning_scale', type=float, default=1.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=1.0)
    parser.add_argument('--pixel_gate_init', type=float, default=6.0,
                        help='Initial logit for pixel gate (sigmoid(logit) is initial gate value)')
    parser.add_argument('--lpips_weight', type=float, default=0.0,
                        help='Optional LPIPS loss weight in training')
    parser.add_argument('--lpips_resize', type=int, default=256,
                        help='Resize for LPIPS loss (0 = use original training resolution)')
    parser.add_argument('--lpips_apply_prob', type=float, default=0.25,
                        help='Probability of applying LPIPS loss on each train step')
    parser.add_argument('--empty_cache_steps', type=int, default=0,
                        help='Call gc/empty_cache every N training steps (0 to disable)')
    
    # Validation/eval start point (img2img-style interpolation from LR + noise)
    parser.add_argument('--strength', type=float, default=1,
                        help='Validation/eval strength (1.0 = pure noise start, 0.8 = skip first 20% steps)')
    parser.add_argument('--val_num_steps', type=int, default=10)
    
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

    if args.scale < 1:
        parser.error("--scale must be >= 1")
    if args.resolution % args.scale != 0:
        parser.error("--resolution must be divisible by --scale")
    if args.degrade_mode == 'paired' and not args.lr_dir:
        parser.error("--lr_dir is required when --degrade_mode is 'paired'")
    
    if args.freeze_controlnet:
        args.train_controlnet = False
    
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(args.seed)
    if is_main and args.degrade_mode == 'realesrgan' and not args.resume:
        print(
            "[Warning] degrade_mode=realesrgan without --resume. "
            "Recommended 2-stage workflow: pretrain with paired, then resume for realesrgan fine-tuning."
        )
    triton_cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if triton_cache_dir:
        try:
            os.makedirs(triton_cache_dir, exist_ok=True)
        except OSError:
            pass
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _fmt_tag(v):
        s = f"{v:g}" if isinstance(v, float) else str(v)
        return s.replace("-", "m").replace(".", "p")

    exp_name = (
        f"{timestamp}"
        f"_deg{args.degrade_mode}"
        f"_x{args.scale}"
        f"_str{_fmt_tag(args.strength)}"
        f"_pw{_fmt_tag(args.pixel_weight)}"
        f"_gate{_fmt_tag(args.pixel_gate_init)}"
        f"_lpw{_fmt_tag(args.lpips_weight)}"
        f"_lpp{_fmt_tag(args.lpips_apply_prob)}"
        f"_crop{args.num_crops}"
    )
    save_dir = os.path.join(args.save_dir, exp_name)
    
    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "=" * 70)
        print("FLUX SR Training - 对齐官方 Diffusers 流程")
        print("=" * 70)
        print(f"\nHR Dir: {args.hr_dir}")
        print(f"LR Dir: {args.lr_dir}")
        print(f"Degrade Mode: {args.degrade_mode}")
        print(f"Scale: x{args.scale}")
        print(f"Resolution: {args.resolution}")
        print(f"Num Crops: {args.num_crops}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Pixel Weight: {args.pixel_weight}")
        print(f"Pixel Gate Init (logit): {args.pixel_gate_init}")
        print(f"Pixel Gate Init (sigmoid): {1.0 / (1.0 + math.exp(-args.pixel_gate_init)):.4f}")
        print(f"Conditioning Scale: {args.conditioning_scale}")
        print(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})")
        print(f"Empty Cache Steps: {args.empty_cache_steps}")
        if triton_cache_dir:
            print(f"TRITON_CACHE_DIR: {triton_cache_dir}")
        print(f"Strength: {args.strength} (推理时跳过 {(1-args.strength)*100:.0f}% 步数)")
        print(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]")
        print(f"Learning Rate: {args.lr} (Pixel branch: {args.lr * 10})")
        if args.degrade_mode == 'realesrgan':
            print(
                f"RealESRGAN Degrade: blur_p={args.realesrgan_blur_prob}, "
                f"second_blur_p={args.realesrgan_second_blur_prob}, "
                f"final_sinc_p={args.realesrgan_final_sinc_prob}"
            )
        print(f"Save Dir: {save_dir}")
        print("=" * 70 + "\n")
    
    # Create model
    system = DualStreamFLUXSR(
        args.model_name, device, args.pretrained_controlnet,
        train_controlnet=args.train_controlnet, pixel_weight=args.pixel_weight,
        control_guidance_start=args.control_guidance_start,
        control_guidance_end=args.control_guidance_end,
        conditioning_scale=args.conditioning_scale,
    )
    with torch.no_grad():
        system.pixel_gate_logit.fill_(args.pixel_gate_init)

    lpips_model = None
    if args.lpips_weight > 0:
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package is required when --lpips_weight > 0. Install with: pip install lpips"
            )
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        lpips_model.requires_grad_(False)
        if is_main:
            print(
                f"[Loss] LPIPS enabled: weight={args.lpips_weight}, "
                f"resize={args.lpips_resize}, prob={args.lpips_apply_prob}"
            )
    
    # Enable gradient checkpointing
    if hasattr(system.transformer, 'enable_gradient_checkpointing'):
        system.transformer.enable_gradient_checkpointing()
    if hasattr(system.controlnet, 'enable_gradient_checkpointing'):
        system.controlnet.enable_gradient_checkpointing()
    
    # Optimizer groups (pixel branch 10x LR)
    pixel_branch_params = system.get_pixel_branch_params()
    if args.train_controlnet:
        optimizer_grouped_parameters = [
            {"params": system.controlnet.parameters(), "lr": args.lr},
            {"params": pixel_branch_params, "lr": args.lr * 10.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {"params": pixel_branch_params, "lr": args.lr * 10.0}
        ]
    
    if is_main:
        total_params = sum(p.numel() for p in system.get_trainable_params())
        print(f"[Training] Trainable parameters: {total_params:,}")
    
    # Datasets
    realesrgan_cfg = {
        'blur_prob': args.realesrgan_blur_prob,
        'second_blur_prob': args.realesrgan_second_blur_prob,
        'gaussian_noise_prob1': args.realesrgan_gaussian_noise_prob1,
        'gaussian_noise_prob2': args.realesrgan_gaussian_noise_prob2,
        'gray_noise_prob1': args.realesrgan_gray_noise_prob1,
        'gray_noise_prob2': args.realesrgan_gray_noise_prob2,
        'final_sinc_prob': args.realesrgan_final_sinc_prob,
        'blur_sigma': args.realesrgan_blur_sigma1,
        'blur_sigma2': args.realesrgan_blur_sigma2,
        'resize_prob1': args.realesrgan_resize_prob1,
        'resize_prob2': args.realesrgan_resize_prob2,
        'resize_range1': args.realesrgan_resize_range1,
        'resize_range2': args.realesrgan_resize_range2,
        'noise_sigma1': args.realesrgan_noise_sigma1,
        'noise_sigma2': args.realesrgan_noise_sigma2,
        'poisson_scale1': args.realesrgan_poisson_scale1,
        'poisson_scale2': args.realesrgan_poisson_scale2,
        'jpeg_range1': args.realesrgan_jpeg_range1,
        'jpeg_range2': args.realesrgan_jpeg_range2,
    }
    train_lr_dir = args.lr_dir if args.degrade_mode == 'paired' else None
    train_dataset = SRDataset(
        args.hr_dir, train_lr_dir, args.resolution,
        num_crops=args.num_crops, is_val=False,
        degrade_mode=args.degrade_mode, scale=args.scale,
        realesrgan_cfg=realesrgan_cfg,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = None
    if args.val_hr_dir:
        val_dataset = SRDataset(
            args.val_hr_dir, args.val_lr_dir, args.resolution,
            num_crops=1, is_val=True, degrade_mode='paired', scale=args.scale
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        if is_main:
            if not args.val_lr_dir:
                print("[Data] val_lr_dir not provided: validation LR will use bicubic degradation from HR.")
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

    # DeepSpeed ZeRO-2 与 accelerate.accumulate(no_sync) 不兼容
    # 这种情况下交给 DeepSpeed 自己管理 grad accumulation
    use_accumulate = accelerator.distributed_type != DistributedType.DEEPSPEED
    if is_main and not use_accumulate:
        print("[Training] DeepSpeed detected: disable accelerator.accumulate() to avoid no_sync assertion.")
    
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
        if 'pixel_fuse_proj' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_fuse_proj'].items()}
            unwrapped.pixel_fuse_proj.load_state_dict(state)
        if 'pixel_gate_logit' in ckpt:
            gate = ckpt['pixel_gate_logit']
            if isinstance(gate, torch.Tensor):
                gate = gate.to(device=unwrapped.pixel_gate_logit.device, dtype=unwrapped.pixel_gate_logit.dtype)
                unwrapped.pixel_gate_logit.data.copy_(gate)
            else:
                unwrapped.pixel_gate_logit.data.fill_(float(gate))
        elif 'pixel_fuse_proj' not in ckpt:
            # Backward-compatible resume for older checkpoints (legacy direct-add behavior).
            with torch.no_grad():
                unwrapped._reset_pixel_fuse_proj_to_identity()
                unwrapped.pixel_gate_logit.fill_(6.0)
            if is_main:
                print("[Resume] Legacy checkpoint detected: initialize gated fusion to near-identity.")
        elif is_main:
            print(
                "[Resume] pixel_gate_logit missing in checkpoint; "
                f"keep current --pixel_gate_init value ({args.pixel_gate_init})."
            )
        if args.train_controlnet and 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            unwrapped.controlnet.load_state_dict(state)
        
        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        if 'control_guidance_start' in ckpt:
            args.control_guidance_start = ckpt['control_guidance_start']
            unwrapped.control_guidance_start = ckpt['control_guidance_start']
        if 'control_guidance_end' in ckpt:
            args.control_guidance_end = ckpt['control_guidance_end']
            unwrapped.control_guidance_end = ckpt['control_guidance_end']
        if 'conditioning_scale' in ckpt:
            unwrapped.conditioning_scale = ckpt['conditioning_scale']
        if is_main:
            print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}")
    
    # Log file
    log_path = os.path.join(save_dir, 'training_log.txt')
    if is_main:
        with open(log_path, 'w') as f:
            f.write("FLUX SR Training - Official Scheduler\n")
            f.write("=" * 60 + "\n")
            f.write(f"Degrade Mode: {args.degrade_mode}\n")
            f.write(f"Scale: x{args.scale}\n")
            f.write(f"Strength: {args.strength}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            f.write(f"Pixel Gate Init (logit): {args.pixel_gate_init}\n")
            f.write(f"Pixel Gate Init (sigmoid): {1.0 / (1.0 + math.exp(-args.pixel_gate_init)):.6f}\n")
            f.write(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})\n\n")
            if args.degrade_mode == 'realesrgan':
                f.write(
                    "RealESRGAN Degrade: "
                    f"blur_p={args.realesrgan_blur_prob}, "
                    f"second_blur_p={args.realesrgan_second_blur_prob}, "
                    f"final_sinc_p={args.realesrgan_final_sinc_prob}\n"
                )
                f.write(
                    "RealESRGAN Noise/JPEG: "
                    f"n1={args.realesrgan_noise_sigma1}, n2={args.realesrgan_noise_sigma2}, "
                    f"j1={args.realesrgan_jpeg_range1}, j2={args.realesrgan_jpeg_range2}\n\n"
                )
            f.write(f"Empty Cache Steps: {args.empty_cache_steps}\n\n")
            f.write(f"Conditioning Scale: {args.conditioning_scale}\n")
            f.write(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]\n\n")
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        unwrapped.pixel_fuse_proj.train()
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
            
            accumulate_ctx = accelerator.accumulate(system) if use_accumulate else nullcontext()
            with accumulate_ctx:
                loss = compute_flow_matching_loss(
                    system,
                    hr_lat,
                    lr_lat,
                    lr,
                    hr_pixel=hr,
                    guidance=args.guidance,
                    lpips_model=lpips_model,
                    lpips_weight=args.lpips_weight,
                    lpips_resize=args.lpips_resize,
                    lpips_apply_prob=args.lpips_apply_prob,
                )
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if (use_accumulate and accelerator.sync_gradients) or (not use_accumulate):
                lr_scheduler.step()
            
            loss_item = loss.item()
            epoch_losses.append(loss_item)
            pbar.set_postfix({'loss': f'{loss_item:.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'})
            global_step += 1

            del hr, lr, hr_lat, lr_lat, loss
            if args.empty_cache_steps > 0 and (global_step % args.empty_cache_steps == 0):
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        val_psnr = 0.0
        if val_loader and (epoch + 1) % args.val_interval == 0:
            if is_main:
                val_psnr = validate(system, accelerator, val_loader, device,
                                   num_samples=5, num_steps=args.val_num_steps,
                                   guidance=args.guidance, strength=args.strength)
        
        if is_main:
            lr_current = lr_scheduler.get_last_lr()[0]
            gate_value = torch.sigmoid(unwrapped.pixel_gate_logit.detach().float()).item()
            log_line = (
                f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, "
                f"LR={lr_current:.2e}, Gate={gate_value:.4f}\n"
            )
            with open(log_path, 'a') as f:
                f.write(log_line)
            
            print(
                f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, "
                f"lr={lr_current:.2e}, gate={gate_value:.4f}"
            )
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.strength,
                               args.control_guidance_start, args.control_guidance_end,
                               os.path.join(save_dir, 'best_model.pt'),
                               lpips_weight=args.lpips_weight,
                               lpips_apply_prob=args.lpips_apply_prob)
                print(f"  → New best PSNR: {best_psnr:.2f} dB")
            
            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.strength,
                               args.control_guidance_start, args.control_guidance_end,
                               os.path.join(save_dir, f'epoch{epoch+1}.pt'),
                               lpips_weight=args.lpips_weight,
                               lpips_apply_prob=args.lpips_apply_prob)
            
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
    
    # Final save
    if is_main:
        save_checkpoint(system, accelerator, args.epochs - 1, avg_loss, best_psnr,
                       args.pixel_weight, args.strength,
                       args.control_guidance_start, args.control_guidance_end,
                       os.path.join(save_dir, 'final_model.pt'),
                       lpips_weight=args.lpips_weight,
                       lpips_apply_prob=args.lpips_apply_prob)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
