#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training (with Degradation Pipeline)
======================================================================

This is the NEW training entry.
- Supports `--degrade_mode {paired, bicubic, realesrgan}`
- `realesrgan` mode performs on-the-fly second-order degradation from HR patches
- Keeps scheduler semantics aligned with official FlowMatch usage
- FLUX LoRA training is enabled by default

Typical commands (DF2K → Mix16K workflow):

    NOTE on resume semantics:
    - default `--epochs_mode absolute` (backward compatible): --epochs is target absolute epoch.
    - `--epochs_mode stage`: --epochs means "epochs to run in THIS stage".
    lr_scheduler and optimizer restart fresh on resume by default (see
    --resume_optimizer / --resume_lr_scheduler to override).

1) Stage 1: paired bicubic pretraining on DF2K
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \
        train_dual_control.py \
        --hr_dir Data/DF2K_HR \
        --lr_dir Data/DF2K_LR_bicubic_X4 \
        --degrade_mode paired --scale 4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --batch_size 4 --epochs 40 --num_crops 2 --lr 1e-5 \
        --warmup_epochs 5 \
        --strength 1 \
        --lpips_weight 0 \
        --empty_cache_steps 50

2) Stage 2: realesrgan degradation fine-tuning on Mix16K
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \
        train_dual_control.py \
        --hr_dir Data/Mix16K_HR \
        --lr_dir Data/Mix16K_LR_bicubic_X4 \
        --degrade_mode realesrgan --scale 4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --resume checkpoints/dual_control/<stage1_exp>/best_model.pt \
        --epochs_mode stage \
        --reset_pixel_gate \
        --controlnet_lr_scale 0.1 \
        --batch_size 4 --epochs 20 --num_crops 2 --lr 5e-6 \
        --warmup_epochs 2 \
        --strength 1 \
        --lpips_weight 0.05 --lpips_resize 256 --lpips_apply_prob 0.1 \
        --lpips_max_sigma 0.7 \
        --realesrgan_paired_prob 0.15 --realesrgan_bicubic_prob 0.10 \
        --usm_mode realesrgan --usm_weight 0.3 \
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

# PEFT is required because LoRA is enabled by default.
try:
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None
    PEFT_AVAILABLE = False

if hasattr(Image, "Resampling"):
    PIL_RESAMPLING = Image.Resampling
else:
    PIL_RESAMPLING = Image


# ============================================================================
# LoRA target module presets
# ============================================================================

LORA_TARGET_REGEX_OMINICONTROL = (
    r"(.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*single_transformer_blocks\.[0-9]+\.norm\.linear"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_q)"
)

LORA_TARGET_REGEX_ATTN_QKVO = (
    r"(.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.(to_q|to_k|to_v|to_out\.0)"
    r"|.*single_transformer_blocks\.[0-9]+\.norm\.linear"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.(to_q|to_k|to_v))"
)

LORA_TARGET_REGEX_QK_ONLY = (
    r"(.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_q)"
)

LORA_TARGET_PRESETS = {
    'ominicontrol': LORA_TARGET_REGEX_OMINICONTROL,
    'attn_qkvo': LORA_TARGET_REGEX_ATTN_QKVO,
    'qk_only': LORA_TARGET_REGEX_QK_ONLY,
}


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
        geom_aug=False,
        usm_mode='off',
        usm_weight=0.0,
        usm_radius=1.0,
        usm_threshold=10.0,
        usm_apply_prob=1.0,
    ):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.resolution = resolution
        self.num_crops = num_crops
        self.scale = int(scale)
        self.is_val = is_val
        self.degrade_mode = degrade_mode
        self.realesrgan_cfg = dict(realesrgan_cfg or {})
        self.geom_aug = bool(geom_aug)
        self.usm_mode = usm_mode
        self.usm_weight = float(usm_weight)
        self.usm_radius = float(usm_radius)
        self.usm_threshold = float(usm_threshold)
        self.usm_apply_prob = float(usm_apply_prob)
        self._resample_map = {
            'area': PIL_RESAMPLING.BOX,
            'bilinear': PIL_RESAMPLING.BILINEAR,
            'bicubic': PIL_RESAMPLING.BICUBIC,
            'lanczos': PIL_RESAMPLING.LANCZOS,
        }
        self._resample_choices = ['area', 'bilinear', 'bicubic', 'lanczos']
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Real-ESRGAN default sigmas ([0.2, 3.0] / [0.2, 1.5]) are calibrated for
        # 256px GT. At larger resolutions (e.g. 512), the same sigma produces
        # visually weaker blur. We multiply by (resolution / 256) unless the
        # user passes an explicit blur_sigma_scale.
        cfg_scale = float(self.realesrgan_cfg.get('blur_sigma_scale', 0.0) or 0.0)
        self._blur_sigma_scale = cfg_scale if cfg_scale > 0 else max(1.0, self.resolution / 256.0)
    
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
        s = self._blur_sigma_scale

        blur_prob = float(cfg.get('blur_prob', 0.8))
        blur_sigma_raw = cfg.get('blur_sigma', [0.2, 3.0])
        blur_sigma = [float(blur_sigma_raw[0]) * s, float(blur_sigma_raw[1]) * s]
        resize_prob1 = cfg.get('resize_prob1', [0.2, 0.7, 0.1])
        resize_range1 = cfg.get('resize_range1', [0.15, 1.5])
        gaussian_noise_prob1 = float(cfg.get('gaussian_noise_prob1', 0.5))
        noise_sigma1 = cfg.get('noise_sigma1', [1.0, 30.0])
        poisson_scale1 = cfg.get('poisson_scale1', [0.05, 3.0])
        gray_noise_prob1 = float(cfg.get('gray_noise_prob1', 0.4))
        jpeg_range1 = cfg.get('jpeg_range1', [30, 95])

        second_blur_prob = float(cfg.get('second_blur_prob', 0.8))
        blur_sigma2_raw = cfg.get('blur_sigma2', [0.2, 1.5])
        blur_sigma2 = [float(blur_sigma2_raw[0]) * s, float(blur_sigma2_raw[1]) * s]
        resize_prob2 = cfg.get('resize_prob2', [0.3, 0.4, 0.3])
        resize_range2 = cfg.get('resize_range2', [0.3, 1.2])
        gaussian_noise_prob2 = float(cfg.get('gaussian_noise_prob2', 0.5))
        noise_sigma2 = cfg.get('noise_sigma2', [1.0, 25.0])
        poisson_scale2 = cfg.get('poisson_scale2', [0.05, 2.5])
        gray_noise_prob2 = float(cfg.get('gray_noise_prob2', 0.4))
        jpeg_range2 = cfg.get('jpeg_range2', [30, 95])
        final_sinc_prob = float(cfg.get('final_sinc_prob', 0.8))

        target_lr = self.resolution // self.scale
        # Allow intermediate steps to go below target_lr (down to ~half),
        # so noise/JPEG get applied at genuinely low resolution and pick up
        # real aliasing/compression artifacts before the final upsample.
        mid_min = max(8, target_lr // 2)
        lq = hr_crop

        if np.random.rand() < blur_prob:
            lq = self._random_blur(lq, blur_sigma)
        lq = self._random_resize(lq, resize_prob1, resize_range1, min_size=mid_min)
        lq = self._random_noise(lq, gaussian_noise_prob1, noise_sigma1, poisson_scale1, gray_noise_prob1)
        lq = self._jpeg_compress(lq, jpeg_range1)

        if np.random.rand() < second_blur_prob:
            lq = self._random_blur(lq, blur_sigma2)
        lq = self._random_resize(lq, resize_prob2, resize_range2, min_size=mid_min)
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
        # Final upsample to HR resolution uses a RANDOM resample kernel so the
        # model doesn't collapse to "the upsample is always bicubic", which
        # hides most of the synthesized artifacts at test time.
        return lq.resize((self.resolution, self.resolution), self._sample_resample())

    @staticmethod
    def _random_augment(*imgs):
        """Apply identical random H-flip, V-flip, 90-degree rotation to all images."""
        hflip = np.random.rand() < 0.5
        vflip = np.random.rand() < 0.5
        rot90 = np.random.rand() < 0.5
        out = []
        for img in imgs:
            if hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if vflip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if rot90:
                img = img.transpose(Image.ROTATE_90)
            out.append(img)
        return out

    @staticmethod
    def _usm_sharp(img, weight=0.5, radius=1.0, threshold=10.0):
        """Apply mild USM sharpening to GT, tuned for SR training stability."""
        if weight <= 0:
            return img
        blur = img.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(radius))))
        arr = np.array(img).astype(np.float32)
        arr_blur = np.array(blur).astype(np.float32)
        diff = arr - arr_blur
        mask = np.abs(diff) > float(threshold)
        arr[mask] = arr[mask] + float(weight) * diff[mask]
        return Image.fromarray(np.clip(np.round(arr), 0, 255).astype(np.uint8), mode='RGB')
    
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
                mix_paired_prob = float(self.realesrgan_cfg.get('paired_prob', 0.0))
                mix_bicubic_prob = float(self.realesrgan_cfg.get('bicubic_prob', 0.0))
                mix_rand = np.random.rand()

                if mix_rand < mix_paired_prob:
                    lr_path = self._find_lr_file(hr_name)
                    if lr_path:
                        lr_img = Image.open(lr_path).convert('RGB')
                        hr_crop, lr_up = self._paired_crop(hr_img, lr_img)
                    else:
                        hr_crop = self._crop_hr_only(hr_img)
                        lr_up = self._degrade_realesrgan(hr_crop)
                elif mix_rand < (mix_paired_prob + mix_bicubic_prob):
                    hr_crop = self._crop_hr_only(hr_img)
                    lr_up = self._bicubic_degrade(hr_crop)
                else:
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

            if self.geom_aug:
                hr_crop, lr_up = self._random_augment(hr_crop, lr_up)

            usm_enabled = (
                self.usm_weight > 0
                and self.usm_apply_prob > 0
                and np.random.rand() < self.usm_apply_prob
                and (self.usm_mode == 'all' or (self.usm_mode == 'realesrgan' and self.degrade_mode == 'realesrgan'))
            )
            if usm_enabled:
                hr_crop = self._usm_sharp(
                    hr_crop,
                    weight=self.usm_weight,
                    radius=self.usm_radius,
                    threshold=self.usm_threshold,
                )
        
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
                 conditioning_scale=1.0,
                 use_lora=True, lora_rank=16, lora_alpha=16,
                 lora_dropout=0.0, lora_target_regex=LORA_TARGET_REGEX_OMINICONTROL,
                 lora_init_weights='gaussian'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.control_guidance_start = control_guidance_start
        self.control_guidance_end = control_guidance_end
        self.conditioning_scale = conditioning_scale

        # LoRA config
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_regex = lora_target_regex
        self.lora_init_weights = lora_init_weights
        
        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self.pixel_fuse_proj = None
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
        try:
            # Enable xformers on base Transformer before LoRA wrapping.
            self.transformer.enable_xformers_memory_efficient_attention()
            if local_rank == 0:
                print("[Flash] Enabled xformers memory efficient attention on Transformer")
        except Exception:
            if local_rank == 0:
                print("[Flash] Transformer: using PyTorch 2.0 SDPA")

        if self.use_lora:
            self._apply_lora_to_transformer(local_rank=local_rank)

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
        # Concat-based fusion: [lr_lat || pixel_feat] (32ch) -> 16ch via 1x1 conv.
        # Identity-init on the first 16 input channels keeps lr_lat passthrough at start;
        # the pixel_feat slice starts at 0 and the PixelFeatureExtractor's zero_conv is
        # already zero, so the fused output equals lr_lat until training learns otherwise.
        self.pixel_fuse_proj = nn.Conv2d(32, 16, kernel_size=1).to(self.device).to(dtype)
        self._reset_pixel_fuse_proj_to_identity()
        self.pixel_extractor.train()
        self.pixel_fuse_proj.train()
        
        # Enable Flash Attention
        try:
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
        """Identity-init the lr_lat passthrough and zero the pixel_feat branch.

        The 1x1 conv has in_channels = lr_lat (16) + pixel_feat (16) = 32 and
        out_channels = 16. We want the initial output to equal lr_lat, so we
        set the first 16 input channels to identity and the last 16 to zero.
        """
        if self.pixel_fuse_proj is None:
            return
        with torch.no_grad():
            self.pixel_fuse_proj.weight.zero_()
            self.pixel_fuse_proj.bias.zero_()
            out_ch = self.pixel_fuse_proj.out_channels
            in_ch = self.pixel_fuse_proj.in_channels
            passthrough = min(out_ch, in_ch // 2 if in_ch >= 2 * out_ch else in_ch)
            idx = torch.arange(passthrough, device=self.pixel_fuse_proj.weight.device)
            self.pixel_fuse_proj.weight[idx, idx, 0, 0] = 1.0

    def _apply_lora_to_transformer(self, local_rank=0):
        """Wrap Transformer with a PEFT LoRA adapter."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT is required for LoRA training. Install with: pip install peft>=0.10"
            )

        if local_rank == 0:
            print(
                f"[LoRA] Applying LoRA to Transformer: rank={self.lora_rank}, "
                f"alpha={self.lora_alpha}, dropout={self.lora_dropout}"
            )
            print(f"[LoRA] Target regex: {self.lora_target_regex}")

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_regex,
            lora_dropout=self.lora_dropout,
            bias="none",
            init_lora_weights=self.lora_init_weights,
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.train()

        # Validate injection on ALL ranks. If a regex typo matched zero modules,
        # we want every rank to raise — otherwise non-main ranks would silently
        # proceed with zero trainable params and the collective would hang
        # later in accelerator.prepare / optimizer.step.
        try:
            lora_sd = get_peft_model_state_dict(self.transformer)
            n_lora_tensors = len(lora_sd)
        except Exception as e:
            n_lora_tensors = 0
            if local_rank == 0:
                print(f"[LoRA][WARN] get_peft_model_state_dict failed: {e}")
        n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        if n_trainable == 0 or n_lora_tensors == 0:
            raise RuntimeError(
                f"[LoRA] target_modules matched zero modules "
                f"(trainable={n_trainable}, state_tensors={n_lora_tensors}). "
                f"Check --lora_target_preset / regex."
            )
        if local_rank == 0:
            print(f"[LoRA] trainable params: {n_trainable:,}, state tensors: {n_lora_tensors}")

    def _get_transformer_base(self):
        """Return base FluxTransformer2DModel regardless of PEFT wrapping."""
        if not self.use_lora:
            return self.transformer
        if hasattr(self.transformer, "get_base_model"):
            try:
                return self.transformer.get_base_model()
            except Exception:
                pass
        if hasattr(self.transformer, "base_model") and hasattr(self.transformer.base_model, "model"):
            return self.transformer.base_model.model
        return self.transformer

    def enable_transformer_gradient_checkpointing(self):
        base = self._get_transformer_base()
        if hasattr(base, "enable_gradient_checkpointing"):
            base.enable_gradient_checkpointing()
            # PEFT + gradient checkpointing: if the base model is frozen and
            # hidden_states entering a checkpointed block has requires_grad=False,
            # reentrant checkpointing can silently drop LoRA grads. Force the
            # inputs to require grad so the recomputed graph reaches LoRA params.
            if self.use_lora and hasattr(base, "enable_input_require_grads"):
                try:
                    base.enable_input_require_grads()
                except Exception:
                    pass
            return True
        return False

    def enable_controlnet_gradient_checkpointing(self):
        if hasattr(self.controlnet, "enable_gradient_checkpointing"):
            self.controlnet.enable_gradient_checkpointing()
            return True
        return False

    def get_pixel_branch_params(self):
        params = list(self.pixel_extractor.parameters())
        params += list(self.pixel_fuse_proj.parameters())
        return params

    def get_lora_params(self):
        if not self.use_lora:
            return []
        return [p for p in self.transformer.parameters() if p.requires_grad]
    
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

        # Concat fusion: [lr_lat || pixel_weight*pixel_feat] -> 1x1 conv -> 16ch.
        # Identity init (see _reset_pixel_fuse_proj_to_identity) makes the initial
        # output equal to lr_lat, after which the network learns the mixing itself.
        fused_cond = self.pixel_fuse_proj(
            torch.cat([lr_lat.to(dtype), (self.pixel_weight * pixel_feat).to(dtype)], dim=1)
        ).to(dtype)
        
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
        if self.use_lora:
            params += self.get_lora_params()
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
    lpips_max_sigma=0.7,
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
    # Only apply when sigma is low enough that the one-step reconstruction
    # (pred_hr = noisy - sigma * v_pred) is meaningful -- at high sigma the
    # decoded image is essentially noise and LPIPS contributes garbage grads.
    sigma_mean = float(sigma.float().mean().item())
    do_lpips = (
        lpips_model is not None
        and lpips_weight > 0
        and hr_pixel is not None
        and sigma_mean < float(lpips_max_sigma)
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
def validate(system, accelerator, val_loader, device, num_samples=10, 
             num_steps=20, guidance=3.5, strength=0.7, lpips_model=None):
    """验证（使用官方 scheduler）"""
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.pixel_fuse_proj.eval()
    unwrapped.controlnet.eval()
    if unwrapped.use_lora:
        unwrapped.transformer.eval()
    
    psnr_list = []
    lpips_list = []
    max_samples = None if (num_samples is None or num_samples <= 0) else int(num_samples)
    
    for i, batch in enumerate(val_loader):
        if max_samples is not None and i >= max_samples:
            break
        
        hr = batch['hr'].to(device).to(torch.bfloat16)
        lr = batch['lr'].to(device).to(torch.bfloat16)
        
        hr_lat = unwrapped.encode(hr)
        lr_lat = unwrapped.encode(lr)
        
        sr_lat = unwrapped.inference(lr_lat, lr, num_steps=num_steps, guidance=guidance, strength=strength)
        sr = unwrapped.decode(sr_lat)
        
        psnr_list.append(calculate_psnr(sr.float(), hr.float()))
        if lpips_model is not None:
            lpips_val = lpips_model(
                sr.float().clamp(-1, 1),
                hr.float().clamp(-1, 1),
            ).mean().float().item()
            lpips_list.append(lpips_val)
        del hr, lr, hr_lat, lr_lat, sr_lat, sr
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 恢复训练模式
    unwrapped.pixel_extractor.train()
    unwrapped.pixel_fuse_proj.train()
    if unwrapped.train_controlnet:
        unwrapped.controlnet.train()
    if unwrapped.use_lora:
        unwrapped.transformer.train()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'psnr': float(np.mean(psnr_list)) if psnr_list else 0.0,
        'lpips': float(np.mean(lpips_list)) if lpips_list else None,
    }
def _load_pixel_fuse_proj_with_migration(unwrapped, ckpt, is_main=False):
    """Load pixel_fuse_proj from a checkpoint, migrating legacy gated-fusion
    checkpoints to the new concat-based layout.

    Legacy layout:
        pixel_fuse_proj: Conv2d(16, 16, 1) applied to pixel_feat
        pixel_gate_logit: scalar; effective contribution = sigmoid(gate) * proj(pixel_feat)
        fused = lr_lat + pixel_weight * sigmoid(gate) * conv_old(pixel_feat)

    New layout:
        pixel_fuse_proj: Conv2d(32, 16, 1) applied to cat([lr_lat, pixel_weight*pixel_feat])
        fused = W_lr @ lr_lat + W_px @ (pixel_weight * pixel_feat) + b

    Migration preserves the learned direction:
        W_lr[i, j] = I (so lr_lat passthrough is preserved)
        W_px = sigmoid(gate) * conv_old.weight / pixel_weight_effective
        b    = sigmoid(gate) * conv_old.bias
    Since the forward multiplies pixel_feat by self.pixel_weight before the conv,
    and the legacy forward also applies self.pixel_weight, we can set W_px =
    sigmoid(gate) * conv_old.weight directly (both paths carry the same
    pixel_weight factor).
    """
    if 'pixel_fuse_proj' not in ckpt:
        if is_main:
            print("[Resume] pixel_fuse_proj missing; keeping fresh identity init.")
        return

    state = {k.replace('module.', ''): v for k, v in ckpt['pixel_fuse_proj'].items()}
    target = unwrapped.pixel_fuse_proj
    target_w = target.weight
    src_w = state.get('weight')
    src_b = state.get('bias')

    is_legacy = (
        src_w is not None
        and src_w.shape == (16, 16, 1, 1)
        and target_w.shape == (16, 32, 1, 1)
    )

    if not is_legacy and src_w is not None and src_w.shape == target_w.shape:
        target.load_state_dict(state)
        if is_main:
            fusion_type = ckpt.get('fusion_type', 'concat')
            print(f"[Resume] pixel_fuse_proj loaded (fusion_type={fusion_type}).")
        return

    if is_legacy:
        gate_raw = ckpt.get('pixel_gate_logit', None)
        if gate_raw is None:
            gate_sig = 1.0
        else:
            if isinstance(gate_raw, torch.Tensor):
                gate_val = float(gate_raw.detach().float().item())
            else:
                gate_val = float(gate_raw)
            gate_sig = 1.0 / (1.0 + math.exp(-gate_val))

        with torch.no_grad():
            target.weight.zero_()
            target.bias.zero_()
            # Identity on first 16 input channels (lr_lat passthrough).
            idx = torch.arange(16, device=target_w.device)
            target.weight[idx, idx, 0, 0] = 1.0
            # Scaled legacy conv on the last 16 input channels (pixel branch).
            legacy_w = src_w.to(device=target_w.device, dtype=target_w.dtype) * gate_sig
            target.weight[:, 16:32, :, :].copy_(legacy_w)
            if src_b is not None:
                legacy_b = src_b.to(device=target.bias.device, dtype=target.bias.dtype) * gate_sig
                target.bias.copy_(legacy_b)
        if is_main:
            print(
                f"[Resume] Legacy gated-fusion checkpoint migrated to concat layout "
                f"(sigmoid(gate)={gate_sig:.4f})."
            )
        return

    if is_main:
        print(
            f"[Resume][WARN] pixel_fuse_proj shape mismatch: "
            f"ckpt={tuple(src_w.shape) if src_w is not None else None}, "
            f"model={tuple(target_w.shape)}. Keeping fresh identity init."
        )


def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, strength,
                    control_guidance_start, control_guidance_end, path,
                    lpips_weight=0.0, lpips_apply_prob=0.25,
                    val_lpips=None, best_metric='psnr', best_metric_value=None,
                    lora_config=None,
                    optimizer=None, lr_scheduler=None, global_step=None):
    """Save training checkpoint.

    Also persists optimizer / lr_scheduler state and global_step so resume
    preserves AdamW momentum, warmup position, and cosine schedule phase.
    This matters in particular for LoRA runs, which typically rely on warmup;
    without these, resume would reset LR and momentum, wiping the adapter.
    """
    unwrapped = accelerator.unwrap_model(system)
    payload = {
        'epoch': epoch,
        'loss': loss,
        'psnr': psnr,
        'val_lpips': val_lpips,
        'best_metric': best_metric,
        'best_metric_value': best_metric_value,
        'pixel_weight': pixel_weight,
        'conditioning_scale': unwrapped.conditioning_scale,
        'strength': strength,
        'lpips_weight': lpips_weight,
        'lpips_apply_prob': lpips_apply_prob,
        'control_guidance_start': control_guidance_start,
        'control_guidance_end': control_guidance_end,
        'pixel_extractor': unwrapped.pixel_extractor.state_dict(),
        'pixel_fuse_proj': unwrapped.pixel_fuse_proj.state_dict(),
        'fusion_type': 'concat',
        'controlnet': unwrapped.controlnet.state_dict(),
        'use_lora': unwrapped.use_lora,
    }
    if unwrapped.use_lora:
        lora_state = get_peft_model_state_dict(unwrapped.transformer)
        payload['lora_state_dict'] = {k: v.detach().cpu() for k, v in lora_state.items()}
        payload['lora_config'] = lora_config or {
            'rank': unwrapped.lora_rank,
            'alpha': unwrapped.lora_alpha,
            'dropout': unwrapped.lora_dropout,
            'target_regex': unwrapped.lora_target_regex,
            'init_weights': unwrapped.lora_init_weights,
        }
        try:
            import peft as _peft
            payload['peft_version'] = getattr(_peft, '__version__', None)
        except Exception:
            pass
    if optimizer is not None:
        try:
            payload['optimizer_state_dict'] = optimizer.state_dict()
        except Exception as e:
            print(f"[Checkpoint][WARN] failed to serialize optimizer state: {e}")
    if lr_scheduler is not None:
        try:
            payload['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        except Exception as e:
            print(f"[Checkpoint][WARN] failed to serialize lr_scheduler state: {e}")
    if global_step is not None:
        payload['global_step'] = int(global_step)
    torch.save(payload, path)


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
    parser.add_argument('--realesrgan_paired_prob', type=float, default=0.0,
                        help='In realesrgan mode, probability of using paired LR when available (quality-anchor mix)')
    parser.add_argument('--realesrgan_bicubic_prob', type=float, default=0.0,
                        help='In realesrgan mode, probability of using bicubic degradation (stability mix)')
    
    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default=None)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--conditioning_scale', type=float, default=1.0)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--epochs_mode', type=str, default='absolute', choices=['absolute', 'stage'],
                        help='Epoch interpretation on resume: absolute=target global epoch (legacy), stage=run this many additional epochs.')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--control_guidance_start', type=float, default=0.0)
    parser.add_argument('--control_guidance_end', type=float, default=1.0)
    parser.add_argument('--pixel_gate_init', type=float, default=6.0,
                        help='[DEPRECATED] Legacy gated-fusion initial logit. No-op now that '
                             'fusion is concat+1x1. Kept for CLI backward compatibility.')
    parser.add_argument('--lpips_weight', type=float, default=0.0,
                        help='Optional LPIPS loss weight in training')
    parser.add_argument('--lpips_resize', type=int, default=256,
                        help='Resize for LPIPS loss (0 = use original training resolution)')
    parser.add_argument('--lpips_apply_prob', type=float, default=0.25,
                        help='Probability of applying LPIPS loss on each train step')
    parser.add_argument('--geom_aug', action='store_true',
                        help='Enable random flip/rotation augmentation during training')
    parser.add_argument('--usm_mode', type=str, default='off', choices=['off', 'realesrgan', 'all'],
                        help='USM sharpening scope for GT: off, only in realesrgan training, or all modes')
    parser.add_argument('--usm_weight', type=float, default=0.0,
                        help='USM sharpening weight (0 disables USM)')
    parser.add_argument('--usm_radius', type=float, default=1.0,
                        help='USM Gaussian blur radius (PIL radius)')
    parser.add_argument('--usm_threshold', type=float, default=10.0,
                        help='USM threshold in [0,255] space')
    parser.add_argument('--usm_apply_prob', type=float, default=1.0,
                        help='Probability to apply USM on each training sample')
    parser.add_argument('--empty_cache_steps', type=int, default=0,
                        help='Call gc/empty_cache every N training steps (0 to disable)')
    
    # Validation/eval start point (img2img-style interpolation from LR + noise)
    parser.add_argument('--strength', type=float, default=1,
                        help='Validation/eval strength (1.0 = pure noise start, 0.8 = skip first 20%% steps)')
    parser.add_argument('--val_num_steps', type=int, default=10)
    parser.add_argument('--val_num_samples', type=int, default=5,
                        help='Number of validation samples per epoch (<=0 means full validation set)')
    parser.add_argument('--val_calc_lpips', action='store_true',
                        help='Also compute LPIPS on validation set')
    parser.add_argument('--best_metric', type=str, default='psnr', choices=['psnr', 'lpips'],
                        help='Metric used for best checkpoint selection')
    parser.add_argument('--reset_best_on_resume', action='store_true',
                        help='When resuming (e.g., Stage2), reset best metric tracking instead of inheriting Stage1 best')

    # LoRA (always-on by default; keep hidden flag for backward compatibility)
    parser.add_argument('--use_lora', action='store_true', default=True, help=argparse.SUPPRESS)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_lr', type=float, default=1e-4,
                        help='Learning rate for LoRA params (typically > --lr)')
    parser.add_argument('--lora_target_preset', type=str, default='ominicontrol',
                        choices=list(LORA_TARGET_PRESETS.keys()),
                        help='Which preset of target modules regex to use')
    parser.add_argument('--lora_init_weights', type=str, default='gaussian',
                        choices=['gaussian', 'default', 'true', 'false'],
                        help='PEFT init_lora_weights option')
    parser.add_argument('--dry_run_lora', action='store_true', default=False,
                        help='Initialize model + LoRA, print matched modules and exit')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/dual_control')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true', default=False,
                        help='When resuming, also restore optimizer momentum (default: off; Stage 2 typically wants a fresh optimizer).')
    parser.add_argument('--resume_lr_scheduler', action='store_true', default=False,
                        help='When resuming, restore the LambdaLR step counter. Default OFF: the new stage gets a fresh warmup + cosine over --epochs.')
    parser.add_argument('--reset_pixel_gate', action='store_true', default=False,
                        help='On resume, reset pixel_fuse_proj back to identity init '
                             '(lr_lat passthrough, pixel branch zeroed). Useful when switching '
                             'Stage 1 -> Stage 2 to re-learn the mixing from a clean starting point.')
    parser.add_argument('--controlnet_lr_scale', type=float, default=1.0,
                        help='Multiplier applied to --lr for the ControlNet param group (use <1 for Stage 2 fine-tuning, e.g. 0.1).')

    # Degradation: resolution-aware scaling
    parser.add_argument('--blur_sigma_scale', type=float, default=0.0,
                        help='Multiplier applied to realesrgan_blur_sigma1/2. 0.0 = auto (resolution/256).')

    # LPIPS: only apply when denoising is mostly done (low sigma)
    parser.add_argument('--lpips_max_sigma', type=float, default=0.7,
                        help='Only compute LPIPS when sampled sigma < this threshold (decoded pred_hr is garbage at high sigma).')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_controlnet', action='store_true', default=True)
    parser.add_argument('--freeze_controlnet', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.scale < 1:
        parser.error("--scale must be >= 1")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.resolution % args.scale != 0:
        parser.error("--resolution must be divisible by --scale")
    if args.usm_weight < 0:
        parser.error("--usm_weight must be >= 0")
    if args.usm_radius < 0:
        parser.error("--usm_radius must be >= 0")
    if not (0.0 <= args.usm_apply_prob <= 1.0):
        parser.error("--usm_apply_prob must be within [0, 1]")
    if not (0.0 <= args.realesrgan_paired_prob <= 1.0):
        parser.error("--realesrgan_paired_prob must be within [0, 1]")
    if not (0.0 <= args.realesrgan_bicubic_prob <= 1.0):
        parser.error("--realesrgan_bicubic_prob must be within [0, 1]")
    if args.realesrgan_paired_prob + args.realesrgan_bicubic_prob > 1.0:
        parser.error("--realesrgan_paired_prob + --realesrgan_bicubic_prob must be <= 1")
    if args.controlnet_lr_scale < 0:
        parser.error("--controlnet_lr_scale must be >= 0")
    if args.blur_sigma_scale < 0:
        parser.error("--blur_sigma_scale must be >= 0")
    if args.lpips_max_sigma < 0:
        parser.error("--lpips_max_sigma must be >= 0")
    if args.degrade_mode == 'paired' and not args.lr_dir:
        parser.error("--lr_dir is required when --degrade_mode is 'paired'")
    if args.best_metric == 'lpips':
        args.val_calc_lpips = True
    if args.best_metric in ('psnr', 'lpips') and not args.val_hr_dir:
        parser.error("--val_hr_dir is required when selecting best checkpoints by validation metrics")
    
    if args.freeze_controlnet:
        args.train_controlnet = False

    lora_target_regex = LORA_TARGET_PRESETS[args.lora_target_preset]
    if args.lora_init_weights in ('true', 'false'):
        lora_init_weights = (args.lora_init_weights == 'true')
    else:
        lora_init_weights = args.lora_init_weights
    args.use_lora = True
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for LoRA training. Install with: pip install peft>=0.10"
        )
    
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(args.seed)
    if is_main and args.degrade_mode == 'realesrgan' and not args.resume:
        print(
            "[Warning] degrade_mode=realesrgan without --resume. "
            "Recommended 2-stage workflow: pretrain with paired, then resume for realesrgan fine-tuning."
        )
    if is_main and args.degrade_mode in ('paired', 'bicubic') and args.lpips_weight > 0:
        print("[Warning] LPIPS in Stage-1 style training can reduce PSNR. Consider --lpips_weight 0.")
    if is_main and args.degrade_mode in ('paired', 'bicubic') and args.usm_mode != 'off' and args.usm_weight > 0:
        print("[Warning] USM on paired/bicubic training can reduce PSNR. Consider --usm_mode off for Stage 1.")
    if is_main and args.usm_mode == 'off' and args.usm_weight > 0:
        print("[Warning] --usm_weight is set but --usm_mode=off, so USM is disabled.")
    if is_main and args.usm_mode != 'off' and args.usm_weight <= 0:
        print("[Warning] --usm_mode is enabled but --usm_weight <= 0, so USM has no effect.")
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
        f"_concat"
        f"_lpw{_fmt_tag(args.lpips_weight)}"
        f"_lpp{_fmt_tag(args.lpips_apply_prob)}"
        f"_aug{int(args.geom_aug)}"
        f"_usm{_fmt_tag(args.usm_mode)}"
        f"_usmw{_fmt_tag(args.usm_weight)}"
        f"_bm{_fmt_tag(args.best_metric)}"
        f"_crop{args.num_crops}"
    )
    if args.use_lora:
        exp_name += (
            f"_lora"
            f"_r{args.lora_rank}"
            f"_a{args.lora_alpha}"
            f"_d{_fmt_tag(args.lora_dropout)}"
            f"_llr{_fmt_tag(args.lora_lr)}"
            f"_tp{args.lora_target_preset}"
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
        print(f"Pixel Fusion: concat+1x1 conv (identity-init lr_lat passthrough)")
        print(f"Conditioning Scale: {args.conditioning_scale}")
        print(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})")
        print(f"Geometric Aug: {args.geom_aug}")
        print(
            f"USM: mode={args.usm_mode}, weight={args.usm_weight}, radius={args.usm_radius}, "
            f"threshold={args.usm_threshold}, prob={args.usm_apply_prob}"
        )
        print(f"Empty Cache Steps: {args.empty_cache_steps}")
        if triton_cache_dir:
            print(f"TRITON_CACHE_DIR: {triton_cache_dir}")
        print(f"Strength: {args.strength} (推理时跳过 {(1-args.strength)*100:.0f}% 步数)")
        print(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]")
        print(f"Validation: steps={args.val_num_steps}, samples={args.val_num_samples}, lpips={args.val_calc_lpips}")
        print(f"Best Metric: {args.best_metric}, Reset Best On Resume: {args.reset_best_on_resume}")
        print(f"Learning Rate: {args.lr} (Pixel branch: {args.lr * 10})")
        if args.use_lora:
            print(
                f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                f"dropout={args.lora_dropout}, lr={args.lora_lr}, "
                f"preset={args.lora_target_preset}"
            )
        else:
            print("LoRA: disabled")
        if args.degrade_mode == 'realesrgan':
            print(
                f"RealESRGAN Degrade: blur_p={args.realesrgan_blur_prob}, "
                f"second_blur_p={args.realesrgan_second_blur_prob}, "
                f"final_sinc_p={args.realesrgan_final_sinc_prob}"
            )
            print(
                f"RealESRGAN Mix: paired_p={args.realesrgan_paired_prob}, "
                f"bicubic_p={args.realesrgan_bicubic_prob}"
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
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_regex=lora_target_regex,
        lora_init_weights=lora_init_weights,
    )
    if args.dry_run_lora:
        if is_main:
            wrapped = []
            for n, m in system.transformer.named_modules():
                if hasattr(m, 'lora_A'):
                    try:
                        if len(m.lora_A) > 0:
                            wrapped.append(n)
                    except TypeError:
                        wrapped.append(n)
            print("\n[dry_run_lora] LoRA regex diagnostic")
            print(f"Target preset: {args.lora_target_preset}")
            print(f"Wrapped modules: {len(wrapped)}")
            for n in wrapped[:200]:
                print(f"  {n}")
            if len(wrapped) > 200:
                print(f"  ... ({len(wrapped) - 200} more)")
        raise SystemExit(0)

    if args.dry_run_lora:
        if is_main:
            wrapped = []
            for n, m in system.transformer.named_modules():
                if hasattr(m, 'lora_A'):
                    try:
                        if len(m.lora_A) > 0:
                            wrapped.append(n)
                    except TypeError:
                        wrapped.append(n)
            print("\n[dry_run_lora] LoRA regex diagnostic")
            print(f"Target preset: {args.lora_target_preset}")
            print(f"Wrapped modules: {len(wrapped)}")
            for n in wrapped[:200]:
                print(f"  {n}")
            if len(wrapped) > 200:
                print(f"  ... ({len(wrapped) - 200} more)")
        raise SystemExit(0)

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

    val_lpips_model = None
    if args.val_calc_lpips:
        if not LPIPS_AVAILABLE:
            raise ImportError(
                "lpips package is required when --val_calc_lpips is enabled. Install with: pip install lpips"
            )
        if lpips_model is not None:
            val_lpips_model = lpips_model
        else:
            val_lpips_model = lpips.LPIPS(net='alex').to(device)
            val_lpips_model.eval()
            val_lpips_model.requires_grad_(False)
        if is_main:
            print("[Val] LPIPS metric enabled for validation.")
    
    # Enable gradient checkpointing
    if system.enable_transformer_gradient_checkpointing() and is_main:
        print("[GradCkpt] Transformer gradient checkpointing enabled")
    if system.enable_controlnet_gradient_checkpointing() and is_main:
        print("[GradCkpt] ControlNet gradient checkpointing enabled")
    
    # Optimizer groups (pixel branch 10x LR; controlnet optionally scaled down for Stage 2)
    pixel_branch_params = system.get_pixel_branch_params()
    lora_params = system.get_lora_params()
    controlnet_lr = args.lr * float(args.controlnet_lr_scale)
    optimizer_grouped_parameters = []
    if args.train_controlnet:
        optimizer_grouped_parameters.append(
            {"params": system.controlnet.parameters(), "lr": controlnet_lr}
        )
    optimizer_grouped_parameters.append(
        {"params": pixel_branch_params, "lr": args.lr * 10.0}
    )
    if args.use_lora:
        if len(lora_params) == 0:
            raise RuntimeError(
                "[LoRA] no trainable LoRA params were found. "
                "Check target_modules regex vs. Transformer module names."
            )
        optimizer_grouped_parameters.append(
            {"params": lora_params, "lr": args.lora_lr}
        )
    if is_main:
        print(
            f"[Training] LR groups: controlnet={controlnet_lr:.2e} "
            f"(scale={args.controlnet_lr_scale}), pixel={args.lr * 10.0:.2e}, "
            f"lora={args.lora_lr:.2e}"
        )
    
    if is_main:
        total_params = sum(p.numel() for p in system.get_trainable_params())
        n_pixel = sum(p.numel() for p in pixel_branch_params)
        n_cn = sum(p.numel() for p in system.controlnet.parameters()) if args.train_controlnet else 0
        n_lora = sum(p.numel() for p in lora_params)
        print(
            f"[Training] Trainable params: total={total_params:,} "
            f"(controlnet={n_cn:,}, pixel={n_pixel:,}, lora={n_lora:,})"
        )
    
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
        'paired_prob': args.realesrgan_paired_prob,
        'bicubic_prob': args.realesrgan_bicubic_prob,
        'blur_sigma_scale': args.blur_sigma_scale,
    }
    # In realesrgan mode we still want lr_dir available for the paired-mix
    # branch (paired_prob / bicubic_prob). Previously we silently set it to
    # None, so --realesrgan_paired_prob had no effect.
    if args.degrade_mode == 'paired':
        train_lr_dir = args.lr_dir
    elif args.degrade_mode == 'realesrgan':
        train_lr_dir = args.lr_dir  # optional; _find_lr_file handles missing files
    else:
        train_lr_dir = None
    train_dataset = SRDataset(
        args.hr_dir, train_lr_dir, args.resolution,
        num_crops=args.num_crops, is_val=False,
        degrade_mode=args.degrade_mode, scale=args.scale,
        realesrgan_cfg=realesrgan_cfg,
        geom_aug=args.geom_aug,
        usm_mode=args.usm_mode,
        usm_weight=args.usm_weight,
        usm_radius=args.usm_radius,
        usm_threshold=args.usm_threshold,
        usm_apply_prob=args.usm_apply_prob,
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
    best_lpips = float('inf')
    
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        unwrapped = accelerator.unwrap_model(system)
        if 'pixel_extractor' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
            unwrapped.pixel_extractor.load_state_dict(state)
        _load_pixel_fuse_proj_with_migration(unwrapped, ckpt, is_main=is_main)
        if args.reset_pixel_gate:
            with torch.no_grad():
                unwrapped._reset_pixel_fuse_proj_to_identity()
            if is_main:
                print("[Resume] --reset_pixel_gate: pixel_fuse_proj reset to identity (lr_lat passthrough).")
        if args.train_controlnet and 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            unwrapped.controlnet.load_state_dict(state)

        ckpt_uses_lora = bool(ckpt.get('use_lora', False)) and ('lora_state_dict' in ckpt)
        if unwrapped.use_lora and ckpt_uses_lora:
            ckpt_cfg = ckpt.get('lora_config', {}) or {}
            cur_cfg = {
                'rank': unwrapped.lora_rank,
                'alpha': unwrapped.lora_alpha,
                'target_regex': unwrapped.lora_target_regex,
                'target_preset': args.lora_target_preset,
            }
            # Hard-check rank + alpha + regex. Alpha matters because PEFT
            # scaling = alpha / rank; silently changing alpha at resume would
            # rescale every adapter delta. target_preset is redundant with
            # target_regex but we check both for clearer error messages.
            hard_mismatch = []
            for key in ('rank', 'alpha', 'target_regex', 'target_preset'):
                if key in ckpt_cfg and ckpt_cfg[key] != cur_cfg[key]:
                    hard_mismatch.append((key, ckpt_cfg[key], cur_cfg[key]))
            if hard_mismatch:
                raise RuntimeError(
                    "[LoRA][Resume] Incompatible LoRA config:\n"
                    + "\n".join([f"  - {k}: ckpt={v1!r}, current={v2!r}" for k, v1, v2 in hard_mismatch])
                )
            if set_peft_model_state_dict is None:
                raise ImportError("PEFT is required to resume a LoRA checkpoint.")
            set_peft_model_state_dict(unwrapped.transformer, ckpt['lora_state_dict'])
            if is_main:
                print(
                    f"[Resume] Loaded LoRA adapter: {len(ckpt['lora_state_dict'])} tensors, "
                    f"config={ckpt.get('lora_config', {})}"
                )
        elif unwrapped.use_lora and not ckpt_uses_lora:
            if is_main:
                print("[Resume] Current run uses LoRA but checkpoint has no LoRA state; LoRA stays at init.")
        # Optimizer / lr_scheduler restore policy for 2-stage workflows:
        # Stage 2 typically wants a fresh warmup + fresh cosine over the new
        # --epochs (different LR, different data, different duration). So by
        # default we do NOT restore optimizer / lr_scheduler state on resume.
        # Pass --resume_optimizer / --resume_lr_scheduler to keep old behavior.
        if args.resume_optimizer and 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if is_main:
                    print("[Resume] Restored optimizer state (--resume_optimizer).")
            except Exception as e:
                if is_main:
                    print(f"[Resume][WARN] optimizer state_dict load failed ({e}); optimizer starts fresh.")
        elif is_main:
            print("[Resume] Optimizer starts fresh (pass --resume_optimizer to keep AdamW momentum).")
        if args.resume_lr_scheduler and 'lr_scheduler_state_dict' in ckpt:
            try:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
                if is_main:
                    print(f"[Resume] Restored lr_scheduler (last_lr={lr_scheduler.get_last_lr()}).")
            except Exception as e:
                if is_main:
                    print(f"[Resume][WARN] lr_scheduler state_dict load failed ({e}); scheduler starts fresh.")
        elif is_main:
            print(
                "[Resume] lr_scheduler starts fresh (warmup + cosine over --epochs of this run). "
                "Pass --resume_lr_scheduler to carry over the old cosine phase."
            )
        resumed_global_step = int(ckpt.get('global_step', 0) or 0)

        start_epoch = ckpt.get('epoch', 0) + 1
        best_psnr = ckpt.get('psnr', 0.0)
        ckpt_lpips = ckpt.get('val_lpips', None)
        if ckpt_lpips is not None:
            best_lpips = float(ckpt_lpips)
        if 'control_guidance_start' in ckpt:
            args.control_guidance_start = ckpt['control_guidance_start']
            unwrapped.control_guidance_start = ckpt['control_guidance_start']
        if 'control_guidance_end' in ckpt:
            args.control_guidance_end = ckpt['control_guidance_end']
            unwrapped.control_guidance_end = ckpt['control_guidance_end']
        if 'conditioning_scale' in ckpt:
            unwrapped.conditioning_scale = ckpt['conditioning_scale']
        if args.reset_best_on_resume:
            best_psnr = -float('inf')
            best_lpips = float('inf')
        if is_main:
            if args.reset_best_on_resume:
                print(f"[Resume] Starting from epoch {start_epoch}, best metric tracking reset.")
            else:
                lpips_msg = f", best LPIPS: {best_lpips:.4f}" if np.isfinite(best_lpips) else ""
                print(f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}{lpips_msg}")
    
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
            f.write(f"Pixel Fusion: concat+1x1 conv (identity-init lr_lat passthrough)\n")
            f.write(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})\n\n")
            f.write(f"Geometric Aug: {args.geom_aug}\n")
            f.write(
                f"USM: mode={args.usm_mode}, weight={args.usm_weight}, radius={args.usm_radius}, "
                f"threshold={args.usm_threshold}, prob={args.usm_apply_prob}\n\n"
            )
            f.write(
                f"Validation: steps={args.val_num_steps}, samples={args.val_num_samples}, "
                f"lpips={args.val_calc_lpips}\n"
            )
            f.write(
                f"Best Metric: {args.best_metric}, Reset Best On Resume: {args.reset_best_on_resume}\n\n"
            )
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
                f.write(
                    "RealESRGAN Mix: "
                    f"paired_p={args.realesrgan_paired_prob}, "
                    f"bicubic_p={args.realesrgan_bicubic_prob}\n\n"
                )
            f.write(f"Empty Cache Steps: {args.empty_cache_steps}\n\n")
            f.write(f"Conditioning Scale: {args.conditioning_scale}\n")
            f.write(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]\n")
            if args.use_lora:
                f.write(
                    f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                    f"dropout={args.lora_dropout}, lr={args.lora_lr}, "
                    f"preset={args.lora_target_preset}\n\n"
                )
            else:
                f.write("LoRA: disabled\n\n")
    
    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    # Resume global_step so empty_cache cycles and any step-indexed logic stay
    # consistent across restarts; 0 on a fresh run.
    global_step = resumed_global_step if args.resume else 0
    if args.resume and is_main:
        print(f"[Resume] global_step resumed at {global_step}")

    if args.epochs_mode == 'stage':
        # Stage semantics: run `args.epochs` more epochs from start_epoch.
        end_epoch = start_epoch + args.epochs
        if is_main:
            print(
                f"[Training] epochs_mode=stage: {start_epoch} -> {end_epoch} "
                f"(this run: {args.epochs} epochs)"
            )
    else:
        # Legacy absolute semantics: --epochs is target global end epoch.
        end_epoch = args.epochs
        if end_epoch <= start_epoch:
            raise RuntimeError(
                f"--epochs_mode=absolute but --epochs={args.epochs} <= resumed start_epoch={start_epoch}. "
                "Increase --epochs or use --epochs_mode stage."
            )
        if is_main:
            print(
                f"[Training] epochs_mode=absolute: {start_epoch} -> {end_epoch} "
                f"(target absolute epoch={args.epochs})"
            )
    for epoch in range(start_epoch, end_epoch):
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        unwrapped.pixel_fuse_proj.train()
        if args.train_controlnet:
            unwrapped.controlnet.train()
        if args.use_lora:
            unwrapped.transformer.train()
        
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch}", disable=not is_main)
        
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
                    lpips_max_sigma=args.lpips_max_sigma,
                )
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if (use_accumulate and accelerator.sync_gradients) or (not use_accumulate):
                lr_scheduler.step()
            
            loss_item = loss.item()
            epoch_losses.append(loss_item)
            postfix = {'loss': f'{loss_item:.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'}
            if args.use_lora and len(lr_scheduler.get_last_lr()) >= 2:
                postfix['lora_lr'] = f'{lr_scheduler.get_last_lr()[-1]:.2e}'
            pbar.set_postfix(postfix)
            global_step += 1

            del hr, lr, hr_lat, lr_lat, loss
            if args.empty_cache_steps > 0 and (global_step % args.empty_cache_steps == 0):
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_loss = np.mean(epoch_losses)
        
        # Validation
        val_psnr = 0.0
        val_lpips = None
        if val_loader and (epoch + 1) % args.val_interval == 0:
            if is_main:
                val_metrics = validate(
                    system, accelerator, val_loader, device,
                    num_samples=args.val_num_samples, num_steps=args.val_num_steps,
                    guidance=args.guidance, strength=args.strength,
                    lpips_model=val_lpips_model if args.val_calc_lpips else None,
                )
                val_psnr = val_metrics['psnr']
                val_lpips = val_metrics['lpips']
        
        if is_main:
            lr_current = lr_scheduler.get_last_lr()[0]
            # Diagnostic for concat fusion: frobenius norm of the pixel_feat input
            # slice (last 16 input channels) relative to the lr_lat passthrough slice.
            # Ratio >> 0 means the net has learned to use the pixel branch.
            w = unwrapped.pixel_fuse_proj.weight.detach().float()
            in_half = w.shape[1] // 2
            lr_norm = w[:, :in_half].norm().item()
            px_norm = w[:, in_half:].norm().item()
            pixel_ratio = px_norm / (lr_norm + 1e-8)
            if val_lpips is not None:
                log_line = (
                    f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, "
                    f"LPIPS={val_lpips:.4f}, LR={lr_current:.2e}, PxRatio={pixel_ratio:.4f}\n"
                )
            else:
                log_line = (
                    f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, "
                    f"LR={lr_current:.2e}, PxRatio={pixel_ratio:.4f}\n"
                )
            if args.use_lora and len(lr_scheduler.get_last_lr()) >= 2:
                log_line = log_line.rstrip("\n") + f", LoRA_LR={lr_scheduler.get_last_lr()[-1]:.2e}\n"
            with open(log_path, 'a') as f:
                f.write(log_line)
            
            if val_lpips is not None:
                print(
                    f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, "
                    f"val_lpips={val_lpips:.4f}, lr={lr_current:.2e}, gate={gate_value:.4f}"
                    + (f", lora_lr={lr_scheduler.get_last_lr()[-1]:.2e}" if args.use_lora and len(lr_scheduler.get_last_lr()) >= 2 else "")
                )
            else:
                print(
                    f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, "
                    f"lr={lr_current:.2e}, gate={gate_value:.4f}"
                    + (f", lora_lr={lr_scheduler.get_last_lr()[-1]:.2e}" if args.use_lora and len(lr_scheduler.get_last_lr()) >= 2 else "")
                )
            
            improved = False
            if args.best_metric == 'psnr':
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    improved = True
                best_value = best_psnr
            else:
                if val_lpips is not None and val_lpips < best_lpips:
                    best_lpips = val_lpips
                    improved = True
                best_value = best_lpips if np.isfinite(best_lpips) else None

            lora_cfg_for_ckpt = None
            if args.use_lora:
                lora_cfg_for_ckpt = {
                    'rank': args.lora_rank,
                    'alpha': args.lora_alpha,
                    'dropout': args.lora_dropout,
                    'target_regex': lora_target_regex,
                    'target_preset': args.lora_target_preset,
                    'init_weights': args.lora_init_weights,
                    'lora_lr': args.lora_lr,
                }

            if improved:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.strength,
                               args.control_guidance_start, args.control_guidance_end,
                               os.path.join(save_dir, 'best_model.pt'),
                               lpips_weight=args.lpips_weight,
                               lpips_apply_prob=args.lpips_apply_prob,
                               val_lpips=val_lpips,
                               best_metric=args.best_metric,
                               best_metric_value=best_value,
                               lora_config=lora_cfg_for_ckpt,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               global_step=global_step)
                if args.best_metric == 'lpips' and best_value is not None:
                    print(f"  -> New best LPIPS: {best_value:.4f}")
                else:
                    print(f"  -> New best PSNR: {best_value:.2f} dB")

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.strength,
                               args.control_guidance_start, args.control_guidance_end,
                               os.path.join(save_dir, f'epoch{epoch+1}.pt'),
                               lpips_weight=args.lpips_weight,
                               lpips_apply_prob=args.lpips_apply_prob,
                               val_lpips=val_lpips,
                               best_metric=args.best_metric,
                               best_metric_value=(val_lpips if args.best_metric == 'lpips' else val_psnr),
                               lora_config=lora_cfg_for_ckpt,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               global_step=global_step)
            
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()
    
    # Final save
    if is_main:
        if not np.isfinite(best_psnr):
            best_psnr = 0.0
        final_best_value = best_psnr if args.best_metric == 'psnr' else (best_lpips if np.isfinite(best_lpips) else None)
        lora_cfg_for_ckpt = None
        if args.use_lora:
            lora_cfg_for_ckpt = {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'target_regex': lora_target_regex,
                'target_preset': args.lora_target_preset,
                'init_weights': args.lora_init_weights,
                'lora_lr': args.lora_lr,
            }
        save_checkpoint(system, accelerator, end_epoch - 1, avg_loss, best_psnr,
                       args.pixel_weight, args.strength,
                       args.control_guidance_start, args.control_guidance_end,
                       os.path.join(save_dir, 'final_model.pt'),
                       lpips_weight=args.lpips_weight,
                       lpips_apply_prob=args.lpips_apply_prob,
                       val_lpips=(best_lpips if np.isfinite(best_lpips) else None),
                       best_metric=args.best_metric,
                       best_metric_value=final_best_value,
                       lora_config=lora_cfg_for_ckpt,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       global_step=global_step)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        if args.best_metric == 'lpips' and final_best_value is not None:
            print(f"Best LPIPS: {final_best_value:.4f}")
            print(f"Best PSNR (tracked): {best_psnr:.2f} dB")
        else:
            print(f"Best PSNR: {best_psnr:.2f} dB")
            if np.isfinite(best_lpips):
                print(f"Best LPIPS (tracked): {best_lpips:.4f}")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
