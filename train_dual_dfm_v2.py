#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training -- DFM + Gated Fusion (v2)
======================================================================

A/B variant of ``train_dual_dfm.py``. Core architectural difference:
the Level-1 pixel fusion is switched from concat+1x1 back to the
original legacy *gated residual add*:

    fused_cond = lr_lat + pixel_weight * sigmoid(pixel_gate_logit)
                          * pixel_fuse_proj(pixel_feat)

Rationale: in concat+identity-init fusion, the pixel-slice of
``pixel_fuse_proj`` is zero-init, so the pixel branch contributes
nothing at step 0 and has to slowly grow ``W_px`` before it starts
helping. Gated fusion with ``pixel_gate_init`` > 0 (default 4 →
sigmoid = 0.98) starts with the pixel branch ~fully open at step 1,
which observationally gives 2-3 dB faster PSNR convergence in the
early epochs.

Everything else (PixelFeatureExtractor stage1/stage2/stage3, DFM
adapters, dataset, scheduler, LoRA, loss, resume, eval) is kept
identical to train_dual_dfm.py so A/B comparisons isolate the fusion
change.

Checkpoints saved by this file set ``fusion_type='gated'`` and store a
``pixel_gate_logit`` scalar. On resume they best-effort migrate from
concat-layout DFM checkpoints (lr_lat identity on the first 16 input
channels is stripped; W_px slice is folded into the 16x16 conv).

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
        train_dual_dfm_v2.py \
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
        train_dual_dfm_v2.py \
        --hr_dir Data/Mix16K_HR \
        --lr_dir Data/Mix16K_LR_bicubic_X4 \
        --degrade_mode realesrgan --scale 4 \
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --resume checkpoints/dual_dfm/<stage1_exp>/best_model.pt \
        --epochs_mode stage \
        --reset_pixel_gate \
        --controlnet_lr_scale 0.1 \
        --batch_size 4 --epochs 20 --num_crops 2 --lr 5e-6 \
        --warmup_epochs 2 \
        --strength 1 \
        --lpips_weight 0.05 --lpips_resize 256 --lpips_apply_prob 0.1 \
        --lpips_max_sigma 0.7 \
        --empty_cache_steps 50
   (RealESRGAN / USM / blur-sigma params live in REALESRGAN_CFG / USM_CFG constants.)
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


def infer_dataset_name(*candidates):
    """Infer a canonical dataset name from free-form path/name candidates."""
    mapping = (
        ('mix24k', 'MIX24K'),
        ('mix16k', 'MIX16K'),
        ('df2k', 'DF2K'),
        ('div2k', 'DIV2K'),
        ('urban100', 'Urban100'),
        ('urban', 'Urban100'),
        ('drealsr', 'DRealSR'),
        ('realsr', 'RealSR'),
    )
    for value in candidates:
        if not value:
            continue
        low = str(value).lower()
        for key, dataset_name in mapping:
            if key in low:
                return dataset_name
    return 'Unknown'


# ============================================================================
# Fixed degradation / USM config (edit here; not exposed on CLI)
# ============================================================================

REALESRGAN_CFG = {
    'blur_prob': 0.8,
    'second_blur_prob': 0.8,
    'gaussian_noise_prob1': 0.5,
    'gaussian_noise_prob2': 0.5,
    'gray_noise_prob1': 0.4,
    'gray_noise_prob2': 0.4,
    'final_sinc_prob': 0.8,
    'blur_sigma': [0.2, 3.0],
    'blur_sigma2': [0.2, 1.5],
    'resize_prob1': [0.2, 0.7, 0.1],
    'resize_prob2': [0.3, 0.4, 0.3],
    'resize_range1': [0.15, 1.5],
    'resize_range2': [0.3, 1.2],
    'noise_sigma1': [1.0, 30.0],
    'noise_sigma2': [1.0, 25.0],
    'poisson_scale1': [0.05, 3.0],
    'poisson_scale2': [0.05, 2.5],
    'jpeg_range1': [30, 95],
    'jpeg_range2': [30, 95],
    'paired_prob': 0.0,
    'bicubic_prob': 0.0,
    'blur_sigma_scale': 0.0,
}

USM_CFG = {
    'mode': 'off',
    'weight': 0.0,
    'radius': 1.0,
    'threshold': 10.0,
    'apply_prob': 1.0,
}


# ============================================================================
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """Stage-grouped pixel encoder (A/B variant of ``train_dual_control.py``).

    Structure (three conv blocks + projection), designed for clean DFM tap
    exposure:

        input (3ch, H)
          -> stage1      (H   -> H/2,  ends at 32ch)   <- "s1" tap
          -> stage2      (H/2 -> H/4,  ends at 64ch)   <- "s2" tap
          -> stage3_body (H/4 -> H/8,  ends at 256ch)  <- "s3" tap
          -> stage3_proj (256ch -> latent_channels at H/8)
          -> zero_conv   (keeps condition-side init at zero)

    Differences vs the flat 18-layer ``nn.Sequential`` in train_dual_control.py:

    1. Named submodules (self.stage1 / stage2 / stage3_body / stage3_proj)
       instead of indexing into a single Sequential. Enables per-stage freeze
       and per-stage LR; makes profilers / ``named_modules()`` self-documenting.

    2. Taps are returned directly from local variables in forward() -- no
       magic indices, no ``_TAP_IDX`` lookup table.

    3. s3 channel width is 256 (vs 128 in the flat variant) to give DFM up0
       a thicker conditioning signal. The condition-side latent projection
       (stage3_proj) still outputs ``latent_channels`` (16 by default) before
       the zero_conv, so the condition path is unchanged in width.

    Note: state_dict keys use the ``stage*`` namespace above and are NOT
    compatible with train_dual_control.py's flat ``encoder.N.*`` checkpoints.
    Train from scratch when switching to this variant.
    """

    _TAP_CHANNELS = {'s1': 32, 's2': 64, 's3': 256}

    def __init__(self, latent_channels=16):
        super().__init__()

        # stage1: 3 -> 32, H -> H/2
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        # stage2: 32 -> 64, H/2 -> H/4
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        # stage3_body: 64 -> 256, H/4 -> H/8  (plan's wider tap)
        self.stage3_body = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
        )
        # stage3_proj: 256 -> latent_channels at H/8 (condition-side branch)
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )

        # Output 1x1 conv for the condition-side branch. Name kept as
        # ``zero_conv`` for checkpoint-key stability, but init is kaiming --
        # zero-init here combined with the zero-init pixel slice of
        # ``pixel_fuse_proj`` created a dead-gradient chain (both
        # ``pixel_feat`` and its downstream weights were 0, so neither side
        # received gradient). ``pixel_fuse_proj`` alone guarantees the
        # ``fused_cond = lr_lat`` identity at init.
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.zero_conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x, return_features=False):
        """Forward.

        Args:
            x: (B, 3, H, W) pixel input in [-1, 1].
            return_features: if True, return ``(out, taps_dict)`` where
                taps_dict is {'s1':32ch H/2, 's2':64ch H/4, 's3':256ch H/8}.
                These feed the decoder-side DFM adapters.

        Returns:
            Default: condition-side feature (latent_channels, H/8, W/8).
            With return_features: (condition_feat, {'s1','s2','s3'}).
        """
        s1 = self.stage1(x)          # 32ch,  H/2
        s2 = self.stage2(s1)         # 64ch,  H/4
        s3 = self.stage3_body(s2)    # 256ch, H/8
        out = self.zero_conv(self.stage3_proj(s3))  # latent_channels, H/8
        if return_features:
            return out, {'s1': s1, 's2': s2, 's3': s3}
        return out


class DFMAdapter(nn.Module):
    """Decoder-side Feature Modulation (residual + SiLU + zero-conv).

    Adds a zero-init residual injection to a VAE-decoder activation using a
    pixel-space feature tap from :class:`PixelFeatureExtractor`:

        decoder_feat <- decoder_feat + zero_conv(silu(align(pixel_feat)))

    ``zero_conv`` is zero-initialised, so at init the residual is exactly 0
    and the frozen VAE decoder produces bit-wise identical output. Additive
    injection leaves the decoder's pretrained activation scale untouched
    (unlike SPADE affine, where a learned multiplicative ``scale`` can push
    activations off-distribution). The intermediate SiLU gives the adapter
    one extra nonlinearity for local feature remapping.

    Args:
        feat_ch:    channel count of the incoming pixel feature map
                    (e.g. 256/64/32 for s3/s2/s1 taps).
        decoder_ch: channel count of the VAE decoder activation at the
                    injection point. Read at runtime from
                    ``vae.decoder.up_blocks[i].resnets[0].conv1.in_channels``
                    so we do not hard-code it for different VAE variants.
    """

    def __init__(self, feat_ch: int, decoder_ch: int):
        super().__init__()
        self.align = nn.Conv2d(feat_ch, decoder_ch, kernel_size=3, padding=1)
        self.zero_conv = nn.Conv2d(decoder_ch, decoder_ch, kernel_size=1)

        # Identity-at-init: zero out the output conv so the residual is 0.
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        # ``align`` is allowed to be non-zero since its output feeds zero_conv
        # (zero-init) -> overall residual contribution is still zero at init.
        nn.init.kaiming_normal_(self.align.weight, nonlinearity='relu')
        nn.init.zeros_(self.align.bias)

    def forward(self, decoder_feat: torch.Tensor, pixel_feat: torch.Tensor) -> torch.Tensor:
        if pixel_feat.shape[-2:] != decoder_feat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat,
                size=decoder_feat.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
        # Cast pixel_feat to decoder_feat dtype (decoder may run in fp16/bf16).
        if pixel_feat.dtype != decoder_feat.dtype:
            pixel_feat = pixel_feat.to(decoder_feat.dtype)
        residual = self.zero_conv(F.silu(self.align(pixel_feat)))
        return decoder_feat + residual


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
                 lora_init_weights='gaussian',
                 use_dfm=False, dfm_pixel_weight=0.05, dfm_lpips_weight=0.0,
                 dfm_sigma_min=0.05, dfm_sigma_gate=0.7,
                 dfm_detach_v=False,
                 pixel_gate_init=4.0):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.train_controlnet = train_controlnet
        self.pixel_weight = pixel_weight
        self.pixel_gate_init = float(pixel_gate_init)
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

        # DFM config (see Phase 1 of plan). Kept in the system so checkpoint /
        # resume / eval paths can discover what was trained with.
        self.use_dfm = bool(use_dfm)
        self.dfm_pixel_weight = float(dfm_pixel_weight)
        self.dfm_lpips_weight = float(dfm_lpips_weight)
        # Pixel-space loss numerical guard: we recover z0_pred from a predicted v
        # via z0 = z_sigma - sigma * v, which blows up at small sigma. Clamp the
        # sigma used in that formula and (optionally) gate the loss to steps
        # above `dfm_sigma_gate`.
        self.dfm_sigma_min = float(dfm_sigma_min)
        self.dfm_sigma_gate = float(dfm_sigma_gate)
        # When True, the v_pred used for z0_pred reconstruction inside
        # decode_with_dfm is detached. This isolates DFM adapter gradients from
        # the FM head and keeps the LPIPS/L1 signal from back-propagating into
        # the transformer / ControlNet. Defaults to False so the pixel-space
        # grounding also refines the FM head.
        self.dfm_detach_v = bool(dfm_detach_v)

        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self.pixel_fuse_proj = None
        self.pixel_gate_logit = None
        self.scheduler = None
        self._cached_embeds = None
        # DFM adapter dict: key is 'up0'/'up1'/'up2', value is DFMAdapter.
        # Populated inside _load_models after the VAE is on device.
        self.dfm_adapters = nn.ModuleDict()
        # Ordered mapping from VAE decoder up_block index -> PixelFeatureExtractor tap.
        # up_blocks[0] (H/8) <- s3 (128ch), [1] (H/4) <- s2 (64ch), [2] (H/2) <- s1 (32ch).
        # up_blocks[3] (H) has no matching tap and is left untouched.
        self._dfm_up_to_tap = {0: 's3', 1: 's2', 2: 's1'}

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
        # Gated fusion (v2): fused = lr_lat + pixel_weight * sigmoid(gate) * proj(pixel_feat).
        # Conv is identity-init so at step 0 proj(pixel_feat) == pixel_feat; with
        # pixel_gate_init=4 (sigmoid=0.98) the pixel branch is ~fully open from step 1,
        # unlike concat+identity where W_px=0 starves the branch of gradient early on.
        self.pixel_fuse_proj = nn.Conv2d(16, 16, kernel_size=1).to(self.device).to(dtype)
        self._reset_pixel_fuse_proj_to_identity()
        # Keep the scalar gate in fp32 for stable optimization; cast on use.
        self.pixel_gate_logit = nn.Parameter(
            torch.tensor(self.pixel_gate_init, device=self.device, dtype=torch.float32)
        )
        self.pixel_extractor.train()
        self.pixel_fuse_proj.train()

        # DFM adapters — registered after VAE + pixel_extractor exist so that
        # decoder channel counts can be read from the live module rather than
        # hard-coded. The VAE decoder is kept frozen; only the adapters train.
        if self.use_dfm:
            self._build_dfm_adapters(dtype=dtype, local_rank=local_rank)
        
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
        """Identity-init the 16->16 projection applied to pixel_feat.

        Combined with sigmoid(pixel_gate_logit) ~ 0.98 at init, this gives
        fused_cond = lr_lat + pixel_weight * 0.98 * pixel_feat from step 1,
        so the pixel branch starts participating immediately.
        """
        if self.pixel_fuse_proj is None:
            return
        with torch.no_grad():
            self.pixel_fuse_proj.weight.zero_()
            self.pixel_fuse_proj.bias.zero_()
            channels = min(self.pixel_fuse_proj.out_channels, self.pixel_fuse_proj.in_channels)
            idx = torch.arange(channels, device=self.pixel_fuse_proj.weight.device)
            self.pixel_fuse_proj.weight[idx, idx, 0, 0] = 1.0

    def _reset_pixel_gate_logit(self):
        """Reset the sigmoid-gate to --pixel_gate_init value."""
        if self.pixel_gate_logit is None:
            return
        with torch.no_grad():
            self.pixel_gate_logit.fill_(self.pixel_gate_init)

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
        if self.pixel_gate_logit is not None:
            params.append(self.pixel_gate_logit)
        # DFM adapters live on the pixel-side branch conceptually
        # (they consume pixel features to modulate the decoder).
        if self.use_dfm and len(self.dfm_adapters) > 0:
            params += list(self.dfm_adapters.parameters())
        return params

    def get_dfm_params(self):
        """Return DFM adapter params only (empty list if DFM disabled).

        Exposed separately in case the optimizer wants its own LR group.
        """
        if not self.use_dfm or len(self.dfm_adapters) == 0:
            return []
        return list(self.dfm_adapters.parameters())

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

    # ------------------------------------------------------------------ DFM
    def _build_dfm_adapters(self, dtype, local_rank: int = 0):
        """Instantiate DFM adapters by reading decoder channels at runtime.

        Requires ``self.vae`` and ``self.pixel_extractor`` to already exist.
        Channels:
            - pixel-side feat_ch comes from PixelFeatureExtractor._TAP_CHANNELS
              (s1=32, s2=64, s3=256).
            - decoder-side decoder_ch is
              ``self.vae.decoder.up_blocks[i].resnets[0].conv1.in_channels`` —
              introspected so the code keeps working across VAE variants.
        """
        decoder = self.vae.decoder
        tap_ch = PixelFeatureExtractor._TAP_CHANNELS  # {'s1':32,'s2':64,'s3':256}

        for up_idx, tap_name in self._dfm_up_to_tap.items():
            if up_idx >= len(decoder.up_blocks):
                if local_rank == 0:
                    print(f"[DFM][WARN] up_blocks[{up_idx}] does not exist; skipping")
                continue
            up_block = decoder.up_blocks[up_idx]
            try:
                dec_ch = up_block.resnets[0].conv1.in_channels
            except AttributeError as e:
                raise RuntimeError(
                    f"[DFM] cannot infer decoder in_channels at up_blocks[{up_idx}]: {e}. "
                    f"Update DFM introspection for this VAE variant."
                )

            adapter = DFMAdapter(feat_ch=tap_ch[tap_name], decoder_ch=dec_ch)
            adapter = adapter.to(self.device).to(dtype)
            # DFM adapters are the only thing bringing gradient into the
            # otherwise-frozen VAE decoder path. Keep them in train mode.
            adapter.train()
            self.dfm_adapters[f'up{up_idx}'] = adapter
            if local_rank == 0:
                print(
                    f"[DFM] up_blocks[{up_idx}] <- tap {tap_name}: "
                    f"feat_ch={tap_ch[tap_name]} -> decoder_ch={dec_ch}"
                )

        if local_rank == 0:
            n_params = sum(p.numel() for p in self.dfm_adapters.parameters())
            print(f"[DFM] total adapter params: {n_params:,}")

    def decode_with_dfm(self, lat, pixel_taps):
        """Decode latent to image, injecting DFM modulation before each up_block.

        Mirrors :meth:`decode` for the latent-space scaling convention:
        ``lat`` is assumed to be in the same (scaled) space as ``self.encode``
        returns. We un-scale, then replay
        :func:`diffusers.models.autoencoders.vae.Decoder.forward` with DFM
        adapters applied just before each up_block listed in
        ``self._dfm_up_to_tap``.

        Args:
            lat:         (B, 16, H/8, W/8) latent in the *scaled* convention.
            pixel_taps:  dict with keys 's1'/'s2'/'s3' from
                         ``PixelFeatureExtractor(..., return_features=True)``.

        Returns:
            (B, 3, H, W) image in the VAE's output range.
        """
        if not self.use_dfm or len(self.dfm_adapters) == 0 or pixel_taps is None:
            # Fallback: identical to self.decode(lat), but do it here so the
            # caller can uniformly call decode_with_dfm whether or not DFM is
            # enabled (makes eval-path branching simpler).
            # We also fall back when ``pixel_taps`` is None -- e.g. during
            # validate() before the first inference() call populates
            # self._last_pixel_taps, or after we explicitly cleared the cache.
            return self.decode(lat)

        # --- un-scale (matches self.decode) ---
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            z = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            z = lat / self.vae.config.scaling_factor
        z = z.to(self.vae.dtype)

        # --- replay Decoder.forward (diffusers>=0.30) with DFM hooks ---
        dec = self.vae.decoder
        # Use gradient checkpointing on up_blocks during training to keep the
        # extra decoder forward memory-affordable (the VAE decoder on 512px HR
        # holds ~1-2 GB of activations without ckpt). We do it manually here
        # since ``vae.decoder.forward`` only turns on ckpt when both flags
        # ``self.training and self.gradient_checkpointing`` are True, and the
        # VAE is kept in eval() with frozen weights -- we don't want to flip
        # those flags on the shared module.
        use_ckpt = self.training and torch.is_grad_enabled()

        sample = dec.conv_in(z)
        # mid_block signature: (sample, latent_embeds=None)
        if use_ckpt:
            sample = torch.utils.checkpoint.checkpoint(
                dec.mid_block, sample, None, use_reentrant=False,
            )
        else:
            sample = dec.mid_block(sample, None)
        upscale_dtype = next(iter(dec.up_blocks.parameters())).dtype
        sample = sample.to(upscale_dtype)

        for i, up_block in enumerate(dec.up_blocks):
            tap_name = self._dfm_up_to_tap.get(i)
            if tap_name is not None and f'up{i}' in self.dfm_adapters:
                feat = pixel_taps.get(tap_name)
                if feat is not None:
                    sample = self.dfm_adapters[f'up{i}'](sample, feat)
            if use_ckpt:
                sample = torch.utils.checkpoint.checkpoint(
                    up_block, sample, None, use_reentrant=False,
                )
            else:
                sample = up_block(sample, None)

        # post-process (latent_embeds is always None on FLUX VAE). The tail
        # holds a full H×W activation, so during training we also run it
        # through gradient checkpointing to cap peak memory.
        if use_ckpt:
            def _tail(x, dec=dec):
                x = dec.conv_norm_out(x)
                x = dec.conv_act(x)
                x = dec.conv_out(x)
                return x
            sample = torch.utils.checkpoint.checkpoint(_tail, sample, use_reentrant=False)
        else:
            sample = dec.conv_norm_out(sample)
            sample = dec.conv_act(sample)
            sample = dec.conv_out(sample)
        return sample

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
    
    def forward(self, noisy, lr_lat, lr_pixel, timestep, guidance=3.5, controlnet_scale=1.0,
                return_dfm_taps=False):
        """
        Forward pass: predict velocity

        Args:
            noisy: 当前 noisy latent
            lr_lat: LR 图像的 latent
            lr_pixel: LR 图像的 pixel tensor
            timestep: 🌟 官方格式的 timestep（已经 / 1000）
            guidance: CFG guidance scale
            return_dfm_taps: if True, also return the pixel-extractor multi-scale
                taps dict (keys s1/s2/s3). Used by the DFM pixel-space loss to
                avoid re-running the pixel extractor.
        """
        B, C, H, W = noisy.shape
        device = noisy.device
        dtype = torch.bfloat16

        # Pixel features -- collect multi-scale taps only when explicitly
        # requested (training DFM loss path). In inference/validation we cache
        # taps once in `inference()` to avoid per-step extractor overhead.
        need_taps = bool(return_dfm_taps)
        if need_taps:
            pixel_feat, pixel_taps = self.pixel_extractor(lr_pixel, return_features=True)
        else:
            pixel_feat = self.pixel_extractor(lr_pixel)
            pixel_taps = None

        if pixel_feat.shape[-2:] != lr_lat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=lr_lat.shape[-2:], mode='bilinear', align_corners=False
            )

        # Gated residual fusion (v2):
        #   fused_cond = lr_lat + pixel_weight * sigmoid(gate) * proj(pixel_feat)
        # proj is identity-init (see _reset_pixel_fuse_proj_to_identity); with
        # pixel_gate_init=4.0 the sigmoid gate starts at ~0.98, so the pixel
        # branch contributes from step 1 without a cold-start zero-gradient
        # phase.
        projected = self.pixel_fuse_proj(pixel_feat.to(dtype))
        gate = torch.sigmoid(self.pixel_gate_logit.float()).to(dtype)
        fused_cond = (lr_lat.to(dtype) + self.pixel_weight * gate * projected).to(dtype)
        
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

        v_pred = self._unpack(out, H, W)
        if return_dfm_taps:
            return v_pred, pixel_taps
        return v_pred
    
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

        # Cache taps once per sample for downstream decode_with_dfm().
        if self.use_dfm:
            _, self._last_pixel_taps = self.pixel_extractor(lr_pixel, return_features=True)
        else:
            self._last_pixel_taps = None
        
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
    use_dfm = bool(getattr(unwrapped, 'use_dfm', False)) and hr_pixel is not None
    if use_dfm:
        v_pred, pixel_taps = system(
            noisy, lr_lat, lr_pixel, sigma, guidance, return_dfm_taps=True
        )
    else:
        v_pred = system(noisy, lr_lat, lr_pixel, sigma, guidance)
        pixel_taps = None

    # Base FM loss
    loss = F.mse_loss(v_pred.float(), target_v.float())

    # -------------------------------------------------------------------------
    # Pixel-space losses (baseline LPIPS + DFM L1/LPIPS)
    # -------------------------------------------------------------------------
    # Both terms live on ``pred_hr = decode(pred_hr_lat)`` where
    #   pred_hr_lat = noisy - sigma_expand * v_pred   (algebraically exact)
    #
    # To avoid running the VAE decoder twice per step when DFM is on, we
    # merge the two decode paths:
    #   - DFM OFF: keep baseline LPIPS path (uses unwrapped.decode -> vae.decoder).
    #   - DFM ON: decode through ``decode_with_dfm`` once. At init DFM is the
    #     identity (zero-init residual branch), so this is numerically identical
    #     to ``decode`` and can carry BOTH the pixel-L1 supervision (for DFM
    #     adapters) and the optional LPIPS term. We therefore skip the extra
    #     baseline LPIPS decode.
    #
    # Gating:
    #   - Baseline LPIPS uses the existing batch-mean ``lpips_max_sigma`` +
    #     random ``lpips_apply_prob`` gate (original behaviour).
    #   - DFM pixel loss uses a PER-SAMPLE sigma mask so ranks stay in sync
    #     under DDP even when a subset of samples is above the gate. The
    #     decode ALWAYS runs when DFM is on (to keep the grad graph shape
    #     consistent across ranks) -- we just zero out the contribution of
    #     high-sigma samples.
    sigma_flt = sigma.float()
    do_dfm_pixel = (
        use_dfm
        and unwrapped.dfm_pixel_weight > 0
        and pixel_taps is not None
    )

    # Baseline LPIPS trigger (only used when DFM is OFF, to avoid double decode).
    do_baseline_lpips = (
        lpips_model is not None
        and lpips_weight > 0
        and hr_pixel is not None
        and not do_dfm_pixel
        and float(sigma_flt.mean().item()) < float(lpips_max_sigma)
        and (lpips_apply_prob >= 1.0 or torch.rand(1, device=device).item() < lpips_apply_prob)
    )

    if do_dfm_pixel:
        # --- Merged DFM pixel decode ---
        # FM identity: z_hr = noisy - sigma * v  (no division, numerically
        # exact for any sigma in [0, 1]).
        v_for_decode = v_pred.detach() if unwrapped.dfm_detach_v else v_pred
        z0_pred = noisy - sigma_expand * v_for_decode
        pixel_pred = unwrapped.decode_with_dfm(z0_pred, pixel_taps)

        pred_f = pixel_pred.float()
        target_f = hr_pixel.float()

        # Per-sample mask: 1 where sigma <= dfm_sigma_gate, else 0. DDP-safe.
        gate = float(getattr(unwrapped, 'dfm_sigma_gate', 0.0))
        if gate > 0.0:
            mask = (sigma_flt <= gate).view(B, 1, 1, 1).to(pred_f.dtype)
            mask_sum = mask.sum().clamp(min=1.0)
            numel_per_sample = pred_f.shape[1] * pred_f.shape[2] * pred_f.shape[3]
            pix_l1 = (mask * (pred_f - target_f).abs()).sum() / (mask_sum * numel_per_sample)
        else:
            # gate <= 0 -> never gate; equivalent to standard F.l1_loss (mean
            # over all elements, matches legacy behaviour for A/B parity).
            pix_l1 = F.l1_loss(pred_f, target_f)

        dfm_loss_term = pix_l1

        # DFM LPIPS term (re-uses the same decoded tensor -> no double decode)
        dfm_lpips_w = float(getattr(unwrapped, 'dfm_lpips_weight', 0.0))
        if lpips_model is not None and dfm_lpips_w > 0:
            pp = pred_f.clamp(-1, 1)
            tt = target_f.clamp(-1, 1)
            if lpips_resize is not None and lpips_resize > 0:
                size = (int(lpips_resize), int(lpips_resize))
                pp = F.interpolate(pp, size=size, mode='bilinear', align_corners=False)
                tt = F.interpolate(tt, size=size, mode='bilinear', align_corners=False)
            dfm_loss_term = dfm_loss_term + dfm_lpips_w * lpips_model(pp, tt).mean().float()
            del pp, tt

        # Baseline LPIPS can piggy-back on the SAME decode (pred_hr == pixel_pred
        # at DFM init; after training they differ by the DFM modulation, but
        # the LPIPS objective is still "decoded pred should look like HR").
        if (
            lpips_model is not None
            and lpips_weight > 0
            and hr_pixel is not None
            and float(sigma_flt.mean().item()) < float(lpips_max_sigma)
            and (lpips_apply_prob >= 1.0 or torch.rand(1, device=device).item() < lpips_apply_prob)
        ):
            pp = pred_f.clamp(-1, 1)
            tt = target_f.clamp(-1, 1)
            if lpips_resize is not None and lpips_resize > 0:
                size = (int(lpips_resize), int(lpips_resize))
                pp = F.interpolate(pp, size=size, mode='bilinear', align_corners=False)
                tt = F.interpolate(tt, size=size, mode='bilinear', align_corners=False)
            loss = loss + float(lpips_weight) * lpips_model(pp, tt).mean().float()
            del pp, tt

        loss = loss + float(unwrapped.dfm_pixel_weight) * dfm_loss_term
        del z0_pred, pixel_pred, pred_f, target_f, pix_l1, dfm_loss_term, v_for_decode

    elif do_baseline_lpips:
        # DFM off: original baseline LPIPS path (single decode, no DFM).
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
        sr = unwrapped.decode_with_dfm(sr_lat, getattr(unwrapped, '_last_pixel_taps', None))
        
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
    # Release the cached pixel taps dict from the last inference() call so
    # large H/2/H/4/H/8 tensors don't sit on the GPU between val and the next
    # training step. Any future decode_with_dfm() call with None falls back to
    # plain decode() (see patch in that method).
    if hasattr(unwrapped, '_last_pixel_taps'):
        unwrapped._last_pixel_taps = None
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return {
        'psnr': float(np.mean(psnr_list)) if psnr_list else 0.0,
        'lpips': float(np.mean(lpips_list)) if lpips_list else None,
    }
def _strip_module_prefix(state_dict):
    """Strip an optional DDP/DeepSpeed ``module.`` prefix from state_dict keys."""
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _load_pixel_fuse_proj_with_migration(unwrapped, ckpt, is_main=False):
    """Load pixel_fuse_proj (v2 gated layout), migrating from legacy concat
    checkpoints when present.

    v2 layout (this file):
        pixel_fuse_proj: Conv2d(16, 16, 1) applied to pixel_feat
        pixel_gate_logit: scalar; fused = lr_lat + pixel_weight * sigmoid(gate) * conv(pixel_feat)

    Legacy concat layout (train_dual_dfm.py):
        pixel_fuse_proj: Conv2d(32, 16, 1) on cat([lr_lat, pixel_weight*pixel_feat])
        fused = W_lr @ lr_lat + W_px @ (pixel_weight * pixel_feat) + b
        (no pixel_gate_logit)

    Migration from concat -> gated:
        Take the last-16 input-channel slice of the concat weight as the gated
        conv weight; the first-16 slice (lr_lat passthrough) is absorbed into
        the residual add. The gate is set to sigmoid(gate) = 1.0 (logit large),
        so the migrated pixel branch matches the concat contribution at init.
    """
    if 'pixel_fuse_proj' not in ckpt:
        if is_main:
            print("[Resume] pixel_fuse_proj missing; keeping fresh identity init.")
        return

    state = _strip_module_prefix(ckpt['pixel_fuse_proj'])
    target = unwrapped.pixel_fuse_proj
    target_w = target.weight
    src_w = state.get('weight')
    src_b = state.get('bias')

    is_concat_legacy = (
        src_w is not None
        and src_w.shape == (16, 32, 1, 1)
        and target_w.shape == (16, 16, 1, 1)
    )

    if not is_concat_legacy and src_w is not None and src_w.shape == target_w.shape:
        target.load_state_dict(state)
        gate_raw = ckpt.get('pixel_gate_logit', None)
        if gate_raw is not None and unwrapped.pixel_gate_logit is not None:
            with torch.no_grad():
                if isinstance(gate_raw, torch.Tensor):
                    gate_raw = gate_raw.to(
                        device=unwrapped.pixel_gate_logit.device,
                        dtype=unwrapped.pixel_gate_logit.dtype,
                    )
                    unwrapped.pixel_gate_logit.copy_(gate_raw)
                else:
                    unwrapped.pixel_gate_logit.fill_(float(gate_raw))
        if is_main:
            fusion_type = ckpt.get('fusion_type', 'gated')
            gate_sig = torch.sigmoid(unwrapped.pixel_gate_logit.detach().float()).item() \
                if unwrapped.pixel_gate_logit is not None else 1.0
            print(f"[Resume] pixel_fuse_proj loaded (fusion_type={fusion_type}, "
                  f"sigmoid(gate)={gate_sig:.4f}).")
        return

    if is_concat_legacy:
        with torch.no_grad():
            # Take pixel-branch slice from concat weight (last 16 input channels).
            pixel_slice = src_w[:, 16:32, :, :].to(
                device=target_w.device, dtype=target_w.dtype
            )
            target.weight.copy_(pixel_slice)
            if src_b is not None:
                target.bias.copy_(src_b.to(device=target.bias.device, dtype=target.bias.dtype))
            # Gate ~ 1 so the migrated branch contributes at roughly the same
            # amplitude as the concat checkpoint's W_px slice.
            if unwrapped.pixel_gate_logit is not None:
                unwrapped.pixel_gate_logit.fill_(6.0)  # sigmoid(6) ~ 0.9975
        if is_main:
            print(
                "[Resume] Concat checkpoint migrated to gated layout "
                "(pixel slice -> 16x16 conv; gate logit set to 6.0 ~ sigmoid 0.998)."
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
                    optimizer=None, lr_scheduler=None, global_step=None,
                    training_data=None):
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
        'pixel_gate_logit': unwrapped.pixel_gate_logit.detach().cpu()
            if unwrapped.pixel_gate_logit is not None else None,
        'fusion_type': 'gated',
        'controlnet': unwrapped.controlnet.state_dict(),
        'use_lora': unwrapped.use_lora,
        'use_dfm': bool(getattr(unwrapped, 'use_dfm', False)),
    }
    # DFM state + config (see Phase 1.5). We persist both so that resume can
    # (a) reload adapter weights verbatim and (b) refuse to mix DFM-trained
    # adapters into a run that declares --use_dfm=0.
    if getattr(unwrapped, 'use_dfm', False):
        payload['dfm_adapters'] = unwrapped.dfm_adapters.state_dict()
        payload['dfm_config'] = {
            'pixel_weight': unwrapped.dfm_pixel_weight,
            'lpips_weight': unwrapped.dfm_lpips_weight,
            'sigma_min': unwrapped.dfm_sigma_min,
            'sigma_gate': unwrapped.dfm_sigma_gate,
            'detach_v': bool(getattr(unwrapped, 'dfm_detach_v', False)),
            'up_to_tap': dict(unwrapped._dfm_up_to_tap),
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
    if training_data is not None:
        payload['train_data'] = training_data
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

    # RealESRGAN degradation params live in ``REALESRGAN_CFG`` module constant.

    # Model
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--pretrained_controlnet', type=str, default=None)
    parser.add_argument('--pixel_weight', type=float, default=1.0)
    parser.add_argument('--pixel_gate_init', type=float, default=4.0,
                        help='Initial logit for sigmoid gate on pixel branch '
                             '(v2 gated fusion). 4.0 => sigmoid ~ 0.982, i.e. '
                             'pixel branch ~fully open from step 1.')
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
    parser.add_argument('--lpips_weight', type=float, default=0.0,
                        help='Optional LPIPS loss weight in training')
    parser.add_argument('--lpips_resize', type=int, default=256,
                        help='Resize for LPIPS loss (0 = use original training resolution)')
    parser.add_argument('--lpips_apply_prob', type=float, default=0.25,
                        help='Probability of applying LPIPS loss on each train step')
    parser.add_argument('--geom_aug', action='store_true',
                        help='Enable random flip/rotation augmentation during training')
    # USM sharpening params live in ``USM_CFG`` module constant.
    parser.add_argument('--empty_cache_steps', type=int, default=0,
                        help='Call gc/empty_cache every N training steps (0 to disable)')
    
    # Validation/eval start point (img2img-style interpolation from LR + noise)
    parser.add_argument('--strength', type=float, default=1,
                        help='Validation/eval strength (1.0 = pure noise start, 0.8 = skip first 20%% steps)')
    parser.add_argument('--val_num_steps', type=int, default=20)
    parser.add_argument('--val_num_samples', type=int, default=10,
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

    # DFM (Decoder-side Feature Modulation) is always ON in this script.
    parser.add_argument('--dfm_pixel_weight', type=float, default=0.05,
                        help='Weight on DFM pixel-space L1 loss (relative to FM MSE). '
                             'Start at 0.05; too large can overwhelm the FM signal.')
    parser.add_argument('--dfm_lpips_weight', type=float, default=0.0,
                        help='Optional LPIPS term inside the DFM pixel loss (reuses '
                             '--use_lpips model). 0 disables.')
    parser.add_argument('--dfm_sigma_min', type=float, default=0.05,
                        help='[Deprecated] kept for checkpoint compatibility; '
                             'z0 reconstruction now always uses raw sampled sigma.')
    parser.add_argument('--dfm_sigma_gate', type=float, default=0.7,
                        help='Per-sample sigma gate for DFM pixel loss: samples with '
                             'sigma > gate contribute 0 (decoded z0 is mostly noise). '
                             '0 or negative disables gating. Default 0.7 keeps the low/mid '
                             'sigma half of each batch and is DDP-safe (per-sample mask, '
                             'decode always runs so the grad graph is the same on every rank).')
    parser.add_argument('--dfm_detach_v', action='store_true', default=False,
                        help='Detach v_pred before reconstructing z0 for the DFM decode. '
                             'Isolates DFM adapter gradients from the FM head + transformer; '
                             'use when you want to tune DFM without perturbing the diffusion '
                             'backbone (e.g. frozen Stage-2).')
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/dual_dfmv2')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_optimizer', action='store_true', default=False,
                        help='When resuming, also restore optimizer momentum (default: off; Stage 2 typically wants a fresh optimizer).')
    parser.add_argument('--resume_lr_scheduler', action='store_true', default=False,
                        help='When resuming, restore the LambdaLR step counter. Default OFF: the new stage gets a fresh warmup + cosine over --epochs.')
    parser.add_argument('--reset_pixel_gate', action='store_true', default=False,
                        help='On resume, reset pixel_fuse_proj to identity and '
                             'pixel_gate_logit back to --pixel_gate_init. Useful when '
                             'switching Stage 1 -> Stage 2 to re-learn the mixing.')
    parser.add_argument('--controlnet_lr_scale', type=float, default=1.0,
                        help='Multiplier applied to --lr for the ControlNet param group (use <1 for Stage 2 fine-tuning, e.g. 0.1).')

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
    if args.controlnet_lr_scale < 0:
        parser.error("--controlnet_lr_scale must be >= 0")
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
        f"_aug{int(args.geom_aug)}"
        f"_bm{_fmt_tag(args.best_metric)}"
        f"_crop{args.num_crops}"
        f"_dfm"
        f"_dpw{_fmt_tag(args.dfm_pixel_weight)}"
        f"_dlw{_fmt_tag(args.dfm_lpips_weight)}"
        f"_dsg{_fmt_tag(args.dfm_sigma_gate)}"
    )
    if args.dfm_detach_v:
        exp_name += "_detachv"
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
    train_data_info = {
        'train_hr_dir': args.hr_dir,
        'train_lr_dir': args.lr_dir,
        'val_hr_dir': args.val_hr_dir,
        'val_lr_dir': args.val_lr_dir,
        'degrade_mode': args.degrade_mode,
        'scale': int(args.scale),
        'resolution': int(args.resolution),
        'num_crops': int(args.num_crops),
    }
    
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
        gate_sig_print = 1.0 / (1.0 + math.exp(-args.pixel_gate_init))
        print(f"Pixel Fusion: gated residual add (identity-init conv, "
              f"logit={args.pixel_gate_init}, sigmoid={gate_sig_print:.4f})")
        print(f"Conditioning Scale: {args.conditioning_scale}")
        print(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})")
        print(f"Geometric Aug: {args.geom_aug}")
        print(f"Empty Cache Steps: {args.empty_cache_steps}")
        if triton_cache_dir:
            print(f"TRITON_CACHE_DIR: {triton_cache_dir}")
        print(f"Strength: {args.strength} (推理时跳过 {(1-args.strength)*100:.0f}% 步数)")
        print(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]")
        print(f"Validation: steps={args.val_num_steps}, samples={args.val_num_samples}, lpips={args.val_calc_lpips}")
        print(f"Best Metric: {args.best_metric}, Reset Best On Resume: {args.reset_best_on_resume}")
        print(f"Learning Rate: {args.lr} (Pixel branch: {args.lr * 10})")
        print(
            f"DFM: ON (pixel_w={args.dfm_pixel_weight}, "
            f"lpips_w={args.dfm_lpips_weight}, sigma_gate={args.dfm_sigma_gate}, "
            f"detach_v={args.dfm_detach_v})"
        )
        print(
            f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
            f"dropout={args.lora_dropout}, lr={args.lora_lr}, "
            f"preset={args.lora_target_preset}"
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
        use_dfm=True,
        dfm_pixel_weight=args.dfm_pixel_weight,
        dfm_lpips_weight=args.dfm_lpips_weight,
        dfm_sigma_min=args.dfm_sigma_min,
        dfm_sigma_gate=args.dfm_sigma_gate,
        dfm_detach_v=args.dfm_detach_v,
        pixel_gate_init=args.pixel_gate_init,
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
    if args.degrade_mode == 'bicubic':
        train_lr_dir = None
    else:
        train_lr_dir = args.lr_dir  # realesrgan also accepts optional paired mix
    train_dataset = SRDataset(
        args.hr_dir, train_lr_dir, args.resolution,
        num_crops=args.num_crops, is_val=False,
        degrade_mode=args.degrade_mode, scale=args.scale,
        realesrgan_cfg=REALESRGAN_CFG,
        geom_aug=args.geom_aug,
        usm_mode=USM_CFG['mode'],
        usm_weight=USM_CFG['weight'],
        usm_radius=USM_CFG['radius'],
        usm_threshold=USM_CFG['threshold'],
        usm_apply_prob=USM_CFG['apply_prob'],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    
    val_loader = None
    val_dataset = None
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
    
    steps_per_epoch = len(train_loader)
    num_training_steps = args.epochs * steps_per_epoch
    num_warmup_steps = args.warmup_epochs * steps_per_epoch
    val_samples = len(val_dataset) if val_dataset is not None else 0
    train_dataset_name = infer_dataset_name(args.hr_dir, train_lr_dir, args.lr_dir)
    val_dataset_name = infer_dataset_name(args.val_hr_dir, args.val_lr_dir)
    world_size = int(getattr(accelerator, 'num_processes', 1))
    grad_accum_steps = int(getattr(accelerator, 'gradient_accumulation_steps', 1))
    global_batch = int(args.batch_size) * world_size
    effective_batch = global_batch * grad_accum_steps
    controlnet_init = args.pretrained_controlnet or "jasperai/Flux.1-dev-Controlnet-Upscaler"
    
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
            state = _strip_module_prefix(ckpt['pixel_extractor'])
            if any(k.startswith("encoder.") for k in state.keys()):
                raise RuntimeError(
                    "[Resume] Incompatible pixel_extractor checkpoint format: found legacy "
                    "'encoder.*' keys from train_dual_control.py, but train_dual_dfm.py "
                    "expects 'stage*' keys. Please resume from a train_dual_dfm checkpoint "
                    "or start a fresh run."
                )
            unwrapped.pixel_extractor.load_state_dict(state, strict=True)
        _load_pixel_fuse_proj_with_migration(unwrapped, ckpt, is_main=is_main)
        if args.reset_pixel_gate:
            with torch.no_grad():
                unwrapped._reset_pixel_fuse_proj_to_identity()
                unwrapped._reset_pixel_gate_logit()
            if is_main:
                gate_sig = torch.sigmoid(
                    unwrapped.pixel_gate_logit.detach().float()
                ).item() if unwrapped.pixel_gate_logit is not None else 1.0
                print(
                    "[Resume] --reset_pixel_gate: pixel_fuse_proj reset to identity, "
                    f"pixel_gate_logit reset (sigmoid={gate_sig:.4f})."
                )
        if args.train_controlnet and 'controlnet' in ckpt:
            state = _strip_module_prefix(ckpt['controlnet'])
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

        # --- DFM resume (Phase 1.5) ---
        ckpt_uses_dfm = bool(ckpt.get('use_dfm', False)) and ('dfm_adapters' in ckpt)
        if unwrapped.use_dfm and ckpt_uses_dfm:
            ckpt_dfm_cfg = ckpt.get('dfm_config', {}) or {}
            # Check that the up_block -> tap mapping matches; if the mapping
            # changed, adapter keys won't line up and silent reload would give
            # nonsense at the wrong VAE stage.
            ckpt_map = dict(ckpt_dfm_cfg.get('up_to_tap', {}) or {})
            # Coerce ckpt_map keys to int (json/pickle roundtrip may preserve int,
            # but be defensive).
            ckpt_map = {int(k): v for k, v in ckpt_map.items()}
            if ckpt_map and ckpt_map != unwrapped._dfm_up_to_tap:
                raise RuntimeError(
                    f"[DFM][Resume] up_block->tap mapping changed: "
                    f"ckpt={ckpt_map}, current={unwrapped._dfm_up_to_tap}"
                )
            try:
                state = _strip_module_prefix(ckpt['dfm_adapters'])
                unwrapped.dfm_adapters.load_state_dict(state, strict=True)
                if is_main:
                    print(f"[Resume] Loaded DFM adapters: {len(state)} tensors, config={ckpt_dfm_cfg}")
            except Exception as e:
                raise RuntimeError(f"[DFM][Resume] failed to load dfm_adapters: {e}")
        elif unwrapped.use_dfm and not ckpt_uses_dfm:
            if is_main:
                print("[Resume] Current run uses DFM but checkpoint has no DFM state; "
                      "adapters stay at zero-init (safe identity).")
        elif (not unwrapped.use_dfm) and ckpt_uses_dfm:
            if is_main:
                print("[Resume][WARN] Checkpoint has DFM state but --use_dfm=0; DFM state ignored.")
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
            f.write(f"Train Dataset: {train_dataset_name}\n")
            f.write(f"Val Dataset: {val_dataset_name}\n")
            f.write(f"Train HR Dir: {args.hr_dir}\n")
            f.write(f"Train LR Dir: {train_lr_dir}\n")
            f.write(f"Val HR Dir: {args.val_hr_dir}\n")
            f.write(f"Val LR Dir: {args.val_lr_dir}\n")
            f.write(
                f"Samples: train={len(train_dataset)}, val={val_samples} | "
                f"Steps/Epoch: {steps_per_epoch}, Total Steps: {num_training_steps}\n"
            )
            f.write(
                f"Batch: per_device={args.batch_size}, world_size={world_size}, "
                f"global={global_batch}, grad_accum={grad_accum_steps}, effective={effective_batch}\n"
            )
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"ControlNet Init: {controlnet_init}\n")
            f.write(f"Resume From: {args.resume}\n")
            f.write(f"Degrade Mode: {args.degrade_mode}\n")
            f.write(f"Scale: x{args.scale}\n")
            f.write(f"Strength: {args.strength}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            _gs = 1.0 / (1.0 + math.exp(-args.pixel_gate_init))
            f.write(
                f"Pixel Fusion: gated residual add "
                f"(identity-init conv, logit={args.pixel_gate_init}, sigmoid={_gs:.4f})\n"
            )
            f.write(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})\n\n")
            f.write(f"Geometric Aug: {args.geom_aug}\n")
            f.write(
                f"Validation: steps={args.val_num_steps}, samples={args.val_num_samples}, "
                f"lpips={args.val_calc_lpips}\n"
            )
            f.write(
                f"Best Metric: {args.best_metric}, Reset Best On Resume: {args.reset_best_on_resume}\n\n"
            )
            f.write(f"Empty Cache Steps: {args.empty_cache_steps}\n\n")
            f.write(f"Conditioning Scale: {args.conditioning_scale}\n")
            f.write(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]\n")
            f.write(
                f"DFM: ON, pixel_w={args.dfm_pixel_weight}, "
                f"lpips_w={args.dfm_lpips_weight}, sigma_gate={args.dfm_sigma_gate}, "
                f"detach_v={args.dfm_detach_v}\n"
            )
            f.write(
                f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                f"dropout={args.lora_dropout}, lr={args.lora_lr}, "
                f"preset={args.lora_target_preset}\n\n"
            )
    
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
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
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
            # Diagnostic for gated fusion (v2): current sigmoid(gate) value. Starts
            # at ~sigmoid(pixel_gate_init), typically ~0.98, so the pixel branch is
            # nearly fully open from step 1. Drift away from init indicates the net
            # is actively tuning the mix.
            gate_value = torch.sigmoid(
                unwrapped.pixel_gate_logit.detach().float()
            ).item() if unwrapped.pixel_gate_logit is not None else 0.0

            # DFM diagnostic: total Frobenius norm of the zero_conv output layers
            # across all adapters. Starts at 0 (zero-init) and grows as the DFM
            # branch learns to modulate the VAE decoder. Stays ~0 => either
            # dfm_pixel_weight too small, sigma_gate too tight, or DFM is a no-op.
            dfm_zc_norm = None
            if getattr(unwrapped, 'use_dfm', False) and len(unwrapped.dfm_adapters) > 0:
                dfm_zc_norm = sum(
                    a.zero_conv.weight.detach().float().norm().item()
                    for a in unwrapped.dfm_adapters.values()
                )

            extra_tag = f", Gate={gate_value:.4f}"
            if dfm_zc_norm is not None:
                extra_tag += f", DFM_zc={dfm_zc_norm:.4f}"
            if val_lpips is not None:
                log_line = (
                    f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, "
                    f"LPIPS={val_lpips:.4f}, LR={lr_current:.2e}{extra_tag}\n"
                )
            else:
                log_line = (
                    f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, "
                    f"LR={lr_current:.2e}{extra_tag}\n"
                )
            if args.use_lora and len(lr_scheduler.get_last_lr()) >= 2:
                log_line = log_line.rstrip("\n") + f", LoRA_LR={lr_scheduler.get_last_lr()[-1]:.2e}\n"
            with open(log_path, 'a') as f:
                f.write(log_line)

            dfm_print = f", dfm_zc={dfm_zc_norm:.4f}" if dfm_zc_norm is not None else ""
            if val_lpips is not None:
                print(
                    f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, "
                    f"val_lpips={val_lpips:.4f}, lr={lr_current:.2e}, gate={gate_value:.4f}"
                    + dfm_print
                    + (f", lora_lr={lr_scheduler.get_last_lr()[-1]:.2e}" if args.use_lora and len(lr_scheduler.get_last_lr()) >= 2 else "")
                )
            else:
                print(
                    f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, "
                    f"lr={lr_current:.2e}, gate={gate_value:.4f}"
                    + dfm_print
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
                               global_step=global_step,
                               training_data=train_data_info)
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
                               global_step=global_step,
                               training_data=train_data_info)
            
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
                       global_step=global_step,
                       training_data=train_data_info)
        
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

    # Explicitly end the Accelerator training state so DDP/NCCL resources are
    # cleaned up before process exit (PyTorch 2.4+ warns if not destroyed).
    accelerator.wait_for_everyone()
    try:
        accelerator.end_training()
    except Exception:
        pass

def _destroy_process_group_safely():
    """Best-effort teardown for torch.distributed process groups."""
    try:
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


if __name__ == '__main__':
    try:
        main()
    finally:
        _destroy_process_group_safely()
