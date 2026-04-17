#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR ControlNet Training + LoRA (based on V1)
======================================================================

This file is built on top of `train_dual_control_v1.py`, adding optional
LoRA adapters on the FLUX Transformer. The Transformer base weights remain
fully frozen; only LoRA adapters are trained (together with ControlNet and
pixel branch).

Target modules follow the OminiControl recipe:
    https://github.com/Yuanshi9815/OminiControl/blob/main/train/config/compact_token_representation.yaml

    dual blocks  (transformer_blocks.*):
        norm1.linear, attn.to_q, attn.to_k
    single blocks (single_transformer_blocks.*):
        norm.linear,  attn.to_q, attn.to_k

Why this subset:
    - Only adapt "what to attend to" (Q, K) and "how conditioning modulates
      features" (AdaLN linear).
    - Keep V / output projections / MLP untouched → preserve FLUX's generation
      prior. Very conservative, well-suited for SR where we don't want to
      rebuild the Transformer from scratch.

Key points:
    - No degradation pipeline (paired bicubic, same as V1)
    - Expects paired LR/HR input (`--lr_dir` is required)
    - `--use_lora` defaults to False → behavior identical to V1
    - When `--use_lora` is set, LoRA adapter (~25M params at rank=16) is
      added to the Transformer and trained with its own LR.

Typical command (V1-compatible, no LoRA):
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \\
        train_dual_lora.py \\
        --hr_dir Data/DIV2K/DIV2K_train_HR \\
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \\
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \\
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \\
        --batch_size 4 --epochs 40 --num_crops 2 \\
        --lr 1e-5 --strength 1 --pixel_gate_init 4

Typical command (with LoRA):
    accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \\
        train_dual_lora.py \\
        --hr_dir Data/DIV2K/DIV2K_train_HR \\
        --lr_dir Data/DIV2K/DIV2K_train_LR_bicubic_X4 \\
        --val_hr_dir Data/DIV2K/DIV2K_valid_HR \\
        --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \\
        --batch_size 4 --epochs 40 --num_crops 2 \\
        --lr 1e-5 --strength 1 --pixel_gate_init 4 \\
        --use_lora --lora_rank 16 --lora_alpha 16 --lora_lr 1e-4
"""


import os
import gc
import math
import argparse
import numpy as np
from contextlib import nullcontext
from datetime import datetime
from PIL import Image

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

# PEFT is optional; only required when --use_lora is set.
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


# ============================================================================
# LoRA target module presets
# ============================================================================

# OminiControl-compatible regex. Matches:
#   .*(?<!single_)transformer_blocks.[0-9]+.(norm1.linear|attn.to_q|attn.to_k)
#   .*single_transformer_blocks.[0-9]+.(norm.linear|attn.to_q|attn.to_k)
LORA_TARGET_REGEX_OMINICONTROL = (
    r"(.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*single_transformer_blocks\.[0-9]+\.norm\.linear"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_q)"
)

# Aggressive preset: full attention (Q/K/V/O) + AdaLN linear.
LORA_TARGET_REGEX_ATTN_QKVO = (
    r"(.*(?<!single_)transformer_blocks\.[0-9]+\.norm1\.linear"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.(to_q|to_k|to_v|to_out\.0)"
    r"|.*single_transformer_blocks\.[0-9]+\.norm\.linear"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.(to_q|to_k|to_v))"
)

# Minimal preset: only Q, K (no AdaLN).
LORA_TARGET_REGEX_QK_ONLY = (
    r"(.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*(?<!single_)transformer_blocks\.[0-9]+\.attn\.to_q"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_k"
    r"|.*single_transformer_blocks\.[0-9]+\.attn\.to_q)"
)

LORA_TARGET_PRESETS = {
    'ominicontrol': LORA_TARGET_REGEX_OMINICONTROL,
    'attn_qkvo':    LORA_TARGET_REGEX_ATTN_QKVO,
    'qk_only':      LORA_TARGET_REGEX_QK_ONLY,
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
        lr_w, lr_h = lr_img.size
        crop_size = self.resolution
        lr_crop_size = crop_size // self.scale

        if hr_w >= crop_size and hr_h >= crop_size:
            if self.is_val:
                # Center crop aligned to scale to keep HR/LR perfectly matched
                x = (hr_w - crop_size) // 2
                y = (hr_h - crop_size) // 2
                x = x - (x % self.scale)
                y = y - (y % self.scale)
            else:
                # Sample on LR grid first, then map to HR grid
                lr_x = np.random.randint(0, lr_w - lr_crop_size + 1)
                lr_y = np.random.randint(0, lr_h - lr_crop_size + 1)
                x = lr_x * self.scale
                y = lr_y * self.scale

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
# Dual-Stream FLUX SR System - Official Scheduler + optional LoRA
# ============================================================================

class DualStreamFLUXSR(nn.Module):
    """
    Dual-Stream FLUX SR System - 使用官方 Scheduler

    Optional LoRA on the Transformer (see `use_lora` / `lora_*` args).
    When `use_lora=False` this class is behaviorally identical to V1.
    """

    def __init__(self, model_name, device, pretrained_controlnet=None,
                 train_controlnet=True, pixel_weight=1.0,
                 control_guidance_start=0.0, control_guidance_end=1.0,
                 conditioning_scale=1.0,
                 use_lora=False, lora_rank=16, lora_alpha=16,
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
        import time

        dtype = torch.bfloat16
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank > 0:
            time.sleep(local_rank * 5)

        # Load Scheduler
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

        # Cache text embeddings
        print(f"[Rank {local_rank}] Caching text embeddings...")
        self._cache_text_embeddings()

        # Load Transformer (frozen base; LoRA applied after)
        print(f"[Rank {local_rank}] Loading FLUX Transformer...")
        self.transformer = self._from_pretrained_with_dtype(
            FluxTransformer2DModel, self.model_name, subfolder="transformer", dtype=dtype
        ).to(self.device)
        # Freeze all base params first.
        self.transformer.requires_grad_(False)
        self.transformer.eval()

        # Enable xformers on the base Transformer BEFORE LoRA wrap to avoid
        # attribute-forwarding subtleties in PeftModel wrappers.
        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            if local_rank == 0:
                print("[Flash] ✓ Enabled xformers memory efficient attention on Transformer")
        except Exception:
            if local_rank == 0:
                print("[Flash] Transformer: using PyTorch 2.0 SDPA")

        # Apply LoRA if requested.
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

        try:
            self.controlnet.enable_xformers_memory_efficient_attention()
            if local_rank == 0:
                print("[Flash] ✓ Enabled xformers memory efficient attention on ControlNet")
        except Exception:
            if local_rank == 0:
                print("[Flash] ControlNet: using PyTorch 2.0 SDPA")

        # Pixel Feature Extractor
        print(f"[Rank {local_rank}] Initializing Pixel Feature Extractor...")
        self.pixel_extractor = PixelFeatureExtractor(latent_channels=16).to(self.device).to(dtype)
        self.pixel_fuse_proj = nn.Conv2d(16, 16, kernel_size=1).to(self.device).to(dtype)
        self._reset_pixel_fuse_proj_to_identity()
        self.pixel_gate_logit = nn.Parameter(torch.tensor(-1.0, device=self.device))
        self.pixel_extractor.train()
        self.pixel_fuse_proj.train()

        print(f"[Rank {local_rank}] ✓ All models loaded")

    def _apply_lora_to_transformer(self, local_rank=0):
        """Wrap self.transformer with a PEFT LoRA adapter."""
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT is required for --use_lora. Install with: pip install peft>=0.10"
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
        # get_peft_model returns a PeftModel that wraps the base Transformer.
        # After this call:
        #   - Base (non-LoRA) params remain requires_grad=False
        #   - LoRA adapter params are requires_grad=True
        self.transformer = get_peft_model(self.transformer, lora_config)

        # IMPORTANT: flip wrapper to train() so:
        #   - LoRA dropout actually takes effect when lora_dropout > 0
        #   - Diffusers' gradient_checkpointing (gated by self.training) activates
        # The base FLUX Transformer has no BatchNorm/Dropout so switching to train()
        # is a no-op for the frozen weights — only LoRA's extra paths observe it.
        self.transformer.train()

        if local_rank == 0:
            # Sanity check: print trainable param count (LoRA only).
            try:
                self.transformer.print_trainable_parameters()
            except Exception:
                # Some peft versions may require explicit call.
                n_trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
                n_total = sum(p.numel() for p in self.transformer.parameters())
                print(
                    f"[LoRA] trainable params: {n_trainable:,} "
                    f"|| all params: {n_total:,} "
                    f"|| trainable%: {100.0 * n_trainable / max(1, n_total):.4f}"
                )

            # Module-count sanity: verify that LoRA actually matched modules.
            # Don't rely on class-name heuristics (varies across peft versions).
            # Instead use two robust signals:
            #   1. get_peft_model_state_dict returns LoRA-only tensors — must be non-empty.
            #   2. Modules with a `lora_A` attribute (BaseTunerLayer API) — must be > 0.
            try:
                lora_sd = get_peft_model_state_dict(self.transformer)
                n_lora_tensors = len(lora_sd)
            except Exception as e:
                lora_sd = {}
                n_lora_tensors = 0
                print(f"[LoRA][WARN] get_peft_model_state_dict failed: {e}")

            lora_layer_modules = []
            for n, m in self.transformer.named_modules():
                has_lora_a = hasattr(m, 'lora_A')
                if has_lora_a:
                    # lora_A may be ModuleDict / dict / ParameterDict; must be non-empty.
                    lora_a_attr = getattr(m, 'lora_A')
                    try:
                        if len(lora_a_attr) > 0:
                            lora_layer_modules.append(n)
                    except TypeError:
                        # lora_A not container-like; count it anyway.
                        lora_layer_modules.append(n)

            n_trainable_lora = sum(
                p.numel() for p in self.transformer.parameters() if p.requires_grad
            )

            print(
                f"[LoRA] Matched {len(lora_layer_modules)} LoRA-wrapped layers "
                f"(state tensors={n_lora_tensors}, trainable params={n_trainable_lora:,})"
            )
            if n_lora_tensors == 0 or n_trainable_lora == 0:
                raise RuntimeError(
                    "[LoRA] target_modules appears to have matched ZERO modules "
                    f"(state_dict_tensors={n_lora_tensors}, "
                    f"trainable_params={n_trainable_lora}, "
                    f"wrapped_layers={len(lora_layer_modules)}). "
                    "Check that the regex fits the installed diffusers FLUX module names. "
                    "Run with --dry_run_lora to inspect the regex match."
                )

    def _get_transformer_base(self):
        """
        Return the underlying FluxTransformer2DModel regardless of LoRA wrapping.
        Useful for operations like `enable_gradient_checkpointing` which live on
        the base model (not on PeftModel).
        """
        if not self.use_lora:
            return self.transformer
        # PeftModel exposes get_base_model() (or .base_model.model)
        trans = self.transformer
        if hasattr(trans, 'get_base_model'):
            try:
                return trans.get_base_model()
            except Exception:
                pass
        if hasattr(trans, 'base_model') and hasattr(trans.base_model, 'model'):
            return trans.base_model.model
        return trans

    def enable_transformer_gradient_checkpointing(self):
        base = self._get_transformer_base()
        if hasattr(base, 'enable_gradient_checkpointing'):
            base.enable_gradient_checkpointing()
            return True
        return False

    def enable_controlnet_gradient_checkpointing(self):
        if hasattr(self.controlnet, 'enable_gradient_checkpointing'):
            self.controlnet.enable_gradient_checkpointing()
            return True
        return False

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

    # ------------------------------------------------------------------
    # Trainable param groups
    # ------------------------------------------------------------------
    def get_pixel_branch_params(self):
        params = list(self.pixel_extractor.parameters())
        params += list(self.pixel_fuse_proj.parameters())
        params.append(self.pixel_gate_logit)
        return params

    def get_lora_params(self):
        """Return only LoRA adapter params (empty list when not using LoRA)."""
        if not self.use_lora:
            return []
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def get_trainable_params(self):
        params = self.get_pixel_branch_params()
        if self.train_controlnet:
            params += list(self.controlnet.parameters())
        if self.use_lora:
            params += self.get_lora_params()
        return params

    # ------------------------------------------------------------------
    # Latent encode / decode
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # FLUX token packing helpers
    # ------------------------------------------------------------------
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
                pass
        self.scheduler.set_timesteps(num_steps, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, noisy, lr_lat, lr_pixel, timestep, guidance=3.5, controlnet_scale=1.0):
        """
        Forward pass: predict velocity

        Args:
            noisy: current noisy latent
            lr_lat: LR latent
            lr_pixel: LR pixel tensor
            timestep: sigma (already / 1000 semantics)
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

        # Transformer (wrapped with LoRA PeftModel if use_lora=True — the call
        # signature is unchanged because PeftModel forwards **kwargs to the base
        # model).
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, strength=0.7):
        """
        Official scheduler-based inference.

        Args:
            strength: 官方 img2img 语义
                     1.0 = 纯噪声起点（完整去噪）
                     0.7 = 跳过前 30% 步数
        """
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16

        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)

        self._set_scheduler_timesteps(num_steps, device, lr_lat)
        timesteps = self.scheduler.timesteps

        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        if len(timesteps) == 0:
            raise ValueError(
                f"No timesteps left after applying strength={strength}. "
                f"Please increase num_steps (current: {num_steps}) or strength."
            )

        self.scheduler.set_begin_index(t_start)

        noise = torch.randn_like(lr_lat)
        timestep_batch = timesteps[:1].expand(B)
        latents = self.scheduler.scale_noise(lr_lat, timestep_batch, noise)

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
# Training Functions
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
    Flow Matching Loss (FLUX official scheduler alignment).

    Training:
        1. sample sigma from scheduler.sigmas (preserves shift / dynamic_shift)
        2. noisy = sigma * noise + (1 - sigma) * hr_lat
        3. target_v = noise - hr_lat
        4. model input timestep = sigma
    """
    B = hr_lat.shape[0]
    device = hr_lat.device
    dtype = torch.bfloat16

    unwrapped = system.module if hasattr(system, 'module') else system
    unwrapped._set_scheduler_timesteps(num_train_timesteps, device, hr_lat)
    sched_sigmas = unwrapped.scheduler.sigmas[:-1] if unwrapped.scheduler.sigmas.shape[0] > 1 else unwrapped.scheduler.sigmas
    sigma_idx = torch.randint(0, sched_sigmas.shape[0], (B,), device=device)
    sigma = sched_sigmas.to(device=device, dtype=dtype)[sigma_idx]
    noise = torch.randn_like(hr_lat)

    sigma_expand = sigma.view(B, 1, 1, 1)
    noisy = sigma_expand * noise + (1 - sigma_expand) * hr_lat
    target_v = noise - hr_lat

    # IMPORTANT: keep forward on wrapped `system` so DDP/DeepSpeed hooks remain active.
    v_pred = system(noisy, lr_lat, lr_pixel, sigma, guidance)

    loss = F.mse_loss(v_pred.float(), target_v.float())

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
def validate(system, accelerator, val_loader, device, num_samples=10,
             num_steps=20, guidance=3.5, strength=0.7):
    """Validation with official scheduler-based inference."""
    unwrapped = accelerator.unwrap_model(system)
    unwrapped.pixel_extractor.eval()
    unwrapped.pixel_fuse_proj.eval()
    unwrapped.controlnet.eval()
    # Transformer: when LoRA is active the wrapper was flipped to train() during
    # training; put it back to eval() for validation so LoRA dropout is off.
    if unwrapped.use_lora:
        unwrapped.transformer.eval()

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

    # Restore training mode
    unwrapped.pixel_extractor.train()
    unwrapped.pixel_fuse_proj.train()
    if unwrapped.train_controlnet:
        unwrapped.controlnet.train()
    if unwrapped.use_lora:
        unwrapped.transformer.train()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return np.mean(psnr_list) if psnr_list else 0.0


# ============================================================================
# Checkpoint save / load
# ============================================================================

def save_checkpoint(system, accelerator, epoch, loss, psnr, pixel_weight, strength,
                    control_guidance_start, control_guidance_end, path,
                    lpips_weight=0.0, lpips_apply_prob=0.25,
                    lora_config=None,
                    optimizer=None, lr_scheduler=None, global_step=None):
    """Save checkpoint. When LoRA is active, also store the LoRA state dict.

    If `optimizer` / `lr_scheduler` / `global_step` are provided, their states are
    also persisted so `--resume` can continue the LR schedule without discontinuity.
    """
    unwrapped = accelerator.unwrap_model(system)
    payload = {
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
        'use_lora': unwrapped.use_lora,
    }
    if unwrapped.use_lora:
        # Extract only the LoRA adapter weights (not the full Transformer!)
        lora_state = get_peft_model_state_dict(unwrapped.transformer)
        # Move to CPU for smaller checkpoint files.
        payload['lora_state_dict'] = {k: v.detach().cpu() for k, v in lora_state.items()}
        payload['lora_config'] = lora_config or {
            'rank': unwrapped.lora_rank,
            'alpha': unwrapped.lora_alpha,
            'dropout': unwrapped.lora_dropout,
            'target_regex': unwrapped.lora_target_regex,
            'init_weights': unwrapped.lora_init_weights,
        }

    if optimizer is not None:
        try:
            payload['optimizer'] = optimizer.state_dict()
        except Exception as e:
            print(f"[Checkpoint][WARN] optimizer state not saved: {e}")
    if lr_scheduler is not None:
        try:
            payload['lr_scheduler'] = lr_scheduler.state_dict()
        except Exception as e:
            print(f"[Checkpoint][WARN] lr_scheduler state not saved: {e}")
    if global_step is not None:
        payload['global_step'] = int(global_step)

    torch.save(payload, path)


def load_checkpoint_into_system(ckpt, system, accelerator, args, is_main=False,
                                optimizer=None, lr_scheduler=None):
    """
    Load the weights from `ckpt` dict into `system` (already wrapped in accelerator).
    Optionally restores optimizer / lr_scheduler state.
    Returns: (start_epoch, best_psnr, global_step)
    """
    unwrapped = accelerator.unwrap_model(system)

    # LoRA compat pre-check: warn if config differs, bail if rank/preset mismatch
    # would make the state dict unloadable.
    if unwrapped.use_lora and ckpt.get('use_lora', False):
        ckpt_cfg = ckpt.get('lora_config', {}) or {}
        cur_cfg = {
            'rank': unwrapped.lora_rank,
            'alpha': unwrapped.lora_alpha,
            'dropout': unwrapped.lora_dropout,
            'target_preset': getattr(args, 'lora_target_preset', None),
            'target_regex': unwrapped.lora_target_regex,
        }
        mismatches = []
        for key in ('rank', 'alpha', 'target_preset', 'target_regex'):
            if key in ckpt_cfg and key in cur_cfg and ckpt_cfg[key] != cur_cfg[key]:
                mismatches.append((key, ckpt_cfg[key], cur_cfg[key]))
        if is_main and mismatches:
            print("[LoRA][Resume][WARN] Config mismatch between checkpoint and current run:")
            for key, v_ckpt, v_cur in mismatches:
                print(f"  - {key}: ckpt={v_ckpt!r}  current={v_cur!r}")
            # Hard fail for rank/target_regex: state dict shapes will break.
            hard_fail_keys = {k for k, _, _ in mismatches if k in ('rank', 'target_regex')}
            if hard_fail_keys:
                raise RuntimeError(
                    f"[LoRA][Resume] Incompatible LoRA config for keys {hard_fail_keys}. "
                    "Use matching --lora_rank / --lora_target_preset or resume into a "
                    "fresh run (omit --resume)."
                )

    if 'pixel_extractor' in ckpt:
        state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
        unwrapped.pixel_extractor.load_state_dict(state)
    if 'pixel_fuse_proj' in ckpt:
        state = {k.replace('module.', ''): v for k, v in ckpt['pixel_fuse_proj'].items()}
        unwrapped.pixel_fuse_proj.load_state_dict(state)
    if 'pixel_gate_logit' in ckpt:
        gate = ckpt['pixel_gate_logit']
        if isinstance(gate, torch.Tensor):
            gate = gate.to(device=unwrapped.pixel_gate_logit.device,
                           dtype=unwrapped.pixel_gate_logit.dtype)
            unwrapped.pixel_gate_logit.data.copy_(gate)
        else:
            unwrapped.pixel_gate_logit.data.fill_(float(gate))
    elif 'pixel_fuse_proj' not in ckpt:
        with torch.no_grad():
            unwrapped._reset_pixel_fuse_proj_to_identity()
            unwrapped.pixel_gate_logit.fill_(6.0)
        if is_main:
            print("[Resume] Legacy checkpoint detected: initialize gated fusion to near-identity.")

    if args.train_controlnet and 'controlnet' in ckpt:
        state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
        unwrapped.controlnet.load_state_dict(state)

    # LoRA resume
    ckpt_uses_lora = bool(ckpt.get('use_lora', False)) and ('lora_state_dict' in ckpt)
    if unwrapped.use_lora and ckpt_uses_lora:
        if set_peft_model_state_dict is None:
            raise ImportError("PEFT is required to resume a LoRA checkpoint.")
        set_peft_model_state_dict(unwrapped.transformer, ckpt['lora_state_dict'])
        if is_main:
            cfg = ckpt.get('lora_config', {})
            print(
                f"[Resume] Loaded LoRA adapter: "
                f"{len(ckpt['lora_state_dict'])} tensors, config={cfg}"
            )
    elif unwrapped.use_lora and not ckpt_uses_lora:
        if is_main:
            print("[Resume] Current run uses LoRA but checkpoint has no LoRA state; LoRA stays at init.")
    elif (not unwrapped.use_lora) and ckpt_uses_lora:
        if is_main:
            print("[Resume] Checkpoint contains LoRA state but --use_lora is off; ignoring LoRA weights.")

    start_epoch = ckpt.get('epoch', 0) + 1
    best_psnr = ckpt.get('psnr', 0.0)
    global_step = int(ckpt.get('global_step', 0) or 0)

    if 'control_guidance_start' in ckpt:
        args.control_guidance_start = ckpt['control_guidance_start']
        unwrapped.control_guidance_start = ckpt['control_guidance_start']
    if 'control_guidance_end' in ckpt:
        args.control_guidance_end = ckpt['control_guidance_end']
        unwrapped.control_guidance_end = ckpt['control_guidance_end']
    if 'conditioning_scale' in ckpt:
        unwrapped.conditioning_scale = ckpt['conditioning_scale']

    # Restore optimizer / lr_scheduler so resume-from-ckpt preserves LR trajectory.
    if optimizer is not None and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            if is_main:
                print("[Resume] Optimizer state restored.")
        except Exception as e:
            if is_main:
                print(f"[Resume][WARN] Optimizer state NOT restored ({e}); starting fresh.")
    elif is_main and optimizer is not None:
        print("[Resume][WARN] Checkpoint has no optimizer state; starting fresh.")

    if lr_scheduler is not None and 'lr_scheduler' in ckpt:
        try:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            if is_main:
                print(
                    f"[Resume] LR scheduler state restored "
                    f"(last_epoch={getattr(lr_scheduler, 'last_epoch', '?')})."
                )
        except Exception as e:
            if is_main:
                print(f"[Resume][WARN] LR scheduler state NOT restored ({e}); will "
                      f"re-init and skip forward to the new epoch.")
    elif is_main and lr_scheduler is not None:
        print("[Resume][WARN] Checkpoint has no LR scheduler state; starting fresh.")

    return start_epoch, best_psnr, global_step


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Dual-Stream FLUX SR Training + LoRA (based on V1, paired bicubic)'
    )

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
    parser.add_argument('--val_num_steps', type=int, default=20,
                        help='Validation num inference steps (default 20; lower values '
                             'significantly understate PSNR for FLUX)')

    # LoRA
    parser.add_argument('--use_lora', action='store_true', default=False,
                        help='Enable LoRA adapters on the FLUX Transformer')
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
                        help='Initialize model + LoRA, print matched modules / param '
                             'counts, and exit (useful to validate target_modules regex).')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/dual_lora')
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

    # Resolve LoRA target regex from preset.
    lora_target_regex = LORA_TARGET_PRESETS[args.lora_target_preset]
    # PEFT accepts strings 'gaussian'/'default' as well as bool; pass through.
    if args.lora_init_weights in ('true', 'false'):
        lora_init_weights = (args.lora_init_weights == 'true')
    else:
        lora_init_weights = args.lora_init_weights

    if args.use_lora and not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT is required for --use_lora. Install with: pip install peft>=0.10"
        )

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main = accelerator.is_main_process
    set_seed(args.seed)
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
        f"_str{_fmt_tag(args.strength)}"
        f"_pw{_fmt_tag(args.pixel_weight)}"
        f"_gate{_fmt_tag(args.pixel_gate_init)}"
        f"_lpw{_fmt_tag(args.lpips_weight)}"
        f"_lpp{_fmt_tag(args.lpips_apply_prob)}"
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
        print("FLUX SR Training + LoRA (based on V1)")
        print("=" * 70)
        print(f"\nHR Dir: {args.hr_dir}")
        print(f"LR Dir: {args.lr_dir}")
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
        if args.use_lora:
            print(
                f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                f"dropout={args.lora_dropout}, lr={args.lora_lr}, "
                f"preset={args.lora_target_preset}"
            )
        else:
            print("LoRA: disabled (behavior identical to V1)")
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
    with torch.no_grad():
        system.pixel_gate_logit.fill_(args.pixel_gate_init)

    # --dry_run_lora: print full target-module diagnostic and exit.
    if args.dry_run_lora:
        if not args.use_lora:
            raise SystemExit("[dry_run_lora] requires --use_lora to be set.")
        if is_main:
            print("\n" + "=" * 70)
            print("[dry_run_lora] LoRA regex diagnostic")
            print("=" * 70)
            print(f"Target preset: {args.lora_target_preset}")
            print(f"Target regex:  {lora_target_regex}")
            # Enumerate wrapped LoRA layers
            wrapped = []
            for n, m in system.transformer.named_modules():
                if hasattr(m, 'lora_A'):
                    try:
                        if len(m.lora_A) > 0:
                            wrapped.append(n)
                    except TypeError:
                        wrapped.append(n)
            print(f"\nWrapped modules ({len(wrapped)}):")
            for n in wrapped[:200]:
                print(f"  {n}")
            if len(wrapped) > 200:
                print(f"  ... ({len(wrapped) - 200} more)")
            try:
                sd = get_peft_model_state_dict(system.transformer)
                print(f"\nLoRA state tensors: {len(sd)}")
            except Exception as e:
                print(f"\nget_peft_model_state_dict failed: {e}")
            n_trainable = sum(p.numel() for p in system.transformer.parameters() if p.requires_grad)
            print(f"LoRA trainable params: {n_trainable:,}")
            print("=" * 70)
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

    # Enable gradient checkpointing (LoRA-aware: acts on the base FLUX Transformer)
    if system.enable_transformer_gradient_checkpointing():
        if is_main:
            print("[GradCkpt] ✓ Transformer gradient checkpointing enabled")
    if system.enable_controlnet_gradient_checkpointing():
        if is_main:
            print("[GradCkpt] ✓ ControlNet gradient checkpointing enabled")

    # Optimizer groups
    pixel_branch_params = system.get_pixel_branch_params()
    lora_params = system.get_lora_params()

    optimizer_grouped_parameters = []
    if args.train_controlnet:
        optimizer_grouped_parameters.append(
            {"params": system.controlnet.parameters(), "lr": args.lr}
        )
    optimizer_grouped_parameters.append(
        {"params": pixel_branch_params, "lr": args.lr * 10.0}
    )
    if args.use_lora:
        if len(lora_params) == 0:
            raise RuntimeError(
                "[LoRA] --use_lora was set but no trainable LoRA params were found. "
                "Check target_modules regex vs. Transformer module names."
            )
        optimizer_grouped_parameters.append(
            {"params": lora_params, "lr": args.lora_lr}
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

    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)

    num_training_steps = args.epochs * len(train_loader)
    num_warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Prepare with accelerate
    system, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        system, optimizer, train_loader, lr_scheduler
    )

    # DeepSpeed ZeRO-2 与 accelerate.accumulate(no_sync) 不兼容
    use_accumulate = accelerator.distributed_type != DistributedType.DEEPSPEED
    if is_main and not use_accumulate:
        print("[Training] DeepSpeed detected: disable accelerator.accumulate() to avoid no_sync assertion.")

    # Resume
    start_epoch = 0
    best_psnr = 0.0

    global_step = 0
    if args.resume:
        if is_main:
            print(f"[Resume] Loading from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch, best_psnr, global_step = load_checkpoint_into_system(
            ckpt, system, accelerator, args, is_main=is_main,
            optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        if is_main:
            print(
                f"[Resume] Starting from epoch {start_epoch}, best PSNR: {best_psnr:.2f}, "
                f"global_step={global_step}"
            )

    # Log file
    log_path = os.path.join(save_dir, 'training_log.txt')
    if is_main:
        with open(log_path, 'w') as f:
            f.write("FLUX SR Training + LoRA\n")
            f.write("=" * 60 + "\n")
            f.write(f"Strength: {args.strength}\n")
            f.write(f"Pixel Weight: {args.pixel_weight}\n")
            f.write(f"Pixel Gate Init (logit): {args.pixel_gate_init}\n")
            f.write(f"Pixel Gate Init (sigmoid): {1.0 / (1.0 + math.exp(-args.pixel_gate_init)):.6f}\n")
            f.write(f"LPIPS Weight: {args.lpips_weight} (resize={args.lpips_resize}, prob={args.lpips_apply_prob})\n\n")
            f.write(f"Empty Cache Steps: {args.empty_cache_steps}\n\n")
            f.write(f"Conditioning Scale: {args.conditioning_scale}\n")
            f.write(f"Control Guidance Window: [{args.control_guidance_start}, {args.control_guidance_end}]\n")
            f.write(f"Train ControlNet: {args.train_controlnet}\n")
            if args.use_lora:
                f.write(
                    f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                    f"dropout={args.lora_dropout}, lr={args.lora_lr}, "
                    f"preset={args.lora_target_preset}\n"
                )
            else:
                f.write("LoRA: disabled\n")
            f.write("\n")

    # Training loop
    if is_main:
        print("\n[Training] Starting...\n")
    # global_step already initialized above (possibly restored from resume).
    avg_loss = 0.0
    for epoch in range(start_epoch, args.epochs):
        unwrapped = accelerator.unwrap_model(system)
        unwrapped.pixel_extractor.train()
        unwrapped.pixel_fuse_proj.train()
        if args.train_controlnet:
            unwrapped.controlnet.train()
        if args.use_lora:
            # Keep LoRA wrapper in train() so LoRA dropout + grad-ckpt activate.
            # Base weights are frozen regardless of this flag — FLUX has no
            # BatchNorm/Dropout outside LoRA paths.
            unwrapped.transformer.train()

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
            postfix = {'loss': f'{loss_item:.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'}
            if args.use_lora and len(lr_scheduler.get_last_lr()) >= 3:
                # The last group is LoRA's LR.
                postfix['lora_lr'] = f'{lr_scheduler.get_last_lr()[-1]:.2e}'
            pbar.set_postfix(postfix)
            global_step += 1

            del hr, lr, hr_lat, lr_lat, loss
            if args.empty_cache_steps > 0 and (global_step % args.empty_cache_steps == 0):
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Validation
        val_psnr = 0.0
        if val_loader and (epoch + 1) % args.val_interval == 0:
            if is_main:
                val_psnr = validate(system, accelerator, val_loader, device,
                                   num_samples=10, num_steps=args.val_num_steps,
                                   guidance=args.guidance, strength=args.strength)

        if is_main:
            lr_current = lr_scheduler.get_last_lr()[0]
            gate_value = torch.sigmoid(unwrapped.pixel_gate_logit.detach().float()).item()
            log_line = (
                f"Epoch {epoch+1}: Loss={avg_loss:.6f}, PSNR={val_psnr:.2f}, "
                f"LR={lr_current:.2e}, Gate={gate_value:.4f}"
            )
            if args.use_lora:
                log_line += f", LoRA_LR={lr_scheduler.get_last_lr()[-1]:.2e}"
            log_line += "\n"
            with open(log_path, 'a') as f:
                f.write(log_line)

            print(
                f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_psnr={val_psnr:.2f} dB, "
                f"lr={lr_current:.2e}, gate={gate_value:.4f}"
                + (f", lora_lr={lr_scheduler.get_last_lr()[-1]:.2e}" if args.use_lora else "")
            )

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

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.strength,
                               args.control_guidance_start, args.control_guidance_end,
                               os.path.join(save_dir, 'best_model.pt'),
                               lpips_weight=args.lpips_weight,
                               lpips_apply_prob=args.lpips_apply_prob,
                               lora_config=lora_cfg_for_ckpt,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               global_step=global_step)
                print(f"  → New best PSNR: {best_psnr:.2f} dB")

            if (epoch + 1) % args.save_interval == 0:
                save_checkpoint(system, accelerator, epoch, avg_loss, val_psnr,
                               args.pixel_weight, args.strength,
                               args.control_guidance_start, args.control_guidance_end,
                               os.path.join(save_dir, f'epoch{epoch+1}.pt'),
                               lpips_weight=args.lpips_weight,
                               lpips_apply_prob=args.lpips_apply_prob,
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
        save_checkpoint(system, accelerator, args.epochs - 1, avg_loss, best_psnr,
                       args.pixel_weight, args.strength,
                       args.control_guidance_start, args.control_guidance_end,
                       os.path.join(save_dir, 'final_model.pt'),
                       lpips_weight=args.lpips_weight,
                       lpips_apply_prob=args.lpips_apply_prob,
                       lora_config=lora_cfg_for_ckpt)

        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {save_dir}")
        print("=" * 70)


if __name__ == '__main__':
    main()
