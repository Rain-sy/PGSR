#!/usr/bin/env python
"""
======================================================================
PGSR CLEAR FLUX SR Evaluation - aligned with official Diffusers pipeline
======================================================================

Companion script for train_pgsr_clear.py
Automatically detects and loads LoRA adapters when checkpoint contains
`use_lora=True` and `lora_state_dict`.

Usage:
    CUDA_VISIBLE_DEVICES=0 python evaluate_pgsr_clear.py \
        --checkpoint checkpoints/dual_control/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --iqa_device cuda \
        --lpips_device cuda
"""

import os
import sys
import gc
import math
import argparse
import json
import shutil
import tempfile
import time
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import FlowMatchEulerDiscreteScheduler

# Make CLEAR/attention_processor.py importable when this script is launched
# from the repo root. CLEAR is optional and only imported when requested.
_CLEAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLEAR")
if os.path.isdir(_CLEAR_DIR) and _CLEAR_DIR not in sys.path:
    sys.path.insert(0, _CLEAR_DIR)

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Note: lpips not installed. Run: pip install lpips")

try:
    from skimage.metrics import structural_similarity as sk_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    sk_ssim = None
    SKIMAGE_AVAILABLE = False
    print("Note: scikit-image not installed; falling back to global SSIM. Run: pip install scikit-image")

# pyiqa supplies MUSIQ / MANIQA / CLIP-IQA / NIQE / DISTS / FID. Optional —
# evaluation falls back to PSNR/SSIM/LPIPS only when not installed.
try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    pyiqa = None
    PYIQA_AVAILABLE = False
    print("Note: pyiqa not installed. NR-IQA (MUSIQ/MANIQA/CLIP-IQA/NIQE/DISTS/FID) "
          "disabled. Install: pip install pyiqa")

# PEFT is optional; only required when checkpoint contains a LoRA adapter.
try:
    from peft import LoraConfig, get_peft_model
    from peft.utils import set_peft_model_state_dict
    PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None
    get_peft_model = None
    set_peft_model_state_dict = None
    PEFT_AVAILABLE = False


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
# Pixel Feature Extractor
# ============================================================================

class PixelFeatureExtractor(nn.Module):
    """Inference-time extractor mirrored from ``train_dual_dfm.py``.

    Uses the stage-grouped architecture with widened ``s3`` tap (256ch):
      s1: 32ch @ H/2, s2: 64ch @ H/4, s3: 256ch @ H/8.
    """

    _TAP_CHANNELS = {'s1': 32, 's2': 64, 's3': 256}

    def __init__(self, latent_channels=16):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.stage3_body = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
        )
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(256, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, latent_channels),
            nn.SiLU(),
        )

        # Name kept as ``zero_conv`` for checkpoint-key compatibility, but
        # weights are kaiming-init -- see train_dual_dfm.py for the
        # dead-gradient explanation.
        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.zero_conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x, return_features=False):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3_body(s2)
        out = self.zero_conv(self.stage3_proj(s3))
        if return_features:
            return out, {'s1': s1, 's2': s2, 's3': s3}
        return out


class DFMAdapter(nn.Module):
    """Inference-time DFM adapter (mirror of train_dual_dfm.DFMAdapter).

    Residual + zero-conv injection:
        decoder_feat <- decoder_feat + zero_conv(silu(align(pixel_feat)))
    """

    def __init__(self, feat_ch: int, decoder_ch: int):
        super().__init__()
        self.align = nn.Conv2d(feat_ch, decoder_ch, kernel_size=3, padding=1)
        self.zero_conv = nn.Conv2d(decoder_ch, decoder_ch, kernel_size=1)
        # Matches train-side init. Checkpoint values overwrite these immediately.
        nn.init.zeros_(self.zero_conv.weight); nn.init.zeros_(self.zero_conv.bias)
        nn.init.kaiming_normal_(self.align.weight, nonlinearity='relu')
        nn.init.zeros_(self.align.bias)

    def forward(self, decoder_feat: torch.Tensor, pixel_feat: torch.Tensor) -> torch.Tensor:
        if pixel_feat.shape[-2:] != decoder_feat.shape[-2:]:
            pixel_feat = F.interpolate(
                pixel_feat, size=decoder_feat.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        if pixel_feat.dtype != decoder_feat.dtype:
            pixel_feat = pixel_feat.to(decoder_feat.dtype)
        return decoder_feat + self.zero_conv(F.silu(self.align(pixel_feat)))


# ============================================================================
# Metrics
# ============================================================================

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calculate_ssim(img1, img2):
    """SSIM on uint8 RGB images. Prefers scikit-image's 11x11 sliding-window
    implementation (standard in SR benchmarks); falls back to a single-window
    global SSIM only if skimage is unavailable."""
    def _global_ssim():
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)
        mu1, mu2 = x.mean(), y.mean()
        sigma1_sq, sigma2_sq = x.var(), y.var()
        sigma12 = ((x - mu1) * (y - mu2)).mean()
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_val)

    if SKIMAGE_AVAILABLE:
        min_side = min(img1.shape[0], img1.shape[1])
        if min_side < 3:
            return _global_ssim()
        win_size = min(11, int(min_side))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            return _global_ssim()
        # channel_axis API differs between skimage versions; try both.
        try:
            return float(sk_ssim(img1, img2, data_range=255, channel_axis=-1, win_size=win_size))
        except TypeError:
            try:
                return float(sk_ssim(img1, img2, data_range=255, multichannel=True, win_size=win_size))
            except ValueError:
                return _global_ssim()
        except ValueError:
            return _global_ssim()
    return _global_ssim()


def pad_tensor_to_multiple(x, multiple=8, mode='replicate'):
    """Pad BCHW tensor so H/W are divisible by ``multiple``."""
    _, _, h, w = x.shape
    pad_h = (-h) % int(multiple)
    pad_w = (-w) % int(multiple)
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), pad_h, pad_w


# ============================================================================
# IQA metric registry (pyiqa-backed: MUSIQ / MANIQA / CLIP-IQA / NIQE / DISTS)
# FID is computed separately from saved PNG directories at the end.
# ============================================================================

# Per-metric config: backend name in pyiqa, type (NR/FR), and convention.
IQA_REGISTRY = {
    'musiq':   {'pyiqa': 'musiq',         'type': 'NR', 'higher_better': True},
    'maniqa':  {'pyiqa': 'maniqa',        'type': 'NR', 'higher_better': True},
    'clipiqa': {'pyiqa': 'clipiqa',       'type': 'NR', 'higher_better': True},
    'niqe':    {'pyiqa': 'niqe',          'type': 'NR', 'higher_better': False},
    'dists':   {'pyiqa': 'dists',         'type': 'FR', 'higher_better': False},
}


class IQAMetricsRunner:
    """Manage pyiqa metrics: lazy init, per-image update, finalize.

    All inputs to ``update`` are uint8 numpy HWC; conversion to BCHW [0,1] on
    ``self.device`` happens internally. Failed metrics push NaN so downstream
    means are robust.
    """

    def __init__(self, device, requested, fr_max_long_edge=1024):
        self.device = device
        # FR metrics like DISTS use VGG features; full-res 2K-4K inputs are
        # both very slow and memory-heavy. Resize so the long edge is at most
        # this value before computing FR metrics. NR metrics already do their
        # own internal resize so this only affects FR.
        self.fr_max_long_edge = int(fr_max_long_edge) if fr_max_long_edge else 0
        self.metrics = {}        # name -> pyiqa Metric module
        self.values_sr = {}      # name -> list of floats (SR vs HR)
        self.values_bic = {}     # name -> list of floats (bicubic vs HR)
        if not PYIQA_AVAILABLE:
            return
        for name in requested:
            if name not in IQA_REGISTRY:
                print(f"[IQA] unknown metric '{name}', skipped.")
                continue
            cfg = IQA_REGISTRY[name]
            try:
                m = pyiqa.create_metric(cfg['pyiqa'], device=device, as_loss=False)
                m.eval()
                self.metrics[name] = m
                self.values_sr[name] = []
                self.values_bic[name] = []
                print(f"[IQA] loaded {name} ({cfg['type']}, "
                      f"{'higher' if cfg['higher_better'] else 'lower'} better)")
            except Exception as e:
                print(f"[IQA] failed to init {name}: {e}")

    @staticmethod
    def _to_tensor01(img_uint8_hwc, device):
        t = torch.from_numpy(img_uint8_hwc).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return t.clamp(0, 1).to(device)

    def _maybe_resize_for_fr(self, t):
        """Cap long edge for FR metrics so DISTS / VGG-based metrics stay fast.
        Returns either the original tensor or a bilinear-resized copy."""
        if self.fr_max_long_edge <= 0:
            return t
        _, _, h, w = t.shape
        long_edge = max(h, w)
        if long_edge <= self.fr_max_long_edge:
            return t
        scale = self.fr_max_long_edge / long_edge
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return F.interpolate(t, size=(new_h, new_w), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def update(self, sr_np, hr_np, bic_np):
        if not self.metrics:
            return
        sr_t = self._to_tensor01(sr_np, self.device)
        bic_t = self._to_tensor01(bic_np, self.device)
        hr_t = None  # lazy: only init for FR metrics
        sr_t_fr = bic_t_fr = hr_t_fr = None  # lazy resize
        for name, m in self.metrics.items():
            cfg = IQA_REGISTRY[name]
            try:
                if cfg['type'] == 'FR':
                    if hr_np is None:
                        sr_score = float('nan')
                        bic_score = float('nan')
                        self.values_sr[name].append(sr_score)
                        self.values_bic[name].append(bic_score)
                        continue
                    if hr_t is None:
                        hr_t = self._to_tensor01(hr_np, self.device)
                    if sr_t_fr is None:
                        sr_t_fr = self._maybe_resize_for_fr(sr_t)
                        bic_t_fr = self._maybe_resize_for_fr(bic_t)
                        hr_t_fr = self._maybe_resize_for_fr(hr_t)
                    sr_score = float(m(sr_t_fr, hr_t_fr).item())
                    bic_score = float(m(bic_t_fr, hr_t_fr).item())
                else:
                    sr_score = float(m(sr_t).item())
                    bic_score = float(m(bic_t).item())
            except Exception as e:
                print(f"[IQA] {name} failed on this image: {e}")
                sr_score = float('nan')
                bic_score = float('nan')
            self.values_sr[name].append(sr_score)
            self.values_bic[name].append(bic_score)
        del sr_t, bic_t
        if hr_t is not None:
            del hr_t
        if sr_t_fr is not None:
            del sr_t_fr, bic_t_fr, hr_t_fr
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def averages(self):
        out = {}
        for name in self.metrics:
            sr_vals = [v for v in self.values_sr[name] if not math.isnan(v)]
            bic_vals = [v for v in self.values_bic[name] if not math.isnan(v)]
            out[name] = {
                'sr': float(np.mean(sr_vals)) if sr_vals else None,
                'bic': float(np.mean(bic_vals)) if bic_vals else None,
                'higher_better': IQA_REGISTRY[name]['higher_better'],
                'type': IQA_REGISTRY[name]['type'],
            }
        return out


def compute_fid(sr_dir, hr_dir, device):
    """Compute FID(sr, hr) using pyiqa. Returns float or None on failure."""
    if not PYIQA_AVAILABLE:
        return None
    try:
        fid_metric = pyiqa.create_metric('fid', device=device)
        score = float(fid_metric(sr_dir, hr_dir))
        del fid_metric
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return score
    except Exception as e:
        print(f"[FID] failed: {e}")
        return None


def center_crop_or_pad_np(img_np, size):
    """Center crop large images and edge-pad small images to a fixed square."""
    if size is None or int(size) <= 0:
        return img_np
    size = int(size)
    h, w = img_np.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    cropped = img_np[y0:min(y0 + size, h), x0:min(x0 + size, w), :]
    pad_h = size - cropped.shape[0]
    pad_w = size - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        cropped = np.pad(
            cropped,
            ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
            mode='edge',
        )
    return cropped


def _crop_or_pad_patch(img_np, y, x, patch_size):
    """Return a fixed-size HWC patch, edge-padding small images if needed."""
    h, w = img_np.shape[:2]
    y = max(0, min(int(y), max(0, h - patch_size)))
    x = max(0, min(int(x), max(0, w - patch_size)))
    patch = img_np[y:min(y + patch_size, h), x:min(x + patch_size, w), :]
    pad_h = patch_size - patch.shape[0]
    pad_w = patch_size - patch.shape[1]
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(
            patch,
            ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
            mode='edge',
        )
    return patch


def _sample_patch_coords(h, w, patch_size, n_per_image, rng):
    """Sample crop coordinates for patch-based distribution metrics."""
    n = max(1, int(n_per_image))
    max_y = max(0, h - patch_size)
    max_x = max(0, w - patch_size)
    if max_y == 0 and max_x == 0:
        return [(0, 0)] * n
    ys = rng.integers(0, max_y + 1, size=n) if max_y > 0 else np.zeros(n, dtype=np.int64)
    xs = rng.integers(0, max_x + 1, size=n) if max_x > 0 else np.zeros(n, dtype=np.int64)
    return list(zip(ys.tolist(), xs.tolist()))


def save_fid_patches(sr_np, hr_np, bic_np, base_name, patch_dirs,
                     patch_size=512, n_per_image=25, patch_pairing='independent',
                     rng_sr=None, rng_bic=None, rng_hr=None):
    """Write SR/HR/bicubic patches for patch-based FID.

    The default protocol is intentionally *independent* patch sampling:
    SR/bicubic patches and HR patches are sampled from their own random
    coordinate streams. FID is a distribution distance, not a paired image
    metric; aligned crops artificially match content/location and can
    noticeably suppress the absolute FID scale.
    """
    rng_sr = rng_sr or np.random.default_rng(42)
    rng_bic = rng_bic or np.random.default_rng(42)
    rng_hr = rng_hr or np.random.default_rng(1000045)
    coords_sr = _sample_patch_coords(
        hr_np.shape[0], hr_np.shape[1], patch_size, n_per_image, rng_sr
    )
    if patch_pairing == 'aligned':
        coords_bic = coords_sr
        coords_hr = coords_sr
    else:
        coords_bic = _sample_patch_coords(
            hr_np.shape[0], hr_np.shape[1], patch_size, n_per_image, rng_bic
        )
        coords_hr = _sample_patch_coords(
            hr_np.shape[0], hr_np.shape[1], patch_size, n_per_image, rng_hr
        )
    for idx, ((y_sr, x_sr), (y_bic, x_bic), (y_hr, x_hr)) in enumerate(
        zip(coords_sr, coords_bic, coords_hr)
    ):
        patch_name = f'{base_name}_p{idx:03d}.png'
        Image.fromarray(_crop_or_pad_patch(sr_np, y_sr, x_sr, patch_size)).save(
            os.path.join(patch_dirs['sr'], patch_name)
        )
        Image.fromarray(_crop_or_pad_patch(hr_np, y_hr, x_hr, patch_size)).save(
            os.path.join(patch_dirs['hr'], patch_name)
        )
        Image.fromarray(_crop_or_pad_patch(bic_np, y_bic, x_bic, patch_size)).save(
            os.path.join(patch_dirs['bic'], patch_name)
        )
    return len(coords_sr)


def clear_memory(device):
    """Aggressively clear Python/CUDA memory."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ============================================================================
# Evaluator
# ============================================================================

class DualStreamEvaluator(nn.Module):
    def __init__(self, model_name, device, checkpoint_path, pixel_weight=1.0,
                 attention_mode='auto'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.pixel_weight = pixel_weight
        self.attention_mode_requested = attention_mode
        self.attention_mode = 'full'

        self.vae = None
        self.transformer = None
        self.controlnet = None
        self.pixel_extractor = None
        self.pixel_fuse_proj = None
        self.fusion_source = 'identity'  # 'concat' | 'migrated' | 'identity'
        self.scheduler = None
        self._cached_embeds = None
        self.strength = 0.7
        self.control_guidance_start = 0.0
        self.control_guidance_end = 1.0
        self.conditioning_scale = 1.0
        self.train_lpips_weight = 0.0
        self.train_lpips_apply_prob = 0.0
        self.train_data = {}
        self.use_lora = False
        self.lora_config_ckpt = None
        self.lora_state_tensors = 0

        # DFM state (filled in by .load() if checkpoint carries dfm_adapters)
        self.use_dfm = False
        self.dfm_adapters = nn.ModuleDict()
        self._dfm_up_to_tap = {0: 's3', 1: 's2', 2: 's1'}

        # Learnable meta text embedding state (filled in by .load() when the
        # checkpoint sets use_learnable_text_embed=True; e.g. ckpts produced
        # by train_pgsr.py / train_pgsr_clear.py). Auto-detected from the ckpt; no CLI flag needed.
        self.use_learnable_text_embed = False
        self.text_embed_format = None
        self.text_embed_tokens = 0
        self.text_embed_scale = 0.1
        self.learnable_text_embed = None

        # CLEAR local-attention state. Filled in by load() when the checkpoint
        # carries CLEAR processor weights and --attention_mode selects CLEAR.
        self.use_clear = False
        self.clear_config = {}
        self.clear_window_size = 16
        self.clear_down_factor = 4
        self.clear_resolution = 512
        self._clear_processors = []
        self._clear_mask_hw = None

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

        # Peek at the checkpoint metadata BEFORE caching text embeddings, so
        # _cache_text_embeddings knows whether to register a learnable
        # nn.Parameter (warm-started from cached empty-string T5). Loaded on
        # CPU to avoid GPU memory pressure; the full ckpt is re-loaded below.
        try:
            _ckpt_peek = torch.load(
                self.checkpoint_path, map_location='cpu', weights_only=False
            )
            self.use_learnable_text_embed = bool(
                _ckpt_peek.get('use_learnable_text_embed', False)
                or _ckpt_peek.get('text_embed_format', None) == 'residual_delta_v1'
            )
            self.text_embed_format = _ckpt_peek.get('text_embed_format', None)
            self.text_embed_tokens = int(_ckpt_peek.get('text_embed_tokens', 0) or 0)
            self.text_embed_scale = float(_ckpt_peek.get('text_embed_scale', 0.1))
            has_clear = bool(
                _ckpt_peek.get('clear_processors', None) is not None
                or _ckpt_peek.get('clear_config', None) is not None
            )
            if self.attention_mode_requested == 'auto':
                self.attention_mode = 'clear' if has_clear else 'full'
            else:
                self.attention_mode = self.attention_mode_requested
            del _ckpt_peek
        except Exception as _peek_err:
            print(f"[Eval][WARN] Could not peek at checkpoint for "
                  f"use_learnable_text_embed flag: {_peek_err}; assuming False.")
            self.use_learnable_text_embed = False
            self.attention_mode = 'full' if self.attention_mode_requested == 'auto' else self.attention_mode_requested

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
        # Concat-based fusion: [lr_lat || pixel_feat] (32ch) -> 16ch.
        self.pixel_fuse_proj = nn.Conv2d(32, 16, kernel_size=1).to(self.device).to(dtype)
        self._reset_pixel_fuse_proj_to_identity()
        self.pixel_fuse_proj.requires_grad_(False)
        self.pixel_fuse_proj.eval()

        print("Loading checkpoint...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if 'pixel_extractor' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['pixel_extractor'].items()}
            if any(k.startswith("encoder.") for k in state.keys()):
                raise RuntimeError(
                    "[Eval] Incompatible checkpoint: pixel_extractor uses legacy "
                    "'encoder.*' keys (train_dual_control style). "
                    "Use evaluate_dual_control.py for that checkpoint, or evaluate a "
                    "train_dual_dfm checkpoint with this script."
                )
            self.pixel_extractor.load_state_dict(state)

        if 'controlnet' in ckpt:
            state = {k.replace('module.', ''): v for k, v in ckpt['controlnet'].items()}
            self.controlnet.load_state_dict(state)

        self._load_pixel_fuse_proj_with_migration(ckpt)

        if 'pixel_weight' in ckpt:
            self.pixel_weight = ckpt['pixel_weight']
        self.conditioning_scale = ckpt.get('conditioning_scale', 1.0)
        self.strength = ckpt.get('strength', 0.7)
        self.control_guidance_start = ckpt.get('control_guidance_start', 0.0)
        self.control_guidance_end = ckpt.get('control_guidance_end', 1.0)
        self.train_lpips_weight = ckpt.get('lpips_weight', 0.0)
        self.train_lpips_apply_prob = ckpt.get('lpips_apply_prob', 0.0)
        train_data = ckpt.get('train_data', {})
        if not isinstance(train_data, dict):
            train_data = {}
        self.train_data = {
            'train_hr_dir': train_data.get('train_hr_dir', ckpt.get('train_hr_dir')),
            'train_lr_dir': train_data.get('train_lr_dir', ckpt.get('train_lr_dir')),
            'val_hr_dir': train_data.get('val_hr_dir', ckpt.get('val_hr_dir')),
            'val_lr_dir': train_data.get('val_lr_dir', ckpt.get('val_lr_dir')),
            'degrade_mode': train_data.get('degrade_mode', ckpt.get('degrade_mode')),
            'degradation_preset': train_data.get('degradation_preset', ckpt.get('degradation_preset')),
            'degradation_cfg': train_data.get('degradation_cfg', ckpt.get('degradation_cfg')),
            'usm_cfg': train_data.get('usm_cfg', ckpt.get('usm_cfg')),
            'scale': train_data.get('scale', ckpt.get('scale')),
            'resolution': train_data.get('resolution', ckpt.get('resolution')),
            'num_crops': train_data.get('num_crops', ckpt.get('num_crops')),
        }

        print(f"Checkpoint: epoch={ckpt.get('epoch', '?')}, psnr={ckpt.get('psnr', 0):.2f}")
        print(f"Pixel Weight: {self.pixel_weight}, Strength: {self.strength}")
        print(f"Pixel Fusion: concat+1x1 conv (source={self.fusion_source})")
        print(f"Train LPIPS: weight={self.train_lpips_weight}, prob={self.train_lpips_apply_prob}")
        if self.train_data.get('train_hr_dir'):
            print(
                f"Train Data: hr={self.train_data.get('train_hr_dir')}, "
                f"lr={self.train_data.get('train_lr_dir')}, "
                f"degrade={self.train_data.get('degrade_mode')}, "
                f"preset={self.train_data.get('degradation_preset')}, "
                f"scale=x{self.train_data.get('scale')}"
            )
        print(f"Conditioning Scale: {self.conditioning_scale}")
        print(f"Control Guidance Window: [{self.control_guidance_start}, {self.control_guidance_end}]")

        try:
            self.transformer.enable_xformers_memory_efficient_attention()
            self.controlnet.enable_xformers_memory_efficient_attention()
        except:
            pass

        self.use_lora = bool(ckpt.get('use_lora', False)) and ('lora_state_dict' in ckpt)
        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "[LoRA] Checkpoint contains a LoRA adapter but `peft` is not installed. "
                    "Install with: pip install peft>=0.10"
                )
            self.lora_config_ckpt = ckpt.get('lora_config') or {}
            ckpt_rank = int(self.lora_config_ckpt.get('rank', 16))
            ckpt_alpha = int(self.lora_config_ckpt.get('alpha', ckpt_rank))
            ckpt_regex = self.lora_config_ckpt.get('target_regex')
            ckpt_preset = self.lora_config_ckpt.get('target_preset')
            ckpt_init = self.lora_config_ckpt.get('init_weights', 'gaussian')
            if ckpt_init in ('true', 'false'):
                ckpt_init = (ckpt_init == 'true')
            if not ckpt_regex and ckpt_preset in LORA_TARGET_PRESETS:
                ckpt_regex = LORA_TARGET_PRESETS[ckpt_preset]
            elif not ckpt_regex:
                ckpt_regex = LORA_TARGET_REGEX_OMINICONTROL

            lora_cfg = LoraConfig(
                r=ckpt_rank,
                lora_alpha=ckpt_alpha,
                target_modules=ckpt_regex,
                lora_dropout=0.0,
                bias="none",
                init_lora_weights=ckpt_init,
            )
            self.transformer = get_peft_model(self.transformer, lora_cfg)
            self.transformer.eval()
            self.transformer.requires_grad_(False)

            lora_state = {k: v.to(self.device) for k, v in ckpt['lora_state_dict'].items()}
            set_peft_model_state_dict(self.transformer, lora_state)
            self.lora_state_tensors = len(lora_state)
            print(
                f"[LoRA] ACTIVE: loaded {self.lora_state_tensors} tensors "
                f"(rank={ckpt_rank}, alpha={ckpt_alpha}, preset={ckpt_preset})"
            )
        else:
            print("[LoRA] inactive (checkpoint has no adapter)")

        self._setup_clear_attention(ckpt, dtype)

        # --- DFM adapter load (Phase 1.5, eval side) ---
        self.use_dfm = bool(ckpt.get('use_dfm', False)) and ('dfm_adapters' in ckpt)
        if self.use_dfm:
            dfm_cfg = ckpt.get('dfm_config', {}) or {}
            ckpt_map = dict(dfm_cfg.get('up_to_tap', {}) or {})
            ckpt_map = {int(k): v for k, v in ckpt_map.items()}
            if ckpt_map:
                self._dfm_up_to_tap = ckpt_map

            decoder = self.vae.decoder
            tap_ch = PixelFeatureExtractor._TAP_CHANNELS
            self.dfm_adapters = nn.ModuleDict()
            for up_idx, tap_name in self._dfm_up_to_tap.items():
                if up_idx >= len(decoder.up_blocks):
                    print(f"[DFM][WARN] up_blocks[{up_idx}] missing; skipping")
                    continue
                dec_ch = decoder.up_blocks[up_idx].resnets[0].conv1.in_channels
                adapter = DFMAdapter(feat_ch=tap_ch[tap_name], decoder_ch=dec_ch)
                self.dfm_adapters[f'up{up_idx}'] = adapter.to(self.device).to(dtype)

            state = {k.replace('module.', ''): v for k, v in ckpt['dfm_adapters'].items()}
            self.dfm_adapters.load_state_dict(state, strict=True)
            self.dfm_adapters.eval()
            self.dfm_adapters.requires_grad_(False)
            n_params = sum(p.numel() for p in self.dfm_adapters.parameters())
            print(f"[DFM] ACTIVE: loaded {len(state)} tensors ({n_params:,} params), "
                  f"up_to_tap={self._dfm_up_to_tap}")
        else:
            print("[DFM] inactive (checkpoint has no adapters)")

        # --- Learnable / residual text embedding load (eval side) ---
        # `_cache_text_embeddings` creates a full prompt Parameter warm-started
        # from empty T5. New train_pgsr checkpoints store a small residual
        # delta; old checkpoints store a full learned prompt.
        if self.use_learnable_text_embed and self.learnable_text_embed is not None:
            text_format = ckpt.get('text_embed_format', self.text_embed_format)
            text_delta_raw = ckpt.get('learnable_text_delta', None)
            text_embed_raw = ckpt.get('learnable_text_embed', None)
            if text_format == 'residual_delta_v1' and text_delta_raw is not None:
                with torch.no_grad():
                    text_delta_raw = text_delta_raw.to(
                        device=self.learnable_text_embed.device,
                        dtype=torch.float32,
                    )
                    n = min(
                        int(text_delta_raw.shape[1]),
                        int(self.learnable_text_embed.shape[1]),
                    )
                    if text_delta_raw.shape[-1] == self.learnable_text_embed.shape[-1]:
                        self.learnable_text_embed.data.copy_(
                            self._cached_embeds['prompt'].to(
                                device=self.learnable_text_embed.device,
                                dtype=self.learnable_text_embed.dtype,
                            )
                        )
                        self.learnable_text_embed.data[:, :n, :] += (
                            float(ckpt.get('text_embed_scale', self.text_embed_scale))
                            * text_delta_raw[:, :n, :].to(self.learnable_text_embed.dtype)
                        )
                        print(
                            f"[TextEmbed] residual_delta_v1 ACTIVE: "
                            f"tokens={n}, scale={float(ckpt.get('text_embed_scale', self.text_embed_scale)):g}, "
                            f"delta_shape={tuple(text_delta_raw.shape)}"
                        )
                    else:
                        print(
                            f"[TextEmbed][WARN] residual delta hidden dim mismatch: "
                            f"ckpt {tuple(text_delta_raw.shape)} vs "
                            f"model {tuple(self.learnable_text_embed.shape)}; "
                            f"falling back to warm-start init."
                        )
            elif text_embed_raw is not None:
                with torch.no_grad():
                    text_embed_raw = text_embed_raw.to(
                        device=self.learnable_text_embed.device,
                        dtype=self.learnable_text_embed.dtype,
                    )
                    if text_embed_raw.shape == self.learnable_text_embed.shape:
                        self.learnable_text_embed.data.copy_(text_embed_raw)
                        print(
                            f"[TextEmbed] ACTIVE: loaded learnable_text_embed "
                            f"shape={tuple(text_embed_raw.shape)} "
                            f"(freeze={bool(ckpt.get('text_embed_freeze', False))})"
                        )
                    else:
                        print(
                            f"[TextEmbed][WARN] shape mismatch: "
                            f"ckpt {tuple(text_embed_raw.shape)} vs "
                            f"model {tuple(self.learnable_text_embed.shape)}; "
                            f"falling back to warm-start init."
                        )
            else:
                print("[TextEmbed][WARN] use_learnable_text_embed=True in "
                      "checkpoint but no text prior tensor found; "
                      "using warm-start init from cached empty-string T5.")
        else:
            print("[TextEmbed] inactive (checkpoint trained with cached empty T5)")

    def _reset_pixel_fuse_proj_to_identity(self):
        """Identity-init lr_lat passthrough (first 16 in-channels); zero pixel branch."""
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

    def _load_pixel_fuse_proj_with_migration(self, ckpt):
        """Load pixel_fuse_proj from ckpt; migrate legacy gated-fusion format if needed."""
        if 'pixel_fuse_proj' not in ckpt:
            self.fusion_source = 'identity'
            print("[Eval] pixel_fuse_proj missing in checkpoint; using identity init.")
            return

        state = {k.replace('module.', ''): v for k, v in ckpt['pixel_fuse_proj'].items()}
        target = self.pixel_fuse_proj
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
            self.fusion_source = ckpt.get('fusion_type', 'concat')
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
                idx = torch.arange(16, device=target_w.device)
                target.weight[idx, idx, 0, 0] = 1.0
                legacy_w = src_w.to(device=target_w.device, dtype=target_w.dtype) * gate_sig
                target.weight[:, 16:32, :, :].copy_(legacy_w)
                if src_b is not None:
                    legacy_b = src_b.to(device=target.bias.device, dtype=target.bias.dtype) * gate_sig
                    target.bias.copy_(legacy_b)
            self.fusion_source = f'migrated(gate={gate_sig:.3f})'
            print(f"[Eval] Legacy gated-fusion checkpoint migrated to concat layout (sigmoid(gate)={gate_sig:.4f}).")
            return

        self.fusion_source = 'identity'
        print(
            f"[Eval][WARN] pixel_fuse_proj shape mismatch: "
            f"ckpt={tuple(src_w.shape) if src_w is not None else None}, "
            f"model={tuple(target_w.shape)}. Using identity init."
        )

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

    def _init_clear_mask(self, patch_h, patch_w):
        """Initialize CLEAR's global flex-attention mask for the current grid."""
        if not self.use_clear:
            return
        patch_h = int(patch_h)
        patch_w = int(patch_w)
        down_factor = int(self.clear_down_factor)
        if down_factor > 1 and (patch_h % down_factor != 0 or patch_w % down_factor != 0):
            raise RuntimeError(
                f"CLEAR down_factor={down_factor} requires packed image grid "
                f"height/width divisible by {down_factor}; got {patch_h}x{patch_w}. "
                "Use --benchmark_square_size or a tile_size divisible by "
                f"{16 * down_factor}."
            )
        try:
            import attention_processor
            from attention_processor import (
                init_local_downsample_mask_flex,
                init_local_mask_flex,
            )
        except ImportError as e:
            raise ImportError(
                "[CLEAR] cannot import CLEAR/attention_processor.py. "
                "Make sure CLEAR/ exists at the repo root."
            ) from e

        device_str = str(self.device)
        if down_factor > 1:
            init_local_downsample_mask_flex(
                height=patch_h, width=patch_w, text_length=512,
                window_size=int(self.clear_window_size),
                down_factor=down_factor,
                device=device_str,
            )
        else:
            init_local_mask_flex(
                height=patch_h, width=patch_w, text_length=512,
                window_size=int(self.clear_window_size),
                device=device_str,
            )
        attention_processor.HEIGHT = patch_h
        attention_processor.WIDTH = patch_w
        self._clear_mask_hw = (patch_h, patch_w)

    def _ensure_clear_mask(self, latent_h, latent_w):
        """Refresh CLEAR mask when tile/input size changes."""
        if not self.use_clear:
            return
        patch_h = int(latent_h) // 2
        patch_w = int(latent_w) // 2
        if self._clear_mask_hw != (patch_h, patch_w):
            self._init_clear_mask(patch_h, patch_w)

    def _setup_clear_attention(self, ckpt, dtype):
        """Restore CLEAR processors from checkpoint when selected."""
        clear_state = ckpt.get('clear_processors', None)
        clear_cfg = ckpt.get('clear_config', {}) or {}
        has_clear = clear_state is not None or bool(clear_cfg)

        if self.attention_mode == 'full':
            self.use_clear = False
            print("[CLEAR] inactive: using full attention")
            return
        if self.attention_mode != 'clear':
            raise ValueError(f"Unknown attention mode: {self.attention_mode}")
        if not has_clear:
            raise RuntimeError(
                "--attention_mode=clear was requested, but the checkpoint "
                "does not contain CLEAR processor state/config."
            )

        try:
            from attention_processor import (
                LocalDownsampleFlexAttnProcessor,
                LocalFlexAttnProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "[CLEAR] cannot import CLEAR/attention_processor.py. "
                "Make sure CLEAR/ exists at the repo root."
            ) from e

        self.use_clear = True
        self.clear_config = dict(clear_cfg)
        self.clear_window_size = int(clear_cfg.get('window_size', 16))
        self.clear_down_factor = int(clear_cfg.get('down_factor', 4))
        self.clear_resolution = int(clear_cfg.get('resolution', 512))
        self._clear_processors = []

        default_patch = max(1, self.clear_resolution // 16)
        self._init_clear_mask(default_patch, default_patch)

        base = self._get_transformer_base()
        n_replaced = 0
        state_list = list(clear_state or [])
        for name, module in base.named_modules():
            if not (
                hasattr(module, 'set_processor')
                and 'transformer_blocks.' in name
                and 'single' not in name
            ):
                continue
            if self.clear_down_factor > 1:
                processor = LocalDownsampleFlexAttnProcessor(
                    down_factor=self.clear_down_factor
                )
            else:
                processor = LocalFlexAttnProcessor()

            if n_replaced < len(state_list):
                state = {k.replace('module.', ''): v for k, v in state_list[n_replaced].items()}
                try:
                    processor.load_state_dict(state, strict=False)
                except AttributeError:
                    # LocalFlexAttnProcessor has no trainable state.
                    pass
            if hasattr(processor, 'to'):
                processor = processor.to(self.device, dtype)
            if hasattr(processor, 'requires_grad_'):
                processor.requires_grad_(False)
            module.set_processor(processor)
            self._clear_processors.append(processor)
            n_replaced += 1

        print(
            f"[CLEAR] ACTIVE: replaced {n_replaced} dual-block processors "
            f"(window={self.clear_window_size}, down_factor={self.clear_down_factor}, "
            f"ckpt_states={len(state_list)})"
        )

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

        # If the checkpoint was trained with a learnable meta text embedding
        # (e.g. by train_pgsr.py), register a Parameter with shape
        # (1, 512, 4096) and warm-start from the cached empty-string T5
        # output. The actual learned weights are copied in below from
        # `ckpt['learnable_text_embed']`. We keep the warm-start fallback so
        # eval still runs even if the key is missing.
        if self.use_learnable_text_embed:
            self.learnable_text_embed = nn.Parameter(
                t5_out[0].detach().to(torch.float32),
                requires_grad=False,
            )
            self.learnable_text_embed.data = (
                self.learnable_text_embed.data.to(self.device)
            )

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

    @torch.no_grad()
    def decode_with_dfm(self, lat, pixel_taps):
        """Decode latent with DFM injection (inference-time mirror of training
        ``DualStreamFLUXSR.decode_with_dfm``). Falls back to :meth:`decode`
        when DFM adapters are not loaded."""
        if not self.use_dfm or len(self.dfm_adapters) == 0 or pixel_taps is None:
            return self.decode(lat)

        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            z = (lat / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        else:
            z = lat / self.vae.config.scaling_factor
        z = z.to(self.vae.dtype)

        dec = self.vae.decoder
        sample = dec.conv_in(z)
        sample = dec.mid_block(sample, None)
        upscale_dtype = next(iter(dec.up_blocks.parameters())).dtype
        sample = sample.to(upscale_dtype)

        for i, up_block in enumerate(dec.up_blocks):
            tap_name = self._dfm_up_to_tap.get(i)
            if tap_name is not None and f'up{i}' in self.dfm_adapters:
                feat = pixel_taps.get(tap_name)
                if feat is not None:
                    sample = self.dfm_adapters[f'up{i}'](sample, feat)
            sample = up_block(sample, None)

        sample = dec.conv_norm_out(sample)
        sample = dec.conv_act(sample)
        sample = dec.conv_out(sample)
        return sample

    def _pack(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, (H // 2) * (W // 2), C * 4)

    def _unpack(self, x, H, W):
        B, _, D = x.shape
        C = D // 4
        x = x.view(B, H // 2, W // 2, C, 2, 2).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)

    @staticmethod
    def _pad_to_even_hw(x):
        """Pad feature map to even H/W for FLUX pack/unpack."""
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return x, pad_h, pad_w

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

        # Concat fusion: [lr_lat || pixel_weight*pixel_feat] -> 1x1 conv -> 16ch.
        fused_cond = self.pixel_fuse_proj(
            torch.cat([lr_lat.to(dtype), (self.pixel_weight * pixel_feat).to(dtype)], dim=1)
        ).to(dtype)
        del pixel_feat

        # FLUX pack() requires even latent H/W. RealSR can produce odd latent sizes.
        noisy_for_pack, pad_h, pad_w = self._pad_to_even_hw(noisy.to(dtype))
        fused_for_pack, _, _ = self._pad_to_even_hw(fused_cond)
        H_pack, W_pack = noisy_for_pack.shape[-2:]
        self._ensure_clear_mask(H_pack, W_pack)

        noisy_packed = self._pack(noisy_for_pack)
        fused_packed = self._pack(fused_for_pack)
        del fused_cond
        img_ids = self._img_ids(H_pack, W_pack, device, dtype)

        pooled = self._cached_embeds['pooled'].expand(B, -1)
        # T5 prompt (= encoder_hidden_states): learnable meta embedding when
        # the checkpoint enabled it (train_pgsr ckpts), else the cached
        # empty-string T5 output. CLIP `pooled` (temb path) and zero
        # `text_ids` (pos embed path) are unchanged regardless.
        if self.use_learnable_text_embed and self.learnable_text_embed is not None:
            prompt = self.learnable_text_embed.to(dtype).expand(B, -1, -1)
        else:
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

        unpacked = self._unpack(out, H_pack, W_pack)
        if pad_h or pad_w:
            unpacked = unpacked[:, :, :H, :W]
        return unpacked

    @torch.no_grad()
    def inference(self, lr_lat, lr_pixel, num_steps=20, guidance=3.5, strength=0.7):
        """Inference using the official scheduler.

        As a side effect, caches the pixel-extractor multi-scale taps on
        ``self._last_pixel_taps`` so the caller can later do
        ``decode_with_dfm(sr_lat, evaluator._last_pixel_taps)``.
        Taps only depend on ``lr_pixel`` and are invariant across denoising
        steps, so we compute them once up front.
        """
        B = lr_lat.shape[0]
        device = lr_lat.device
        dtype = torch.bfloat16

        lr_lat = lr_lat.to(dtype)
        lr_pixel = lr_pixel.to(dtype)

        # Stash taps for downstream decode_with_dfm (harmless when DFM off).
        if self.use_dfm:
            _, self._last_pixel_taps = self.pixel_extractor(lr_pixel, return_features=True)
        else:
            self._last_pixel_taps = None

        # Set timesteps (dynamic shifting may require mu)
        self._set_scheduler_timesteps(num_steps, device, lr_lat)
        timesteps = self.scheduler.timesteps

        # Compute start point based on strength
        init_timestep = min(int(num_steps * strength), num_steps)
        t_start = max(num_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        if len(timesteps) == 0:
            raise ValueError(
                f"No timesteps left after applying strength={strength}. "
                f"Please increase num_steps (current: {num_steps}) or strength."
            )

        # Align with official img2img: set begin_index then call scale_noise
        self.scheduler.set_begin_index(t_start)

        noise = torch.randn_like(lr_lat)

        # Use official scale_noise
        timestep_batch = timesteps[:1].expand(B)
        latents = self.scheduler.scale_noise(lr_lat, timestep_batch, noise)
        del noise

        # Denoising loop
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
                 tile_size=640, overlap=64, strength=0.7):
    _, _, H, W = lr_t.shape

    if H <= tile_size and W <= tile_size:
        lr_lat = evaluator.encode(lr_t)
        sr_lat = evaluator.inference(lr_lat, lr_t, num_steps=num_steps,
                                     guidance=guidance, strength=strength)
        # decode_with_dfm auto-falls-back to decode() when DFM is off.
        result = evaluator.decode_with_dfm(sr_lat, getattr(evaluator, '_last_pixel_taps', None))
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

    def build_tile_blend(tile_h, tile_w, y, y_end, x, x_end):
        """
        Boundary-aware linear blending:
        - Only taper sides that overlap with neighboring tiles.
        - Keep image outer borders untapered to avoid dark/gray frame artifacts.
        """
        blend = torch.ones((1, 1, tile_h, tile_w), dtype=torch.float32)
        if overlap <= 0:
            return blend

        if y > 0:
            n = min(overlap, tile_h)
            ramp = torch.linspace(1.0 / (n + 1), n / (n + 1), n, dtype=torch.float32)
            blend[:, :, :n, :] *= ramp.view(1, 1, n, 1)
        if y_end < H:
            n = min(overlap, tile_h)
            ramp = torch.linspace(n / (n + 1), 1.0 / (n + 1), n, dtype=torch.float32)
            blend[:, :, -n:, :] *= ramp.view(1, 1, n, 1)
        if x > 0:
            n = min(overlap, tile_w)
            ramp = torch.linspace(1.0 / (n + 1), n / (n + 1), n, dtype=torch.float32)
            blend[:, :, :, :n] *= ramp.view(1, 1, 1, n)
        if x_end < W:
            n = min(overlap, tile_w)
            ramp = torch.linspace(n / (n + 1), 1.0 / (n + 1), n, dtype=torch.float32)
            blend[:, :, :, -n:] *= ramp.view(1, 1, 1, n)

        return blend

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
                # Each tile re-runs inference, which refreshes _last_pixel_taps
                # on the evaluator. Safe to reuse here.
                sr_tile = evaluator.decode_with_dfm(
                    sr_lat, getattr(evaluator, '_last_pixel_taps', None)
                )
                sr_tile_cpu = sr_tile[:, :, :tile_h, :tile_w].float().cpu()
                del tile_lat, sr_lat, sr_tile, tile
                clear_memory(device)

                tile_blend = build_tile_blend(tile_h, tile_w, y, y_end, x, x_end)

                out[:, :, y:y_end, x:x_end] += sr_tile_cpu * tile_blend
                weight[:, :, y:y_end, x:x_end] += tile_blend
                del sr_tile_cpu

                pbar.update(1)

    return out / weight.clamp(min=1e-8)


@torch.no_grad()
def run_sr_tiled_with_oom_retry(
    evaluator, lr_t, device, num_steps=20, guidance=3.5,
    tile_size=640, overlap=64, strength=0.7,
    min_tile_size=256
):
    """
    OOM-safe tiled inference:
    - Try current tile_size
    - On OOM, clear memory and halve tile_size until min_tile_size
    """
    current_tile = tile_size
    last_err = None
    while current_tile >= min_tile_size:
        try:
            clear_memory(device)
            return run_sr_tiled(
                evaluator, lr_t, device, num_steps=num_steps, guidance=guidance,
                tile_size=current_tile, overlap=min(overlap, max(0, current_tile // 4)),
                strength=strength
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


@torch.no_grad()
def run_sr_timed(
    evaluator, lr_t, device, num_steps=20, guidance=3.5,
    tile_size=640, overlap=64, strength=0.7, min_tile_size=256,
    warmup=0, repeats=1,
):
    """Run tiled SR with optional warmup/repeats and synchronized timing."""
    warmup = max(0, int(warmup))
    repeats = max(1, int(repeats))
    denoise_steps = max(1, min(int(num_steps), int(int(num_steps) * float(strength))))

    for _ in range(warmup):
        tmp = run_sr_tiled_with_oom_retry(
            evaluator, lr_t, device, num_steps=num_steps, guidance=guidance,
            tile_size=tile_size, min_tile_size=min_tile_size,
            overlap=overlap, strength=strength,
        )
        del tmp
        clear_memory(device)

    wall_times = []
    cuda_times = []
    peak_mem = 0
    result = None
    for rep in range(repeats):
        if result is not None:
            del result
            result = None
            clear_memory(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = end_event = None

        start_wall = time.perf_counter()
        result = run_sr_tiled_with_oom_retry(
            evaluator, lr_t, device, num_steps=num_steps, guidance=guidance,
            tile_size=tile_size, min_tile_size=min_tile_size,
            overlap=overlap, strength=strength,
        )
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize(device)
        end_wall = time.perf_counter()

        wall_times.append(end_wall - start_wall)
        if device.type == 'cuda':
            cuda_times.append(start_event.elapsed_time(end_event) / 1000.0)
            peak_mem = max(peak_mem, int(torch.cuda.max_memory_allocated(device)))
        clear_memory(device)

    stats = {
        'denoise_steps': int(denoise_steps),
        'wall_times_sec': wall_times,
        'wall_mean_sec': float(np.mean(wall_times)) if wall_times else None,
        'wall_std_sec': float(np.std(wall_times)) if len(wall_times) > 1 else 0.0,
        'wall_per_step_sec': (
            float(np.mean(wall_times) / denoise_steps) if wall_times and denoise_steps > 0 else None
        ),
        'cuda_times_sec': cuda_times,
        'cuda_mean_sec': float(np.mean(cuda_times)) if cuda_times else None,
        'cuda_std_sec': float(np.std(cuda_times)) if len(cuda_times) > 1 else 0.0,
        'cuda_per_step_sec': (
            float(np.mean(cuda_times) / denoise_steps) if cuda_times and denoise_steps > 0 else None
        ),
        'peak_mem_mb': float(peak_mem / (1024 ** 2)) if peak_mem else None,
    }
    return result, stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PGSR CLEAR FLUX SR Evaluation')

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hr_dir', type=str, default=None,
                        help='Optional HR directory. Omit for speed-only LR evaluation.')
    parser.add_argument('--lr_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev')
    parser.add_argument('--attention_mode', type=str, default='auto',
                        choices=['auto', 'clear', 'full'],
                        help='auto uses CLEAR when the checkpoint has CLEAR state; '
                             'clear/full force the attention implementation.')
    parser.add_argument('--input_scale', type=int, default=4,
                        help='When --hr_dir is omitted, upsample LR by this factor before SR.')
    parser.add_argument('--benchmark_square_size', type=int, default=0,
                        help='Center-crop or edge-pad HR-space inputs to this square size before inference.')
    parser.add_argument('--timing_warmup', type=int, default=0,
                        help='Warmup runs per image before timing.')
    parser.add_argument('--timing_repeats', type=int, default=1,
                        help='Timed repeats per image. The final repeat is used for saved images/metrics.')

    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=3.5)
    parser.add_argument('--pixel_weight', type=float, default=None)
    parser.add_argument('--strength', type=float, default=None,
                        help='Inference start strength (omit to use checkpoint value)')
    parser.add_argument('--control_guidance_start', type=float, default=None,
                        help='Override ControlNet guidance start ratio in [0,1]. '
                             'Omit to use checkpoint value.')
    parser.add_argument('--control_guidance_end', type=float, default=None,
                        help='Override ControlNet guidance end ratio in [0,1]. '
                             'Omit to use checkpoint value.')
    parser.add_argument('--tile_size', type=int, default=512,
                        help='Tile size for inference. Should match training resolution (512 by default) -- running at a different size shifts the FlowMatch dynamic_shifting sigma schedule and FLUX RoPE positions.')
    parser.add_argument('--min_tile_size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--calc_lpips', dest='calc_lpips', action='store_true',
                        help='Enable LPIPS calculation (default: enabled if lpips package is available)')
    parser.add_argument('--no_calc_lpips', dest='calc_lpips', action='store_false',
                        help='Disable LPIPS calculation')
    parser.add_argument('--lpips_device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.set_defaults(calc_lpips=True)
    # Extra IQA metrics (pyiqa-backed): MUSIQ / MANIQA / CLIP-IQA / NIQE / DISTS.
    # FID is a directory-level metric computed at the end.
    parser.add_argument(
        '--iqa_metrics',
        type=str,
        default='musiq,maniqa,clipiqa,niqe,dists',
        help='Comma-separated IQA metrics to compute. Empty string disables all. '
             'Available: musiq, maniqa, clipiqa, niqe, dists',
    )
    parser.add_argument('--iqa_device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device for pyiqa metrics. cpu is slower but safe; '
                             'cuda is fast but uses extra GPU memory on top of FLUX.')
    parser.add_argument('--iqa_fr_max_long_edge', type=int, default=1024,
                        help='Cap long edge for FR metrics (e.g. DISTS) to avoid '
                             'multi-minute VGG forward on 2K-4K images. Set 0 to disable.')
    parser.add_argument('--calc_fid', dest='calc_fid', action='store_true',
                        help='Compute FID(SR, HR) at the end using --fid_mode.')
    parser.add_argument('--no_calc_fid', dest='calc_fid', action='store_false')
    parser.set_defaults(calc_fid=True)
    parser.add_argument('--fid_device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--fid_mode', type=str, default='patch',
                        choices=['patch', 'full', 'both'],
                        help='FID protocol. patch crops many fixed-size patches per image; '
                             'full preserves the previous full-image pyiqa FID path.')
    parser.add_argument('--fid_patch_size', type=int, default=512)
    parser.add_argument('--fid_patches_per_image', type=int, default=25,
                        help='Number of patches sampled per evaluated image for patch FID.')
    parser.add_argument('--fid_patch_pairing', type=str, default='independent',
                        choices=['independent', 'aligned'],
                        help='Patch-FID sampling protocol. independent samples SR/BIC and HR '
                             'patch coordinates from separate random streams and is the paper-table '
                             'default; aligned reuses coordinates and usually gives lower FID.')
    parser.add_argument('--fid_patch_seed', type=int, default=42)

    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', dest='save_images', action='store_true', default=True)
    parser.add_argument('--no_save_images', dest='save_images', action='store_false')
    parser.add_argument('--save_metrics_json', action='store_true',
                        help='Also save metrics.json for programmatic table collection. '
                             'Default is off so evaluate outputs contain only predictions/ '
                             'and results.txt.')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    if args.control_guidance_start is not None and not (0.0 <= args.control_guidance_start <= 1.0):
        parser.error("--control_guidance_start must be within [0, 1]")
    if args.control_guidance_end is not None and not (0.0 <= args.control_guidance_end <= 1.0):
        parser.error("--control_guidance_end must be within [0, 1]")
    if (
        args.control_guidance_start is not None
        and args.control_guidance_end is not None
        and args.control_guidance_start > args.control_guidance_end
    ):
        parser.error("--control_guidance_start must be <= --control_guidance_end")
    if args.fid_patch_size < 1:
        parser.error("--fid_patch_size must be >= 1")
    if args.fid_patches_per_image < 1:
        parser.error("--fid_patches_per_image must be >= 1")
    if args.input_scale < 1:
        parser.error("--input_scale must be >= 1")
    if args.benchmark_square_size < 0:
        parser.error("--benchmark_square_size must be >= 0")
    if args.timing_warmup < 0:
        parser.error("--timing_warmup must be >= 0")
    if args.timing_repeats < 1:
        parser.error("--timing_repeats must be >= 1")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    has_hr = bool(args.hr_dir)
    if not has_hr:
        args.calc_lpips = False
        args.calc_fid = False

    # Auto-detect dataset tag when not provided, then normalize to canonical name.
    if args.dataset is None:
        args.dataset = infer_dataset_name(args.hr_dir, args.lr_dir)
    dataset_name = infer_dataset_name(args.dataset, args.hr_dir, args.lr_dir)

    # Load model
    initial_pixel_weight = args.pixel_weight if args.pixel_weight is not None else 1.0
    evaluator = DualStreamEvaluator(
        args.model_name, device, args.checkpoint, initial_pixel_weight,
        attention_mode=args.attention_mode,
    )
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

    def _fmt_tag(v):
        s = f"{v:g}" if isinstance(v, float) else str(v)
        return s.replace("-", "m").replace(".", "p")

    if args.exp_name:
        exp_name = args.exp_name
    else:
        fusion_tag = "concat"
        fusion_source_tag = str(evaluator.fusion_source).replace("(", "").replace(")", "").replace(" ", "_")
        exp_name = (
            f"{ts}"
            f"_str{_fmt_tag(strength)}"
            f"_step{args.num_steps}"
            f"_g{_fmt_tag(args.guidance)}"
            f"_pw{_fmt_tag(evaluator.pixel_weight)}"
            f"_cgs{_fmt_tag(control_guidance_start)}"
            f"_cge{_fmt_tag(control_guidance_end)}"
            f"_attn{evaluator.attention_mode}"
            f"_{fusion_tag}"
            f"_fus{fusion_source_tag}"
            f"_trlpw{_fmt_tag(evaluator.train_lpips_weight)}"
        )
        if evaluator.use_lora:
            cfg = evaluator.lora_config_ckpt or {}
            exp_name += (
                f"_lora"
                f"_r{cfg.get('rank', '?')}"
                f"_a{cfg.get('alpha', '?')}"
                f"_tp{cfg.get('target_preset', '?')}"
            )

    if evaluator.use_dfm and evaluator.use_lora:
        method_tag = 'DualDFMLoRA'
    elif evaluator.use_dfm:
        method_tag = 'DualDFM'
    elif evaluator.use_lora:
        method_tag = 'DualLoRA'
    else:
        method_tag = 'DualControl'
    output_dir = os.path.join(args.output_base, args.dataset, method_tag, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)

    print("=" * 70)
    print("PGSR CLEAR FLUX SR Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset} (canonical: {dataset_name})")
    print(f"Strength: {strength}")
    print(f"Control Guidance Window: [{control_guidance_start}, {control_guidance_end}]")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    print(f"Train LPIPS: weight={evaluator.train_lpips_weight}, prob={evaluator.train_lpips_apply_prob}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}")
    print(f"Attention: {evaluator.attention_mode} (requested={args.attention_mode})")
    print(f"Timing: warmup={args.timing_warmup}, repeats={args.timing_repeats}")
    if args.benchmark_square_size > 0:
        print(f"Benchmark square: {args.benchmark_square_size}x{args.benchmark_square_size}")
    if args.calc_fid and PYIQA_AVAILABLE:
        print(
            f"FID: mode={args.fid_mode}, patch_size={args.fid_patch_size}, "
            f"patches_per_image={args.fid_patches_per_image}, "
            f"patch_pairing={args.fid_patch_pairing}"
        )
    print(f"DFM: {'ACTIVE' if evaluator.use_dfm else 'inactive'}")
    if evaluator.use_lora:
        cfg = evaluator.lora_config_ckpt or {}
        print(
            f"LoRA: ACTIVE (tensors={evaluator.lora_state_tensors}, "
            f"rank={cfg.get('rank', '?')}, alpha={cfg.get('alpha', '?')}, "
            f"preset={cfg.get('target_preset', '?')})"
        )
    else:
        print("LoRA: inactive")
    print(f"Output: {output_dir}")
    print("=" * 70)

    lpips_fn = None
    lpips_device = torch.device('cpu')
    if args.calc_lpips:
        if not LPIPS_AVAILABLE:
            print("Warning: LPIPS enabled but lpips package not available, skip LPIPS.")
        else:
            if args.lpips_device == 'cuda' and torch.cuda.is_available():
                lpips_device = torch.device('cuda')
            else:
                lpips_device = torch.device('cpu')
            lpips_fn = lpips.LPIPS(net='alex').to(lpips_device)
            lpips_fn.eval()

    iqa_device = torch.device(args.iqa_device if (args.iqa_device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    requested_iqa = [m.strip().lower() for m in args.iqa_metrics.split(',') if m.strip()]
    iqa_runner = IQAMetricsRunner(iqa_device, requested_iqa,
                                  fr_max_long_edge=args.iqa_fr_max_long_edge)

    # Save FID staging files to a temporary directory, not output_dir.
    # output_dir should only keep predictions/ plus the human-readable log by
    # default. Patch FID is the default because full-resolution DIV2K images
    # can produce unrealistically tiny FID values after Inception's 299px resize.
    will_compute_fid = bool(args.calc_fid and PYIQA_AVAILABLE)
    compute_full_fid = bool(will_compute_fid and args.fid_mode in ('full', 'both'))
    compute_patch_fid = bool(will_compute_fid and args.fid_mode in ('patch', 'both'))
    fid_tmp_ctx = tempfile.TemporaryDirectory(prefix='dfm_eval_fid_') if will_compute_fid else None
    fid_tmp_root = fid_tmp_ctx.name if fid_tmp_ctx is not None else None
    if args.save_images:
        fid_sr_dir = os.path.join(output_dir, 'predictions')
    elif fid_tmp_root:
        fid_sr_dir = os.path.join(fid_tmp_root, 'fid_sr')
    else:
        fid_sr_dir = None
    fid_hr_dir = os.path.join(fid_tmp_root, 'fid_hr') if fid_tmp_root else None
    fid_bic_dir = os.path.join(fid_tmp_root, 'fid_bicubic') if fid_tmp_root else None
    fid_patch_dirs = {
        'sr': os.path.join(fid_tmp_root, 'fid_patches_sr') if fid_tmp_root else None,
        'hr': os.path.join(fid_tmp_root, 'fid_patches_hr') if fid_tmp_root else None,
        'bic': os.path.join(fid_tmp_root, 'fid_patches_bicubic') if fid_tmp_root else None,
    }
    fid_patch_rng_sr = np.random.default_rng(args.fid_patch_seed)
    fid_patch_rng_bic = np.random.default_rng(args.fid_patch_seed + 7919)
    fid_patch_rng_hr = np.random.default_rng(args.fid_patch_seed + 1000003)
    fid_patch_count = 0
    if compute_full_fid:
        os.makedirs(fid_sr_dir, exist_ok=True)
        os.makedirs(fid_hr_dir, exist_ok=True)
        os.makedirs(fid_bic_dir, exist_ok=True)
    if compute_patch_fid:
        for d in fid_patch_dirs.values():
            os.makedirs(d, exist_ok=True)

    lr_files = sorted([f for f in os.listdir(args.lr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if has_hr:
        hr_files = sorted([f for f in os.listdir(args.hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        eval_files = hr_files
    else:
        hr_files = []
        eval_files = lr_files

    print(f"\nEvaluating {len(eval_files)} images...\n")

    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_bic_list, ssim_bic_list, lpips_bic_list = [], [], []
    filenames = []
    timing_rows = []

    for eval_name in tqdm(eval_files, desc="Evaluating"):
        if has_hr:
            hf = eval_name
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
        else:
            lf = eval_name
            base_name = os.path.splitext(lf)[0]

        filenames.append(base_name)

        lr_img = Image.open(os.path.join(args.lr_dir, lf)).convert('RGB')
        if has_hr:
            hr_img = Image.open(os.path.join(args.hr_dir, hf)).convert('RGB')
            hr_np = np.array(hr_img)
            H, W = hr_np.shape[0], hr_np.shape[1]
            lr_bicubic = lr_img.resize((W, H), Image.BICUBIC)
        else:
            hr_np = None
            target_w = int(lr_img.size[0] * args.input_scale)
            target_h = int(lr_img.size[1] * args.input_scale)
            lr_bicubic = lr_img.resize((target_w, target_h), Image.BICUBIC)
        lr_bicubic_np = np.array(lr_bicubic)
        if args.benchmark_square_size > 0:
            lr_bicubic_np = center_crop_or_pad_np(lr_bicubic_np, args.benchmark_square_size)
            if hr_np is not None:
                hr_np = center_crop_or_pad_np(hr_np, args.benchmark_square_size)
        H, W = lr_bicubic_np.shape[0], lr_bicubic_np.shape[1]

        lr_t = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        lr_t = lr_t.to(device).to(torch.bfloat16)
        # The FLUX VAE downsamples by 8x; odd/non-divisible image sizes like
        # Urban100's 322x512 otherwise decode back to 320x512. Pad before
        # encode/inference and crop back after decode so metrics stay aligned.
        pad_multiple = 8
        if evaluator.use_clear:
            pad_multiple = max(pad_multiple, 16 * int(evaluator.clear_down_factor))
        lr_t_infer, _, _ = pad_tensor_to_multiple(lr_t, multiple=pad_multiple, mode='replicate')

        sr_t, timing = run_sr_timed(
            evaluator, lr_t_infer, device,
            num_steps=args.num_steps,
            guidance=args.guidance,
            tile_size=args.tile_size,
            min_tile_size=args.min_tile_size,
            overlap=args.overlap,
            strength=strength,
            warmup=args.timing_warmup,
            repeats=args.timing_repeats,
        )
        timing['filename'] = base_name
        timing['height'] = int(H)
        timing['width'] = int(W)
        timing_rows.append(timing)
        sr_t = sr_t[:, :, :H, :W]
        sr_np = ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)

        if hr_np is not None:
            psnr_val = calculate_psnr(sr_np, hr_np)
            ssim_val = calculate_ssim(sr_np, hr_np)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            psnr_bic = calculate_psnr(lr_bicubic_np, hr_np)
            ssim_bic = calculate_ssim(lr_bicubic_np, hr_np)
            psnr_bic_list.append(psnr_bic)
            ssim_bic_list.append(ssim_bic)

        if lpips_fn and hr_np is not None:
            hr_t = torch.from_numpy(hr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
            lr_t_lpips = torch.from_numpy(lr_bicubic_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

            hr_t = hr_t.to(lpips_device)
            lr_t_lpips = lr_t_lpips.to(lpips_device)
            lpips_val = lpips_fn(sr_t.float().clamp(-1, 1).to(lpips_device), hr_t).item()
            lpips_bic = lpips_fn(lr_t_lpips, hr_t).item()
            lpips_list.append(lpips_val)
            lpips_bic_list.append(lpips_bic)

        # Extra IQA metrics (MUSIQ / MANIQA / CLIP-IQA / NIQE / DISTS).
        iqa_runner.update(sr_np, hr_np, lr_bicubic_np)

        if args.save_images:
            Image.fromarray(sr_np).save(os.path.join(output_dir, 'predictions', f'{base_name}.png'))

        # Mirror SR / HR / bicubic to FID dirs (only needed for full-image FID).
        if compute_full_fid and hr_np is not None:
            if not args.save_images:
                Image.fromarray(sr_np).save(os.path.join(fid_sr_dir, f'{base_name}.png'))
            Image.fromarray(hr_np).save(os.path.join(fid_hr_dir, f'{base_name}.png'))
            Image.fromarray(lr_bicubic_np).save(os.path.join(fid_bic_dir, f'{base_name}.png'))
        if compute_patch_fid and hr_np is not None:
            fid_patch_count += save_fid_patches(
                sr_np, hr_np, lr_bicubic_np, base_name, fid_patch_dirs,
                patch_size=args.fid_patch_size,
                n_per_image=args.fid_patches_per_image,
                patch_pairing=args.fid_patch_pairing,
                rng_sr=fid_patch_rng_sr,
                rng_bic=fid_patch_rng_bic,
                rng_hr=fid_patch_rng_hr,
            )

        del lr_t, sr_t
        if lpips_fn and hr_np is not None:
            del hr_t, lr_t_lpips
        del lr_bicubic_np, sr_np
        if hr_np is not None:
            del hr_np
        clear_memory(device)

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_psnr_bic = float(np.mean(psnr_bic_list)) if psnr_bic_list else 0.0
    avg_ssim_bic = float(np.mean(ssim_bic_list)) if ssim_bic_list else 0.0
    avg_lpips = np.mean(lpips_list) if lpips_list else 0
    avg_lpips_bic = np.mean(lpips_bic_list) if lpips_bic_list else 0
    timing_wall = [r['wall_mean_sec'] for r in timing_rows if r.get('wall_mean_sec') is not None]
    timing_cuda = [r['cuda_mean_sec'] for r in timing_rows if r.get('cuda_mean_sec') is not None]
    timing_wall_step = [r['wall_per_step_sec'] for r in timing_rows if r.get('wall_per_step_sec') is not None]
    timing_cuda_step = [r['cuda_per_step_sec'] for r in timing_rows if r.get('cuda_per_step_sec') is not None]
    timing_summary = {
        'wall_mean_sec': float(np.mean(timing_wall)) if timing_wall else None,
        'wall_std_sec': float(np.std(timing_wall)) if len(timing_wall) > 1 else 0.0,
        'wall_per_step_mean_sec': float(np.mean(timing_wall_step)) if timing_wall_step else None,
        'cuda_mean_sec': float(np.mean(timing_cuda)) if timing_cuda else None,
        'cuda_std_sec': float(np.std(timing_cuda)) if len(timing_cuda) > 1 else 0.0,
        'cuda_per_step_mean_sec': float(np.mean(timing_cuda_step)) if timing_cuda_step else None,
        'images_per_sec_wall': float(1.0 / np.mean(timing_wall)) if timing_wall and np.mean(timing_wall) > 0 else None,
        'peak_mem_mb': float(max([r.get('peak_mem_mb') or 0.0 for r in timing_rows])) if timing_rows else None,
    }

    iqa_avg = iqa_runner.averages()  # name -> {'sr', 'bic', 'higher_better', 'type'}

    fid_results = {}
    fid_sr = None
    fid_bic = None
    if will_compute_fid and len(psnr_list) > 0:
        fid_device = torch.device(args.fid_device if torch.cuda.is_available() else 'cpu')
        if compute_patch_fid:
            print(
                f"\n[FID][patch] computing on {fid_patch_count} patches "
                f"(size={args.fid_patch_size}, per_image={args.fid_patches_per_image}) ..."
            )
            patch_sr = compute_fid(fid_patch_dirs['sr'], fid_patch_dirs['hr'], fid_device)
            print(f"[FID][patch] SR vs HR  = {patch_sr:.4f}" if patch_sr is not None else "[FID][patch] failed")
            patch_bic = compute_fid(fid_patch_dirs['bic'], fid_patch_dirs['hr'], fid_device)
            print(f"[FID][patch] BIC vs HR = {patch_bic:.4f}" if patch_bic is not None else "[FID][patch] failed")
            fid_results['patch'] = {'sr': patch_sr, 'bic': patch_bic}
        if compute_full_fid:
            print("\n[FID][full] computing FID(SR, HR) ...")
            full_sr = compute_fid(fid_sr_dir, fid_hr_dir, fid_device)
            print(f"[FID][full] SR vs HR  = {full_sr:.4f}" if full_sr is not None else "[FID][full] failed")
            print("[FID][full] computing FID(Bicubic, HR) ...")
            full_bic = compute_fid(fid_bic_dir, fid_hr_dir, fid_device)
            print(f"[FID][full] BIC vs HR = {full_bic:.4f}" if full_bic is not None else "[FID][full] failed")
            fid_results['full'] = {'sr': full_sr, 'bic': full_bic}

        primary_fid = fid_results.get('patch') or fid_results.get('full') or {}
        fid_sr = primary_fid.get('sr')
        fid_bic = primary_fid.get('bic')

        # TemporaryDirectory cleanup below removes all FID staging files.

    def _fmt(v, prec=4):
        return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{prec}f}'

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Strength: {strength}")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    print(f"Pixel Fusion: concat+1x1 conv (source={evaluator.fusion_source})")
    print("---------- Timing ----------")
    print(
        f"Wall: mean={_fmt(timing_summary['wall_mean_sec'])}s, "
        f"per_step={_fmt(timing_summary['wall_per_step_mean_sec'])}s, "
        f"std={_fmt(timing_summary['wall_std_sec'])}s, "
        f"throughput={_fmt(timing_summary['images_per_sec_wall'])} img/s"
    )
    if timing_summary['cuda_mean_sec'] is not None:
        print(
            f"CUDA: mean={_fmt(timing_summary['cuda_mean_sec'])}s, "
            f"per_step={_fmt(timing_summary['cuda_per_step_mean_sec'])}s, "
            f"std={_fmt(timing_summary['cuda_std_sec'])}s, "
            f"peak={_fmt(timing_summary['peak_mem_mb'], 1)} MB"
        )
    if has_hr:
        print("---------- Full-reference (HR available) ----------")
        if lpips_fn:
            print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
            print(f"SR:       PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
            print(f"Delta:    {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, "
                  f"{avg_lpips_bic - avg_lpips:+.4f}")
        else:
            print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}")
            print(f"SR:       PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}")
    else:
        print("---------- Full-reference ----------")
        print("HR unavailable: PSNR/SSIM/LPIPS/FID skipped.")
    if any(v['type'] == 'FR' for v in iqa_avg.values()):
        print("---------- Full-reference (pyiqa) ----------")
        for name, v in iqa_avg.items():
            if v['type'] != 'FR':
                continue
            arrow = '↓' if not v['higher_better'] else '↑'
            print(f"  {name.upper():<8} ({arrow}): SR={_fmt(v['sr'])}, BIC={_fmt(v['bic'])}")
    if any(v['type'] == 'NR' for v in iqa_avg.values()):
        print("---------- No-reference (pyiqa) ----------")
        for name, v in iqa_avg.items():
            if v['type'] != 'NR':
                continue
            arrow = '↑' if v['higher_better'] else '↓'
            print(f"  {name.upper():<8} ({arrow}): SR={_fmt(v['sr'])}, BIC={_fmt(v['bic'])}")
    if fid_results:
        print("---------- Distribution-level ----------")
        if 'patch' in fid_results:
            v = fid_results['patch']
            print(f"  FID-patch (↓): SR={_fmt(v['sr'])},  BIC={_fmt(v['bic'])}")
        if 'full' in fid_results:
            v = fid_results['full']
            print(f"  FID-full  (↓): SR={_fmt(v['sr'])},  BIC={_fmt(v['bic'])}")
    print("=" * 70)

    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("PGSR CLEAR FLUX SR Evaluation\n")
        f.write("=" * 60 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Strength: {strength}\n")
        f.write(f"Control Guidance Window: [{control_guidance_start}, {control_guidance_end}]\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n")
        f.write(f"Attention Mode: {evaluator.attention_mode} (requested={args.attention_mode})\n")
        if evaluator.use_clear:
            f.write(
                f"CLEAR: window={evaluator.clear_window_size}, "
                f"down_factor={evaluator.clear_down_factor}, "
                f"resolution={evaluator.clear_resolution}\n"
            )
        f.write(f"Train LPIPS Weight: {evaluator.train_lpips_weight}\n")
        f.write(f"Train LPIPS Apply Prob: {evaluator.train_lpips_apply_prob}\n")
        f.write(f"Pixel Fusion: concat+1x1 conv (source={evaluator.fusion_source})\n")
        train_data = evaluator.train_data or {}
        if any(v is not None for v in train_data.values()):
            f.write("Train Data (from checkpoint):\n")
            f.write(f"  HR Dir: {train_data.get('train_hr_dir')}\n")
            f.write(f"  LR Dir: {train_data.get('train_lr_dir')}\n")
            f.write(f"  Val HR Dir: {train_data.get('val_hr_dir')}\n")
            f.write(f"  Val LR Dir: {train_data.get('val_lr_dir')}\n")
            f.write(
                f"  Degrade: {train_data.get('degrade_mode')}, "
                f"Scale: x{train_data.get('scale')}, "
                f"Resolution: {train_data.get('resolution')}, "
                f"Num Crops: {train_data.get('num_crops')}\n"
            )
        f.write(f"Dataset Name: {dataset_name}\n")
        f.write(f"Dataset Tag: {args.dataset}\n")
        f.write(f"Eval HR Dir: {args.hr_dir}\n")
        f.write(f"Eval LR Dir: {args.lr_dir}\n")
        f.write(f"Images: {len(filenames)}\n")
        f.write(f"Has HR: {has_hr}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}\n")
        f.write(f"Tile: size={args.tile_size}, min={args.min_tile_size}, overlap={args.overlap}\n")
        f.write(
            f"Timing: warmup={args.timing_warmup}, repeats={args.timing_repeats}, "
            f"wall_mean={_fmt(timing_summary['wall_mean_sec'])}s, "
            f"wall_per_step={_fmt(timing_summary['wall_per_step_mean_sec'])}s, "
            f"cuda_mean={_fmt(timing_summary['cuda_mean_sec'])}s, "
            f"cuda_per_step={_fmt(timing_summary['cuda_per_step_mean_sec'])}s, "
            f"throughput={_fmt(timing_summary['images_per_sec_wall'])} img/s, "
            f"peak_mem={_fmt(timing_summary['peak_mem_mb'], 1)} MB\n"
        )
        if args.benchmark_square_size > 0:
            f.write(f"Benchmark Square Size: {args.benchmark_square_size}\n")
        if will_compute_fid:
            f.write(
                f"FID: mode={args.fid_mode}, patch_size={args.fid_patch_size}, "
                f"patches_per_image={args.fid_patches_per_image}, "
                f"patch_pairing={args.fid_patch_pairing}, "
                f"patch_count={fid_patch_count}\n"
            )
        f.write(f"Save Images: {args.save_images}\n")
        f.write(f"Device: {device}\n")
        if evaluator.use_lora:
            cfg = evaluator.lora_config_ckpt or {}
            f.write(
                f"LoRA: ACTIVE (tensors={evaluator.lora_state_tensors}, "
                f"rank={cfg.get('rank', '?')}, alpha={cfg.get('alpha', '?')}, "
                f"preset={cfg.get('target_preset', '?')})\n"
            )
        else:
            f.write("LoRA: inactive\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Summary (averages):\n")
        f.write("[Timing]\n")
        f.write(
            f"  Wall: mean={_fmt(timing_summary['wall_mean_sec'])}s, "
            f"per_step={_fmt(timing_summary['wall_per_step_mean_sec'])}s, "
            f"std={_fmt(timing_summary['wall_std_sec'])}s, "
            f"throughput={_fmt(timing_summary['images_per_sec_wall'])} img/s\n"
        )
        f.write(
            f"  CUDA: mean={_fmt(timing_summary['cuda_mean_sec'])}s, "
            f"per_step={_fmt(timing_summary['cuda_per_step_mean_sec'])}s, "
            f"std={_fmt(timing_summary['cuda_std_sec'])}s, "
            f"peak={_fmt(timing_summary['peak_mem_mb'], 1)} MB\n"
        )
        f.write("[Full-reference, classical]\n")
        if has_hr and lpips_fn:
            f.write(f"  Bicubic: PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}\n")
            f.write(f"  SR:      PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}\n")
        elif has_hr:
            f.write(f"  Bicubic: PSNR={avg_psnr_bic:.4f}, SSIM={avg_ssim_bic:.4f}\n")
            f.write(f"  SR:      PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}\n")
        else:
            f.write("  HR unavailable: skipped.\n")
        if any(v['type'] == 'FR' for v in iqa_avg.values()):
            f.write("[Full-reference, pyiqa]\n")
            for name, v in iqa_avg.items():
                if v['type'] != 'FR':
                    continue
                arrow = '↓' if not v['higher_better'] else '↑'
                f.write(f"  {name.upper()} ({arrow}): SR={_fmt(v['sr'])}, BIC={_fmt(v['bic'])}\n")
        if any(v['type'] == 'NR' for v in iqa_avg.values()):
            f.write("[No-reference, pyiqa]\n")
            for name, v in iqa_avg.items():
                if v['type'] != 'NR':
                    continue
                arrow = '↑' if v['higher_better'] else '↓'
                f.write(f"  {name.upper()} ({arrow}): SR={_fmt(v['sr'])}, BIC={_fmt(v['bic'])}\n")
        if fid_results:
            f.write("[Distribution-level]\n")
            if 'patch' in fid_results:
                v = fid_results['patch']
                f.write(f"  FID-patch (↓): SR={_fmt(v['sr'])}, BIC={_fmt(v['bic'])}\n")
            if 'full' in fid_results:
                v = fid_results['full']
                f.write(f"  FID-full (↓): SR={_fmt(v['sr'])}, BIC={_fmt(v['bic'])}\n")
        f.write("\n" + "=" * 60 + "\n")

        # Per-image header: PSNR, SSIM, LPIPS, then any pyiqa metrics in order.
        iqa_names = list(iqa_avg.keys())
        header_parts = [
            'filename', 'height', 'width',
            'denoise_steps',
            'wall_mean_sec', 'wall_per_step_sec',
            'cuda_mean_sec', 'cuda_per_step_sec',
            'peak_mem_mb',
        ]
        if has_hr:
            header_parts += ['PSNR', 'dPSNR', 'SSIM', 'dSSIM']
        if has_hr and lpips_fn:
            header_parts += ['LPIPS', 'dLPIPS']
        for n in iqa_names:
            header_parts.append(n.upper())
            header_parts.append(f'{n.upper()}_bic')
        f.write("Per-image (TSV):\n")
        f.write("\t".join(header_parts) + "\n")
        for i, fname in enumerate(filenames):
            tr = timing_rows[i] if i < len(timing_rows) else {}
            row = [
                fname,
                str(tr.get('height', '')),
                str(tr.get('width', '')),
                str(tr.get('denoise_steps', '')),
                _fmt(tr.get('wall_mean_sec')),
                _fmt(tr.get('wall_per_step_sec')),
                _fmt(tr.get('cuda_mean_sec')),
                _fmt(tr.get('cuda_per_step_sec')),
                _fmt(tr.get('peak_mem_mb'), 1),
            ]
            if has_hr:
                row += [
                    f'{psnr_list[i]:.4f}', f'{psnr_list[i] - psnr_bic_list[i]:+.4f}',
                    f'{ssim_list[i]:.4f}', f'{ssim_list[i] - ssim_bic_list[i]:+.4f}',
                ]
            if has_hr and lpips_fn:
                row.append(f'{lpips_list[i]:.4f}')
                row.append(f'{lpips_list[i] - lpips_bic_list[i]:+.4f}')
            for n in iqa_names:
                sr_v = iqa_runner.values_sr[n][i] if i < len(iqa_runner.values_sr[n]) else float('nan')
                bic_v = iqa_runner.values_bic[n][i] if i < len(iqa_runner.values_bic[n]) else float('nan')
                row.append(_fmt(sr_v))
                row.append(_fmt(bic_v))
            f.write("\t".join(row) + "\n")

    summary_json = {
        'checkpoint': args.checkpoint,
        'dataset_name': dataset_name,
        'dataset_tag': args.dataset,
        'images': len(filenames),
        'has_hr': has_hr,
        'attention_mode': evaluator.attention_mode,
        'attention_mode_requested': args.attention_mode,
        'timing': {
            'warmup': args.timing_warmup,
            'repeats': args.timing_repeats,
            'summary': timing_summary,
            'per_image': timing_rows,
        },
        'classical': {
            'sr': {'psnr': (avg_psnr if has_hr else None),
                   'ssim': (avg_ssim if has_hr else None),
                   'lpips': (avg_lpips if has_hr and lpips_fn else None)},
            'bic': {'psnr': (avg_psnr_bic if has_hr else None),
                    'ssim': (avg_ssim_bic if has_hr else None),
                    'lpips': (avg_lpips_bic if has_hr and lpips_fn else None)},
        },
        'iqa': {
            n: {
                'sr': v['sr'], 'bic': v['bic'],
                'type': v['type'], 'higher_better': v['higher_better'],
            }
            for n, v in iqa_avg.items()
        },
        'fid': {
            'primary_mode': 'patch' if 'patch' in fid_results else ('full' if 'full' in fid_results else None),
            'sr': fid_sr,
            'bic': fid_bic,
            'patch': fid_results.get('patch'),
            'full': fid_results.get('full'),
            'patch_size': args.fid_patch_size if compute_patch_fid else None,
            'patches_per_image': args.fid_patches_per_image if compute_patch_fid else None,
            'patch_pairing': args.fid_patch_pairing if compute_patch_fid else None,
            'patch_count': fid_patch_count if compute_patch_fid else None,
        },
    }
    metrics_path = os.path.join(output_dir, 'metrics.json')
    if args.save_metrics_json:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, default=lambda o: None)
    elif os.path.exists(metrics_path):
        os.remove(metrics_path)

    if fid_tmp_ctx is not None:
        fid_tmp_ctx.cleanup()
    for legacy_dir in (
        '_fid_hr',
        '_fid_bicubic',
        '_fid_patches_sr',
        '_fid_patches_hr',
        '_fid_patches_bicubic',
    ):
        legacy_path = os.path.join(output_dir, legacy_dir)
        if os.path.isdir(legacy_path):
            shutil.rmtree(legacy_path, ignore_errors=True)

    print(f"\nResults saved: {output_dir}")
    print(f"  results.txt : {results_path}")
    if args.save_metrics_json:
        print(f"  metrics.json: {metrics_path}")


if __name__ == '__main__':
    main()
