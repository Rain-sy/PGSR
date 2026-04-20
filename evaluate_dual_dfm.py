#!/usr/bin/env python
"""
======================================================================
Dual-Stream FLUX SR Evaluation - aligned with official Diffusers pipeline
======================================================================

Companion script for train_dual_dfm.py
Automatically detects and loads LoRA adapters when checkpoint contains
`use_lora=True` and `lora_state_dict`.

Usage:
    python evaluate_dual_dfm.py \
        --checkpoint checkpoints/dual_control/xxx/best_model.pt \
        --hr_dir Data/DIV2K/DIV2K_valid_HR \
        --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
        --strength 0.7 --num_steps 20
"""

import os
import gc
import math
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

try:
    from skimage.metrics import structural_similarity as sk_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    sk_ssim = None
    SKIMAGE_AVAILABLE = False
    print("Note: scikit-image not installed; falling back to global SSIM. Run: pip install scikit-image")

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

        self.zero_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
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
        
        noisy_packed = self._pack(noisy_for_pack)
        fused_packed = self._pack(fused_for_pack)
        del fused_cond
        img_ids = self._img_ids(H_pack, W_pack, device, dtype)
        
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
    
    parser.add_argument('--output_base', type=str, default='./outputs')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--save_images', dest='save_images', action='store_true', default=True)
    parser.add_argument('--no_save_images', dest='save_images', action='store_false')
    parser.add_argument('--save_comparisons', dest='save_comparisons', action='store_true', default=True)
    parser.add_argument('--no_save_comparisons', dest='save_comparisons', action='store_false')
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
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect dataset
    if args.dataset is None:
        hr_lower = args.hr_dir.lower()
        if 'urban' in hr_lower:
            args.dataset = 'Urban100'
        elif 'div2k' in hr_lower:
            args.dataset = 'DIV2K'
        elif 'drealsr' in hr_lower:
            args.dataset = 'DRealSR'
        elif 'realsr' in hr_lower:
            args.dataset = 'RealSR'
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
    print(f"Train LPIPS: weight={evaluator.train_lpips_weight}, prob={evaluator.train_lpips_apply_prob}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}")
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
    print(f"Pixel Fusion: concat+1x1 conv (source={evaluator.fusion_source})")
    if lpips_fn:
        print(f"Bicubic:  PSNR={avg_psnr_bic:.4f} dB, SSIM={avg_ssim_bic:.4f}, LPIPS={avg_lpips_bic:.4f}")
        print(f"SR:       PSNR={avg_psnr:.4f} dB, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        print(f"Delta:    {avg_psnr - avg_psnr_bic:+.4f} dB, {avg_ssim - avg_ssim_bic:+.4f}, {avg_lpips_bic - avg_lpips:+.4f}")
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
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Images: {len(psnr_list)}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}\n")
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
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (delta {delta:+.2f}), LPIPS={lpips_list[i]:.4f}\n")
            else:
                f.write(f"{fname}: PSNR={psnr_list[i]:.2f} (delta {delta:+.2f})\n")
    
    print(f"\nResults saved: {output_dir}")


if __name__ == '__main__':
    main()
