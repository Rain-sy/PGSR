#!/usr/bin/env python
"""
Single-image Dual-Stream FLUX SR evaluation with an explicit FLUX text prompt.

This script reuses evaluate_dual_lora.py's LoRA-aware model loading and tiled
inference, but replaces the cached empty text embeddings with embeddings from a
user supplied prompt. It is intended for quick prompt ablations on one image.
"""

import argparse
import hashlib
import os
from datetime import datetime

import numpy as np
from PIL import Image
import torch

from evaluate_dual_lora import (
    DualStreamEvaluator,
    calculate_psnr,
    calculate_ssim,
    clear_memory,
    run_sr_tiled_with_oom_retry,
)


DEFAULT_CHECKPOINT = (
    "checkpoints/dual_lora/"
    "20260416_165714_str1_pw1_gate4_lpw0_lpp0p25_crop2_lora_r16_a16_d0_llr0p0001_tpominicontrol/"
    "best_model.pt"
)
DEFAULT_LR_IMAGE = "Data/DIV2K/DIV2K_valid_LR_bicubic_X4/0804x4.png"
DEFAULT_PROMPT = (
    "A high quality super-resolution image. The back of the person's shirt has "
    "clear readable text saying barkrodeo.com."
)


class PromptDualStreamEvaluator(DualStreamEvaluator):
    def __init__(
        self,
        model_name,
        device,
        checkpoint_path,
        pixel_weight=1.0,
        prompt="",
        prompt_mode="both",
    ):
        super().__init__(model_name, device, checkpoint_path, pixel_weight)
        self.prompt = prompt or ""
        self.prompt_mode = prompt_mode

    def _cache_text_embeddings(self):
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        dtype = torch.bfloat16
        clip_prompt = self.prompt if self.prompt_mode in ("both", "clip") else ""
        t5_prompt = self.prompt if self.prompt_mode in ("both", "t5") else ""

        text_enc = CLIPTextModel.from_pretrained(
            self.model_name, subfolder="text_encoder", torch_dtype=dtype
        ).to(self.device)
        tok = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer")

        text_enc_2 = T5EncoderModel.from_pretrained(
            self.model_name, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(self.device)
        tok_2 = T5TokenizerFast.from_pretrained(self.model_name, subfolder="tokenizer_2")

        with torch.no_grad():
            clip_inputs = tok(
                [clip_prompt],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            t5_inputs = tok_2(
                [t5_prompt],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)

            clip_out = text_enc(clip_inputs)
            t5_out = text_enc_2(t5_inputs)
            self._cached_embeds = {
                "pooled": clip_out.pooler_output.to(dtype),
                "prompt": t5_out[0].to(dtype),
                "text_ids": torch.zeros(t5_out[0].shape[1], 3, device=self.device, dtype=dtype),
            }

        print(f"[Prompt] mode={self.prompt_mode}")
        print(f"[Prompt] CLIP: {clip_prompt!r}")
        print(f"[Prompt] T5:   {t5_prompt!r}")

        del text_enc, text_enc_2, tok, tok_2, clip_out, t5_out, clip_inputs, t5_inputs
        clear_memory(self.device)


def _default_hr_from_lr(lr_path):
    directory, filename = os.path.split(lr_path)
    base, ext = os.path.splitext(filename)
    if base.endswith("x4"):
        base = base[:-2]
    elif base.endswith("_x4"):
        base = base[:-3]
    candidates = []
    if "DIV2K_valid_LR_bicubic_X4" in directory:
        candidates.append(
            os.path.join(
                directory.replace("DIV2K_valid_LR_bicubic_X4", "DIV2K_valid_HR"),
                base + ext,
            )
        )
    candidates.append(os.path.join("Data/DIV2K/DIV2K_valid_HR", base + ext))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _safe_tag(text):
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"prompt_{digest}"


def main():
    parser = argparse.ArgumentParser(
        description="Single-image Dual LoRA SR evaluation with a non-empty FLUX prompt."
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lr_image", type=str, default=DEFAULT_LR_IMAGE)
    parser.add_argument("--hr_image", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="both",
        choices=["both", "clip", "t5"],
        help="Which FLUX text branch receives the prompt. The other branch uses empty text.",
    )
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--pixel_weight", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--control_guidance_start", type=float, default=None)
    parser.add_argument("--control_guidance_end", type=float, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--min_tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--blend_mode", type=str, default="linear")
    parser.add_argument("--output_base", type=str, default="./outputs/prompt_single")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--save_bicubic", action="store_true", default=False)
    parser.add_argument("--save_compare", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    initial_pixel_weight = args.pixel_weight if args.pixel_weight is not None else 1.0
    evaluator = PromptDualStreamEvaluator(
        args.model_name,
        device,
        args.checkpoint,
        initial_pixel_weight,
        prompt=args.prompt,
        prompt_mode=args.prompt_mode,
    )
    evaluator.load()

    if args.pixel_weight is not None:
        evaluator.pixel_weight = args.pixel_weight
    strength = args.strength if args.strength is not None else evaluator.strength
    evaluator.control_guidance_start = (
        args.control_guidance_start
        if args.control_guidance_start is not None
        else evaluator.control_guidance_start
    )
    evaluator.control_guidance_end = (
        args.control_guidance_end
        if args.control_guidance_end is not None
        else evaluator.control_guidance_end
    )

    lr_img = Image.open(args.lr_image).convert("RGB")
    hr_path = args.hr_image if args.hr_image is not None else _default_hr_from_lr(args.lr_image)
    hr_img = Image.open(hr_path).convert("RGB") if hr_path else None
    if hr_img is not None:
        target_size = hr_img.size
    else:
        target_size = (lr_img.width * args.scale, lr_img.height * args.scale)

    lr_bicubic = lr_img.resize(target_size, Image.BICUBIC)
    lr_np = np.array(lr_bicubic)
    lr_t = torch.from_numpy(lr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    lr_t = lr_t.to(device).to(torch.bfloat16)

    print("=" * 70)
    print("Single-image Dual-Stream FLUX SR Evaluation (LoRA + Prompt)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LR image: {args.lr_image}")
    print(f"HR image: {hr_path if hr_path else '(none)'}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.num_steps}, Guidance: {args.guidance}, Strength: {strength}")
    print(f"Pixel Weight: {evaluator.pixel_weight}")
    print(f"Control Guidance Window: [{evaluator.control_guidance_start}, {evaluator.control_guidance_end}]")
    print("=" * 70)

    sr_t = run_sr_tiled_with_oom_retry(
        evaluator,
        lr_t,
        device,
        num_steps=args.num_steps,
        guidance=args.guidance,
        tile_size=args.tile_size,
        min_tile_size=args.min_tile_size,
        overlap=args.overlap,
        blend_mode=args.blend_mode,
        strength=strength,
    )
    sr_np = (
        ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5)
        .permute(1, 2, 0)
        .numpy()
        .astype(np.uint8)
    )

    base_name = os.path.splitext(os.path.basename(args.lr_image))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{ts}_{base_name}_{_safe_tag(args.prompt)}_{args.prompt_mode}"
    output_dir = os.path.join(args.output_base, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, f"{base_name}_sr_prompt.png")
    bic_path = os.path.join(output_dir, f"{base_name}_bicubic.png")
    Image.fromarray(sr_np).save(pred_path)
    if args.save_bicubic:
        lr_bicubic.save(bic_path)

    compare_path = None
    if args.save_compare:
        compare_path = os.path.join(output_dir, f"{base_name}_compare.png")
        if hr_img is not None:
            comp = Image.new("RGB", (target_size[0] * 3, target_size[1]))
            comp.paste(lr_bicubic, (0, 0))
            comp.paste(Image.fromarray(sr_np), (target_size[0], 0))
            comp.paste(hr_img, (target_size[0] * 2, 0))
        else:
            comp = Image.new("RGB", (target_size[0] * 2, target_size[1]))
            comp.paste(lr_bicubic, (0, 0))
            comp.paste(Image.fromarray(sr_np), (target_size[0], 0))
        comp.save(compare_path)

    metrics = []
    if hr_img is not None:
        hr_np = np.array(hr_img)
        psnr_sr = calculate_psnr(sr_np, hr_np)
        ssim_sr = calculate_ssim(sr_np, hr_np)
        psnr_bic = calculate_psnr(lr_np, hr_np)
        ssim_bic = calculate_ssim(lr_np, hr_np)
        metrics.append(f"Bicubic: PSNR={psnr_bic:.4f}, SSIM={ssim_bic:.4f}")
        metrics.append(f"Prompt SR: PSNR={psnr_sr:.4f}, SSIM={ssim_sr:.4f}")
        metrics.append(f"Delta: PSNR={psnr_sr - psnr_bic:+.4f}, SSIM={ssim_sr - ssim_bic:+.4f}")

    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("Single-image Dual-Stream FLUX SR Evaluation (LoRA + Prompt)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"LR image: {args.lr_image}\n")
        f.write(f"HR image: {hr_path if hr_path else '(none)'}\n")
        f.write(f"Prompt mode: {args.prompt_mode}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Steps: {args.num_steps}, Guidance: {args.guidance}, Strength: {strength}\n")
        f.write(f"Pixel Weight: {evaluator.pixel_weight}\n")
        f.write(f"Control Guidance Window: [{evaluator.control_guidance_start}, {evaluator.control_guidance_end}]\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Prediction: {pred_path}\n")
        if args.save_bicubic:
            f.write(f"Bicubic: {bic_path}\n")
        if compare_path is not None:
            f.write(f"Comparison: {compare_path}\n")
        if metrics:
            f.write("\n".join(metrics) + "\n")

    for line in metrics:
        print(line)
    print(f"Prediction: {pred_path}")
    if compare_path is not None:
        print(f"Comparison: {compare_path}")
    print(f"Results: {results_path}")

    del lr_t, sr_t
    clear_memory(device)


if __name__ == "__main__":
    main()
