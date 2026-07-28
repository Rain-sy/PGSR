#!/usr/bin/env python
"""
Run a small single-image prompt/parameter sweep for Dual LoRA SR.

Unlike launching evaluate_dual_lora_prompt_single.py repeatedly, this script
encodes all text prompts first, loads the FLUX/ControlNet/LoRA stack once, then
runs the variants back-to-back. It saves only SR predictions by default.
"""

import argparse
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
from evaluate_dual_lora_prompt_single import DEFAULT_CHECKPOINT, DEFAULT_LR_IMAGE


DEFAULT_PROMPT = (
    "Faithful super-resolution of the original photo. Preserve natural clothing "
    "texture and details. The small blue website text on the back of the white "
    "shirt reads barkrodeo.com in lowercase letters."
)


SWEEP_VARIANTS = [
    {
        "name": "run1_both_g2p5_s1",
        "prompt": DEFAULT_PROMPT,
        "prompt_mode": "both",
        "guidance": 2.5,
        "strength": 1.0,
        "seed": 61,
        "num_steps": 20,
    },
    {
        "name": "run2_both_g3_s1",
        "prompt": DEFAULT_PROMPT,
        "prompt_mode": "both",
        "guidance": 3.0,
        "strength": 1.0,
        "seed": 62,
        "num_steps": 20,
    },
    {
        "name": "run3_both_g3p5_s1",
        "prompt": DEFAULT_PROMPT,
        "prompt_mode": "both",
        "guidance": 3.5,
        "strength": 1.0,
        "seed": 63,
        "num_steps": 20,
    },
    {
        "name": "run4_both_g4_s1",
        "prompt": DEFAULT_PROMPT,
        "prompt_mode": "both",
        "guidance": 4.0,
        "strength": 1.0,
        "seed": 64,
        "num_steps": 20,
    },
    {
        "name": "run5_both_g4p5_s1",
        "prompt": DEFAULT_PROMPT,
        "prompt_mode": "both",
        "guidance": 4.5,
        "strength": 1.0,
        "seed": 65,
        "num_steps": 20,
    },
]


def _fmt_float_tag(value):
    return f"{value:.1f}".replace("-", "m").replace(".", "p")


def build_guidance_variants(prompt, start, step, count, strength, seed_start,
                            num_steps, prompt_mode, repeats=1):
    variants = []
    seed = int(seed_start)
    for idx in range(count):
        guidance = round(float(start) + idx * float(step), 10)
        for rep in range(int(repeats)):
            variants.append({
                "name": (
                    f"run{len(variants) + 1:02d}_{prompt_mode}_"
                    f"g{_fmt_float_tag(guidance)}_rep{rep + 1}_s1"
                ),
                "prompt": prompt,
                "prompt_mode": prompt_mode,
                "guidance": guidance,
                "strength": float(strength),
                "seed": seed,
                "num_steps": int(num_steps),
            })
            seed += 1
    return variants


class CachedPromptEvaluator(DualStreamEvaluator):
    def __init__(self, *args, initial_embeds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_prompt_embeds = initial_embeds

    def _cache_text_embeddings(self):
        if self._initial_prompt_embeds is None:
            raise RuntimeError("CachedPromptEvaluator requires precomputed prompt embeddings.")
        self.set_prompt_embeds(self._initial_prompt_embeds)

    def set_prompt_embeds(self, embeds):
        self._cached_embeds = {
            key: value.to(self.device) for key, value in embeds.items()
        }


def _encode_variant_prompts(model_name, device, variants):
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    dtype = torch.bfloat16
    print("[PromptSweep] Pre-encoding text embeddings...")
    text_enc = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    tok = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_enc_2 = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device)
    tok_2 = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer_2")

    encoded = []
    with torch.no_grad():
        for variant in variants:
            prompt = variant["prompt"]
            mode = variant["prompt_mode"]
            clip_prompt = prompt if mode in ("both", "clip") else ""
            t5_prompt = prompt if mode in ("both", "t5") else ""
            clip_inputs = tok(
                [clip_prompt],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            t5_inputs = tok_2(
                [t5_prompt],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            clip_out = text_enc(clip_inputs)
            t5_out = text_enc_2(t5_inputs)
            encoded.append({
                "pooled": clip_out.pooler_output.to(dtype).cpu(),
                "prompt": t5_out[0].to(dtype).cpu(),
                "text_ids": torch.zeros(t5_out[0].shape[1], 3, dtype=dtype),
            })
            print(
                f"[PromptSweep] {variant['name']}: mode={mode}, "
                f"guidance={variant['guidance']}, strength={variant['strength']}, "
                f"seed={variant['seed']}"
            )

    del text_enc, text_enc_2, tok, tok_2
    clear_memory(device)
    return encoded


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


def main():
    parser = argparse.ArgumentParser(description="Five-run prompt sweep for 0804 shirt text.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--lr_image", type=str, default=DEFAULT_LR_IMAGE)
    parser.add_argument("--hr_image", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--output_base", type=str, default="./outputs/prompt_sweep")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--pixel_weight", type=float, default=None)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--min_tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--blend_mode", type=str, default="linear")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt_mode", type=str, default="both", choices=["both", "clip", "t5"])
    parser.add_argument("--guidance_start", type=float, default=None)
    parser.add_argument("--guidance_step", type=float, default=0.1)
    parser.add_argument("--guidance_count", type=int, default=None)
    parser.add_argument("--guidance_repeats", type=int, default=1)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed_start", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    variants = SWEEP_VARIANTS
    if args.guidance_start is not None or args.guidance_count is not None:
        if args.guidance_start is None or args.guidance_count is None:
            raise ValueError("--guidance_start and --guidance_count must be set together.")
        variants = build_guidance_variants(
            args.prompt,
            args.guidance_start,
            args.guidance_step,
            args.guidance_count,
            args.strength,
            args.seed_start,
            args.num_steps,
            args.prompt_mode,
            repeats=args.guidance_repeats,
        )
    encoded_variants = _encode_variant_prompts(args.model_name, device, variants)

    initial_pixel_weight = args.pixel_weight if args.pixel_weight is not None else 1.0
    evaluator = CachedPromptEvaluator(
        args.model_name,
        device,
        args.checkpoint,
        initial_pixel_weight,
        initial_embeds=encoded_variants[0],
    )
    evaluator.load()
    if args.pixel_weight is not None:
        evaluator.pixel_weight = args.pixel_weight

    lr_img = Image.open(args.lr_image).convert("RGB")
    hr_path = args.hr_image if args.hr_image is not None else _default_hr_from_lr(args.lr_image)
    hr_img = Image.open(hr_path).convert("RGB") if hr_path else None
    target_size = hr_img.size if hr_img is not None else (
        lr_img.width * args.scale,
        lr_img.height * args.scale,
    )
    lr_bicubic = lr_img.resize(target_size, Image.BICUBIC)
    lr_np = np.array(lr_bicubic)
    lr_t = torch.from_numpy(lr_np).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    lr_t = lr_t.to(device).to(torch.bfloat16)

    base_name = os.path.splitext(os.path.basename(args.lr_image))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"{ts}_{base_name}_barkrodeo_sweep"
    output_dir = os.path.join(args.output_base, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Dual LoRA Prompt Sweep")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LR image: {args.lr_image}")
    print(f"HR image: {hr_path if hr_path else '(none)'}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    lines = []
    hr_np = np.array(hr_img) if hr_img is not None else None
    for variant, embeds in zip(variants, encoded_variants):
        print(f"\n[PromptSweep] Running {variant['name']}")
        evaluator.set_prompt_embeds(embeds)
        torch.manual_seed(int(variant["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(variant["seed"]))

        sr_t = run_sr_tiled_with_oom_retry(
            evaluator,
            lr_t,
            device,
            num_steps=int(variant["num_steps"]),
            guidance=float(variant["guidance"]),
            tile_size=args.tile_size,
            min_tile_size=args.min_tile_size,
            overlap=args.overlap,
            blend_mode=args.blend_mode,
            strength=float(variant["strength"]),
        )
        sr_np = (
            ((sr_t[0].float().cpu().clamp(-1, 1) + 1) * 127.5)
            .permute(1, 2, 0)
            .numpy()
            .astype(np.uint8)
        )
        pred_path = os.path.join(output_dir, f"{base_name}_{variant['name']}.png")
        Image.fromarray(sr_np).save(pred_path)

        metric_text = ""
        if hr_np is not None:
            metric_text = (
                f", PSNR={calculate_psnr(sr_np, hr_np):.4f}, "
                f"SSIM={calculate_ssim(sr_np, hr_np):.4f}"
            )
        line = (
            f"{variant['name']}: prompt_mode={variant['prompt_mode']}, "
            f"guidance={variant['guidance']}, strength={variant['strength']}, "
            f"seed={variant['seed']}{metric_text}, path={pred_path}"
        )
        lines.append(line)
        print(line)
        del sr_t, sr_np
        clear_memory(device)

    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write("Dual LoRA Prompt Sweep\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"LR image: {args.lr_image}\n")
        f.write(f"HR image: {hr_path if hr_path else '(none)'}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Output: {output_dir}\n\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
