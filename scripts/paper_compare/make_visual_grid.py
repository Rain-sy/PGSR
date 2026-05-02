#!/usr/bin/env python
"""Build qualitative comparison grids from normalized method outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import dataset_dirs, ensure_dir, list_images, load_yaml, repo_path, workspace_dir, write_json


def load_crop_config(path: str | Path) -> dict:
    return load_yaml(path)


def image_for_method(manifest: dict, method: str, dataset: str, stem: str, lr_path: Path, hr_dir: Path | None) -> Path | None:
    if method == "LR/Bicubic":
        return lr_path
    if method == "HR":
        if hr_dir is None:
            return None
        matches = [p for p in list_images(hr_dir) if p.stem == stem]
        return matches[0] if matches else None
    p = workspace_dir(manifest, "outputs", method, dataset, f"{stem}.png")
    return p if p.exists() else None


def crop_image(img: Image.Image, crop: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = crop
    x = max(0, min(x, max(0, img.width - 1)))
    y = max(0, min(y, max(0, img.height - 1)))
    w = max(1, min(w, img.width - x))
    h = max(1, min(h, img.height - y))
    return img.crop((x, y, x + w, y + h))


def default_crop(img: Image.Image, crop_size: int) -> tuple[int, int, int, int]:
    size = min(crop_size, img.width, img.height)
    return ((img.width - size) // 2, (img.height - size) // 2, size, size)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(245, 245, 245), font=font)


def build_grid(images: list[tuple[str, Image.Image]], crop_size: int, gap: int = 8, label_h: int = 34) -> Image.Image:
    width = len(images) * crop_size + (len(images) - 1) * gap
    height = label_h + crop_size
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, img in images:
        thumb = img.resize((crop_size, crop_size), Image.BICUBIC)
        draw_label(draw, (x + 4, 5), label)
        canvas.paste(thumb, (x, label_h))
        x += crop_size + gap
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["DIV2K", "RealSR", "DRealSR"])
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--crops", default="configs/paper_compare/crops.yaml")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method columns. Defaults to crops.yaml columns.")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    crop_cfg = load_crop_config(args.crops)
    lr_dir, hr_dir = dataset_dirs(manifest, args.dataset)
    lr_images = list_images(lr_dir)
    if not lr_images:
        raise SystemExit(f"No LR images found in {lr_dir}")
    if args.max_images:
        lr_images = lr_images[: args.max_images]

    default_cfg = crop_cfg.get("default", {})
    dataset_cfg = crop_cfg.get("datasets", {}).get(args.dataset, {})
    columns = args.methods.split(",") if args.methods else default_cfg.get("columns", [])
    columns = [c.strip() for c in columns if c.strip()]
    crop_size = args.crop_size or int(default_cfg.get("crop_size", 256))
    per_image_crops = dataset_cfg.get("images", {}) or {}

    out_dir = ensure_dir(workspace_dir(manifest, "figures", args.dataset))
    report = {}
    for lr_path in lr_images:
        stem = lr_path.stem
        base_img = Image.open(lr_path).convert("RGB")
        crop = per_image_crops.get(stem)
        if crop is None:
            crop = default_crop(base_img, crop_size)
        else:
            crop = tuple(int(x) for x in crop)

        panels = []
        missing = []
        for method in columns:
            src = image_for_method(manifest, method, args.dataset, stem, lr_path, hr_dir)
            if src is None:
                missing.append(method)
                continue
            img = Image.open(src).convert("RGB")
            panels.append((method, crop_image(img, crop)))
        if not panels:
            report[stem] = {"status": "missing_all", "missing": missing}
            continue
        grid = build_grid(panels, crop_size)
        grid.save(out_dir / f"{stem}.png")
        grid.save(out_dir / f"{stem}.pdf")
        report[stem] = {"status": "ok", "crop": list(crop), "missing": missing}

    write_json(out_dir / "visual_grid_report.json", report)
    print(f"[DONE] figures -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
