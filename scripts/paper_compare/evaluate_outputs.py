#!/usr/bin/env python
"""Evaluate precomputed SR outputs for the paper comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from common import ensure_dir, list_images, repo_path

try:
    import lpips
except ImportError:
    lpips = None

try:
    import pyiqa
except ImportError:
    pyiqa = None

try:
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError:
    sk_ssim = None


IQA_REGISTRY = {
    "dists": {"pyiqa": "dists", "type": "FR", "higher_better": False},
    "niqe": {"pyiqa": "niqe", "type": "NR", "higher_better": False},
    "musiq": {"pyiqa": "musiq", "type": "NR", "higher_better": True},
    "maniqa": {"pyiqa": "maniqa", "type": "NR", "higher_better": True},
    "clipiqa": {"pyiqa": "clipiqa", "type": "NR", "higher_better": True},
}


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if sk_ssim is None:
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)
        mu1, mu2 = x.mean(), y.mean()
        sigma1_sq, sigma2_sq = x.var(), y.var()
        sigma12 = ((x - mu1) * (y - mu2)).mean()
        return float(
            ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2))
            / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        )
    win_size = min(11, img1.shape[0], img1.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return calculate_ssim(img1, img2)
    try:
        return float(sk_ssim(img1, img2, data_range=255, channel_axis=-1, win_size=win_size))
    except TypeError:
        return float(sk_ssim(img1, img2, data_range=255, multichannel=True, win_size=win_size))


def to_tensor_01(img: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).div(255.0).clamp(0, 1).to(device)


def resize_long_edge(t: torch.Tensor, max_long_edge: int) -> torch.Tensor:
    if max_long_edge <= 0:
        return t
    _, _, h, w = t.shape
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return t
    scale = max_long_edge / long_edge
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)


def match_by_hr_stem(root: Path, hr_stem: str) -> Path | None:
    candidates = [
        root / f"{hr_stem}{ext}"
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    ]
    candidates.extend(
        root / f"{hr_stem}{suffix}{ext}"
        for suffix in ("x4", "_x4", "_LR4", "X4", "_X4")
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    )
    for p in candidates:
        if p.exists():
            return p
    matches = sorted([p for p in list_images(root) if p.stem == hr_stem or hr_stem in p.stem])
    return matches[0] if matches else None


def crop_or_pad_patch(img: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y = max(0, min(int(y), max(0, h - patch_size)))
    x = max(0, min(int(x), max(0, w - patch_size)))
    patch = img[y : min(y + patch_size, h), x : min(x + patch_size, w), :]
    pad_h = patch_size - patch.shape[0]
    pad_w = patch_size - patch.shape[1]
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)), mode="edge")
    return patch


def sample_patch_coords(h: int, w: int, patch_size: int, n_per_image: int, rng) -> list[tuple[int, int]]:
    max_y = max(0, h - patch_size)
    max_x = max(0, w - patch_size)
    if max_y == 0 and max_x == 0:
        return [(0, 0)] * n_per_image
    ys = rng.integers(0, max_y + 1, size=n_per_image) if max_y > 0 else np.zeros(n_per_image, dtype=np.int64)
    xs = rng.integers(0, max_x + 1, size=n_per_image) if max_x > 0 else np.zeros(n_per_image, dtype=np.int64)
    return list(zip(ys.tolist(), xs.tolist()))


def compute_fid(path_a: Path, path_b: Path, device: torch.device) -> float | None:
    if pyiqa is None:
        return None
    try:
        metric = pyiqa.create_metric("fid", device=device)
        score = float(metric(str(path_a), str(path_b)))
        del metric
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return score
    except Exception as exc:
        print(f"[FID] failed: {exc}")
        return None


def mean_or_none(values: list[float]) -> float | None:
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-dir", required=True)
    parser.add_argument("--hr-dir", required=True)
    parser.add_argument("--lr-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips-device", default="cuda")
    parser.add_argument("--iqa-device", default="cuda")
    parser.add_argument("--fid-device", default="cuda")
    parser.add_argument("--iqa-metrics", default="dists,niqe,musiq,maniqa,clipiqa")
    parser.add_argument("--fr-max-long-edge", type=int, default=1024)
    parser.add_argument("--fid-mode", choices=["none", "patch", "full", "both"], default="patch")
    parser.add_argument("--fid-patch-size", type=int, default=512)
    parser.add_argument("--fid-patches-per-image", type=int, default=25)
    parser.add_argument("--fid-patch-seed", type=int, default=42)
    parser.add_argument(
        "--fid-patch-pairing",
        choices=["independent", "aligned"],
        default="independent",
        help=(
            "Patch sampling protocol for FID. 'independent' samples separate random "
            "patch coordinates for SR/BIC and HR, which is closer to distributional "
            "FID protocols. 'aligned' uses identical coordinates and usually gives "
            "lower values because image content is spatially matched."
        ),
    )
    args = parser.parse_args()

    sr_dir = repo_path(args.sr_dir)
    hr_dir = repo_path(args.hr_dir)
    lr_dir = repo_path(args.lr_dir)
    out_dir = ensure_dir(args.out_dir)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    lpips_device = torch.device(args.lpips_device if args.lpips_device == "cpu" or torch.cuda.is_available() else "cpu")
    iqa_device = torch.device(args.iqa_device if args.iqa_device == "cpu" or torch.cuda.is_available() else "cpu")
    fid_device = torch.device(args.fid_device if args.fid_device == "cpu" or torch.cuda.is_available() else "cpu")

    lpips_fn = None
    if lpips is not None:
        lpips_fn = lpips.LPIPS(net="alex").to(lpips_device).eval()

    iqa_metrics = {}
    iqa_values = {name: [] for name in IQA_REGISTRY}
    iqa_bic_values = {name: [] for name in IQA_REGISTRY}
    if pyiqa is not None:
        for name in [m.strip().lower() for m in args.iqa_metrics.split(",") if m.strip()]:
            if name not in IQA_REGISTRY:
                continue
            try:
                iqa_metrics[name] = pyiqa.create_metric(IQA_REGISTRY[name]["pyiqa"], device=iqa_device, as_loss=False).eval()
                print(f"[IQA] loaded {name}")
            except Exception as exc:
                print(f"[IQA] failed to load {name}: {exc}")

    psnr_values, ssim_values, lpips_values = [], [], []
    psnr_bic_values, ssim_bic_values, lpips_bic_values = [], [], []
    rows = []
    rng_sr = np.random.default_rng(args.fid_patch_seed)
    rng_bic = np.random.default_rng(args.fid_patch_seed)
    rng_hr = np.random.default_rng(args.fid_patch_seed if args.fid_patch_pairing == "aligned" else args.fid_patch_seed + 1000003)
    fid_patch_count = 0
    patch_dirs = {
        "sr": out_dir / "_fid_patches_sr",
        "hr": out_dir / "_fid_patches_hr",
        "bic": out_dir / "_fid_patches_bicubic",
    }
    full_dirs = {
        "sr": out_dir / "_fid_full_sr",
        "hr": out_dir / "_fid_full_hr",
        "bic": out_dir / "_fid_full_bicubic",
    }
    if args.fid_mode in ("patch", "both"):
        for d in patch_dirs.values():
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
    if args.fid_mode in ("full", "both"):
        for d in full_dirs.values():
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    hr_files = list_images(hr_dir)
    for hr_path in tqdm(hr_files, desc=f"{args.method}/{args.dataset}"):
        sr_path = match_by_hr_stem(sr_dir, hr_path.stem)
        lr_path = match_by_hr_stem(lr_dir, hr_path.stem)
        if sr_path is None or lr_path is None:
            print(f"[WARN] missing pair for {hr_path.name}: sr={sr_path}, lr={lr_path}")
            continue
        hr_np = np.array(Image.open(hr_path).convert("RGB"))
        sr_img = Image.open(sr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        if sr_img.size != (hr_np.shape[1], hr_np.shape[0]):
            sr_img = sr_img.resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC)
        bic_img = lr_img.resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC)
        sr_np = np.array(sr_img)
        bic_np = np.array(bic_img)

        psnr = calculate_psnr(sr_np, hr_np)
        ssim = calculate_ssim(sr_np, hr_np)
        psnr_bic = calculate_psnr(bic_np, hr_np)
        ssim_bic = calculate_ssim(bic_np, hr_np)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        psnr_bic_values.append(psnr_bic)
        ssim_bic_values.append(ssim_bic)

        lp = lp_bic = None
        if lpips_fn is not None:
            with torch.no_grad():
                sr_t = to_tensor_01(sr_np, lpips_device).mul(2).sub(1)
                hr_t = to_tensor_01(hr_np, lpips_device).mul(2).sub(1)
                bic_t = to_tensor_01(bic_np, lpips_device).mul(2).sub(1)
                lp = float(lpips_fn(sr_t, hr_t).item())
                lp_bic = float(lpips_fn(bic_t, hr_t).item())
            lpips_values.append(lp)
            lpips_bic_values.append(lp_bic)
            del sr_t, hr_t, bic_t

        for name, metric in iqa_metrics.items():
            cfg = IQA_REGISTRY[name]
            try:
                with torch.no_grad():
                    sr_t = to_tensor_01(sr_np, iqa_device)
                    bic_t = to_tensor_01(bic_np, iqa_device)
                    if cfg["type"] == "FR":
                        hr_t = to_tensor_01(hr_np, iqa_device)
                        sr_score = float(metric(resize_long_edge(sr_t, args.fr_max_long_edge), resize_long_edge(hr_t, args.fr_max_long_edge)).item())
                        bic_score = float(metric(resize_long_edge(bic_t, args.fr_max_long_edge), resize_long_edge(hr_t, args.fr_max_long_edge)).item())
                        del hr_t
                    else:
                        sr_score = float(metric(sr_t).item())
                        bic_score = float(metric(bic_t).item())
                    del sr_t, bic_t
            except Exception as exc:
                print(f"[IQA] {name} failed on {hr_path.name}: {exc}")
                sr_score = bic_score = float("nan")
            iqa_values[name].append(sr_score)
            iqa_bic_values[name].append(bic_score)

        if args.fid_mode in ("patch", "both"):
            coords_sr = sample_patch_coords(hr_np.shape[0], hr_np.shape[1], args.fid_patch_size, args.fid_patches_per_image, rng_sr)
            coords_bic = coords_sr if args.fid_patch_pairing == "aligned" else sample_patch_coords(hr_np.shape[0], hr_np.shape[1], args.fid_patch_size, args.fid_patches_per_image, rng_bic)
            coords_hr = coords_sr if args.fid_patch_pairing == "aligned" else sample_patch_coords(hr_np.shape[0], hr_np.shape[1], args.fid_patch_size, args.fid_patches_per_image, rng_hr)
            for idx, ((y_sr, x_sr), (y_bic, x_bic), (y_hr, x_hr)) in enumerate(zip(coords_sr, coords_bic, coords_hr)):
                name = f"{hr_path.stem}_p{idx:03d}.png"
                Image.fromarray(crop_or_pad_patch(sr_np, y_sr, x_sr, args.fid_patch_size)).save(patch_dirs["sr"] / name)
                Image.fromarray(crop_or_pad_patch(hr_np, y_hr, x_hr, args.fid_patch_size)).save(patch_dirs["hr"] / name)
                Image.fromarray(crop_or_pad_patch(bic_np, y_bic, x_bic, args.fid_patch_size)).save(patch_dirs["bic"] / name)
            fid_patch_count += len(coords_sr)
        if args.fid_mode in ("full", "both"):
            Image.fromarray(sr_np).save(full_dirs["sr"] / f"{hr_path.stem}.png")
            Image.fromarray(hr_np).save(full_dirs["hr"] / f"{hr_path.stem}.png")
            Image.fromarray(bic_np).save(full_dirs["bic"] / f"{hr_path.stem}.png")

        rows.append({
            "file": hr_path.name,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp,
            "bic_psnr": psnr_bic,
            "bic_ssim": ssim_bic,
            "bic_lpips": lp_bic,
        })
        if device.type == "cuda" or lpips_device.type == "cuda" or iqa_device.type == "cuda":
            torch.cuda.empty_cache()

    fid_results = {}
    if args.fid_mode in ("patch", "both"):
        fid_results["patch"] = {
            "sr": compute_fid(patch_dirs["sr"], patch_dirs["hr"], fid_device),
            "bic": compute_fid(patch_dirs["bic"], patch_dirs["hr"], fid_device),
        }
    if args.fid_mode in ("full", "both"):
        fid_results["full"] = {
            "sr": compute_fid(full_dirs["sr"], full_dirs["hr"], fid_device),
            "bic": compute_fid(full_dirs["bic"], full_dirs["hr"], fid_device),
        }
    primary = fid_results.get("patch") or fid_results.get("full") or {}

    summary = {
        "method": args.method,
        "dataset": args.dataset,
        "images": len(rows),
        "classical": {
            "sr": {"psnr": mean_or_none(psnr_values), "ssim": mean_or_none(ssim_values), "lpips": mean_or_none(lpips_values)},
            "bic": {"psnr": mean_or_none(psnr_bic_values), "ssim": mean_or_none(ssim_bic_values), "lpips": mean_or_none(lpips_bic_values)},
        },
        "iqa": {
            name: {
                "sr": mean_or_none(iqa_values[name]),
                "bic": mean_or_none(iqa_bic_values[name]),
                "type": IQA_REGISTRY[name]["type"],
                "higher_better": IQA_REGISTRY[name]["higher_better"],
            }
            for name in iqa_metrics
        },
        "fid": {
            "primary_mode": "patch" if "patch" in fid_results else ("full" if "full" in fid_results else None),
            "sr": primary.get("sr"),
            "bic": primary.get("bic"),
            "patch": fid_results.get("patch"),
            "full": fid_results.get("full"),
            "patch_size": args.fid_patch_size if args.fid_mode in ("patch", "both") else None,
            "patches_per_image": args.fid_patches_per_image if args.fid_mode in ("patch", "both") else None,
            "patch_pairing": args.fid_patch_pairing if args.fid_mode in ("patch", "both") else None,
            "patch_count": fid_patch_count if args.fid_mode in ("patch", "both") else None,
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)

    for d in list(patch_dirs.values()) + list(full_dirs.values()):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
