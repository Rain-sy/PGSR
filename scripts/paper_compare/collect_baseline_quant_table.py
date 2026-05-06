#!/usr/bin/env python
"""Collect locally recomputed baseline metrics into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import repo_path


DATASET_ORDER = ["DIV2K", "RealSR", "DRealSR"]
EXPECTED_IMAGES = {"DIV2K": 100, "RealSR": 100, "DRealSR": 93}
METHOD_ORDER = [
    "ResShift",
    "StableSR",
    "DiffBIR",
    "SeeSR_g1p5_s40_wavelet",
    "PASD",
    "OSEDiff",
    "SinSR",
    "PiSA-SR",
    "PGSR",
]
METRICS = ["PSNR", "SSIM", "LPIPS", "DISTS", "FID", "NIQE", "MUSIQ", "MANIQA", "CLIP-IQA"]


def metric_path(root: Path, method: str, dataset: str) -> Path:
    return root / method / dataset / "metrics.json"


def get_nested(data: dict, *keys):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def fmt(value) -> str:
    if value is None:
        return "--"
    if isinstance(value, (int, float)):
        if abs(float(value)) >= 10:
            return f"{float(value):.2f}"
        return f"{float(value):.4f}"
    return str(value)


def row_from_metrics(method: str, dataset: str, data: dict, valid: bool) -> dict:
    return {
        "Dataset": dataset,
        "Method": method,
        "ValidFID": "yes" if valid else "no",
        "Images": data.get("images"),
        "PSNR": get_nested(data, "classical", "sr", "psnr"),
        "SSIM": get_nested(data, "classical", "sr", "ssim"),
        "LPIPS": get_nested(data, "classical", "sr", "lpips"),
        "DISTS": get_nested(data, "iqa", "dists", "sr"),
        "FID": get_nested(data, "fid", "sr"),
        "NIQE": get_nested(data, "iqa", "niqe", "sr"),
        "MUSIQ": get_nested(data, "iqa", "musiq", "sr"),
        "MANIQA": get_nested(data, "iqa", "maniqa", "sr"),
        "CLIP-IQA": get_nested(data, "iqa", "clipiqa", "sr"),
        "FID_Mode": get_nested(data, "fid", "primary_mode"),
        "FID_Pairing": get_nested(data, "fid", "patch_pairing"),
        "FID_PatchSize": get_nested(data, "fid", "patch_size"),
        "FID_PatchesPerImage": get_nested(data, "fid", "patches_per_image"),
        "FID_PatchCount": get_nested(data, "fid", "patch_count"),
    }


def latex_row(row: dict) -> str:
    values = [fmt(row[m]) for m in METRICS]
    return f"{row['Dataset']} & {row['Method']} & " + " & ".join(values) + r" \\"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", default="experiments/paper_compare_quant/metrics/baselines")
    parser.add_argument("--output-prefix", default="experiments/paper_compare_quant/metrics/baseline_quant_table")
    parser.add_argument("--include-invalid", action="store_true", help="Include rows whose FID was not computed with independent patch pairing.")
    args = parser.parse_args()

    root = repo_path(args.metrics_root)
    rows = []
    missing = []
    invalid = []
    for dataset in DATASET_ORDER:
        for method in METHOD_ORDER:
            path = metric_path(root, method, dataset)
            if not path.exists():
                missing.append((method, dataset))
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fid = data.get("fid", {})
            expected_images = EXPECTED_IMAGES.get(dataset)
            complete = expected_images is None or data.get("images") == expected_images
            valid = (
                complete
                and fid.get("primary_mode") == "patch"
                and fid.get("patch_pairing") == "independent"
                and fid.get("sr") is not None
            )
            if not valid:
                invalid.append((method, dataset, fid.get("patch_pairing")))
                if not args.include_invalid:
                    continue
            rows.append(row_from_metrics(method, dataset, data, valid))

    out_prefix = repo_path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Dataset",
        "Method",
        "ValidFID",
        "Images",
        *METRICS,
        "FID_Mode",
        "FID_Pairing",
        "FID_PatchSize",
        "FID_PatchesPerImage",
        "FID_PatchCount",
    ]
    with open(out_prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_prefix.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(latex_row(row) for row in rows) + ("\n" if rows else ""))
    with open(out_prefix.with_name(out_prefix.name + "_status.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "valid_rows": len(rows),
                "missing": [{"method": m, "dataset": d} for m, d in missing],
                "invalid_fid": [{"method": m, "dataset": d, "patch_pairing": p} for m, d, p in invalid],
            },
            f,
            indent=2,
        )
    print(f"[DONE] wrote {out_prefix.with_suffix('.csv')}")
    print(f"[DONE] wrote {out_prefix.with_suffix('.tex')}")
    print(f"[DONE] wrote {out_prefix.with_name(out_prefix.name + '_status.json')}")
    print(f"[INFO] valid rows={len(rows)}, missing={len(missing)}, invalid_fid={len(invalid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
