#!/usr/bin/env python
"""Merge published baseline numbers with locally recomputed PGSR metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import load_yaml, repo_path, workspace_dir

METHOD_ORDER = [
    "ResShift",
    "StableSR",
    "DiffBIR",
    "SeeSR",
    "PASD",
    "PiSA-SR",
    "GuideSR",
    "FlowSR",
    "FluxSR",
    "PGSR",
]

DATASET_ALIASES = {
    "DIV2K": "DIV2K-Val",
    "DIV2K-Val": "DIV2K-Val",
    "RealSR": "RealSR",
    "DRealSR": "DRealSR",
}


def fmt(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def load_pgsr_metrics(manifest: dict, dataset: str) -> list[float | None] | None:
    metrics_path = workspace_dir(manifest, "metrics", "PGSR", dataset, "metrics.json")
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classical = data.get("classical", {}).get("sr", {})
    iqa = data.get("iqa", {})
    fid = data.get("fid", {})
    return [
        classical.get("psnr"),
        classical.get("ssim"),
        classical.get("lpips"),
        iqa.get("dists", {}).get("sr"),
        fid.get("sr"),
        iqa.get("niqe", {}).get("sr"),
        iqa.get("musiq", {}).get("sr"),
        iqa.get("maniqa", {}).get("sr"),
        iqa.get("clipiqa", {}).get("sr"),
    ]


def latex_row(dataset: str, method: str, values: list) -> str:
    prefix = f"& \\textbf{{PGSR (ours)}}" if method == "PGSR" else f"& {method}"
    return prefix + " & " + " & ".join(fmt(v) for v in values) + r" \\"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--published", default="configs/paper_compare/published_results.yaml")
    parser.add_argument("--output-prefix", default="experiments/paper_compare/metrics/main_table")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    published = load_yaml(args.published)
    metrics = published["metrics"]
    rows = []
    latex_lines = []

    for dataset_key, dataset_values in published["datasets"].items():
        local_dataset = "DIV2K" if dataset_key == "DIV2K-Val" else dataset_key
        pgsr = load_pgsr_metrics(manifest, local_dataset)
        if pgsr is not None:
            dataset_values = dict(dataset_values)
            dataset_values["PGSR"] = pgsr
        for method in METHOD_ORDER:
            if method not in dataset_values:
                continue
            values = dataset_values[method]
            rows.append({"Dataset": dataset_key, "Method": method, **dict(zip(metrics, values))})
            latex_lines.append(latex_row(dataset_key, method, values))

    out_prefix = repo_path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Method"] + metrics)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_prefix.with_suffix(".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")
    print(f"[DONE] wrote {out_prefix.with_suffix('.csv')}")
    print(f"[DONE] wrote {out_prefix.with_suffix('.tex')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
