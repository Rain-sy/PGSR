#!/usr/bin/env python
"""Run one baseline/dataset quantitative job, then delete intermediate images."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import dataset_dirs, list_images, load_yaml, repo_path, workspace_dir


def run(cmd: list[str], cwd: Path, env=None) -> int:
    print("CMD:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method")
    parser.add_argument("dataset", choices=["DIV2K", "RealSR", "DRealSR"])
    parser.add_argument("--manifest", default="configs/paper_compare/baselines_quant.yaml")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--metric-env", default="IR")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--fid-mode", choices=["none", "patch", "full", "both"], default="patch")
    parser.add_argument("--fid-patch-size", type=int, default=512)
    parser.add_argument("--fid-patches-per-image", type=int, default=25)
    parser.add_argument("--fid-patch-pairing", choices=["independent", "aligned"], default="independent")
    parser.add_argument("--iqa-metrics", default="dists,niqe,musiq,maniqa,clipiqa")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    lr_dir, hr_dir = dataset_dirs(manifest, args.dataset)
    if hr_dir is None:
        raise SystemExit(f"{args.dataset} has no HR dir configured")

    sr_dir = workspace_dir(manifest, "outputs", args.method, args.dataset)
    raw_dir = workspace_dir(manifest, "outputs", "_raw", args.method, args.dataset)
    shutil.rmtree(sr_dir, ignore_errors=True)
    shutil.rmtree(raw_dir, ignore_errors=True)

    env_cmd = [
        "python",
        "scripts/paper_compare/run_baseline.py",
        args.method,
        args.dataset,
        "--manifest",
        args.manifest,
        "--execute",
        "--allow-needs-command",
        "--gpu",
        args.gpu,
    ]
    rc = run(env_cmd, repo_path("."), env=None)
    if rc != 0:
        return rc

    expected_images = len(list_images(lr_dir))
    produced_images = len(list_images(sr_dir))
    if produced_images != expected_images:
        print(
            f"[ERROR] {args.method}/{args.dataset} produced {produced_images}/{expected_images} "
            "normalized outputs; refusing to compute metrics on an incomplete set.",
            flush=True,
        )
        return 4

    metric_dir = workspace_dir(manifest, "metrics", "baselines", args.method, args.dataset)
    metric_cmd = [
        "conda",
        "run",
        "-n",
        args.metric_env,
        "python",
        "scripts/paper_compare/evaluate_outputs.py",
        "--sr-dir",
        str(sr_dir),
        "--hr-dir",
        str(hr_dir),
        "--lr-dir",
        str(lr_dir),
        "--dataset",
        args.dataset,
        "--method",
        args.method,
        "--out-dir",
        str(metric_dir),
        "--device",
        "cuda",
        "--lpips-device",
        "cuda",
        "--iqa-device",
        "cuda",
        "--fid-device",
        "cuda",
        "--fid-mode",
        args.fid_mode,
        "--fid-patch-size",
        str(args.fid_patch_size),
        "--fid-patches-per-image",
        str(args.fid_patches_per_image),
        "--fid-patch-pairing",
        args.fid_patch_pairing,
        "--iqa-metrics",
        args.iqa_metrics,
    ]
    metric_env = os.environ.copy()
    metric_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    rc = run(metric_cmd, repo_path("."), env=metric_env)
    if rc != 0:
        return rc
    metrics_json = metric_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if int(metrics.get("images") or 0) <= 0:
            print(f"[ERROR] metric evaluation found 0 images for {args.method}/{args.dataset}; keeping outputs for debugging.")
            return 3

    if not args.keep_images:
        shutil.rmtree(sr_dir, ignore_errors=True)
        shutil.rmtree(raw_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
