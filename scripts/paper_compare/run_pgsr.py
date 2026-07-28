#!/usr/bin/env python
"""Run PGSR evaluation and collect outputs into the paper comparison workspace."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import (
    REPO_ROOT,
    copy_tree_flat,
    dataset_dirs,
    ensure_dir,
    load_yaml,
    repo_path,
    workspace_dir,
)


def _arg(cmd: list[str], name: str, value) -> None:
    if value is not None:
        cmd += [name, str(value)]


def find_predictions(eval_base: Path, dataset: str, exp_name: str) -> Path:
    candidates = sorted((eval_base / dataset).glob(f"*/{exp_name}/predictions"))
    if not candidates:
        raise SystemExit(f"Could not find PGSR predictions under {eval_base}/{dataset}/*/{exp_name}/predictions")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["DIV2K", "RealSR", "DRealSR"])
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--checkpoint-config", default="configs/paper_compare/pgsr_checkpoints.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    ckpt_cfg = load_yaml(args.checkpoint_config)
    lr_dir, hr_dir = dataset_dirs(manifest, args.dataset)
    if hr_dir is None or not hr_dir.exists():
        raise SystemExit(f"PGSR evaluation requires HR images for {args.dataset}: {hr_dir}")

    defaults = ckpt_cfg.get("default_args", {})
    checkpoint = args.checkpoint or ckpt_cfg.get("checkpoints", {}).get(args.dataset, {}).get("path")
    if not checkpoint:
        raise SystemExit(f"No PGSR checkpoint configured for {args.dataset}")
    checkpoint = repo_path(checkpoint)

    exp_name = args.exp_name or f"PGSR_{args.dataset}"
    eval_base = workspace_dir(manifest, "metrics", "pgsr_eval")
    final_output_dir = workspace_dir(manifest, "outputs", "PGSR", args.dataset)
    final_metrics_dir = workspace_dir(manifest, "metrics", "PGSR", args.dataset)
    ensure_dir(eval_base)
    ensure_dir(final_output_dir)
    ensure_dir(final_metrics_dir)

    cmd = [
        "python",
        "evaluate_pgsr.py",
        "--checkpoint",
        str(checkpoint),
        "--hr_dir",
        str(hr_dir),
        "--lr_dir",
        str(lr_dir),
        "--dataset",
        args.dataset,
        "--output_base",
        str(eval_base),
        "--exp_name",
        exp_name,
        "--save_images",
        "--save_metrics_json",
    ]
    for key, flag in [
        ("num_steps", "--num_steps"),
        ("guidance", "--guidance"),
        ("strength", "--strength"),
        ("tile_size", "--tile_size"),
        ("min_tile_size", "--min_tile_size"),
        ("overlap", "--overlap"),
        ("fid_mode", "--fid_mode"),
        ("fid_patch_size", "--fid_patch_size"),
        ("fid_patches_per_image", "--fid_patches_per_image"),
        ("fid_patch_pairing", "--fid_patch_pairing"),
    ]:
        _arg(cmd, flag, defaults.get(key))

    print("CMD:", " ".join(cmd))
    if not args.execute:
        return 0

    env = None
    if args.gpu:
        import os

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False).returncode
    if rc != 0:
        return rc

    pred_dir = find_predictions(eval_base, args.dataset, exp_name)
    if not args.no_copy:
        copied = copy_tree_flat(pred_dir, final_output_dir)
        print(f"[DONE] copied {copied} PGSR predictions -> {final_output_dir}")

    run_dir = pred_dir.parent
    for name in ("results.txt", "metrics.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, final_metrics_dir / name)
    print(f"[DONE] PGSR metrics -> {final_metrics_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
