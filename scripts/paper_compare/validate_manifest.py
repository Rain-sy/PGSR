#!/usr/bin/env python
"""Validate paper comparison inputs, manifest fields, and normalized outputs."""

from __future__ import annotations

import argparse
import sys

from PIL import Image

from common import dataset_dirs, list_images, load_yaml, workspace_dir


def check_dataset(manifest: dict, dataset: str, methods: list[str]) -> int:
    problems = 0
    lr_dir, hr_dir = dataset_dirs(manifest, dataset)
    lr_images = list_images(lr_dir)
    if not lr_images:
        print(f"[ERR] {dataset}: no LR images in {lr_dir}")
        return 1
    print(f"[OK] {dataset}: {len(lr_images)} LR images")
    lr_stems = {p.stem for p in lr_images}
    if hr_dir and hr_dir.exists():
        hr_stems = {p.stem for p in list_images(hr_dir)}
        missing_hr = sorted(lr_stems - hr_stems)
        if missing_hr:
            problems += 1
            print(f"[WARN] {dataset}: {len(missing_hr)} LR images missing HR")

    for method in methods:
        out_dir = workspace_dir(manifest, "outputs", method, dataset)
        outputs = list_images(out_dir)
        out_stems = {p.stem for p in outputs}
        missing = sorted(lr_stems - out_stems)
        if missing:
            problems += 1
            print(f"[WARN] {dataset}/{method}: missing {len(missing)} outputs")
            continue
        for lr_path in lr_images[:3]:
            out_path = out_dir / f"{lr_path.stem}.png"
            if not out_path.exists():
                continue
            lr = Image.open(lr_path)
            sr = Image.open(out_path)
            if sr.width != lr.width * 4 or sr.height != lr.height * 4:
                print(f"[WARN] {dataset}/{method}/{out_path.name}: expected 4x LR, got {sr.size} vs LR {lr.size}")
                problems += 1
                break
        print(f"[OK] {dataset}/{method}: {len(outputs)} outputs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--datasets", default="DIV2K,RealSR,DRealSR")
    parser.add_argument("--methods", default="ResShift,StableSR,DiffBIR,SeeSR,PiSA-SR,FluxSR,PGSR")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for method in methods:
        if method not in manifest.get("methods", {}):
            raise SystemExit(f"Unknown method in --methods: {method}")

    problems = 0
    for dataset in datasets:
        problems += check_dataset(manifest, dataset, methods)
    print(f"[DONE] validation finished with {problems} warning/error groups")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
