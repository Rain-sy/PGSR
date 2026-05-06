#!/usr/bin/env python
"""Report whether a local quant metric result is complete and valid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import repo_path


REQUIRED_IQA = ["dists", "niqe", "musiq", "maniqa", "clipiqa"]
EXPECTED_IMAGES = {"DIV2K": 100, "RealSR": 100, "DRealSR": 93}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method")
    parser.add_argument("dataset")
    parser.add_argument("--metrics-root", default="experiments/paper_compare_quant/metrics/baselines")
    args = parser.parse_args()

    path = repo_path(args.metrics_root) / args.method / args.dataset / "metrics.json"
    if not path.exists():
        print("missing")
        return 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    expected_images = EXPECTED_IMAGES.get(args.dataset)
    if expected_images is not None and data.get("images") != expected_images:
        print(f"incomplete_images {data.get('images')}/{expected_images}")
        return 4
    fid = data.get("fid", {})
    if fid.get("primary_mode") != "patch" or fid.get("patch_pairing") != "independent" or fid.get("sr") is None:
        print(f"invalid_fid pairing={fid.get('patch_pairing')}")
        return 2
    iqa = data.get("iqa", {})
    missing_iqa = [name for name in REQUIRED_IQA if iqa.get(name, {}).get("sr") is None]
    if missing_iqa:
        print("missing_iqa " + ",".join(missing_iqa))
        return 3
    print("valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
