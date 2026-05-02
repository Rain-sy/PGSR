#!/usr/bin/env python
"""Run all configured external baselines for one or more datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys

from common import load_yaml


DEFAULT_METHODS = ["ResShift", "StableSR", "DiffBIR", "SeeSR", "PiSA-SR", "FluxSR"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--datasets", default="DIV2K,RealSR,DRealSR")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--allow-needs-command", action="store_true")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    methods_cfg = manifest.get("methods", {})
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rc = 0
    for dataset in datasets:
        for method in methods:
            cfg = methods_cfg.get(method, {})
            status = cfg.get("status")
            if status == "pending_public_code":
                print(f"[SKIP] {method}/{dataset}: pending_public_code")
                continue
            cmd = [
                "python",
                "scripts/paper_compare/run_baseline.py",
                method,
                dataset,
                "--manifest",
                args.manifest,
            ]
            if args.execute:
                cmd.append("--execute")
            if args.gpu:
                cmd += ["--gpu", args.gpu]
            if args.allow_needs_command:
                cmd.append("--allow-needs-command")
            print("\n==>", " ".join(cmd))
            cur = subprocess.run(cmd, check=False).returncode
            if cur != 0:
                rc = cur
    return rc


if __name__ == "__main__":
    sys.exit(main())
