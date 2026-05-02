#!/usr/bin/env python
"""Run or dry-run one external SR baseline from the paper comparison manifest."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common import dataset_dirs, ensure_dir, list_images, load_yaml, normalize_outputs, repo_path, run_command, workspace_dir, write_json


RUNNABLE_STATUSES = {"runnable"}


def build_command(method_cfg: dict, input_dir: Path, raw_output_dir: Path) -> tuple[list[str], Path]:
    repo_root = repo_path(Path("external_baselines") / method_cfg["repo_dir"])
    entrypoint = method_cfg.get("entrypoint")
    if not entrypoint:
        raise SystemExit(
            f"Method has no entrypoint yet. Status={method_cfg.get('status')}; "
            "fill configs/paper_compare/baselines.yaml first."
        )

    python_bin = method_cfg.get("python", "python")
    cmd = [python_bin, entrypoint]
    cmd += [method_cfg["input_arg"], str(input_dir)]
    cmd += [method_cfg["output_arg"], str(raw_output_dir)]
    cmd += [str(x) for x in method_cfg.get("extra_args", [])]

    env_name = method_cfg.get("env_name")
    if env_name:
        cmd = ["conda", "run", "-n", env_name] + cmd
    return cmd, repo_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method")
    parser.add_argument("dataset", choices=["DIV2K", "RealSR", "DRealSR"])
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--execute", action="store_true", help="Actually run the baseline command.")
    parser.add_argument("--allow-needs-command", action="store_true",
                        help="Try methods marked public_needs_command after you fill their command fields.")
    parser.add_argument("--gpu", default=None, help="Optional CUDA_VISIBLE_DEVICES value for execution.")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    methods = manifest.get("methods", {})
    if args.method not in methods:
        raise SystemExit(f"Unknown method '{args.method}'. Known: {', '.join(sorted(methods))}")
    method_cfg = methods[args.method]
    status = method_cfg.get("status", "unknown")
    if status not in RUNNABLE_STATUSES and not args.allow_needs_command:
        print(f"[SKIP] {args.method}: status={status}")
        print(method_cfg.get("notes", "Use --allow-needs-command only after completing the manifest command."))
        return 0

    lr_dir, _ = dataset_dirs(manifest, args.dataset)
    lr_images = list_images(lr_dir)
    if not lr_images and args.execute:
        raise SystemExit(f"No LR images found in {lr_dir}")
    if not lr_images:
        print(f"[DRY-RUN][WARN] No LR images found yet in {lr_dir}")

    raw_dir = workspace_dir(manifest, "outputs", "_raw", args.method, args.dataset)
    final_dir = workspace_dir(manifest, "outputs", args.method, args.dataset)
    if args.execute:
        ensure_dir(raw_dir)
        ensure_dir(final_dir)

    cmd, cwd = build_command(method_cfg, lr_dir, raw_dir)
    if not cwd.exists():
        print(f"[SETUP] Missing repo dir: {cwd}")
        if method_cfg.get("repo_url"):
            print(f"Clone command: git clone {method_cfg['repo_url']} {cwd}")
        return 2 if args.execute else 0

    env_updates = {"CUDA_VISIBLE_DEVICES": args.gpu} if args.gpu else None
    rc = run_command(cmd, cwd=cwd, execute=args.execute, env_updates=env_updates)
    if rc != 0:
        return rc

    if args.execute:
        report = normalize_outputs(raw_dir, final_dir, lr_images)
        write_json(workspace_dir(manifest, "outputs", args.method, args.dataset, "normalize_report.json"), report)
        missing = [k for k, v in report.items() if v == "missing"]
        print(f"[DONE] normalized {len(report) - len(missing)}/{len(report)} outputs -> {final_dir}")
        if missing:
            print("[WARN] missing outputs:", ", ".join(missing[:20]))
        if not args.keep_raw:
            shutil.rmtree(raw_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
