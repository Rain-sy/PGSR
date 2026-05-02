#!/usr/bin/env python
"""Clone/check external baseline repositories from the comparison manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import load_yaml, repo_path


def run(cmd: list[str], execute: bool, cwd: Path | None = None) -> int:
    print("CMD:", " ".join(cmd))
    if not execute:
        return 0
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/paper_compare/baselines.yaml")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated method names. Default: all methods with repo_url.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkout", action="store_true",
                        help="After cloning/existing repo check, checkout the manifest commit/ref when set.")
    args = parser.parse_args()

    manifest = load_yaml(args.manifest)
    external_root = repo_path(manifest.get("external_root", "external_baselines"))
    methods = manifest.get("methods", {})
    selected = [m.strip() for m in args.methods.split(",")] if args.methods else sorted(methods)

    rc = 0
    for method in selected:
        cfg = methods.get(method)
        if not cfg:
            print(f"[SKIP] unknown method: {method}")
            rc = 1
            continue
        repo_url = cfg.get("repo_url")
        repo_dir = cfg.get("repo_dir")
        if not repo_url or not repo_dir:
            print(f"[SKIP] {method}: no public repo_url in manifest (status={cfg.get('status')})")
            continue

        target = external_root / repo_dir
        if target.exists():
            print(f"[OK] {method}: repo exists at {target}")
        else:
            rc = run(["git", "clone", repo_url, str(target)], args.execute) or rc
            if rc != 0:
                continue

        ref = cfg.get("commit")
        if args.checkout and ref and ref not in ("main", "master", "local"):
            rc = run(["git", "checkout", str(ref)], args.execute, cwd=target) or rc

        env_name = cfg.get("env_name")
        print(f"[INFO] {method}: env={env_name}, status={cfg.get('status')}")
        if cfg.get("notes"):
            print(f"[NOTE] {cfg['notes']}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
