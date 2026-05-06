#!/usr/bin/env python
"""Shared helpers for paper comparison scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required for paper_compare configs. Install with: pip install pyyaml"
        ) from exc
    with open(repo_path(path), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    path = repo_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(path: str | Path) -> list[Path]:
    root = repo_path(path)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def recursive_images(path: str | Path) -> list[Path]:
    root = repo_path(path)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def dataset_dirs(manifest: dict[str, Any], dataset: str) -> tuple[Path, Path | None]:
    datasets = manifest.get("datasets", {})
    if dataset not in datasets:
        known = ", ".join(sorted(datasets))
        raise SystemExit(f"Unknown dataset '{dataset}'. Known: {known}")
    cfg = datasets[dataset]
    lr_dir = repo_path(cfg["lr_dir"])
    hr_dir = repo_path(cfg["hr_dir"]) if cfg.get("hr_dir") else None
    return lr_dir, hr_dir


def workspace_dir(manifest: dict[str, Any], *parts: str) -> Path:
    return repo_path(Path(manifest.get("workspace", "experiments/paper_compare"), *parts))


def copy_image_as_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        if src.suffix.lower() == ".png" and dst.suffix.lower() == ".png":
            shutil.copy2(src, dst)
            return
        raise SystemExit("Pillow is required to normalize non-PNG baseline outputs.") from None

    Image.open(src).convert("RGB").save(dst)


def normalize_outputs(raw_dir: Path, final_dir: Path, lr_images: Iterable[Path]) -> dict[str, str]:
    """Copy method outputs to final_dir using LR basenames as canonical names."""
    raw_images = recursive_images(raw_dir)
    by_stem: dict[str, list[Path]] = {}
    for p in raw_images:
        by_stem.setdefault(p.stem, []).append(p)

    report: dict[str, str] = {}
    final_dir.mkdir(parents=True, exist_ok=True)
    for lr in lr_images:
        candidates = by_stem.get(lr.stem, [])
        if not candidates:
            candidates = [p for p in raw_images if lr.stem in p.stem]
        if not candidates:
            report[lr.name] = "missing"
            continue
        src = sorted(candidates, key=lambda p: (len(p.name), str(p)))[0]
        copy_image_as_png(src, final_dir / f"{lr.stem}.png")
        report[lr.name] = str(src)
    return report


def run_command(cmd: list[str], cwd: Path, execute: bool, env_updates: dict[str, str] | None = None) -> int:
    print("CWD:", cwd)
    print("CMD:", " ".join(cmd))
    if not execute:
        return 0
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    return subprocess.run(cmd, cwd=str(cwd), env=env, check=False).returncode


def write_json(path: str | Path, payload: Any) -> None:
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def copy_tree_flat(src_dir: Path, dst_dir: Path) -> int:
    count = 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in recursive_images(src_dir):
        copy_image_as_png(src, dst_dir / f"{src.stem}.png")
        count += 1
    return count
