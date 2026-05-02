#!/usr/bin/env python
"""Download public baseline checkpoints for paper comparison experiments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from common import repo_path


@dataclass(frozen=True)
class DirectAsset:
    method: str
    url: str
    dst: str


@dataclass(frozen=True)
class GitRepoAsset:
    method: str
    url: str
    dst: str


@dataclass(frozen=True)
class GDownAsset:
    method: str
    url: str
    dst: str
    folder: bool = False


@dataclass(frozen=True)
class SymlinkAsset:
    method: str
    src: str
    dst: str


DIRECT_ASSETS = [
    DirectAsset(
        "ResShift",
        "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
        "external_baselines/ResShift/weights/autoencoder_vq_f4.pth",
    ),
    DirectAsset(
        "ResShift",
        "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s4_v3.pth",
        "external_baselines/ResShift/weights/resshift_realsrx4_s4_v3.pth",
    ),
    DirectAsset(
        "StableSR",
        "https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_turbo.ckpt",
        "external_baselines/StableSR/stablesr_turbo.ckpt",
    ),
    DirectAsset(
        "StableSR",
        "https://huggingface.co/Iceclear/StableSR/resolve/main/vqgan_cfw_00011.ckpt",
        "external_baselines/StableSR/vqgan_cfw_00011.ckpt",
    ),
    DirectAsset(
        "StableSR",
        "https://huggingface.co/Iceclear/StableSR/resolve/main/stablesr_768v_000139.ckpt",
        "external_baselines/StableSR/stablesr_768v_000139.ckpt",
    ),
    DirectAsset(
        "DiffBIR",
        "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt",
        "external_baselines/DiffBIR/weights/DiffBIR_v2.1.pt",
    ),
    DirectAsset(
        "DiffBIR",
        "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
        "external_baselines/DiffBIR/weights/realesrgan_s4_swinir_100k.pth",
    ),
    DirectAsset(
        "DiffBIR",
        "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
        "external_baselines/DiffBIR/weights/sd2.1-base-zsnr-laionaes5.ckpt",
    ),
    DirectAsset(
        "DiffBIR",
        "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
        "external_baselines/DiffBIR/weights/BSRNet.pth",
    ),
    DirectAsset(
        "SeeSR",
        "https://huggingface.co/CSWRY/SeeSR/resolve/main/DAPE.pth",
        "external_baselines/SeeSR/preset/models/DAPE.pth",
    ),
    DirectAsset(
        "SeeSR",
        "https://huggingface.co/CSWRY/SeeSR/resolve/main/seesr/scaler.pt",
        "external_baselines/SeeSR/preset/models/seesr/scaler.pt",
    ),
    DirectAsset(
        "SeeSR",
        "https://huggingface.co/CSWRY/SeeSR/resolve/main/seesr/controlnet/config.json",
        "external_baselines/SeeSR/preset/models/seesr/controlnet/config.json",
    ),
    DirectAsset(
        "SeeSR",
        "https://huggingface.co/CSWRY/SeeSR/resolve/main/seesr/controlnet/diffusion_pytorch_model.safetensors",
        "external_baselines/SeeSR/preset/models/seesr/controlnet/diffusion_pytorch_model.safetensors",
    ),
    DirectAsset(
        "SeeSR",
        "https://huggingface.co/CSWRY/SeeSR/resolve/main/seesr/unet/config.json",
        "external_baselines/SeeSR/preset/models/seesr/unet/config.json",
    ),
    DirectAsset(
        "SeeSR",
        "https://huggingface.co/CSWRY/SeeSR/resolve/main/seesr/unet/diffusion_pytorch_model.safetensors",
        "external_baselines/SeeSR/preset/models/seesr/unet/diffusion_pytorch_model.safetensors",
    ),
    DirectAsset(
        "PiSA-SR",
        "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
        "external_baselines/PiSA-SR/src/ram_pretrain_model/ram_swin_large_14m.pth",
    ),
    DirectAsset(
        "OSEDiff",
        "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth",
        "external_baselines/OSEDiff/preset/models/ram_swin_large_14m.pth",
    ),
    DirectAsset(
        "PASD",
        "https://huggingface.co/yangtao9009/PASD/resolve/main/pasd/checkpoint-100000/scaler.pt",
        "external_baselines/PASD/runs/pasd/checkpoint-100000/scaler.pt",
    ),
    DirectAsset(
        "PASD",
        "https://huggingface.co/yangtao9009/PASD/resolve/main/pasd/checkpoint-100000/controlnet/config.json",
        "external_baselines/PASD/runs/pasd/checkpoint-100000/controlnet/config.json",
    ),
    DirectAsset(
        "PASD",
        "https://huggingface.co/yangtao9009/PASD/resolve/main/pasd/checkpoint-100000/controlnet/diffusion_pytorch_model.safetensors",
        "external_baselines/PASD/runs/pasd/checkpoint-100000/controlnet/diffusion_pytorch_model.safetensors",
    ),
    DirectAsset(
        "PASD",
        "https://huggingface.co/yangtao9009/PASD/resolve/main/pasd/checkpoint-100000/unet/config.json",
        "external_baselines/PASD/runs/pasd/checkpoint-100000/unet/config.json",
    ),
    DirectAsset(
        "PASD",
        "https://huggingface.co/yangtao9009/PASD/resolve/main/pasd/checkpoint-100000/unet/diffusion_pytorch_model.safetensors",
        "external_baselines/PASD/runs/pasd/checkpoint-100000/unet/diffusion_pytorch_model.safetensors",
    ),
]


GIT_REPO_ASSETS = [
    GitRepoAsset(
        "PiSA-SR",
        "https://huggingface.co/Manojb/stable-diffusion-2-1-base",
        "external_baselines/weights/hf/stable-diffusion-2-1-base-manojb",
    ),
    GitRepoAsset(
        "OSEDiff",
        "https://huggingface.co/Manojb/stable-diffusion-2-1-base",
        "external_baselines/weights/hf/stable-diffusion-2-1-base-manojb",
    ),
    GitRepoAsset(
        "SeeSR",
        "https://huggingface.co/Manojb/stable-diffusion-2-base",
        "external_baselines/weights/hf/stable-diffusion-2-base",
    ),
    GitRepoAsset(
        "PASD",
        "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
        "external_baselines/weights/hf/stable-diffusion-v1-5-official",
    ),
]


GDOWN_ASSETS = [
    GDownAsset(
        "PiSA-SR",
        "https://drive.google.com/drive/folders/1oLetijWNd59xwJE5oU-eXylQBifxWdss?usp=drive_link",
        "external_baselines/PiSA-SR/preset/models",
        folder=True,
    ),
]


SYMLINK_ASSETS = [
    SymlinkAsset(
        "PiSA-SR",
        "external_baselines/weights/hf/stable-diffusion-2-1-base-manojb",
        "external_baselines/PiSA-SR/preset/models/stable-diffusion-2-1-base",
    ),
    SymlinkAsset(
        "OSEDiff",
        "external_baselines/weights/hf/stable-diffusion-2-1-base-manojb",
        "external_baselines/OSEDiff/preset/models/stable-diffusion-2-1-base",
    ),
    SymlinkAsset(
        "SeeSR",
        "external_baselines/weights/hf/stable-diffusion-2-base",
        "external_baselines/SeeSR/preset/models/stable-diffusion-2-base",
    ),
]


def run(cmd: list[str], *, execute: bool, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    print("CMD:", " ".join(cmd))
    if not execute:
        return 0
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=False).returncode


def selected(methods: set[str] | None, method: str) -> bool:
    return methods is None or method in methods


def file_done(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def download_direct(asset: DirectAsset, execute: bool) -> int:
    dst = repo_path(asset.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if file_done(dst):
        print(f"[OK] {asset.method}: {dst} exists ({dst.stat().st_size} bytes)")
        return 0
    return run(
        [
            "wget",
            "-c",
            "--tries=5",
            "--timeout=30",
            "--progress=bar:force:noscroll",
            "-O",
            str(dst),
            asset.url,
        ],
        execute=execute,
    )


def clone_hf_repo(asset: GitRepoAsset, execute: bool) -> int:
    dst = repo_path(asset.dst)
    if (dst / ".git").exists():
        print(f"[OK] {asset.method}: HF repo exists at {dst}")
        rc = run(["git", "lfs", "pull"], execute=execute, cwd=dst)
        return rc
    dst.parent.mkdir(parents=True, exist_ok=True)
    return run(["git", "clone", asset.url, str(dst)], execute=execute)


def download_gdown(asset: GDownAsset, execute: bool) -> int:
    dst = repo_path(asset.dst)
    dst.mkdir(parents=True, exist_ok=True)
    gdown = shutil.which("gdown") or "/jumbo/yuwingtai/sy/miniconda3/bin/gdown"
    cmd = [gdown]
    if asset.folder:
        cmd.append("--folder")
    cmd.extend([asset.url, "-O", str(dst)])
    return run(cmd, execute=execute)


def make_symlink(asset: SymlinkAsset, execute: bool) -> int:
    src = repo_path(asset.src)
    dst = repo_path(asset.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = os.path.relpath(src, dst.parent)
    if dst.is_symlink():
        current = os.readlink(dst)
        if current == rel_src:
            print(f"[OK] {asset.method}: link exists at {dst} -> {current}")
            return 0
        print(f"RELINK: {dst} {current} -> {rel_src}")
        if execute:
            dst.unlink()
    elif dst.exists():
        print(f"[OK] {asset.method}: non-symlink target exists at {dst}")
        return 0
    print(f"LINK: {dst} -> {rel_src}")
    if execute:
        dst.symlink_to(rel_src, target_is_directory=True)
    return 0


def parse_methods(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=None, help="Comma-separated methods. Default: all known assets.")
    parser.add_argument("--execute", action="store_true", help="Actually download/clone/link assets.")
    parser.add_argument("--skip-gdown", action="store_true", help="Skip Google Drive assets.")
    parser.add_argument("--skip-hf-repos", action="store_true", help="Skip large HuggingFace diffusers repos.")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    rc = 0

    for asset in DIRECT_ASSETS:
        if selected(methods, asset.method):
            rc = download_direct(asset, args.execute) or rc

    if not args.skip_hf_repos:
        seen: set[str] = set()
        for asset in GIT_REPO_ASSETS:
            if not selected(methods, asset.method) or asset.dst in seen:
                continue
            seen.add(asset.dst)
            rc = clone_hf_repo(asset, args.execute) or rc

    if not args.skip_gdown:
        for asset in GDOWN_ASSETS:
            if selected(methods, asset.method):
                rc = download_gdown(asset, args.execute) or rc

    for asset in SYMLINK_ASSETS:
        if selected(methods, asset.method):
            rc = make_symlink(asset, args.execute) or rc

    if methods and "FluxSR" in methods:
        print("[SKIP] FluxSR: official repo currently has no released inference weights.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
