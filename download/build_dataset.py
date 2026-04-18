#!/usr/bin/env python
"""
Build a mixed SR dataset from multiple HR-only sources.

Compared with the old script, this version:
1. Supports recursive image discovery (needed for OST_train/category/*.png).
2. Lets you add extra datasets from CLI without editing the file every time.
3. Supports built-in presets: Mix16K / DF2K / Mix24K.

Examples:
    python download/build_dataset.py

    python download/build_dataset.py --preset df2k

    python download/build_dataset.py --preset mix24k

    python download/build_dataset.py \
        --target_name Mix18K \
        --source OST=./Data/OST/OST_train:2000 \
        --source TEXT=./Data/TextDataset:2000

    python download/build_dataset.py \
        --sources_json ./download/mix_sources.json \
        --target_name RealMix20K
"""

import argparse
import shutil
import json
import os
import random
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEFAULT_SOURCES = {
    "DIV2K": {"path": "./Data/DIV2K/DIV2K_train_HR", "count": 800},
    "Flickr2K": {"path": "./Data/Flickr2K/Flickr2K", "count": 2650},
    "LSDIR": {"path": "./Data/LSDIR/LSDIR_HR", "count": 8000},
    "FFHQ": {"path": "./Data/FFHQ", "count": 4550},
}

DF2K_SOURCES = {
    "DIV2K": {"path": "./Data/DIV2K/DIV2K_train_HR", "count": 800},
    "Flickr2K": {"path": "./Data/Flickr2K/Flickr2K", "count": 2650},
}

MIX24K_SOURCES = {
    "DIV2K": {
        "path": "./Data/DIV2K/DIV2K_train_HR",
        "count": 800,
        "sample_mode": "head",
    },
    "Flickr2K": {
        "path": "./Data/Flickr2K/Flickr2K",
        "count": 2650,
        "sample_mode": "head",
    },
    "LSDIR": {
        "path": "./Data/LSDIR/LSDIR_HR_subset",
        "fallback_path": "./Data/LSDIR/LSDIR_HR",
        "count": 11000,
        "fallback_count": 8000,
        "sample_mode": "head",
        "min_short_side": 1024,
        "fallback_min_short_side": 512,
    },
    "OST": {
        "path": "./Data/OST",
        "count": 5500,
        "sample_mode": "random",
        "min_short_side": 768,
    },
    "FFHQ": {
        "path": "./Data/FFHQ",
        "count": 2500,
        "sample_mode": "random",
    },
    "TEXT": {
        "path": "./Data/TextOCR",
        "count": 1550,
        "sample_mode": "random",
        "min_short_side": 512,
    },
}

PRESET_SOURCES = {
    "mix16k": DEFAULT_SOURCES,
    "df2k": DF2K_SOURCES,
    "mix24k": MIX24K_SOURCES,
}


def collect_image_files(source_dir, recursive=True):
    source_path = Path(source_dir)
    if recursive:
        candidates = source_path.rglob("*")
    else:
        candidates = source_path.iterdir()
    return sorted(
        p for p in candidates
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def load_sources(args):
    base_sources = PRESET_SOURCES[args.preset]
    sources = {
        name: {
            "path": info["path"],
            "fallback_path": info.get("fallback_path"),
            "count": info.get("count"),
            "fallback_count": info.get("fallback_count"),
            "recursive": info.get("recursive", True),
            "sample_mode": info.get("sample_mode", "random"),
            "min_short_side": info.get("min_short_side"),
            "fallback_min_short_side": info.get("fallback_min_short_side"),
        }
        for name, info in base_sources.items()
    }

    if args.sources_json:
        json_path = Path(args.sources_json)
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        for name, info in loaded.items():
            sources[name] = {
                "path": info["path"],
                "fallback_path": info.get("fallback_path"),
                "count": info.get("count"),
                "fallback_count": info.get("fallback_count"),
                "recursive": info.get("recursive", True),
                "sample_mode": info.get("sample_mode", "random"),
                "min_short_side": info.get("min_short_side"),
                "fallback_min_short_side": info.get("fallback_min_short_side"),
            }

    for spec in args.source:
        name, config = parse_source_spec(spec)
        sources[name] = config

    return sources


def parse_source_spec(spec):
    """
    Format:
        NAME=/path/to/images
        NAME=/path/to/images:COUNT
        NAME=/path/to/images:COUNT:recursive|flat
    """
    if "=" not in spec:
        raise ValueError(f"无效的 --source 格式: {spec}")

    name, rest = spec.split("=", 1)
    if not name.strip():
        raise ValueError(f"数据集名称不能为空: {spec}")

    parts = rest.rsplit(":", 2)
    path = parts[0].strip()
    count = None
    recursive = True

    if len(parts) >= 2 and parts[1].strip():
        count_part = parts[1].strip()
        if count_part.lower() not in {"recursive", "flat"}:
            count = int(count_part)
        else:
            recursive = count_part.lower() == "recursive"

    if len(parts) == 3:
        recursive_part = parts[2].strip().lower()
        if recursive_part not in {"recursive", "flat"}:
            raise ValueError(f"递归模式必须是 recursive 或 flat: {spec}")
        recursive = recursive_part == "recursive"

    return name.strip(), {
        "path": path,
        "fallback_path": None,
        "count": count,
        "fallback_count": None,
        "recursive": recursive,
        "sample_mode": "random",
        "min_short_side": None,
        "fallback_min_short_side": None,
    }


def order_files_for_sampling(all_files, rng, sample_mode="random"):
    if sample_mode not in {"random", "head"}:
        raise ValueError(f"不支持的 sample_mode: {sample_mode}，可选 random/head")
    if sample_mode == "head":
        return list(all_files)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    return shuffled


def process_dataset(
    source_dir,
    target_hr_dir,
    target_lr_dir,
    prefix,
    sample_count=None,
    sample_mode="random",
    recursive=True,
    scale=4,
    min_size=None,
    min_short_side=None,
    rng=None,
):
    all_files = collect_image_files(source_dir, recursive=recursive)

    if not all_files:
        print(f"❌ [警告] 在 {source_dir} 中没有找到图片，请检查路径是否正确。")
        return {"processed": 0, "available": 0, "skipped_small": 0, "failed": 0}

    rng = rng or random.Random(42)
    ordered_files = order_files_for_sampling(all_files, rng, sample_mode=sample_mode)
    target_count = sample_count if sample_count is not None and sample_count > 0 else None

    if target_count and target_count < len(all_files):
        mode_text = "头部抽样" if sample_mode == "head" else "随机抽样"
        print(f"[{prefix}] {mode_text}目标 {target_count} 张（可用 {len(all_files)}），按尺寸过滤后尽量补足...")
    else:
        print(f"[{prefix}] 提取全部 {len(ordered_files)} 张图片...")

    processed_count = 0
    skipped_small = 0
    failed = 0
    min_size = min_size or scale

    for img_path in tqdm(ordered_files, desc=f"处理 {prefix}"):
        try:
            hr_img = Image.open(img_path).convert("RGB")
            width, height = hr_img.size

            if min_short_side is not None and min(width, height) < min_short_side:
                skipped_small += 1
                continue
            if width < min_size or height < min_size or width < scale or height < scale:
                skipped_small += 1
                continue

            new_filename = f"{prefix}_{processed_count:05d}.png"
            hr_save_path = os.path.join(target_hr_dir, new_filename)
            lr_save_path = os.path.join(target_lr_dir, new_filename)

            hr_img.save(hr_save_path, format="PNG")

            lr_w = max(1, width // scale)
            lr_h = max(1, height // scale)
            lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
            lr_img.save(lr_save_path, format="PNG")

            processed_count += 1
            if target_count is not None and processed_count >= target_count:
                break
        except Exception as e:
            failed += 1
            print(f"跳过损坏图片 {img_path}: {e}")

    if target_count is not None and processed_count < target_count:
        print(
            f"[{prefix}] ⚠️ 仅得到 {processed_count}/{target_count} 张；"
            "可能是尺寸过滤(min_short_side/min_size)导致可用样本不足。"
        )

    return {
        "processed": processed_count,
        "requested": target_count,
        "available": len(all_files),
        "skipped_small": skipped_small,
        "failed": failed,
    }


def resolve_source_path(info):
    primary = info["path"]
    if os.path.exists(primary):
        return primary, False
    fallback = info.get("fallback_path")
    if fallback and os.path.exists(fallback):
        return fallback, True
    return None, False


def copy_from_mix16k(
    mix_hr_dir,
    mix_lr_dir,
    target_hr_dir,
    target_lr_dir,
    include_prefixes,
):
    """
    Fast path for DF2K: directly copy paired files from existing Mix16K HR/LR.
    """
    mix_hr = Path(mix_hr_dir)
    mix_lr = Path(mix_lr_dir)
    dst_hr = Path(target_hr_dir)
    dst_lr = Path(target_lr_dir)
    include_prefixes = tuple(include_prefixes)

    if not mix_hr.is_dir() or not mix_lr.is_dir():
        return None

    hr_files = sorted(
        f for f in os.listdir(mix_hr)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and f.startswith(include_prefixes)
    )
    lr_set = set(
        f for f in os.listdir(mix_lr)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    copied = 0
    missing_lr = 0
    for fname in tqdm(hr_files, desc="复制 DF2K (from Mix16K)"):
        src_hr = mix_hr / fname
        src_lr = mix_lr / fname
        if fname not in lr_set:
            missing_lr += 1
            continue
        shutil.copy2(src_hr, dst_hr / fname)
        shutil.copy2(src_lr, dst_lr / fname)
        copied += 1

    return {
        "processed": copied,
        "available": len(hr_files),
        "skipped_small": 0,
        "failed": 0,
        "missing_lr": missing_lr,
        "source_mode": "copy_from_mix16k",
    }


def build_manifest(target_name, target_hr, target_lr, scale, seed, sources, results):
    return {
        "target_name": target_name,
        "target_hr": str(target_hr),
        "target_lr": str(target_lr),
        "scale": scale,
        "seed": seed,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            name: {
                "path": info["path"],
                "fallback_path": info.get("fallback_path"),
                "count": info.get("count"),
                "fallback_count": info.get("fallback_count"),
                "recursive": info.get("recursive", True),
                "sample_mode": info.get("sample_mode", "random"),
                "min_short_side": info.get("min_short_side"),
                "fallback_min_short_side": info.get("fallback_min_short_side"),
                **results.get(name, {}),
            }
            for name, info in sources.items()
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="构建混合 SR 训练数据集")
    parser.add_argument("--preset", type=str, default="mix16k", choices=["mix16k", "df2k", "mix24k"],
                        help="内置数据源预设：mix16k / df2k / mix24k")
    parser.add_argument("--sources_json", type=str, default=None,
                        help="JSON 文件路径，格式为 {name: {path, fallback_path, count, fallback_count, recursive, sample_mode, min_short_side, fallback_min_short_side}}")
    parser.add_argument("--source", action="append", default=[],
                        help="追加数据源，格式 NAME=PATH[:COUNT[:recursive|flat]]")
    parser.add_argument("--target_name", type=str, default=None,
                        help="输出数据集名字，不传时按 preset 自动使用 Mix16K / DF2K / Mix24K")
    parser.add_argument("--target_hr", type=str, default=None, help="自定义 HR 输出目录")
    parser.add_argument("--target_lr", type=str, default=None, help="自定义 LR 输出目录")
    parser.add_argument("--scale", type=int, default=4, help="下采样倍数")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样 seed")
    parser.add_argument("--min_size", type=int, default=0,
                        help="跳过宽或高小于该值的图片；0 表示只要求不小于 scale")
    parser.add_argument("--mix_hr_dir", type=str, default="./Data/Mix16K_HR",
                        help="DF2K 复制模式下的 Mix16K HR 目录")
    parser.add_argument("--mix_lr_dir", type=str, default="./Data/Mix16K_LR_bicubic_X4",
                        help="DF2K 复制模式下的 Mix16K LR 目录")
    return parser.parse_args()


def main():
    args = parse_args()
    sources = load_sources(args)
    rng = random.Random(args.seed)
    if args.target_name is None:
        preset_default_name = {
            "df2k": "DF2K",
            "mix24k": "Mix24K",
            "mix16k": "Mix16K",
        }
        args.target_name = preset_default_name.get(args.preset, "Mix16K")

    target_hr = Path(args.target_hr or f"./Data/{args.target_name}_HR")
    target_lr = Path(args.target_lr or f"./Data/{args.target_name}_LR_bicubic_X{args.scale}")
    target_hr.mkdir(parents=True, exist_ok=True)
    target_lr.mkdir(parents=True, exist_ok=True)

    total_images = 0
    results = {}

    print("=" * 70)
    print(f"🚀 开始构建 {args.target_name}")
    print(f"HR 输出: {target_hr}")
    print(f"LR 输出: {target_lr}")
    print(f"Scale: X{args.scale}")
    print("=" * 70)

    if args.preset == "df2k":
        copied_summary = copy_from_mix16k(
            mix_hr_dir=args.mix_hr_dir,
            mix_lr_dir=args.mix_lr_dir,
            target_hr_dir=str(target_hr),
            target_lr_dir=str(target_lr),
            include_prefixes=("DIV2K_", "Flickr2K_"),
        )
        if copied_summary is not None:
            results["DF2K"] = copied_summary
            total_images = copied_summary["processed"]
            manifest = build_manifest(
                args.target_name, target_hr, target_lr, args.scale, args.seed, sources, results
            )
            manifest_path = target_hr.parent / f"{args.target_name}_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

            print("\n" + "=" * 70)
            print("✅ DF2K 已从 Mix16K 直接复制完成")
            print(f"总数: {total_images} (缺失配对 LR: {copied_summary.get('missing_lr', 0)})")
            print(f"HR 路径: {target_hr}")
            print(f"LR 路径: {target_lr}")
            print(f"Manifest: {manifest_path}")
            print("=" * 70)
            return

    for prefix, info in sources.items():
        source_path, used_fallback = resolve_source_path(info)
        if source_path is None:
            print(f"\n❌ [跳过] 找不到文件夹: {info['path']}")
            if info.get("fallback_path"):
                print(f"   fallback_path 也不存在: {info['fallback_path']}")
            print("请检查数据集是否已解压，或者路径是否多了一层目录。")
            results[prefix] = {"processed": 0, "available": 0, "skipped_small": 0, "failed": 0}
            continue
        if used_fallback:
            print(f"\n[{prefix}] 主路径不存在，自动使用 fallback_path: {source_path}")
        effective_count = info.get("count")
        effective_min_short_side = info.get("min_short_side")
        if used_fallback:
            if info.get("fallback_count") is not None:
                effective_count = info.get("fallback_count")
                print(f"[{prefix}] 使用 fallback_count: {effective_count}")
            if info.get("fallback_min_short_side") is not None:
                effective_min_short_side = info.get("fallback_min_short_side")
                print(f"[{prefix}] 使用 fallback_min_short_side: {effective_min_short_side}")

        summary = process_dataset(
            source_path,
            str(target_hr),
            str(target_lr),
            prefix,
            sample_count=effective_count,
            sample_mode=info.get("sample_mode", "random"),
            recursive=info.get("recursive", True),
            scale=args.scale,
            min_size=args.min_size if args.min_size > 0 else None,
            min_short_side=effective_min_short_side,
            rng=rng,
        )
        summary["used_path"] = source_path
        summary["used_fallback_path"] = used_fallback
        summary["effective_count"] = effective_count
        summary["effective_min_short_side"] = effective_min_short_side
        results[prefix] = summary
        total_images += summary["processed"]

    manifest = build_manifest(
        args.target_name, target_hr, target_lr, args.scale, args.seed, sources, results
    )
    manifest_path = target_hr.parent / f"{args.target_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ {args.target_name} 数据集构建完毕！共计成功处理图片: {total_images} 张")
    print(f"HR 路径: {target_hr}")
    print(f"LR 路径: {target_lr}")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
