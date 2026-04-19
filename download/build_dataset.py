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
import hashlib
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
BUILDER_VERSION = "build_dataset_repro_v2"

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

PRESET_TARGET_TOTAL = {
    "mix24k": 24000,
}

# Deterministic top-up order for mix24k when base sources cannot reach 24k.
# This is designed to reproduce the current local Mix24K composition:
#   base + FFHQtopup(3000) + OSTmore(remaining).
MIX24K_TOPUP_PLAN = [
    {
        "source": "FFHQ",
        "prefix": "FFHQtopup",
        "count": 3000,
        "sample_mode": "random",
        "min_short_side": None,
    },
    {
        "source": "OST",
        "prefix": "OSTmore",
        "count": None,  # Fill whatever remains to target_total.
        "sample_mode": "random",
        "min_short_side": None,
        "source_path_override": "./Data/OST/OutdoorSceneTrain_v2",
    },
]


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


def make_rng(base_seed, namespace):
    text = f"{base_seed}:{namespace}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return random.Random(seed_int)


def hash_string_list(values):
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def list_target_images(dir_path):
    path = Path(dir_path)
    if not path.exists():
        return []
    return sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def ensure_clean_target_dirs(target_hr, target_lr, force_rebuild=False):
    hr_existing = list_target_images(target_hr)
    lr_existing = list_target_images(target_lr)
    if not hr_existing and not lr_existing:
        return {"removed_hr": 0, "removed_lr": 0}

    if not force_rebuild:
        raise RuntimeError(
            "目标输出目录中已存在图片文件。为保证可复现，请先清空目录，"
            "或使用 --force_rebuild 让脚本自动清空后重建。"
        )

    for p in hr_existing:
        p.unlink()
    for p in lr_existing:
        p.unlink()
    return {"removed_hr": len(hr_existing), "removed_lr": len(lr_existing)}


def compute_output_signature(target_hr, target_lr):
    hr_files = list_target_images(target_hr)
    lr_files = list_target_images(target_lr)
    hr_names = [p.name for p in hr_files]
    lr_names = [p.name for p in lr_files]
    hr_set = set(hr_names)
    lr_set = set(lr_names)
    common = sorted(hr_set & lr_set)

    h = hashlib.sha256()
    for name in common:
        hr_size = (Path(target_hr) / name).stat().st_size
        lr_size = (Path(target_lr) / name).stat().st_size
        h.update(f"{name}|{hr_size}|{lr_size}\n".encode("utf-8"))

    return {
        "hr_count": len(hr_names),
        "lr_count": len(lr_names),
        "paired_count": len(common),
        "hr_only_count": len(hr_set - lr_set),
        "lr_only_count": len(lr_set - hr_set),
        "paired_size_hash_sha256": h.hexdigest(),
    }


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
    source_files=None,
    exclude_files=None,
    collect_selected_files=False,
    start_index=0,
    strict_scale_alignment=True,
):
    if source_files is None:
        all_files = collect_image_files(source_dir, recursive=recursive)
    else:
        all_files = list(source_files)

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
    skipped_excluded = 0
    failed = 0
    cropped_to_scale = 0
    min_size = min_size or scale
    selected_files = []
    exclude_set = {
        str(Path(p).resolve()) if not isinstance(p, Path) else str(p.resolve())
        for p in (exclude_files or [])
    }

    for raw_path in tqdm(ordered_files, desc=f"处理 {prefix}"):
        img_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        img_key = str(img_path.resolve())
        if img_key in exclude_set:
            skipped_excluded += 1
            continue
        try:
            with Image.open(img_path) as _hr:
                hr_img = _hr.convert("RGB")
            width, height = hr_img.size

            if min_short_side is not None and min(width, height) < min_short_side:
                skipped_small += 1
                continue
            if width < min_size or height < min_size or width < scale or height < scale:
                skipped_small += 1
                continue

            if strict_scale_alignment:
                aligned_w = (width // scale) * scale
                aligned_h = (height // scale) * scale
                if aligned_w < scale or aligned_h < scale:
                    skipped_small += 1
                    continue
                if aligned_w != width or aligned_h != height:
                    # Deterministic center-crop so that HR exactly equals LR * scale.
                    left = (width - aligned_w) // 2
                    top = (height - aligned_h) // 2
                    hr_img = hr_img.crop((left, top, left + aligned_w, top + aligned_h))
                    width, height = hr_img.size
                    cropped_to_scale += 1

            new_filename = f"{prefix}_{start_index + processed_count:05d}.png"
            hr_save_path = os.path.join(target_hr_dir, new_filename)
            lr_save_path = os.path.join(target_lr_dir, new_filename)

            hr_img.save(hr_save_path, format="PNG")

            lr_w = max(1, width // scale)
            lr_h = max(1, height // scale)
            lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
            lr_img.save(lr_save_path, format="PNG")

            selected_files.append(img_key)
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

    summary = {
        "processed": processed_count,
        "requested": target_count,
        "available": len(all_files),
        "skipped_small": skipped_small,
        "skipped_excluded": skipped_excluded,
        "failed": failed,
        "cropped_to_scale": cropped_to_scale,
    }
    if collect_selected_files:
        summary["_selected_files"] = selected_files
    return summary


def resolve_source_path(info):
    primary = info["path"]
    if os.path.exists(primary):
        return primary, False
    fallback = info.get("fallback_path")
    if fallback and os.path.exists(fallback):
        return fallback, True
    return None, False


def run_reproducible_topup(
    args,
    sources,
    target_hr,
    target_lr,
    total_images,
    used_files_by_source,
):
    target_total = args.ensure_target_total
    if target_total <= 0:
        return total_images, {}
    if total_images >= target_total:
        return total_images, {}

    print("\n" + "-" * 70)
    print(f"🔁 启动可复现补齐：当前 {total_images}，目标 {target_total}")
    print("-" * 70)

    topup_results = {}
    for step in MIX24K_TOPUP_PLAN:
        remaining = target_total - total_images
        if remaining <= 0:
            break

        source_name = step["source"]
        info = sources.get(source_name)
        if info is None:
            print(f"[TopUp:{source_name}] ❌ 源不存在于 sources，跳过。")
            continue

        source_path, used_fallback = resolve_source_path(info)
        override = step.get("source_path_override")
        if override:
            if os.path.exists(override):
                source_path = override
                used_fallback = False
            else:
                print(f"[TopUp:{source_name}] ❌ source_path_override 不存在: {override}")
                continue
        if source_path is None:
            print(f"[TopUp:{source_name}] ❌ 找不到路径，跳过。")
            continue

        step_count = step.get("count")
        if step_count is None or step_count <= 0:
            effective_count = remaining
        else:
            effective_count = min(int(step_count), remaining)
        if effective_count <= 0:
            continue

        min_short_side = step.get("min_short_side")
        sample_mode = step.get("sample_mode", info.get("sample_mode", "random"))
        recursive = info.get("recursive", True)
        # Reuse base-source namespace so random ordering is stable and reproducible.
        rng = make_rng(args.seed, f"base::{source_name}")

        used = used_files_by_source.setdefault(source_name, set())
        print(
            f"[TopUp:{source_name}] prefix={step['prefix']} 目标补 {effective_count} 张 "
            f"(remaining={remaining}, 排除已用={len(used)})"
        )

        source_files = collect_image_files(source_path, recursive=recursive)
        summary = process_dataset(
            source_path,
            str(target_hr),
            str(target_lr),
            step["prefix"],
            sample_count=effective_count,
            sample_mode=sample_mode,
            recursive=recursive,
            scale=args.scale,
            min_size=args.min_size if args.min_size > 0 else None,
            min_short_side=min_short_side,
            rng=rng,
            source_files=source_files,
            exclude_files=used,
            collect_selected_files=True,
            strict_scale_alignment=args.strict_scale_alignment,
        )
        selected_list = summary.pop("_selected_files", [])
        selected = set(selected_list)
        used.update(selected)

        summary["selected_count"] = len(selected_list)
        summary["selected_hash"] = hash_string_list(selected_list)
        summary["source"] = source_name
        summary["used_path"] = source_path
        summary["used_fallback_path"] = used_fallback
        summary["effective_count"] = effective_count
        summary["effective_min_short_side"] = min_short_side
        summary["remaining_before"] = remaining
        summary["excluded_used_paths"] = len(used) - len(selected)
        topup_results[step["prefix"]] = summary

        total_images += summary["processed"]
        if summary["processed"] < effective_count:
            print(
                f"[TopUp:{source_name}] ⚠️ 仅补到 {summary['processed']}/{effective_count}，"
                "可能是可用样本不足或过滤过严。"
            )

    if total_images < target_total:
        print(
            f"\n⚠️ 补齐后仍未达到目标总数: {total_images}/{target_total}。"
            "请检查源数据是否不足，或放宽过滤条件。"
        )
    else:
        print(f"\n✅ 可复现补齐完成：{total_images}/{target_total}")

    return total_images, topup_results


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


def build_manifest(
    target_name,
    target_hr,
    target_lr,
    scale,
    seed,
    sources,
    results,
    target_total=None,
    topup_results=None,
    include_timestamp=False,
    reproducibility=None,
    output_signature=None,
):
    manifest = {
        "target_name": target_name,
        "target_hr": str(target_hr),
        "target_lr": str(target_lr),
        "scale": scale,
        "seed": seed,
        "builder_version": BUILDER_VERSION,
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
    if include_timestamp:
        manifest["built_at"] = datetime.now().isoformat(timespec="seconds")
    if target_total is not None and target_total > 0:
        manifest["target_total"] = int(target_total)
    if topup_results:
        manifest["topup_results"] = topup_results
        manifest["topup_plan"] = MIX24K_TOPUP_PLAN
    if reproducibility is not None:
        manifest["reproducibility"] = reproducibility
    if output_signature is not None:
        manifest["output_signature"] = output_signature
    return manifest


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
    parser.add_argument("--strict_scale_alignment", dest="strict_scale_alignment", action="store_true", default=True,
                        help="默认开启：HR 会先中心裁到可被 scale 整除，再生成 LR，保证严格 x4 配对。")
    parser.add_argument("--no_strict_scale_alignment", dest="strict_scale_alignment", action="store_false",
                        help="关闭严格对齐（不推荐，会产生近似 x4 但非严格配对样本）。")
    parser.add_argument("--force_rebuild", action="store_true", default=False,
                        help="当目标目录非空时，先删除其中已有图片再重建（保证可复现）。")
    parser.add_argument("--manifest_timestamp", action="store_true", default=False,
                        help="在 manifest 中写入 built_at 时间戳。默认关闭以保持 manifest 可复现。")
    parser.add_argument("--ensure_target_total", type=int, default=0,
                        help="构建后自动补齐到该总数（0 表示不补齐）。mix24k 默认自动补到 24000。")
    parser.add_argument("--mix_hr_dir", type=str, default="./Data/Mix16K_HR",
                        help="DF2K 复制模式下的 Mix16K HR 目录")
    parser.add_argument("--mix_lr_dir", type=str, default="./Data/Mix16K_LR_bicubic_X4",
                        help="DF2K 复制模式下的 Mix16K LR 目录")
    return parser.parse_args()


def main():
    args = parse_args()
    sources = load_sources(args)
    if args.ensure_target_total <= 0 and args.preset in PRESET_TARGET_TOTAL:
        args.ensure_target_total = PRESET_TARGET_TOTAL[args.preset]
        print(
            f"[{args.preset}] 自动启用可复现补齐: "
            f"--ensure_target_total={args.ensure_target_total}"
        )
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
    cleanup_info = ensure_clean_target_dirs(target_hr, target_lr, force_rebuild=args.force_rebuild)

    total_images = 0
    results = {}
    topup_results = {}
    used_files_by_source = {}

    print("=" * 70)
    print(f"🚀 开始构建 {args.target_name}")
    print(f"HR 输出: {target_hr}")
    print(f"LR 输出: {target_lr}")
    print(f"Scale: X{args.scale}")
    print(f"Strict Scale Alignment: {args.strict_scale_alignment}")
    print(f"Force Rebuild: {args.force_rebuild}")
    print(f"Manifest Timestamp: {args.manifest_timestamp}")
    if cleanup_info["removed_hr"] > 0 or cleanup_info["removed_lr"] > 0:
        print(
            f"已清空旧图片: HR={cleanup_info['removed_hr']} / "
            f"LR={cleanup_info['removed_lr']}"
        )
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
                args.target_name,
                target_hr,
                target_lr,
                args.scale,
                args.seed,
                sources,
                results,
                target_total=args.ensure_target_total,
                topup_results=topup_results,
                include_timestamp=args.manifest_timestamp,
                reproducibility={
                    "strict_scale_alignment": args.strict_scale_alignment,
                    "force_rebuild": args.force_rebuild,
                    "manifest_timestamp": args.manifest_timestamp,
                },
                output_signature=compute_output_signature(target_hr, target_lr),
            )
            manifest_path = target_hr.parent / f"{args.target_name}_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

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
            rng=make_rng(args.seed, f"base::{prefix}"),
            collect_selected_files=True,
            strict_scale_alignment=args.strict_scale_alignment,
        )
        selected_list = summary.pop("_selected_files", [])
        selected = set(selected_list)
        used_files_by_source[prefix] = selected
        summary["selected_count"] = len(selected_list)
        summary["selected_hash"] = hash_string_list(selected_list)
        summary["used_path"] = source_path
        summary["used_fallback_path"] = used_fallback
        summary["effective_count"] = effective_count
        summary["effective_min_short_side"] = effective_min_short_side
        results[prefix] = summary
        total_images += summary["processed"]

    if args.ensure_target_total > 0:
        if args.preset == "mix24k":
            total_images, topup_results = run_reproducible_topup(
                args=args,
                sources=sources,
                target_hr=target_hr,
                target_lr=target_lr,
                total_images=total_images,
                used_files_by_source=used_files_by_source,
            )
        else:
            print("⚠️ --ensure_target_total 当前仅内置支持 preset=mix24k，已跳过补齐。")

    manifest = build_manifest(
        args.target_name,
        target_hr,
        target_lr,
        args.scale,
        args.seed,
        sources,
        results,
        target_total=args.ensure_target_total,
        topup_results=topup_results,
        include_timestamp=args.manifest_timestamp,
        reproducibility={
            "strict_scale_alignment": args.strict_scale_alignment,
            "force_rebuild": args.force_rebuild,
            "manifest_timestamp": args.manifest_timestamp,
        },
        output_signature=compute_output_signature(target_hr, target_lr),
    )
    manifest_path = target_hr.parent / f"{args.target_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("\n" + "=" * 70)
    print(f"✅ {args.target_name} 数据集构建完毕！共计成功处理图片: {total_images} 张")
    print(f"HR 路径: {target_hr}")
    print(f"LR 路径: {target_lr}")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
