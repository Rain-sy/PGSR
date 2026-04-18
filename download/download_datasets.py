#!/usr/bin/env python
"""
Dataset download / extraction helpers for SR training.

Supported tasks:
1. Download OST (Outdoor Scene Training) from the public Google Drive mirror.
2. Extract LSDIR HR images from local parquet files.
3. Download TextOCR (train/val images + annotations) from official public files.
4. Print build-dataset recipes for mixing OST and text-rich datasets.

Examples:
    python download/download_datasets.py ost --out_dir ./Data/OST
    python download/download_datasets.py textocr --out_dir ./Data/TextOCR
    python download/download_datasets.py lsdir_parquet --input_pattern "./Data/LSDIR/data/*.parquet"
    python download/download_datasets.py show_examples

Legacy compatibility:
    python download/download_datasets.py --out_dir ./Data/OST
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


OST_FOLDER_ID = "1LIb631GU3bOyQVTeuALesD8_eoApNniB"
OST_FOLDER_URL = f"https://drive.google.com/drive/folders/{OST_FOLDER_ID}"
TEXTOCR_URLS = {
    "train_val_images.zip": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
    "TextOCR_0.1_train.json": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
    "TextOCR_0.1_val.json": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
}


def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[info] {package_name} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package_name])


def ensure_gdown():
    ensure_package("gdown")


def download_ost_folder(out_dir: Path):
    import gdown

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Downloading OST folder to: {out_dir}")
    print(f"[info] URL: {OST_FOLDER_URL}")

    result = gdown.download_folder(
        url=OST_FOLDER_URL,
        output=str(out_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    if result is None or len(result) == 0:
        raise RuntimeError(
            "gdown returned no files. The folder may be rate-limited or the ID changed.\n"
            "Manual fallback: open the URL in a browser and download the files manually:\n"
            f"  {OST_FOLDER_URL}\n"
        )
    print(f"[ok] Downloaded {len(result)} files.")
    return result


def maybe_extract(out_dir: Path):
    for p in out_dir.iterdir():
        if p.suffix == ".zip":
            print(f"[info] unzip {p.name}")
            shutil.unpack_archive(str(p), str(out_dir))
        elif p.name.endswith(".tar.gz") or p.suffix == ".tgz":
            print(f"[info] untar {p.name}")
            shutil.unpack_archive(str(p), str(out_dir))


def download_file(url: str, out_path: Path, chunk_size=8 * 1024 * 1024):
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Downloading {out_path.name}")
    print(f"[info] URL: {url}")

    with urllib.request.urlopen(url) as response, open(out_path, "wb") as f:
        total = response.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100.0 / total
                print(f"\r[info] {out_path.name}: {downloaded / (1024**2):.1f}MB / {total / (1024**2):.1f}MB ({pct:.1f}%)", end="")
            else:
                print(f"\r[info] {out_path.name}: {downloaded / (1024**2):.1f}MB", end="")
    print()


def run_ost_download(out_dir, no_extract=False):
    out_dir = Path(out_dir).expanduser().resolve()
    ensure_gdown()
    download_ost_folder(out_dir)

    if not no_extract:
        maybe_extract(out_dir)

    print("\n[done] OST ready at:", out_dir)
    print("Expected structure:")
    print("  OST_train/{sky,water,grass,mountain,building,plant,animal}/*.png")
    print("  OST_test/*.png")


def run_textocr_download(out_dir, no_extract=False):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in TEXTOCR_URLS.items():
        target_file = out_dir / fname
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[skip] {fname} already exists: {target_file}")
            continue
        download_file(url, target_file)

    if not no_extract:
        zip_path = out_dir / "train_val_images.zip"
        if zip_path.exists():
            print(f"[info] unzip {zip_path.name}")
            shutil.unpack_archive(str(zip_path), str(out_dir))

    print("\n[done] TextOCR ready at:", out_dir)
    print("Expected structure (any one is fine):")
    print("  TextOCR/train_val_images/*.jpg  (+ json annotations)")
    print("  or TextOCR/train_images/*.jpg   (+ json annotations)")


def extract_lsdir_parquet(input_pattern, output_dir, target_count=8000, image_column=None):
    ensure_package("datasets")
    ensure_package("tqdm")

    from datasets import load_dataset
    from tqdm import tqdm

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📦 正在读取 Parquet 文件，这一步可能需要一些时间...")
    try:
        dataset = load_dataset("parquet", data_files=input_pattern, split="train")
    except Exception as e:
        raise RuntimeError(f"加载 parquet 失败，请检查路径: {e}") from e

    total_available = len(dataset)
    actual_count = min(target_count, total_available)
    chosen_column = image_column or ("image" if "image" in dataset.column_names else dataset.column_names[0])

    print(f"📊 数据集中总共有 {total_available} 张图片")
    print(f"🚀 开始提取前 {actual_count} 张图片到 {output_dir}")
    print(f"🧩 使用图片列: {chosen_column}")

    success_count = 0
    for i in tqdm(range(actual_count), desc="提取 LSDIR"):
        try:
            img = dataset[i][chosen_column]
            if img.mode != "RGB":
                img = img.convert("RGB")
            save_path = output_dir / f"LSDIR_{i:05d}.png"
            img.save(save_path, format="PNG")
            success_count += 1
        except Exception as e:
            print(f"⚠️ 提取第 {i} 张图片时出错: {e}")

    print("\n" + "=" * 60)
    print(f"✅ LSDIR 提取完成，成功保存 {success_count} 张图片到 {output_dir}")
    print("=" * 60)


def show_examples():
    print("=" * 70)
    print("OST 下载")
    print("python download/download_datasets.py ost --out_dir ./Data/OST")
    print()
    print("TextOCR 下载（train/val 图像 + 标注）")
    print("python download/download_datasets.py textocr --out_dir ./Data/TextOCR")
    print()
    print("LSDIR parquet 提取")
    print(
        'python download/download_datasets.py lsdir_parquet '
        '--input_pattern "./Data/LSDIR/data/*.parquet" '
        '--output_dir ./Data/LSDIR/LSDIR_HR --target_count 8000'
    )
    print()
    print("构建默认 Mix16K")
    print("python download/build_dataset.py")
    print()
    print("构建 DF2K（DIV2K + Flickr2K）")
    print("python download/build_dataset.py --preset df2k")
    print()
    print("构建 Mix24K（DIV2K+Flickr2K+LSDIR+OST+FFHQ+TEXT）")
    print("python download/build_dataset.py --preset mix24k")
    print()
    print("在 Mix16K 基础上追加 OST 和文本数据")
    print(
        "python download/build_dataset.py "
        '--target_name Mix20K '
        '--source OST=./Data/OST/OST_train:2000 '
        '--source TEXT=./Data/TextDataset:2000'
    )
    print()
    print("如果你的文本数据集是多层目录，也可以显式写 recursive")
    print(
        "python download/build_dataset.py "
        '--target_name Mix20K_Text '
        '--source TEXT=./Data/TextDataset:2000:recursive'
    )
    print("=" * 70)


def build_parser():
    parser = argparse.ArgumentParser(description="下载或准备 SR 训练数据集")
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["ost", "textocr", "lsdir_parquet", "show_examples"],
        help="要执行的任务。不传时，如果给了 --out_dir，则默认按 ost 处理。",
    )
    parser.add_argument("--out_dir", type=str, default=None,
                        help="OST/TextOCR 下载目录，例如 ./Data/OST 或 ./Data/TextOCR")
    parser.add_argument("--no_extract", action="store_true",
                        help="下载后不自动解压（OST/TextOCR 均适用）")
    parser.add_argument("--input_pattern", type=str, default="./Data/LSDIR/data/*.parquet",
                        help='LSDIR parquet 文件 glob，例如 "./Data/LSDIR/data/*.parquet"')
    parser.add_argument("--output_dir", type=str, default="./Data/LSDIR/LSDIR_HR",
                        help="LSDIR 提取输出目录")
    parser.add_argument("--target_count", type=int, default=8000,
                        help="LSDIR 要提取的图片数量")
    parser.add_argument("--image_column", type=str, default=None,
                        help="parquet 中的图片列名，不传则自动推断")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "show_examples":
        show_examples()
        return

    if args.dataset == "lsdir_parquet":
        extract_lsdir_parquet(
            input_pattern=args.input_pattern,
            output_dir=args.output_dir,
            target_count=args.target_count,
            image_column=args.image_column,
        )
        return

    if args.dataset == "textocr":
        if not args.out_dir:
            parser.error("TextOCR 下载需要传 --out_dir")
        run_textocr_download(args.out_dir, no_extract=args.no_extract)
        return

    if args.dataset == "ost" or (args.dataset is None and args.out_dir):
        if not args.out_dir:
            parser.error("OST 下载需要传 --out_dir")
        run_ost_download(args.out_dir, no_extract=args.no_extract)
        return

    parser.error("请指定任务，例如: ost / textocr / lsdir_parquet / show_examples")


if __name__ == "__main__":
    main()
