"""Export actual LR/Ours pairs to lossless web images; no generated imagery.

Run: python tools/build_demo_assets.py --source NeurIPS_2026_Dual_SR/8k
Requires Pillow. The raw results remain outside the published repository.
"""
import argparse
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"0492": (3128, 2752, 3640, 3264), "0467": (3112, 4624, 3624, 5136)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for sample, box in SAMPLES.items():
        lr = Image.open(args.source / f"{sample}_LR.png").convert("RGB")
        ours = Image.open(args.source / f"{sample}_ours.png").convert("RGB")
        assert ours.size == (lr.width*4, lr.height*4)
        assert all(n % 4 == 0 for n in box)
        assert 0 <= box[0] < box[2] <= ours.width and 0 <= box[1] < box[3] <= ours.height
        lr_box = tuple(n//4 for n in box)
        full_size = (1600, round(1600*ours.height/ours.width))
        for kind, image, roi in (("lr", lr, lr_box), ("ours", ours, box)):
            image.resize(full_size, Image.Resampling.BICUBIC).save(output / f"{sample}-full-{kind}.webp", lossless=True)
            image.crop(roi).resize((768, 768), Image.Resampling.BICUBIC).save(output / f"{sample}-detail-{kind}.webp", lossless=True)
        manifest[sample] = {"lr_size": lr.size, "output_size": ours.size,
                            "output_crop_xyxy": box, "lr_crop_xyxy": lr_box,
                            "display_full_size": full_size, "display_detail_size": [768, 768],
                            "display_resampling": "bicubic for both methods", "encoding": "lossless WebP"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Exported 2 paired examples (full image + aligned detail).")


if __name__ == "__main__":
    main()
