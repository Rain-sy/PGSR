"""Make README-native animated comparisons from actual, aligned LR/Ours crops.

python tools/build_readme_demos.py --source NeurIPS_2026_Dual_SR/8k
Requires Pillow. GIF is a display preview (256 colors), not an evaluation asset.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"0492": (3128, 2752, 3640, 3264), "0467": (3112, 4624, 3624, 5136)}
SIZE, HEADER = 560, 40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    output = ROOT / "assets"
    output.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    manifest = {}
    for sample, box in SAMPLES.items():
        lr = Image.open(args.source / f"{sample}_LR.png").convert("RGB")
        ours = Image.open(args.source / f"{sample}_ours.png").convert("RGB")
        assert ours.size == (lr.width*4, lr.height*4)
        assert all(x % 4 == 0 for x in box)
        lr_box = tuple(x//4 for x in box)
        left = lr.crop(lr_box).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        right = ours.crop(box).resize((SIZE, SIZE), Image.Resampling.BICUBIC)
        # Shared palette avoids color changes as the divider moves.
        palette_source = Image.new("RGB", (SIZE*2, SIZE+HEADER), "white")
        palette_source.paste(left, (0, HEADER))
        palette_source.paste(right, (SIZE, HEADER))
        ImageDraw.Draw(palette_source).rectangle((0, 0, 60, HEADER), fill="black")
        palette = palette_source.quantize(colors=256)
        positions = [round(SIZE*(.08+.84*i/16)) for i in range(17)]
        positions = positions + positions[-2:0:-1]
        frames, durations = [], []
        for index, split in enumerate(positions):
            frame = Image.new("RGB", (SIZE, SIZE+HEADER), "white")
            frame.paste(right, (0, HEADER))
            frame.paste(left.crop((0, 0, split, SIZE)), (0, HEADER))
            draw = ImageDraw.Draw(frame)
            draw.text((12, 10), "LR input", fill="black", font=font)
            draw.text((SIZE-12, 10), "PGSR (ours)", fill="black", font=font, anchor="ra")
            draw.line((split, HEADER, split, SIZE+HEADER), fill="white", width=3)
            y = HEADER+SIZE//2
            draw.ellipse((split-15, y-15, split+15, y+15), fill="white", outline="black", width=1)
            draw.line((split-5, y-6, split-5, y+6), fill="black", width=1)
            draw.line((split+5, y-6, split+5, y+6), fill="black", width=1)
            frames.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))
            durations.append(900 if index in (0, 16) else 100)
        frames[0].save(output / f"demo-{sample}.gif", save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=1)
        manifest[sample] = {"lr_size": lr.size, "ours_size": ours.size, "ours_crop_xyxy": box,
                            "lr_crop_xyxy": lr_box, "resampling": "bicubic (both sides)",
                            "display_size": [SIZE, SIZE], "gif_colors": 256}
    (output / "demo-crops.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("Saved two README-native comparison GIFs.")


if __name__ == "__main__":
    main()
