# PGSR project page

Static, dependency-free website for GitHub Pages (`main`, `/docs`).
Only this directory is published. No model inference or third-party analytics runs
in the browser.

Preview: `python -m http.server 8765 --directory docs`, then open
`http://localhost:8765/`.

To regenerate the paired web images with Pillow:

```bash
python tools/build_demo_assets.py --source /path/to/original/8k
```

The input directory must contain `0467_LR.png`, `0467_ours.png`,
`0492_LR.png`, and `0492_ours.png`. HR references are not used as restored outputs.
Crop coordinates, source sizes, and display resampling are recorded in
`images/manifest.json`. Both sides use identical output-space coordinates and
lossless WebP encoding. No synthetic degradation is added for the demo.
