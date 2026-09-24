# Figure sources

These web-resolution images are exported from the authors' existing paper figures,
without regenerating restoration results or changing the crops:

- `teaser.png`: paper `figures/teaser.pdf` (GT, LR, PiSA-SR, PGSR).
- `realsr-comparison.png`: paper `figures/Canon_002_comparison.pdf`.
- `high-resolution-comparison.png`: paper `8k/8K-res-aligned.pdf`; actual LR and
  PGSR crops share the same region under the x4 coordinate mapping.

The manuscript, rebuttal, raw dataset images, and model weights are not included
in this directory.

`demo-0492.gif` and `demo-0467.gif` are automatically sweeping README previews
from actual LR/PGSR pairs. Rebuild with `tools/build_readme_demos.py` (Pillow).
Coordinates are in `demo-crops.json`; both sides use bicubic display resizing
and the same GIF palette. These animations are not draggable widgets.
