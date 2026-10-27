# Figure sources

These web-resolution images are exported from the authors' existing paper figures,
without regenerating restoration results or changing the crops:

- `teaser.png`: `figures/Teaser/teaser_2x4_input_first.png` (LR input,
  without pixel guidance, PGSR, GT).
- `pipeline.png`: paper `figures/pipeline.png`.
- `div2k-0801-comparison.png`: `figures/Results/DIV2K/0801_comparison.png`.
- `realsr-comparison.png`: paper `figures/Canon_002_comparison.pdf`.
- `high-resolution-comparison.png`: paper `8k/8K-res-aligned.pdf`; actual LR and
  PGSR crops share the same region under the x4 coordinate mapping.

The manuscript, rebuttal, raw dataset images, and model weights are not included
in this directory.

`demo-0492.gif` and `demo-0467.gif` are automatically sweeping README previews
from actual LR/PGSR pairs.
Coordinates are in `demo-crops.json`; both sides use bicubic display resizing
and the same GIF palette. These animations are not draggable widgets.

The independent GitHub Pages project page lives in `docs/`. Its `media/`
directory contains lossless WebP exports of these figures. The interactive
0492 and 0467 comparisons use the same source pairs and coordinates in
`demo-crops.json`, resized with bicubic interpolation to 768 × 768 on both
sides. The underlying restoration outputs are not regenerated, sharpened,
or recolored. Only `docs/` is published by GitHub Pages.

Additional project-page figures come from the paper appendix:
- DIV2K: `0834`, `0805`, `0818`, `0848`.
- RealSR: `Canon_034`, `Nikon_041`, `Nikon_044`, `Nikon_046`.

Their source PNGs are under `figures/Results/{DIV2K,RealSR}/` in the local
manuscript directory. All are exported as lossless WebP with a maximum width
of 2400 pixels, preserving the original method labels and layouts.
