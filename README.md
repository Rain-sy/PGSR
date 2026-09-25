<div align="center">

# When Latents Forget Pixels: Restoring Fidelity in Diffusion Transformer Super-Resolution

**NeurIPS 2026 · Poster**

[Paper (arXiv)](https://arxiv.org/abs/2608.09133) · [Method](#method-overview) · [Visual Results](#visual-results) · [Installation](#installation) · [Inference](#inference) · [Training](#training)

</div>

Official implementation of **Pixel-Grounded Super-Resolution (PGSR)**, which uses LR pixel evidence to guide diffusion and VAE decoding with FLUX.1-dev.

![PGSR teaser: LR input, without pixel guidance, PGSR, and ground truth](assets/teaser.png)

*Left to right: LR input, without pixel guidance, PGSR (ours), and ground truth.*

## News

- **Coming soon:** PGSR checkpoints.
- **2026-09:** PGSR accepted to **NeurIPS 2026 as a poster**!
- **2026-08-10:** Our [paper](https://arxiv.org/abs/2608.09133) is available on arXiv.
- **2026-08-03:** Training and evaluation code organized for release.

## Method Overview

![PGSR pipeline: condition-side trajectory guidance and decoder-side pixel grounding](assets/pipeline.png)

## Visual Results

### DIV2K super-resolution

![DIV2K 0801 comparison with generative super-resolution methods](assets/div2k-0801-comparison.png)

### Real-world super-resolution

![RealSR comparison with generative super-resolution methods](assets/realsr-comparison.png)

### High-resolution demos

**LR (left) / PGSR (right)** with an automatically sweeping divider. Crops are spatially aligned; LR is bicubic-enlarged for display.

**Carved stone · DIV8K 0492 · 4× SR (1680 × 1296 → 6720 × 5184)**

![Animated LR versus PGSR comparison of carved stone](assets/demo-0492.gif)

**Window tracery · DIV8K 0467 · 4× SR (1680 × 1584 → 6720 × 6336)**

![Animated LR versus PGSR comparison of cathedral windows](assets/demo-0467.gif)

<details>
<summary>Full-image context and display notes</summary>

![HR overview with matched LR and PGSR crops](assets/high-resolution-comparison.png)

GIFs are 256-color web previews, not evaluation images. Crop coordinates and
display settings are recorded in [assets/demo-crops.json](assets/demo-crops.json).

</details>

## Installation

```bash
conda create -n pgsr python=3.11 -y
conda activate pgsr

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.36.0 accelerate==0.34.0 transformers==4.57.5 \
  peft==0.18.1 deepspeed lpips scikit-image pyiqa
```

Request access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
and authenticate with Hugging Face before running the code. The default ControlNet
initialization is [Flux.1-dev-Controlnet-Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler).

## Data

- **Stage 1:** DF2K (DIV2K + Flickr2K), with paired bicubic x4 LR-HR images.
- **Stage 2:** the union of DF2K, LSDIR, FFHQ, and OST. LR inputs are synthesized online with the second-order Real-ESRGAN degradation pipeline.
- **Evaluation:** DIV2K validation, RealSR, and DRealSR.

Place downloaded datasets under `Data/`, or pass their locations directly with `--hr_dir`, `--lr_dir`, `--val_hr_dir`, and `--val_lr_dir`.

## Inference

Use your own trained PGSR checkpoint until public weights are released.

### Full attention

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_pgsr.py \
  --checkpoint checkpoints/pgsr/best_model.pt \
  --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
  --hr_dir Data/DIV2K/DIV2K_valid_HR \
  --num_steps 20 \
  --iqa_device cuda \
  --lpips_device cuda
```

Results and metrics are written to `outputs/`. The HR directory is optional when only restored images are needed.

### Sparse attention

Use a matching sparse-attention PGSR checkpoint. The default configuration also
requires the [official CLEAR initialization weights](https://huggingface.co/Huage001/CLEAR)
matching the training window size and downsampling factor:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_pgsr_sparse.py \
  --checkpoint checkpoints/pgsr_sparse/best_model.pt \
  --sparse_ckpt ckpt/clear_local_16_down_4.safetensors \
  --attention_mode sparse \
  --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
  --num_steps 20
```

`--attention_mode auto` detects the attention configuration stored in the checkpoint.
Sparse attention uses PyTorch FlexAttention / Triton and requires a compatible CUDA environment.

## Training

Stage 1 uses paired bicubic data; Stage 2 uses Real-ESRGAN degradation. See [`train_pgsr.py`](train_pgsr.py) for both stages and resume settings.

```bash
accelerate launch --num_processes=8 --gradient_accumulation_steps=8 \
  train_pgsr.py \
  --hr_dir Data/DF2K_HR \
  --lr_dir Data/DF2K_LR_bicubic_X4 \
  --degrade_mode paired \
  --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
  --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
  --scale 4 --resolution 512
```

For sparse-attention training, use `train_pgsr_sparse.py` with the same data
arguments and add `--sparse_ckpt ckpt/clear_local_16_down_4.safetensors`.
Configure the attention pattern with `--sparse_window_size` and `--sparse_down_factor`.
Training resolution must be divisible by `16 * sparse_down_factor`.

## Acknowledgements

This implementation builds on [Diffusers](https://github.com/huggingface/diffusers), [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev), and the [FLUX ControlNet Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler).

## Citation

```bibtex
@article{shi2026latents,
  title={When Latents Forget Pixels: Restoring Fidelity in Diffusion Transformer Super-Resolution},
  author={Shi, Yu and Zhang, Yuyao and Tai, Yu-wing},
  journal={arXiv preprint arXiv:2608.09133},
  year={2026}
}
```
