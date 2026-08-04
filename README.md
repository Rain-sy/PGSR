# PGSR

Official research code for pixel-grounded image super-resolution with a
FLUX.1-dev backbone. The repository keeps the final PGSR and PGSR+CLEAR
training/evaluation entry points at the top level; earlier experimental
variants are retained under `train/` for reference.

## Repository layout

```text
PGSR/
|-- train_pgsr.py              # final PGSR training
|-- evaluate_pgsr.py           # final PGSR evaluation
|-- train_pgsr_clear.py        # PGSR with CLEAR sparse attention
|-- evaluate_pgsr_clear.py     # PGSR+CLEAR evaluation
|-- CLEAR/                     # sparse-attention implementation
|-- configs/                   # Accelerate/DeepSpeed and comparison configs
|-- download/                  # dataset preparation utilities
|-- scripts/paper_compare/     # reproducible baseline/evaluation pipeline
|-- train/                     # archived training/evaluation variants
|-- Data/                      # local datasets (ignored)
|-- checkpoints/               # local model weights (ignored)
|-- outputs/                   # local inference outputs (ignored)
|-- experiments/               # local experiment outputs (mostly ignored)
|-- external_baselines/        # local third-party repositories (ignored)
`-- rebuttal/                  # local rebuttal code/results (ignored)
```

## Environment

The final code was tested with Python 3.11, PyTorch 2.5.1+cu121,
Diffusers 0.36.0, Accelerate 0.34.0, Transformers 4.57.5, and PEFT 0.18.1.
Training additionally uses DeepSpeed ZeRO-2. LPIPS is optional for training;
evaluation can additionally use `scikit-image` and `pyiqa`.

The default pretrained components are:

- `black-forest-labs/FLUX.1-dev`
- `jasperai/Flux.1-dev-Controlnet-Upscaler`

Access to FLUX.1-dev must be configured through Hugging Face before the first
run.

## Dataset preparation

Dataset download and preparation helpers are under `download/`. Prepared data
is expected under `Data/`, which is intentionally excluded from version
control.

```bash
python download/download_datasets.py --help
python download/build_dataset.py --help
```

## Training

The training script supports paired bicubic data and on-the-fly Real-ESRGAN
degradation. A typical two-stage workflow is documented in the module header.

```bash
accelerate launch --config_file configs/accelerate_deepspeed.yaml \
  train_pgsr.py \
  --hr_dir Data/DF2K_HR \
  --lr_dir Data/DF2K_LR_bicubic_X4 \
  --degrade_mode paired \
  --val_hr_dir Data/DIV2K/DIV2K_valid_HR \
  --val_lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
  --scale 4 --resolution 512
```

Use `train_pgsr_clear.py` with the same dataset/checkpoint conventions to train
the sparse-attention variant.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_pgsr.py \
  --checkpoint checkpoints/pgsr/<run>/best_model.pt \
  --hr_dir Data/DIV2K/DIV2K_valid_HR \
  --lr_dir Data/DIV2K/DIV2K_valid_LR_bicubic_X4 \
  --num_steps 20
```

Use `evaluate_pgsr_clear.py` for checkpoints trained with CLEAR. Run any entry
point with `--help` for the complete option set.

## Local-only material

Datasets, checkpoints, generated outputs, third-party baselines, and all
rebuttal-specific code/results are ignored by Git. They remain available in
the local workspace but are not part of the public PGSR repository.
