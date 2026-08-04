# PGSR

## When Latents Forget Pixels: Restoring Fidelity in Diffusion Transformer Super-Resolution

The official implementation of the paper **"When Latents Forget Pixels: Restoring Fidelity in Diffusion Transformer Super-Resolution."**

PGSR preserves pixel-level evidence from the low-resolution input and uses it to guide both the diffusion trajectory and VAE decoding. The implementation is based on FLUX.1-dev and its super-resolution ControlNet.

> Paper and pretrained checkpoints will be released soon.

## Installation

The code was tested with Python 3.11, PyTorch 2.5.1, Diffusers 0.36.0, Accelerate 0.34.0, Transformers 4.57.5, and PEFT 0.18.1.

```bash
conda create -n pgsr python=3.11 -y
conda activate pgsr

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.36.0 accelerate==0.34.0 transformers==4.57.5 \
  peft==0.18.1 deepspeed lpips scikit-image pyiqa
```

Access to [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) is required. By default, PGSR initializes its ControlNet from [Flux.1-dev-Controlnet-Upscaler](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler).

## Data

Dataset download and preparation utilities are provided in `download/`:

```bash
python download/download_datasets.py --help
python download/build_dataset.py --help
```

## Inference

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

## Training

PGSR uses two-stage training: paired bicubic pretraining followed by Real-ESRGAN degradation fine-tuning. The complete commands and resume settings are documented at the top of `train_pgsr.py`.

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

The final entry points are `train_pgsr.py` and `evaluate_pgsr.py`. The corresponding CLEAR sparse-attention variant is provided in `train_pgsr_clear.py` and `evaluate_pgsr_clear.py`.

## Citation

Citation information will be added with the paper release.
