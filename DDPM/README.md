# DDPM Paper Reimplementation

This folder contains a fresh 2D conditional DDPM implementation aligned to the paper `Denoising Diffusion Models for Inpainting of Healthy Brain Tissue`.

Implemented paper defaults:

- Slice-wise 2D training
- Conditional input as `b_i || m_i || x_0`, where the last channel is noised during diffusion
- `T = 1000`
- `batch_size = 8`
- `lr = 1e-4`
- EMA with decay `0.9999`
- U-Net base width `128`
- `num_res_blocks = 2`
- one attention head at downsample factor `16`
- Gaussian post-processing with `sigma = 1.075`

Important limitation with the current dataset:

- The original paper trains with healthy-region masks and corresponding healthy tissue as ground truth.
- The provided local dataset only exposes 2D PNG slices plus binary masks.
- Because of that, this implementation reconstructs the original masked slice from the provided image itself. The training and sampling pipeline follows the paper structure, but the supervision signal is not identical to the BraTS healthy-tissue setup.

Train:

```bash
python -m DDPM.train \
  --train-image-dir data/brain_tumors/train_images \
  --train-mask-dir data/brain_tumors/train_masks \
  --val-image-dir data/brain_tumors/val_images \
  --val-mask-dir data/brain_tumors/val_masks \
  --output-dir DDPM/outputs/paper_run \
  --amp
```

Sample:

```bash
python -m DDPM.sample \
  --image-dir data/brain_tumors/val_images \
  --mask-dir data/brain_tumors/val_masks \
  --checkpoint DDPM/outputs/paper_run/checkpoints/step_005000.pt \
  --output-dir DDPM/samples/val \
  --use-ema \
  --blend-known-region \
  --restore-original-size
```

Output structure:

- `checkpoints/`: training checkpoints
- `previews/`: periodic qualitative samples from validation data
- `raw/`: direct DDPM outputs
- `blended/`: outputs composited with the known region outside the mask
