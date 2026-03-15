# Latent Drifting — Disease-conditioned Manipulation (Brain Tumor)

Implementation of **Disease-conditioned Manipulation** from:

> **"Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis"**  
> Yeganeh et al., arXiv 2412.20651 (2024)

**Task**: Convert brain MRI slices with tumour (positive) into counterfactual healthy brain MRIs (negative), and vice versa.  
**Method**: Pix2Pix Zero + Latent Drifting (LD) on top of Stable Diffusion (Basic Fine-Tuning).

---

## Method Overview

### Latent Drifting (Section 3.3)

LD is a signed scalar hyperparameter δ added to the diffusion process:

```
pθ(xt-1|xt) = N(xt-1 ; µθ(xt,t) + δ,  Σθ(xt,t))     [Eq. 4]
```

- **During fine-tuning (forward process)**: δ is added to the target `z_T`,
  shifting it from `N(0,I)` → `N(δ,I)`.
- **During inference (reverse process)**: δ is added to the predicted mean at
  every denoising step, adapting image statistics to the medical domain.
- **Optimal δ** is found via grid search minimising L1 distance to the
  target distribution (supplement Algorithm).

### Disease-conditioned Manipulation (Section 4.2.2)

```
Source image (brain MRI with tumour)
        │
        ▼
  DDIM Inversion  ──(+δ)──►  z_T
        │
        ▼
  Collect source cross-attention maps A*  (Pix2Pix Zero)
        │
        ▼
  DDIM Denoising with:
    • Target text prompt ("a healthy brain MRI")
    • Cross-attention guidance: minimise ||A_target - A*_source||²  (τ)
    • Latent Drift: z_{t-1} += δ  at each step
        │
        ▼
  Counterfactual healthy brain MRI
```

---

## Directory Structure

```
ld_paper/
├── config.py                # All hyperparameters in one dataclass
├── prompts.py               # 21 diverse prompts per class (tumour / healthy)
├── dataset.py               # Brain tumour MRI dataset (expects tumor/ + healthy/)
├── latent_drifting.py       # Core LD utilities + grid-search for δ
├── attention_utils.py       # Cross-attention store + Pix2Pix Zero guidance loss
├── ddim_inversion.py        # DDIM inversion (source image → noise)
├── pix2pix_zero_ld.py       # Full Pix2Pix Zero + LD pipeline (main model)
├── train_basic_ft.py        # Basic Fine-Tuning of SD U-Net with LD
├── find_delta.py            # Grid search to find optimal δ*
├── run_manipulation.py      # End-to-end inference: tumour → healthy
└── evaluate.py              # FID, KID, SSIM, PSNR, LPIPS metrics
```

---

## Data Preparation

Organise your brain MRI images as follows:

```
data/
├── train/
│   ├── tumor/          ← positive: axial slices with tumour
│   └── healthy/        ← negative: healthy brain slices
├── val/
│   ├── tumor/
│   └── healthy/
└── test/
    ├── tumor/
    └── healthy/
```

Images should be `.png` / `.jpg` and will be automatically resized to 512 × 512.

---

## Step-by-Step Pipeline

### Step 1 — Find Optimal δ (without fine-tuning)

Run on the base SD-v1.4 to find the best δ for your dataset distribution:

```bash
python find_delta.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --source_data_dir ./data/test/tumor \
    --reference_data_dir ./data/test/healthy \
    --num_samples 5 \
    --delta_min -0.2 \
    --delta_max  0.2 \
    --delta_steps 9
```

This writes `optimal_delta.txt` with `δ*`.

### Step 2 — Basic Fine-Tuning with Latent Drifting

Fine-tune the SD U-Net on your brain MRI dataset:

```bash
accelerate launch train_basic_ft.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --train_data_dir ./data/train \
    --output_dir ./checkpoints \
    --delta 0.1 \
    --num_train_epochs 150 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --mixed_precision fp16
```

### Step 3 — Disease-conditioned Manipulation (Tumour → Healthy)

```bash
python run_manipulation.py \
    --model_path ./checkpoints \
    --source_data_dir ./data/test/tumor \
    --source_label tumor \
    --target_label healthy \
    --output_dir ./generated/tumor2healthy \
    --delta 0.1 \
    --tau 0.1 \
    --guidance_scale 7.5 \
    --save_diff
```

### Step 4 — Evaluate

```bash
python evaluate.py \
    --generated_dir ./generated/tumor2healthy/counterfactual \
    --real_dir      ./data/test/healthy \
    --source_dir    ./data/test/tumor \
    --output_file   ./results/metrics.json
```

---

## Key Hyperparameters

| Parameter | Paper value | Description                                           |
|-----------|-------------|-------------------------------------------------------|
| `δ`       | 0.1         | Latent drift (grid-searched over −0.2 → 0.2)         |
| `τ`       | 0.1         | Cross-attention guidance weight (Pix2Pix Zero)        |
| `w`       | 7.5         | Classifier-free guidance scale                        |
| T_inv     | 50          | DDIM inversion steps                                  |
| T_den     | 50          | DDIM denoising steps                                  |
| epochs    | 150         | Basic FT epochs                                       |
| lr        | 5e-6        | U-Net learning rate                                   |
| prompts   | Diverse+PI  | Best prompt type per Tables 3 & 4                     |

---

## Reference

```bibtex
@article{yeganeh2024latentdrifting,
  title   = {Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis},
  author  = {Yeganeh, Yousef and Farshad, Azade and Charisiadis, Ioannis and
             Hasny, Marta and Hartenberger, Martin and Ommer, Björn and
             Navab, Nassir and Adeli, Ehsan},
  journal = {arXiv preprint arXiv:2412.20651},
  year    = {2024}
}
```
