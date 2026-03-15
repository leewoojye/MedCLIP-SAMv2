# DDPM for Brain Tissue Inpainting

이 구현은 논문 "Denoising Diffusion Models for Inpainting of Healthy Brain Tissue" (Durrer et al., 2024)을 정확히 따릅니다.

## 📄 Paper Implementation Summary

### Paper Specifications (Section 3.2)

**Architecture:**
- U-Net with 128 channels in first layer
- 1 attention head at resolution 16
- 113,672,066 total parameters

**Training:**
- **Optimizer**: Adam (LR = 1e-4)
- **Batch size**: 8
- **Total iterations**: 2,850,000 (~2.5 weeks)
- **Timesteps (T)**: 1000
- **EMA**: 0.9999 rate for validation/test evaluation

**Data Preprocessing (Section 3.1):**
- Remove top and bottom 0.1 percentile voxel intensities
- Normalize to [0, 1]
- 2D axial slices
- **Crop to [224, 224] - only background pixels affected**
- Only use slices with non-zero mask

**Postprocessing:**
- Gaussian filter with σ = 1.075

## 📂 New Implementation Files

### 1. `DDPM/unet_paper.py`
Complete U-Net architecture faithful to paper specifications:
- SinusoidalTimeEmbedding: Time step encoding
- ResidualBlock: With time conditioning (scale-shift modulation)
- AttentionBlock: Self-attention with 1 head
- Downsample/Upsample: Resolution scaling
- Full encoder-decoder architecture with skip connections

### 2. `DDPM/simple_ddpm_paper.py` 
Core DDPM model:
- **q_sample()**: Forward diffusion process (Equation 2)
- **p_sample()**: Reverse diffusion step (Equation 5)  
- **forward()**: Training forward pass with noise prediction
- **sample()**: Iterative denoising from noise to image
- Diffusion schedule: Linear β schedule (0.0001 to 0.02)
- **EMA support**: For validation/test evaluation

### 3. `DDPM/train_paper.py`
Training script implementing paper specifications:
- **Batch size**: 8 (paper spec)
- **Iterations**: 2,850,000 (paper spec)
- **Learning rate**: 1e-4 with Adam optimizer
- **EMA class**: Exponential moving average (decay=0.9999)
- **DDPMTrainer**: Full training loop with checkpointing, validation, loss tracking

## ✨ Data Preprocessing: ROI-Aware Cropping

**User-specified modification**: Instead of fixed 224×224 cropping, uses ROI-aware cropping via `MaskAwareCropper`:
- Detects tumor/healthy tissue ROI from mask
- Generates crop with 15% padding around ROI
- Minimizes data loss while removing background
- Maintains paper's intent: "Only background pixels are affected by the cropping"

## 🚀 Usage

### Training

```python
from DDPM.train_paper import DDPMTrainer

trainer = DDPMTrainer(
    data_dir="./data/brain_tumors",
    output_dir="./DDPM_output",
    batch_size=8,
    num_iterations=2_850_000,  # 2.85M as per paper
    learning_rate=1e-4,
    T=1000,
    ema_decay=0.9999,
)

trainer.train()
```

### Or via command line:

```bash
cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2

# Quick test (3 epochs)
python -m DDPM.train_paper \
    --data-dir ./data/brain_tumors \
    --output-dir ./DDPM_output \
    --batch-size 8 \
    --num-epochs 3 \
    --num-iterations 2_850_000 \
    --learning-rate 1e-4

# Or with source activate for conda env
source activate medclipsamv2
python -m DDPM.train_paper \
    --data-dir ./data/brain_tumors \
    --output-dir ./DDPM_output \
    --batch-size 8
```

### Import and use model:

```python
import torch
from DDPM.simple_ddpm_paper import SimpleDDPM

# Create model
model = SimpleDDPM(model_channels=128, T=1000)
model.to("cuda")

# Training forward pass
x_0 = torch.randn(8, 1, 224, 224)  # Ground truth
t = torch.randint(0, 1000, (8,))   # Random timesteps  
context = torch.randn(8, 2, 224, 224)  # baseline + mask

# Forward pass
predicted_noise, actual_noise = model(x_0, t, context)
loss = ((predicted_noise - actual_noise) ** 2).mean()
```

## 📊 Data Format

**Input Data Structure**:
```
data/brain_tumors/
├── train_images/     # T1 scans, PNG format
│   ├── patient_001.png
│   ├── patient_002.png
│   └── ...
└── train_masks/      # Tumor region masks, PNG format
    ├── patient_001.png
    ├── patient_002.png
    └── ...
```

**During Training:**
- Input channels: 3
  - Channel 0: Noisy image (from q_sample)
  - Channel 1: Baseline (masked T1 image)
  - Channel 2: Mask (healthy tissue region)
- Output: Predicted noise (same shape as input image)

## 🔧 Key Differences from Original Code

✓ **Matches paper exactly**:
- 128-channel first layer
- 1 attention head at resolution 16
- 113M+ parameters target
- T=1000 timesteps
- Adam, LR=1e-4
- EMA 0.9999 for eval
- Linear β schedule

✓ **ROI-aware data preprocessing**:
- Uses MaskAwareCropper instead of fixed center crop
- Pads 15% around tumor ROI
- Preserves paper's "background only" cropping intent

## 📝 Paper Equations Implemented

- **Equation 2 (Forward diffusion)**: `q_sample()`
- **Equation 5 (Reverse step)**: `p_sample()`  
- **Equation 6 (MSE Loss)**: Training loop

## ✅ Testing

```bash
# Quick test of architecture
python test_paper_simple.py
```

Expected output:
```
✓ Model created
✓ Forward pass successful
✓ MSE Loss computed
✓ ALL TESTS PASSED
```

## 📚 References

- Original paper: "Denoising Diffusion Models for Inpainting of Healthy Brain Tissue" (Durrer et al., 2024)
- Baseline DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (NIPS 2020)
- U-Net architecture from Wolleb et al., "Diffusion Models for Implicit Image Segmentation Ensembles" (MIDL 2022)

## 🎯 Next Steps

1. Run training with 3 epochs to test pipeline
2. Evaluate on validation set
3. Compare SSIM/PSNR/MSE metrics with paper's results
4.  (Optional) Extend to 3D model as mentioned in paper discussion

---

**Implementation Status**: ✓ Complete and ready for training
**Last Updated**: 2026-03-15
