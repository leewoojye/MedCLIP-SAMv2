"""
Evaluation Metrics for counterfactual brain MRI generation.

Paper (Section 4.1):
  "For the evaluation of the image realism, we calculate the Fréchet
   Inception Distance (FID) and Kernel Inception Distance (KID) between
   the synthetically generated samples and our test set.
   Additionally, we determine the Structure Similarity Index (SSIM) between
   the target and the source image to determine how well the identity of
   the source image is retained."

Metrics implemented
-------------------
  FID   – Fréchet Inception Distance         (image realism vs. real distribution)
  KID   – Kernel Inception Distance          (unbiased FID variant)
  SSIM  – Structural Similarity Index        (source identity preservation)
  PSNR  – Peak Signal-to-Noise Ratio         (pixel-level fidelity)
  LPIPS – Learned Perceptual Image Patch Sim (perceptual fidelity)

Dependencies
------------
  torch, torchvision, scikit-image, numpy
  Optional: lpips (pip install lpips)

Usage
-----
python evaluate.py \
    --generated_dir ./generated/tumor2healthy/counterfactual \
    --real_dir      ./data/test/healthy \
    --source_dir    ./data/test/tumor \
    --output_file   ./results/metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm

# scikit-image for SSIM & PSNR
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_inception_transform = T.Compose([
    T.Resize(299, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_psnr_ssim_transform = T.Compose([
    T.Resize(512, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.CenterCrop(512),
])


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_image_paths(directory: str) -> List[Path]:
    return sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )


def load_pil(path: Path, size: int = 512) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Inception features  (for FID / KID)
# ---------------------------------------------------------------------------

def _get_inception_model(device: torch.device):
    """Load InceptionV3 with pooled features output."""
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()       # output: [B, 2048]
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_inception_features(
    image_paths: List[Path],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Extract 2048-d Inception features for a list of image paths."""
    model = _get_inception_model(device)
    all_feats: List[torch.Tensor] = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors = torch.stack([
            _inception_transform(load_pil(p, 299)) for p in batch_paths
        ]).to(device)
        feats = model(tensors)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)   # [N, 2048]


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    feats_real: torch.Tensor,
    feats_fake: torch.Tensor,
) -> float:
    """
    Compute FID between two sets of Inception features.

    FID = ||µ_r - µ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r·Σ_g)^{1/2})
    """
    import scipy.linalg

    mu1 = feats_real.mean(0).numpy()
    mu2 = feats_fake.mean(0).numpy()
    sigma1 = np.cov(feats_real.numpy(), rowvar=False)
    sigma2 = np.cov(feats_fake.numpy(), rowvar=False)

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


# ---------------------------------------------------------------------------
# KID
# ---------------------------------------------------------------------------

def _polynomial_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
    """Polynomial kernel K(x,y) = (x·y / d + 1)^degree."""
    d = X.shape[1]
    return (X @ Y.T / d + 1) ** degree


def compute_kid(
    feats_real: torch.Tensor,
    feats_fake: torch.Tensor,
    num_subsets: int = 100,
    subset_size: int = 1000,
) -> Tuple[float, float]:
    """
    Compute KID (mean ± std) using polynomial kernel MMD.

    Returns (kid_mean, kid_std).
    """
    rng = np.random.default_rng(42)
    X   = feats_real.numpy().astype(np.float32)
    Y   = feats_fake.numpy().astype(np.float32)
    n   = min(len(X), len(Y), subset_size)
    mmd_values = []

    for _ in range(num_subsets):
        xi = rng.choice(len(X), n, replace=False)
        yi = rng.choice(len(Y), n, replace=False)
        Xs, Ys = X[xi], Y[yi]

        k_xx = _polynomial_kernel(Xs, Xs)
        k_yy = _polynomial_kernel(Ys, Ys)
        k_xy = _polynomial_kernel(Xs, Ys)

        mmd = (k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())
        mmd_values.append(mmd)

    arr = np.array(mmd_values)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# SSIM & PSNR  (source identity preservation)
# ---------------------------------------------------------------------------

def compute_ssim_psnr_batch(
    source_paths: List[Path],
    generated_paths: List[Path],
) -> Tuple[float, float]:
    """
    Compute mean SSIM and PSNR between source and generated image pairs.

    Measures how well the source identity is preserved after editing.
    """
    ssim_vals: List[float] = []
    psnr_vals: List[float] = []

    n = min(len(source_paths), len(generated_paths))
    for i in range(n):
        src  = np.array(load_pil(source_paths[i]))
        gen  = np.array(load_pil(generated_paths[i]))

        ssim_val = skimage_ssim(src, gen, channel_axis=2, data_range=255)
        psnr_val = skimage_psnr(src, gen, data_range=255)

        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)

    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


# ---------------------------------------------------------------------------
# LPIPS  (optional — requires `pip install lpips`)
# ---------------------------------------------------------------------------

def compute_lpips_batch(
    source_paths: List[Path],
    generated_paths: List[Path],
    device: torch.device,
) -> Optional[float]:
    """Compute mean LPIPS (AlexNet) between source and generated pairs."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex").to(device)
        loss_fn.eval()
    except ImportError:
        print("  [LPIPS] `lpips` not installed — skipping. pip install lpips")
        return None

    _transform = T.Compose([
        T.Resize(512, antialias=True),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    vals: List[float] = []
    n = min(len(source_paths), len(generated_paths))

    with torch.no_grad():
        for i in range(n):
            src = _transform(load_pil(source_paths[i])).unsqueeze(0).to(device)
            gen = _transform(load_pil(generated_paths[i])).unsqueeze(0).to(device)
            vals.append(loss_fn(src, gen).item())

    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(
    generated_dir: str,
    real_dir: str,
    source_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        generated_dir : Directory of generated/counterfactual images.
        real_dir      : Directory of real target-domain images (D_GT).
        source_dir    : Directory of source images (for SSIM/PSNR/LPIPS).
        output_file   : JSON file to write results.
        batch_size    : Inception batch size.
        device        : Torch device.

    Returns:
        dict with all computed metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("Evaluation – Latent Drifting (Brain Tumor Counterfactuals)")
    print(f"{'='*60}")

    # ---- Load paths ------------------------------------------------------
    gen_paths  = load_image_paths(generated_dir)
    real_paths = load_image_paths(real_dir)

    print(f"  Generated images : {len(gen_paths)}")
    print(f"  Real images      : {len(real_paths)}")

    results: Dict[str, float] = {}

    # ---- Inception features for FID / KID --------------------------------
    print("\n[1/4] Extracting Inception features …")
    feats_real = extract_inception_features(real_paths, device, batch_size)
    feats_gen  = extract_inception_features(gen_paths,  device, batch_size)

    # ---- FID -------------------------------------------------------------
    print("[2/4] Computing FID …")
    results["FID"] = compute_fid(feats_real, feats_gen)
    print(f"  FID  = {results['FID']:.3f}")

    # ---- KID -------------------------------------------------------------
    print("[3/4] Computing KID …")
    kid_mean, kid_std = compute_kid(feats_real, feats_gen)
    results["KID"]     = kid_mean
    results["KID_std"] = kid_std
    print(f"  KID  = {kid_mean:.5f} ± {kid_std:.5f}")

    # ---- SSIM / PSNR / LPIPS (source ↔ generated pairs) ----------------
    if source_dir is not None:
        src_paths = load_image_paths(source_dir)
        n = min(len(src_paths), len(gen_paths))
        print(f"[4/4] Computing SSIM, PSNR, LPIPS on {n} pairs …")

        ssim_val, psnr_val = compute_ssim_psnr_batch(src_paths[:n], gen_paths[:n])
        results["SSIM"] = ssim_val
        results["PSNR"] = psnr_val
        print(f"  SSIM = {ssim_val:.4f}")
        print(f"  PSNR = {psnr_val:.2f} dB")

        lpips_val = compute_lpips_batch(src_paths[:n], gen_paths[:n], device)
        if lpips_val is not None:
            results["LPIPS"] = lpips_val
            print(f"  LPIPS= {lpips_val:.4f}")
    else:
        print("[4/4] Skipping SSIM/PSNR/LPIPS (no --source_dir provided).")

    # ---- Print summary ---------------------------------------------------
    print(f"\n{'─'*40}")
    print("SUMMARY")
    print(f"{'─'*40}")
    for key, val in results.items():
        print(f"  {key:<10}: {val:.5f}")

    # ---- Save ------------------------------------------------------------
    if output_file is not None:
        os.makedirs(Path(output_file).parent, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate counterfactual generation")
    parser.add_argument("--generated_dir",  type=str, required=True)
    parser.add_argument("--real_dir",       type=str, required=True,
                        help="Real images from D_GT (target domain, e.g. healthy).")
    parser.add_argument("--source_dir",     type=str, default=None,
                        help="Source images (for SSIM/PSNR/LPIPS pair evaluation).")
    parser.add_argument("--output_file",    type=str, default="./results/metrics.json")
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--device",         type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    evaluate(
        generated_dir=args.generated_dir,
        real_dir=args.real_dir,
        source_dir=args.source_dir,
        output_file=args.output_file,
        batch_size=args.batch_size,
        device=device,
    )
