"""
Latent Drifting (LD) core module.

From Section 3.3 of the paper:

  "We define LD as a hyperparameter in diffusion models to adapt the learned
   distribution of a pre-trained model Dθ to a new data distribution D_GT.
   LD is represented by a signed scalar value δ and is introduced to the
   diffusion process.

   For the fine-tuning through Latent Drifting, LD is added to the target
   zT of the forward process, as well as the reverse processes:

       pθ(xt-1|xt) = N(xt-1 ; µθ(xt,t) + δ,  Σθ(xt,t))         (Eq. 4)"

Grid-search algorithm (supplement):
  1. δ ∈ {−0.2, −0.15, …, 0.15, 0.2}
  2. For each δ, generate a small batch; compute L1 distance to target domain.
  3. Select δ* = argmin_δ  L1(generated, target).
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from PIL import Image
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def apply_latent_drift(latents: torch.Tensor, delta: float) -> torch.Tensor:
    """
    Shift the mean of a latent tensor by δ.

    In the reverse process this implements  µ_θ(xt,t) + δ.
    In the forward process this shifts the target zT by δ.

    Args:
        latents : Any shape tensor (typically [B, C, H, W]).
        delta   : Signed scalar δ.

    Returns:
        Shifted tensor of the same shape and dtype.
    """
    if abs(delta) < 1e-9:
        return latents
    return latents + delta


def apply_latent_drift_to_noise(
    noise: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """
    Shift the initial noise zT ~ N(0, I) to zT ~ N(δ, I).

    Used during *fine-tuning* forward process (Section 3.3):
    'LD is added to the target zT of the forward process'.
    """
    return apply_latent_drift(noise, delta)


# ---------------------------------------------------------------------------
# L1 distance between image distributions (grid-search objective)
# ---------------------------------------------------------------------------

_to_tensor = T.Compose([
    T.Resize((64, 64), antialias=True),  # downscale for speed
    T.ToTensor(),
])


def _pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL image to normalised [0,1] tensor on device."""
    return _to_tensor(img.convert("RGB")).to(device)


def compute_l1_distribution_distance(
    generated: List[Image.Image],
    reference: List[Image.Image],
    device: torch.device,
) -> float:
    """
    Estimate L1 distance between two image distributions via Monte Carlo sampling.

    From the paper (Section 3.4):
      "In the absence of the training set, this distance can be estimated via
       Monte Carlo sampling of generated samples from Dθ and existing samples
       in D_GT. Here, we use L1norm as the distance function."

    Args:
        generated  : List of PIL images from the model (Dθ).
        reference  : List of real target-domain images (D_GT).
        device     : Torch device.

    Returns:
        Scalar L1 distance.
    """
    gen_tensors = [_pil_to_tensor(img, device) for img in generated]
    ref_tensors = [_pil_to_tensor(img, device) for img in reference]

    gen_mean = torch.stack(gen_tensors).mean(dim=0)
    ref_mean = torch.stack(ref_tensors).mean(dim=0)

    return F.l1_loss(gen_mean, ref_mean).item()


# ---------------------------------------------------------------------------
# Grid search for optimal δ
# ---------------------------------------------------------------------------

def find_optimal_delta(
    generate_fn,                             # callable: (delta) -> List[PIL.Image]
    reference_images: List[Image.Image],
    delta_range: Tuple[float, float] = (-0.2, 0.2),
    num_grid_steps: int = 9,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> float:
    """
    Grid-search for the optimal latent drift δ*.

    Algorithm (supplement):
      For each δ in the grid:
        1. Generate a small batch with the model conditioned on δ.
        2. Compute L1(generated, reference).
      Return δ* that minimises L1.

    Args:
        generate_fn      : A function (delta: float) -> List[PIL.Image].
                           Should return generated images for the given δ.
        reference_images : Real samples from the target domain D_GT (healthy).
        delta_range      : (min_δ, max_δ) grid boundaries.
        num_grid_steps   : Number of δ values to evaluate.
        device           : Torch device (defaults to CUDA if available).
        verbose          : Print per-δ distances.

    Returns:
        Optimal δ* as a float.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    delta_values = torch.linspace(delta_range[0], delta_range[1], num_grid_steps).tolist()

    best_delta = 0.0
    best_dist = float("inf")
    results = []

    for delta in delta_values:
        generated = generate_fn(delta)
        dist = compute_l1_distribution_distance(generated, reference_images, device)
        results.append((delta, dist))

        if verbose:
            print(f"  δ = {delta:+.3f}  |  L1 = {dist:.5f}")

        if dist < best_dist:
            best_dist = dist
            best_delta = delta

    if verbose:
        print(f"\n  → Optimal δ* = {best_delta:+.3f}  (L1 = {best_dist:.5f})")

    return best_delta
