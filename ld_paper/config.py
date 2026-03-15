"""
Configuration for Latent Drifting (LD) Disease-conditioned Manipulation.

Based on: "Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis"
Task   : Brain Tumor (Positive) -> Healthy (Negative) counterfactual synthesis.
Method : Pix2Pix Zero + Latent Drifting (LD) on Stable Diffusion Basic Fine-Tuning.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------
    model_id: str = "CompVis/stable-diffusion-v1-4"
    # Path to checkpoint after Basic FT with LD; None = use base model
    finetuned_model_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Latent Drifting (LD)           [Section 3.3 of the paper]
    # ------------------------------------------------------------------
    # δ : signed scalar drift applied to mean at every reverse step
    #   pθ(xt-1|xt) = N(xt-1 ; µθ(xt,t) + δ, Σθ(xt,t))
    delta: float = 0.1
    # Range and resolution for grid-search (Algorithm in supplement)
    delta_range: Tuple[float, float] = (-0.2, 0.2)
    delta_grid_steps: int = 9          # 9 points: −0.2,−0.15,...,0.2

    # ------------------------------------------------------------------
    # Basic Fine-Tuning with LD      [Section 4.1]
    # ------------------------------------------------------------------
    output_dir: str = "./checkpoints"
    train_data_dir: str = "./data/train"        # expects tumor/ + healthy/ subdirs
    validation_data_dir: str = ""
    logging_dir: str = "./logs"

    resolution: int = 512
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 150
    max_train_steps: Optional[int] = None       # overrides epochs if set
    learning_rate: float = 5e-6
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 200
    mixed_precision: str = "fp16"               # "no" | "fp16" | "bf16"
    gradient_checkpointing: bool = True
    seed: int = 42

    # LD applied to training: add δ to noisy latents at every step
    apply_ld_to_forward: bool = True

    # ------------------------------------------------------------------
    # Disease-conditioned Manipulation  [Section 4.2.2]
    # Method: Pix2Pix Zero + LD
    # ------------------------------------------------------------------
    num_ddim_inversion_steps: int = 50
    num_ddim_denoising_steps: int = 50
    guidance_scale: float = 7.5          # classifier-free guidance weight
    # τ : cross-attention guidance weight (Pix2Pix Zero)
    cross_attention_guidance_amount: float = 0.1

    # Task direction
    source_label: str = "tumor"          # positive: brain with tumor
    target_label: str = "healthy"        # negative: healthy brain

    # ------------------------------------------------------------------
    # Prompting                       [Section 4.1, Tables 3 & 4]
    # Best: Diverse prompts + Patient Information
    # ------------------------------------------------------------------
    use_diverse_prompts: bool = True
    use_patient_info: bool = False       # set True when metadata is available

    # ------------------------------------------------------------------
    # Inference / Output
    # ------------------------------------------------------------------
    output_images_dir: str = "./generated"
    num_eval_samples: int = 200
    save_intermediate: bool = False      # save per-step latent images

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    real_images_dir: str = "./data/test/healthy"   # 1,000장 IXI 정상 MRI
    fid_batch_size: int = 50
