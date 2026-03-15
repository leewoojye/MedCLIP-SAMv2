"""
DDIM Inversion for Pix2Pix Zero + Latent Drifting.

DDIM Inversion maps a real image x0 back to noise zT by running the
DDIM deterministic process in *reverse time* (forward direction).
The resulting noise trajectory is used as the starting point for
image editing:  editing restarts from zT and denoises toward the
target condition.

We apply Latent Drifting (δ) during inversion so the inverted
representation sits in the same shifted latent space used during
inference (Eq. 4 of the paper).

Reference:
  Song et al. "Denoising Diffusion Implicit Models" (DDIM), ICLR 2021.
  Parmar et al. "Zero-shot Image-to-Image Translation" (Pix2Pix Zero), SIGGRAPH 2023.
  Yeganeh et al. "Latent Drifting in Diffusion Models…" arXiv 2412.20651.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from diffusers import DDIMScheduler
from PIL import Image
import torchvision.transforms as T
import numpy as np

from latent_drifting import apply_latent_drift


# ---------------------------------------------------------------------------
# DDIMInverter
# ---------------------------------------------------------------------------

class DDIMInverter:
    """
    Deterministic DDIM inversion (forward pass, reverse time direction).

    The inverted noise trajectory {zT, z_{T-1}, ..., z_0} is stored
    so that editing can start from any timestep.

    Latent Drifting is applied after each inversion step
    (mirroring its application in the reverse denoising process).
    """

    def __init__(
        self,
        unet,
        vae,
        tokenizer,
        text_encoder,
        scheduler_config: dict,
        num_inversion_steps: int = 50,
        guidance_scale: float = 1.0,       # typically 1.0 for inversion
        delta: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.unet         = unet
        self.vae          = vae
        self.tokenizer    = tokenizer
        self.text_encoder = text_encoder
        self.num_steps    = num_inversion_steps
        self.guidance_scale = guidance_scale
        self.delta        = delta
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype        = dtype

        # Build a dedicated DDIM scheduler for inversion
        self.scheduler = DDIMScheduler.from_config(scheduler_config)
        self.scheduler.set_timesteps(num_inversion_steps)

        self._img_transform = T.Compose([
            T.Resize(512, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to VAE latent space. Returns [1, 4, H/8, W/8]."""
        img_tensor = self._img_transform(image.convert("RGB"))
        img_tensor = img_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)
        latent = self.vae.encode(img_tensor).latent_dist.mean
        return latent * self.vae.config.scaling_factor

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Return CLIP text embedding [2, seq_len, dim] (uncond + cond)."""
        # Conditional
        cond_ids = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        cond_emb = self.text_encoder(cond_ids)[0]

        # Unconditional (empty string)
        uncond_ids = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        uncond_emb = self.text_encoder(uncond_ids)[0]

        return torch.cat([uncond_emb, cond_emb])   # [2, seq, dim]

    # ------------------------------------------------------------------
    # DDIM Inversion core
    # ------------------------------------------------------------------

    @torch.no_grad()
    def invert(
        self,
        image: Image.Image,
        source_prompt: str,
        return_all_latents: bool = True,
    ) -> List[torch.Tensor]:
        """
        Invert `image` to noise using DDIM (forward direction).

        The inversion runs from t=0 → t=T (increasing noise).
        At each step we apply the Latent Drift δ so the inverted
        trajectory is consistent with LD-based inference.

        Args:
            image           : Source PIL image (brain MRI with tumor).
            source_prompt   : Caption for the source image.
            return_all_latents : If True, return the full trajectory
                               [z_0, z_1, ..., z_T]; otherwise [z_T].

        Returns:
            List of latent tensors (each [1, 4, 64, 64]).
            Index 0  = z_0 (encoded image),
            Index -1 = z_T (fully inverted noise → use as start for editing).
        """
        # Encode image and prompt
        latents = self._encode_image(image)
        prompt_embeds = self._encode_prompt(source_prompt)

        all_latents: List[torch.Tensor] = [latents.clone()]

        # DDIM inversion: iterate over REVERSED timestep schedule
        # i.e. from small t to large t (more noise)
        timesteps_forward = reversed(self.scheduler.timesteps)

        for t in timesteps_forward:
            # Classifier-free guidance during inversion
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Compute next (noisier) latent via DDIM inversion formula
            latents = self._ddim_inversion_step(latents, noise_pred, t)

            # Apply Latent Drift to keep trajectory in LD-shifted space
            latents = apply_latent_drift(latents, self.delta)

            if return_all_latents:
                all_latents.append(latents.clone())

        if not return_all_latents:
            return [latents]

        return all_latents   # [z_0, ..., z_T]

    def _ddim_inversion_step(
        self,
        latents: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of DDIM inversion (forward time direction).

        Standard DDIM inversion:
            z_{t+1} = √ᾱ_{t+1} * [ (z_t - √(1-ᾱ_t)*ε̂) / √ᾱ_t ]
                    + √(1-ᾱ_{t+1}) * ε̂
        """
        scheduler = self.scheduler
        alphas    = scheduler.alphas_cumprod.to(self.device, dtype=self.dtype)

        # Clamp to valid range
        t_int = int(t.item())
        T_max = scheduler.config.num_train_timesteps
        t_next_int = min(
            t_int + T_max // self.num_steps,
            T_max - 1,
        )

        alpha_t      = alphas[t_int]
        alpha_t_next = alphas[t_next_int]

        # Predicted x0 (denoised)
        pred_x0 = (latents - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()

        # Next latent
        dir_xt  = (1 - alpha_t_next).sqrt() * noise_pred
        z_next  = alpha_t_next.sqrt() * pred_x0 + dir_xt

        return z_next

    # ------------------------------------------------------------------
    # Decode helper  (for debugging / visualisation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode a latent [1, 4, H/8, W/8] to PIL image."""
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.dtype), return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        image = (image * 255).round().astype(np.uint8)
        return Image.fromarray(image)
