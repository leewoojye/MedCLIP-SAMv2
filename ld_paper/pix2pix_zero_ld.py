"""
Pix2Pix Zero + Latent Drifting (LD) — Disease-conditioned Manipulation.

Paper Sections 4.2.2:
  "Disease-conditioned Manipulation: We use the Pix2Pix Zero model with a
   basic fine-tuned Stable Diffusion model to generate healthy brain MRIs
   from ones diagnosed with Alzheimer's Disease and vice versa.  We generate
   counterfactual images by negating the ground truth label of the 200 test
   samples and conditioning the model on the negated label value and the
   source image."

Adaptation for brain tumors:
  Source : a brain MRI with brain tumor  (positive label)
  Target : a healthy brain MRI           (negative label)

Algorithm
---------
1.  Load SD-v1.4 (optionally a LD-fine-tuned checkpoint).
2.  Register cross-attention store processors on UNet.
3.  For each source image:
    a. DDIM-invert the source image (with δ per Eq. 4).
    b. Run DDIM denoising forward with *source* text; collect A* maps.
    c. Run DDIM denoising with *target* text + cross-attention guidance:
           L_attn = ||A_target(t) – A*_source(t)||²_F
           latent ← latent + τ * ∇_{latent} L_attn
    d. Apply Latent Drift at every denoising step:
           z_{t-1} ← z_{t-1} + δ
4.  Decode final latent → counterfactual healthy brain MRI.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from attention_utils import (
    AttentionStore,
    build_attn_store_processors,
    cross_attention_guidance_loss,
)
from ddim_inversion import DDIMInverter
from latent_drifting import apply_latent_drift


# ---------------------------------------------------------------------------
# Pix2Pix Zero + LD  Pipeline
# ---------------------------------------------------------------------------

class Pix2PixZeroWithLD:
    """
    Disease-conditioned image manipulation using Pix2Pix Zero + Latent Drifting.

    Parameters
    ----------
    pipeline : StableDiffusionPipeline
        Loaded (and optionally fine-tuned) Stable Diffusion pipeline.
    delta : float
        Latent drift parameter δ (default 0.1 per paper's best setting).
    num_ddim_inversion_steps : int
        DDIM inversion steps T_inv.
    num_ddim_denoising_steps : int
        DDIM denoising steps T_den.
    guidance_scale : float
        Classifier-free guidance weight w.
    cross_attention_guidance_amount : float
        τ — weight of the cross-attention guidance gradient update.
    dtype : torch.dtype
        Half or full precision.
    """

    def __init__(
        self,
        pipeline: StableDiffusionPipeline,
        delta: float = 0.1,
        num_ddim_inversion_steps: int = 50,
        num_ddim_denoising_steps: int = 50,
        guidance_scale: float = 7.5,
        cross_attention_guidance_amount: float = 0.1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.pipeline   = pipeline
        self.unet       = pipeline.unet
        self.vae        = pipeline.vae
        self.tokenizer  = pipeline.tokenizer
        self.text_enc   = pipeline.text_encoder
        self.delta      = delta
        self.guidance_scale = guidance_scale
        self.tau        = cross_attention_guidance_amount
        self.dtype      = dtype
        self.device     = pipeline.device

        # ---- Denoising scheduler ----------------------------------------
        self.denoising_scheduler = DDIMScheduler.from_config(
            pipeline.scheduler.config
        )
        self.denoising_scheduler.set_timesteps(num_ddim_denoising_steps)
        self.num_den_steps = num_ddim_denoising_steps

        # ---- Cross-attention store (Pix2Pix Zero) -----------------------
        self.attn_store = AttentionStore()
        build_attn_store_processors(self.unet, self.attn_store)

        # ---- DDIM Inverter -----------------------------------------------
        self.inverter = DDIMInverter(
            unet=self.unet,
            vae=self.vae,
            tokenizer=self.tokenizer,
            text_encoder=self.text_enc,
            scheduler_config=pipeline.scheduler.config,
            num_inversion_steps=num_ddim_inversion_steps,
            guidance_scale=1.0,       # null-text inversion uses w=1
            delta=self.delta,
            device=self.device,
            dtype=self.dtype,
        )

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_prompt(
        self,
        prompt: str,
        do_classifier_free: bool = True,
    ) -> torch.Tensor:
        """Encode `prompt` to text embeddings.  Returns [1|2, seq, dim]."""
        ids = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        cond_emb = self.text_enc(ids)[0]

        if not do_classifier_free:
            return cond_emb

        uncond_ids = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        uncond_emb = self.text_enc(uncond_ids)[0]

        return torch.cat([uncond_emb, cond_emb])

    # ------------------------------------------------------------------
    # Source attention map collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _collect_source_attn_maps(
        self,
        inverted_latents: List[torch.Tensor],
        source_embeddings: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Reconstruct source image from inverted noise, capturing the
        cross-attention maps A* at each denoising step.

        Returns:
            dict[step_index -> dict[layer_key -> Tensor]]
        """
        source_attn_per_step: Dict[int, Dict[str, torch.Tensor]] = {}

        latents = inverted_latents[-1].clone()  # start from z_T
        self.denoising_scheduler.set_timesteps(self.num_den_steps)

        for step_idx, t in enumerate(self.denoising_scheduler.timesteps):
            self.attn_store.reset()
            with self.attn_store:     # activates capture
                latent_input = torch.cat([latents] * 2)
                latent_input = self.denoising_scheduler.scale_model_input(latent_input, t)

                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=source_embeddings,
                    return_dict=False,
                )[0]

            # Store the maps for this step
            source_attn_per_step[step_idx] = self.attn_store.get_maps()

            # Apply CFG
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Step
            latents = self.denoising_scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
            # Apply LD
            latents = apply_latent_drift(latents, self.delta)

        return source_attn_per_step

    # ------------------------------------------------------------------
    # Editing denoising loop with cross-attention guidance + LD
    # ------------------------------------------------------------------

    def _denoising_loop_with_guidance(
        self,
        start_latents: torch.Tensor,
        target_embeddings: torch.Tensor,
        source_attn_per_step: Dict[int, Dict[str, torch.Tensor]],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """
        DDIM denoising from z_T to z_0 using:
          (1) Target text embedding (disease label swap)
          (2) Cross-attention guidance τ·∇L_attn  (Pix2Pix Zero)
          (3) Latent Drift +δ at every step         (LD, Eq. 4)
        """
        latents = start_latents.clone()
        self.denoising_scheduler.set_timesteps(self.num_den_steps)

        for step_idx, t in enumerate(self.denoising_scheduler.timesteps):

            # ------- Forward pass (with gradient for guidance) ----------
            latents = latents.detach().requires_grad_(True)

            latent_input = torch.cat([latents] * 2)
            latent_input = self.denoising_scheduler.scale_model_input(latent_input, t)

            # Capture target attention maps at this step
            self.attn_store.reset()

            # ------- Cross-attention guidance (Pix2Pix Zero) ------------
            # Only applied in the first half of steps (keeps low-frequency)
            src_maps = source_attn_per_step.get(step_idx, {}) if self.tau > 0 else {}
            need_guidance = self.tau > 0 and step_idx < self.num_den_steps // 2 and bool(src_maps)

            with self.attn_store, torch.enable_grad():
                noise_pred = self.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=target_embeddings,
                    return_dict=False,
                )[0]

                target_maps = self.attn_store.get_maps()

                # ------- Classifier-free guidance ---------------------------
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_cfg = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                if need_guidance and target_maps:
                    loss = cross_attention_guidance_loss(
                        src_maps, target_maps, self.device
                    )
                    # Gradient of the attention loss w.r.t. the latents
                    grad = torch.autograd.grad(loss, latents, retain_graph=False)[0]
                    # Update noise prediction (equivalent to nudging latents)
                    noise_pred_cfg = noise_pred_cfg - self.tau * grad.detach()

            # ------- DDIM denoising step --------------------------------
            latents = self.denoising_scheduler.step(
                noise_pred_cfg.detach(),
                t,
                latents.detach(),
                generator=generator,
                return_dict=False,
            )[0]

            # ------- Latent Drifting  (Eq. 4) ---------------------------
            # pθ(xt-1|xt) = N(xt-1 ; µθ(xt,t) + δ,  Σ)
            latents = apply_latent_drift(latents, self.delta)

        return latents.detach()

    # ------------------------------------------------------------------
    # VAE decode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        latents = latents / self.vae.config.scaling_factor
        img = self.vae.decode(latents.to(self.dtype), return_dict=False)[0]
        img = (img / 2 + 0.5).clamp(0, 1)
        img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        img = (img * 255).round().astype(np.uint8)
        pil = Image.fromarray(img)
        # MRI 이미지는 grayscale: RGB 평균으로 단채널 변환 후 RGB로 돌려줌
        # (녹색/채색 아티팩트 방지)
        pil = pil.convert("L").convert("RGB")
        return pil

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def edit(
        self,
        source_image: Image.Image,
        source_prompt: str,
        target_prompt: str,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Perform disease-conditioned manipulation.

        Converts `source_image` (brain MRI with tumor) to a counterfactual
        healthy brain MRI by:
          DDIM Inversion  →  Cross-attention guidance denoising with LD.

        Args:
            source_image   : PIL image (brain MRI, positive/tumor label).
            source_prompt  : Text describing the source (e.g. "a brain MRI with tumor").
            target_prompt  : Text describing the target (e.g. "a healthy brain MRI").
            seed           : RNG seed for reproducibility.
            verbose        : Print progress messages.

        Returns:
            (edited_image, z0_latents)
        """
        generator = torch.Generator(self.device).manual_seed(seed)

        if verbose:
            print(f"  Source: {source_prompt}")
            print(f"  Target: {target_prompt}")
            print(f"  δ = {self.delta:+.3f}  |  τ = {self.tau}  |  w = {self.guidance_scale}")

        # 1. Encode text embeddings
        source_embeddings = self._encode_prompt(source_prompt, do_classifier_free=True)
        target_embeddings = self._encode_prompt(target_prompt, do_classifier_free=True)

        # 2. DDIM inversion (forward pass, reverse time direction)
        if verbose:
            print("  [1/3] DDIM inversion …")
        inverted_latents = self.inverter.invert(
            image=source_image,
            source_prompt=source_prompt,
            return_all_latents=True,
        )
        # inverted_latents[-1] = z_T (fully noised, with LD shift)

        # 3. Reconstruct source to collect A* cross-attention maps
        if verbose:
            print("  [2/3] Collecting source cross-attention maps …")
        source_attn_per_step = self._collect_source_attn_maps(
            inverted_latents, source_embeddings
        )

        # 4. Edit: denoise with target text + attention guidance + LD
        if verbose:
            print("  [3/3] Denoising with target prompt + LD …")
        z0_edited = self._denoising_loop_with_guidance(
            start_latents=inverted_latents[-1],
            target_embeddings=target_embeddings,
            source_attn_per_step=source_attn_per_step,
            generator=generator,
        )

        # 5. Decode
        edited_image = self._decode_latents(z0_edited)

        if verbose:
            print("  Done.")

        return edited_image, z0_edited


# ---------------------------------------------------------------------------
# Convenience: load a pipeline, optionally from a local LD-fine-tuned dir
# ---------------------------------------------------------------------------

def _is_accelerate_checkpoint(path: str) -> bool:
    """accelerate save_state() 형식인지 확인 (optimizer 상태 파일 존재)."""
    import os
    p = Path(path)
    return (p / "optimizer.bin").exists() or (p / "optimizer_states.bin").exists() or \
           any(p.glob("optimizer*"))


def load_pipeline(
    model_id_or_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    base_model_id: str = "CompVis/stable-diffusion-v1-4",
) -> StableDiffusionPipeline:
    """Load a Stable Diffusion pipeline (base, fine-tuned, or accelerate checkpoint).

    accelerate 중간 체크포인트(checkpoint-N/)인 경우:
      base_model_id로 파이프라인을 로드한 뒤 U-Net 가중치만 교체합니다.
    """
    import os
    from safetensors.torch import load_file as safetensors_load

    path = Path(model_id_or_path)
    if path.exists() and _is_accelerate_checkpoint(model_id_or_path):
        # 1) 베이스 파이프라인 로드
        pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        # 2) 체크포인트에서 U-Net 가중치 로드
        unet_safetensors = path / "model.safetensors"
        unet_bin = path / "pytorch_model.bin"
        if unet_safetensors.exists():
            state_dict = safetensors_load(str(unet_safetensors))
        elif unet_bin.exists():
            state_dict = torch.load(str(unet_bin), map_location="cpu")
        else:
            # accelerate는 sub-폴더에 저장하기도 함
            unet_dir = path / "unet"
            if (unet_dir / "model.safetensors").exists():
                state_dict = safetensors_load(str(unet_dir / "model.safetensors"))
            elif (unet_dir / "pytorch_model.bin").exists():
                state_dict = torch.load(str(unet_dir / "pytorch_model.bin"), map_location="cpu")
            else:
                raise FileNotFoundError(
                    f"U-Net 가중치 파일을 찾을 수 없습니다: {path}"
                )
        missing, unexpected = pipeline.unet.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[경고] 누락된 키 {len(missing)}개 (일반적으로 무해)")
        print(f"[로드 완료] accelerate 체크포인트에서 U-Net 가중치 교체: {path.name}")
    else:
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipeline = pipeline.to(device)
    pipeline.enable_attention_slicing()
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipeline
