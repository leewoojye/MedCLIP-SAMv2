"""
Cross-Attention utilities for Pix2Pix Zero guidance.

Pix2Pix Zero (Parmar et al., 2023) preserves image structure during editing
by minimising the difference between source and target cross-attention maps
at each DDIM denoising step.

Architecture:
  AttentionStore   – captures cross-attention probability maps via diffusers
                     attention processors during UNet forward passes.
  AttnStoreProcessor – custom diffusers AttnProcessor that records maps.
  build_attn_store_processors – wires AttnStoreProcessor into every
                     cross-attention layer of a UNet.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


# ---------------------------------------------------------------------------
# Attention Map Storage
# ---------------------------------------------------------------------------

class AttentionStore:
    """
    Thread-safe store for cross-attention probability maps.

    Usage
    -----
    store = AttentionStore()
    # (wire processors – see build_attn_store_processors)
    with store.capture():
        _ = unet(...)
    maps = store.get_maps()   # dict[layer_key -> Tensor[H*W, seq_len]]
    store.reset()
    """

    def __init__(self) -> None:
        self._maps: Dict[str, List[torch.Tensor]] = {}
        self._active: bool = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def capture(self) -> "AttentionStore":
        """Enable map capture for the duration of a `with` block."""
        return self

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, *_):
        self._active = False

    # ------------------------------------------------------------------
    # Internal API (called by AttnStoreProcessor)
    # ------------------------------------------------------------------

    def record(self, layer_key: str, attn_probs: torch.Tensor) -> None:
        """
        Save cross-attention probability tensor.

        attn_probs : shape [batch*heads, H*W, seq_len]
        """
        if not self._active:
            return
        if layer_key not in self._maps:
            self._maps[layer_key] = []
        # If gradient is needed (requires_grad), keep on GPU with grad.
        # Otherwise detach to CPU to save memory (source maps).
        if attn_probs.requires_grad or torch.is_grad_enabled() and attn_probs.grad_fn is not None:
            self._maps[layer_key].append(attn_probs)
        else:
            self._maps[layer_key].append(attn_probs.detach().cpu())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored maps."""
        self._maps = {}

    def get_maps(self) -> Dict[str, torch.Tensor]:
        """
        Return averaged cross-attention maps per layer.

        Returns:
            dict[layer_key -> Tensor[batch*heads, H*W, seq_len]]
        """
        out: Dict[str, torch.Tensor] = {}
        for key, tensors in self._maps.items():
            out[key] = torch.stack(tensors).mean(0)
        return out

    def is_empty(self) -> bool:
        return len(self._maps) == 0


# ---------------------------------------------------------------------------
# Custom Attention Processor
# ---------------------------------------------------------------------------

class AttnStoreProcessor:
    """
    Drop-in replacement for diffusers' default AttnProcessor.

    Records cross-attention probability maps (is_cross=True only)
    into the supplied AttentionStore under `layer_key`.
    """

    def __init__(self, store: AttentionStore, layer_key: str) -> None:
        self.store = store
        self.layer_key = layer_key

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        kv_states = encoder_hidden_states if is_cross else hidden_states
        if attn.norm_cross:
            kv_states = attn.norm_encoder_hidden_states(kv_states)

        key   = attn.to_k(kv_states)
        value = attn.to_v(kv_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # ---- attention scores ----------------------------------------
        attn_probs = attn.get_attention_scores(query, key, attention_mask)

        # ---- record cross-attention maps -----------------------------
        if is_cross:
            self.store.record(self.layer_key, attn_probs)

        # ---- weighted sum --------------------------------------------
        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Wiring helper
# ---------------------------------------------------------------------------

def build_attn_store_processors(
    unet,
    store: AttentionStore,
) -> Dict[str, AttnStoreProcessor]:
    """
    Replace every cross-attention processor in `unet` with an
    AttnStoreProcessor that logs to `store`.

    Returns the dict of processors so callers can restore the originals.

    Call this once after loading the pipeline:
        processors = build_attn_store_processors(pipeline.unet, store)
    """
    processors: Dict[str, AttnStoreProcessor] = {}
    attn_procs: Dict = {}

    for name, module in unet.attn_processors.items():
        # Cross-attention modules contain "attn2" in their key
        if "attn2" in name:
            proc = AttnStoreProcessor(store, layer_key=name)
            attn_procs[name] = proc
            processors[name] = proc
        else:
            attn_procs[name] = module   # keep existing self-attention processor

    unet.set_attn_processor(attn_procs)
    return processors


# ---------------------------------------------------------------------------
# Cross-attention guidance loss  (Pix2Pix Zero objective)
# ---------------------------------------------------------------------------

def cross_attention_guidance_loss(
    source_maps: Dict[str, torch.Tensor],
    target_maps: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the cross-attention guidance loss used in Pix2Pix Zero.

    L_attn = (1/N) Σ_layers  || A_target(l) - A_source(l) ||²_F

    Where A(l) ∈ R^{H*W × seq_len} is the cross-attention probability
    map at layer l.  Source maps are detached (treated as constants).

    Args:
        source_maps : dict from get_maps() during source text run.
        target_maps : dict from get_maps() during target text run (requires_grad).
        device      : computation device.

    Returns:
        Scalar loss tensor (with grad).
    """
    loss = torch.zeros(1, device=device, requires_grad=False)
    count = 0

    for key in source_maps:
        if key not in target_maps:
            continue
        src = source_maps[key].to(device).detach()
        tgt = target_maps[key].to(device)
        loss = loss + F.mse_loss(tgt, src)
        count += 1

    if count > 0:
        loss = loss / count

    return loss
