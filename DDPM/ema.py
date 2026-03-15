from __future__ import annotations

from copy import deepcopy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = dict(model.named_parameters())
        for name, shadow_param in self.shadow.named_parameters():
            shadow_param.data.mul_(self.decay).add_(
                model_state[name].data, alpha=1.0 - self.decay
            )

        model_buffers = dict(model.named_buffers())
        for name, shadow_buffer in self.shadow.named_buffers():
            shadow_buffer.copy_(model_buffers[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict["decay"])
        self.shadow.load_state_dict(state_dict["shadow"])

