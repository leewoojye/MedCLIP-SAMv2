"""Minimal fp16 helpers used by the OpenAI-style U-Net module."""

import torch.nn as nn


def convert_module_to_f16(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


def convert_module_to_f32(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()

