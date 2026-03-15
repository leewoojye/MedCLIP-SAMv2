from __future__ import annotations

from dataclasses import dataclass

from .guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from .guided_diffusion.unet import UNetModel


@dataclass
class PaperModelConfig:
    image_size: int = 224
    in_channels: int = 3
    model_channels: int = 128
    out_channels: int = 1
    num_res_blocks: int = 2
    channel_mult: tuple[int, ...] = (1, 1, 2, 2, 4, 4)
    attention_downsample_rates: tuple[int, ...] = (16,)
    dropout: float = 0.0
    num_heads: int = 1
    num_head_channels: int = -1
    use_checkpoint: bool = False
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    use_fp16: bool = False
    learn_sigma: bool = False


def build_paper_unet(config: PaperModelConfig) -> UNetModel:
    out_channels = 2 if config.learn_sigma else config.out_channels
    return UNetModel(
        image_size=config.image_size,
        in_channels=config.in_channels,
        model_channels=config.model_channels,
        out_channels=out_channels,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_downsample_rates,
        dropout=config.dropout,
        channel_mult=config.channel_mult,
        use_checkpoint=config.use_checkpoint,
        use_fp16=config.use_fp16,
        num_heads=config.num_heads,
        num_head_channels=config.num_head_channels,
        num_heads_upsample=config.num_heads,
        use_scale_shift_norm=config.use_scale_shift_norm,
        resblock_updown=config.resblock_updown,
    )


def build_paper_diffusion(
    diffusion_steps: int = 1000,
    noise_schedule: str = "linear",
    learn_sigma: bool = False,
) -> GaussianDiffusion:
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    model_var_type = (
        ModelVarType.LEARNED_RANGE if learn_sigma else ModelVarType.FIXED_SMALL
    )
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=model_var_type,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )

