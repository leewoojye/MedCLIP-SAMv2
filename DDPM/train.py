from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .data import BrainTumorInpaintingDataset
from .ema import EMA
from .model import PaperModelConfig, build_paper_diffusion, build_paper_unet
from .utils import (
    blend_prediction,
    count_parameters,
    format_step,
    save_debug_montage,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the paper-style 2D DDPM for brain tissue inpainting."
    )
    parser.add_argument("--train-image-dir", required=True)
    parser.add_argument("--train-mask-dir", required=True)
    parser.add_argument("--val-image-dir")
    parser.add_argument("--val-mask-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=2_850_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", default="linear")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=2000)
    parser.add_argument(
        "--train-mask-mode",
        choices=["provided", "synthetic_healthy"],
        default="synthetic_healthy",
    )
    parser.add_argument(
        "--val-mask-mode",
        choices=["provided", "synthetic_healthy"],
        default="synthetic_healthy",
    )
    parser.add_argument(
        "--hide-tumor-in-context",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hole-loss-weight", type=float, default=1.0)
    parser.add_argument("--context-loss-weight", type=float, default=0.05)
    parser.add_argument("--tumor-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--learn-sigma", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BrainTumorInpaintingDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        crop_size=args.crop_size,
        include_target=True,
        mask_mode=args.train_mask_mode,
        hide_tumor_in_context=args.hide_tumor_in_context,
        max_samples=args.max_train_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= args.batch_size,
    )

    val_dataset = None
    if args.val_image_dir and args.val_mask_dir:
        val_dataset = BrainTumorInpaintingDataset(
            image_dir=args.val_image_dir,
            mask_dir=args.val_mask_dir,
            crop_size=args.crop_size,
            include_target=True,
            mask_mode=args.val_mask_mode,
            hide_tumor_in_context=args.hide_tumor_in_context,
            base_seed=args.seed + 10_000,
            max_samples=args.max_val_samples,
        )

    model_config = PaperModelConfig(
        image_size=args.crop_size,
        use_checkpoint=args.use_checkpoint,
        learn_sigma=args.learn_sigma,
    )
    model = build_paper_unet(model_config).to(device)
    diffusion = build_paper_diffusion(
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    ema = EMA(model, decay=args.ema_decay)

    global_step = 0
    if args.resume:
        global_step = load_checkpoint(
            checkpoint_path=Path(args.resume),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
        )

    config_path = output_dir / "paper_config.json"
    config_path.write_text(
        json.dumps(
            {
                **vars(args),
                "device": str(device),
                "parameter_count": count_parameters(model),
            },
            indent=2,
        )
    )

    print("Training paper-style DDPM")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}")
    print(f"Model parameters: {count_parameters(model):,}")

    data_iter = iter(train_loader)
    running_loss = 0.0

    while global_step < args.train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        model.train()
        model_input = batch["model_input"].to(device, non_blocking=True)
        loss_weights = build_loss_weights(
            loss_mask=batch["loss_mask"].to(device, non_blocking=True),
            tumor_mask=batch["tumor_mask"].to(device, non_blocking=True),
            hole_weight=args.hole_loss_weight,
            context_weight=args.context_loss_weight,
            tumor_weight=args.tumor_loss_weight,
        )
        timesteps = torch.randint(
            low=0,
            high=diffusion.num_timesteps,
            size=(model_input.shape[0],),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=scaler.is_enabled()):
            terms, _ = diffusion.training_losses_segmentation(
                model=model,
                classifier=None,
                x_start=model_input,
                t=timesteps,
                loss_weights=loss_weights,
            )
            loss = terms["loss"].mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        global_step += 1
        running_loss += float(loss.item())

        if global_step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            print(f"step={global_step} loss={avg_loss:.6f}")
            running_loss = 0.0

        if val_dataset is not None and global_step % args.sample_every == 0:
            save_preview(
                diffusion=diffusion,
                model=ema.shadow,
                dataset=val_dataset,
                output_path=preview_dir / f"step_{format_step(global_step)}.png",
                device=device,
            )

        if global_step % args.save_every == 0 or global_step == args.train_steps:
            save_checkpoint(
                checkpoint_path=checkpoint_dir / f"step_{format_step(global_step)}.pt",
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                args=args,
            )


@torch.no_grad()
def save_preview(
    diffusion,
    model: torch.nn.Module,
    dataset: BrainTumorInpaintingDataset,
    output_path: Path,
    device: torch.device,
) -> None:
    sample = dataset[0]
    model.eval()
    conditioned = sample["model_input"].unsqueeze(0).to(device)
    generated, _, _ = diffusion.p_sample_loop_known(
        model=model,
        shape=(1, conditioned.shape[1], conditioned.shape[2], conditioned.shape[3]),
        img=conditioned,
        clip_denoised=True,
        progress=False,
    )
    baseline = sample["baseline"].squeeze(0).cpu().numpy()
    mask = sample["mask"].squeeze(0).cpu().numpy()
    tumor_mask = sample["tumor_mask"].squeeze(0).cpu().numpy()
    target = sample["target"].squeeze(0).cpu().numpy()
    raw = generated.squeeze(0).squeeze(0).cpu().numpy()
    blended = blend_prediction(raw, baseline, mask)

    save_debug_montage(
        [
            ("baseline", baseline),
            ("hole_mask", mask),
            ("tumor_mask", tumor_mask),
            ("target", target),
            ("raw", raw),
            ("blended", blended),
        ],
        output_path,
    )

    masked_pixels = float(np.maximum(mask.sum(), 1.0))
    preview_stats = {
        "sample_name": sample["name"],
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "hole_fraction": float(mask.mean()),
        "tumor_fraction": float(tumor_mask.mean()),
        "masked_mse_raw": float((((raw - target) ** 2) * mask).sum() / masked_pixels),
        "masked_mse_blended": float(
            (((blended - target) ** 2) * mask).sum() / masked_pixels
        ),
    }
    output_path.with_suffix(".json").write_text(json.dumps(preview_stats, indent=2))


def build_loss_weights(
    loss_mask: torch.Tensor,
    tumor_mask: torch.Tensor,
    hole_weight: float,
    context_weight: float,
    tumor_weight: float,
) -> torch.Tensor:
    context_mask = (1.0 - loss_mask - tumor_mask).clamp(min=0.0, max=1.0)
    return (
        hole_weight * loss_mask
        + context_weight * context_mask
        + tumor_weight * tumor_mask
    )


def save_checkpoint(
    checkpoint_path: Path,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "global_step": global_step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    ema.load_state_dict(checkpoint["ema"])
    print(f"Resumed from {checkpoint_path}")
    return int(checkpoint["global_step"])


if __name__ == "__main__":
    main()
