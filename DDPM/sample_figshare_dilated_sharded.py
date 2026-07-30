from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

from .model import build_paper_diffusion
from .sample_figshare_dilation_test import load_test_sample
from .sample_figshare_sharded import (
    atomic_save_png,
    collect_pairs,
    load_model,
    resize_float_to_shape,
    seed_sample,
)
from .utils import gaussian_blur_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dilated-mask Figshare samples with one GPU shard."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--dilation-size", type=int, default=9)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context-margin", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--blur-sigma", type=float, default=1.075)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be at least 1.")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, world_size).")
    if args.dilation_size < 3 or args.dilation_size % 2 == 0:
        raise ValueError("--dilation-size must be an odd integer of at least 3.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sharded sampler.")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    blended_dir = output_dir / "blended"
    raw_dir.mkdir(parents=True, exist_ok=True)
    blended_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(image_dir, mask_dir)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    assigned_indices = list(range(args.rank, len(pairs), args.world_size))

    properties = torch.cuda.get_device_properties(0)
    print(
        f"[rank {args.rank}] physical_gpu={args.physical_gpu} "
        f"device={properties.name} samples={len(assigned_indices)} "
        f"dilation={args.dilation_size} weights={args.weights}",
        flush=True,
    )
    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        crop_size=args.crop_size,
        learn_sigma=False,
        weights=args.weights,
    )
    diffusion = build_paper_diffusion(
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=False,
    )
    print(f"[rank {args.rank}] checkpoint loaded", flush=True)

    failures: list[tuple[str, str]] = []
    completed = 0
    skipped = 0
    started_at = time.monotonic()
    for local_index, dataset_index in enumerate(assigned_indices, start=1):
        image_path, mask_path = pairs[dataset_index]
        raw_path = raw_dir / image_path.name
        blended_path = blended_dir / image_path.name
        if args.skip_existing and raw_path.exists() and blended_path.exists():
            skipped += 1
            continue

        try:
            seed_sample(args.seed + dataset_index)
            sample = load_test_sample(
                image_path=image_path,
                mask_path=mask_path,
                crop_size=args.crop_size,
                context_margin=args.context_margin,
                dilation_size=args.dilation_size,
            )
            conditioned = sample["conditioned"].unsqueeze(0).to(device)
            with torch.inference_mode():
                generated, _, _ = diffusion.p_sample_loop_known(
                    model=model,
                    shape=(
                        1,
                        conditioned.shape[1],
                        conditioned.shape[2],
                        conditioned.shape[3],
                    ),
                    img=conditioned,
                    clip_denoised=True,
                    progress=False,
                )

            prediction = generated[0, 0].float().cpu().numpy()
            if args.blur_sigma > 0:
                prediction = gaussian_blur_2d(prediction, args.blur_sigma)

            original = sample["original"]
            applied_mask = sample["applied_mask"]
            top, left, bottom, right = sample["roi"]
            roi_shape = (bottom - top, right - left)
            restored_prediction = resize_float_to_shape(
                prediction,
                shape=roi_shape,
                resample=Image.Resampling.BILINEAR,
            )
            raw = original.copy()
            raw[top:bottom, left:right] = restored_prediction
            blended = original.copy()
            blended_roi = blended[top:bottom, left:right]
            mask_roi = applied_mask[top:bottom, left:right] > 0.5
            blended_roi[mask_roi] = restored_prediction[mask_roi]

            atomic_save_png(raw, raw_path, args.rank)
            atomic_save_png(blended, blended_path, args.rank)
            completed += 1
            if completed % args.log_every == 0 or local_index == len(assigned_indices):
                elapsed = time.monotonic() - started_at
                print(
                    f"[rank {args.rank}] completed={completed} skipped={skipped} "
                    f"position={local_index}/{len(assigned_indices)} "
                    f"file={image_path.name} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
        except Exception as exc:
            failures.append((image_path.name, f"{type(exc).__name__}: {exc}"))
            print(
                f"[rank {args.rank}] FAILED file={image_path.name}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            torch.cuda.empty_cache()

    elapsed = time.monotonic() - started_at
    print(
        f"[rank {args.rank}] finished completed={completed} skipped={skipped} "
        f"failed={len(failures)} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    if failures:
        summary = ", ".join(f"{name} ({reason})" for name, reason in failures[:10])
        raise RuntimeError(
            f"{len(failures)} sample(s) failed on rank {args.rank}: {summary}"
        )


if __name__ == "__main__":
    main()
