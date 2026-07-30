from __future__ import annotations

import argparse
import gc
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import PaperModelConfig, build_paper_diffusion, build_paper_unet
from .utils import (
    binarize_mask,
    gaussian_blur_2d,
    load_grayscale_image,
    normalize_to_unit_interval,
    resize_mask_to_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tumor-to-normal Figshare samples with one independently "
            "launched GPU worker."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--physical-gpu", default="unknown")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context-margin", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--blur-sigma", type=float, default=1.075)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def collect_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    image_paths = {path.name: path for path in image_dir.glob("*.png")}
    mask_paths = {path.name: path for path in mask_dir.glob("*.png")}
    shared_names = sorted(
        image_paths.keys() & mask_paths.keys(),
        key=lambda name: natural_key(Path(name)),
    )
    if not shared_names:
        raise FileNotFoundError(
            f"No paired PNG files found in {image_dir} and {mask_dir}."
        )
    if len(shared_names) != len(image_paths) or len(shared_names) != len(mask_paths):
        raise ValueError(
            "Image/mask filenames do not match exactly: "
            f"images={len(image_paths)}, masks={len(mask_paths)}, "
            f"paired={len(shared_names)}."
        )
    return [(image_paths[name], mask_paths[name]) for name in shared_names]


def tumor_aware_roi(
    mask: np.ndarray,
    crop_size: int,
    context_margin: int,
) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(mask > 0.5)
    if coordinates.size == 0:
        raise ValueError("Tumor mask is empty.")

    height, width = mask.shape
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1
    box_height = int(y_max - y_min)
    box_width = int(x_max - x_min)
    side = max(
        crop_size,
        box_height + 2 * context_margin,
        box_width + 2 * context_margin,
    )
    side = min(side, height, width)

    center_y = (float(y_min) + float(y_max)) / 2.0
    center_x = (float(x_min) + float(x_max)) / 2.0
    top = int(round(center_y - side / 2.0))
    left = int(round(center_x - side / 2.0))
    top = min(max(top, 0), height - side)
    left = min(max(left, 0), width - side)
    bottom = top + side
    right = left + side

    roi_mask = mask[top:bottom, left:right]
    if int((roi_mask > 0.5).sum()) != int((mask > 0.5).sum()):
        raise ValueError(
            "Tumor-aware ROI does not contain the complete tumor mask: "
            f"bbox=({x_min}, {y_min}, {x_max}, {y_max}), "
            f"roi=({left}, {top}, {right}, {bottom})."
        )
    return top, left, bottom, right


def resize_float_image(
    array: np.ndarray,
    output_size: int,
    resample: Image.Resampling,
) -> np.ndarray:
    if array.shape == (output_size, output_size):
        return array.astype(np.float32, copy=True)
    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((output_size, output_size), resample=resample)
    return np.asarray(resized, dtype=np.float32).copy()


def resize_float_to_shape(
    array: np.ndarray,
    shape: tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    if array.shape == shape:
        return array.astype(np.float32, copy=True)
    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((shape[1], shape[0]), resample=resample)
    return np.asarray(resized, dtype=np.float32).copy()


def load_conditioned_sample(
    image_path: Path,
    mask_path: Path,
    crop_size: int,
    context_margin: int,
) -> dict[str, object]:
    original = normalize_to_unit_interval(load_grayscale_image(image_path))
    mask = resize_mask_to_shape(load_grayscale_image(mask_path), original.shape)
    mask = binarize_mask(mask)
    top, left, bottom, right = tumor_aware_roi(
        mask=mask,
        crop_size=crop_size,
        context_margin=context_margin,
    )

    image_roi = original[top:bottom, left:right]
    mask_roi = mask[top:bottom, left:right]
    model_image = resize_float_image(
        image_roi,
        output_size=crop_size,
        resample=Image.Resampling.BILINEAR,
    )
    model_mask = resize_float_image(
        mask_roi,
        output_size=crop_size,
        resample=Image.Resampling.NEAREST,
    )
    model_mask = binarize_mask(model_mask)
    baseline = model_image.copy()
    baseline[model_mask > 0.5] = 0.0
    conditioned = np.stack(
        [baseline, model_mask, np.zeros_like(model_image)],
        axis=0,
    )
    return {
        "conditioned": torch.from_numpy(conditioned).float(),
        "original": original,
        "mask": mask,
        "roi": (top, left, bottom, right),
    }


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    crop_size: int,
    learn_sigma: bool,
    weights: str,
) -> torch.nn.Module:
    model = build_paper_unet(
        PaperModelConfig(image_size=crop_size, learn_sigma=learn_sigma)
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state_dict = (
        checkpoint["ema"]["shadow"] if weights == "ema" else checkpoint["model"]
    )
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    del checkpoint
    gc.collect()
    model.eval()
    model.requires_grad_(False)
    return model


def atomic_save_png(array: np.ndarray, path: Path, rank: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.rank{rank}.tmp")
    image = np.clip(array, 0.0, 1.0)
    rendered = Image.fromarray((image * 255.0).round().astype(np.uint8))
    rendered.save(temporary_path, format="PNG")
    temporary_path.replace(path)


def seed_sample(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main() -> None:
    args = parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be at least 1.")
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, world_size).")
    if args.context_margin < 0:
        raise ValueError("--context-margin must be non-negative.")
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
        f"weights={args.weights}",
        flush=True,
    )
    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        crop_size=args.crop_size,
        learn_sigma=args.learn_sigma,
        weights=args.weights,
    )
    diffusion = build_paper_diffusion(
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
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
            sample = load_conditioned_sample(
                image_path=image_path,
                mask_path=mask_path,
                crop_size=args.crop_size,
                context_margin=args.context_margin,
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
            full_mask = sample["mask"]
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
            mask_roi = full_mask[top:bottom, left:right] > 0.5
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
