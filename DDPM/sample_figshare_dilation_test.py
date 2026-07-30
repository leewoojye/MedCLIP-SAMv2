from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from .model import build_paper_diffusion
from .sample_figshare_sharded import (
    atomic_save_png,
    load_model,
    resize_float_image,
    resize_float_to_shape,
    seed_sample,
    tumor_aware_roi,
)
from .utils import (
    binarize_mask,
    gaussian_blur_2d,
    load_grayscale_image,
    normalize_to_unit_interval,
    resize_mask_to_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate matched Figshare samples for a mask dilation test."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-names", nargs="+", required=True)
    parser.add_argument("--dilation-size", type=int, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context-margin", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--blur-sigma", type=float, default=1.075)
    return parser.parse_args()


def dilate_mask(mask: np.ndarray, dilation_size: int) -> np.ndarray:
    if dilation_size == 0:
        return mask.astype(np.float32, copy=True)
    if dilation_size < 3 or dilation_size % 2 == 0:
        raise ValueError("--dilation-size must be 0 or an odd integer of at least 3.")
    rendered = Image.fromarray((mask > 0.5).astype(np.uint8) * 255)
    dilated = rendered.filter(ImageFilter.MaxFilter(size=dilation_size))
    return (np.asarray(dilated, dtype=np.uint8) > 0).astype(np.float32)


def load_test_sample(
    image_path: Path,
    mask_path: Path,
    crop_size: int,
    context_margin: int,
    dilation_size: int,
) -> dict[str, object]:
    original = normalize_to_unit_interval(load_grayscale_image(image_path))
    original_mask = resize_mask_to_shape(
        load_grayscale_image(mask_path),
        original.shape,
    )
    original_mask = binarize_mask(original_mask)

    # Keep the ROI identical across dilation variants. The context margin is
    # wider than the largest tested dilation radius, so the expanded mask fits.
    top, left, bottom, right = tumor_aware_roi(
        mask=original_mask,
        crop_size=crop_size,
        context_margin=context_margin,
    )
    applied_mask = dilate_mask(original_mask, dilation_size)
    roi_mask_pixels = int(
        (applied_mask[top:bottom, left:right] > 0.5).sum()
    )
    if roi_mask_pixels != int((applied_mask > 0.5).sum()):
        raise ValueError(
            f"Dilated mask does not fit the fixed ROI for {image_path.name}."
        )

    image_roi = original[top:bottom, left:right]
    mask_roi = applied_mask[top:bottom, left:right]
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
        "original_mask": original_mask,
        "applied_mask": applied_mask,
        "roi": (top, left, bottom, right),
    }


def sample_seed(base_seed: int, name: str) -> int:
    stem = Path(name).stem
    return base_seed + int(stem) - 1 if stem.isdigit() else base_seed


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the dilation test.")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    blended_dir = output_dir / "blended"
    applied_mask_dir = output_dir / "applied_masks"
    for directory in (raw_dir, blended_dir, applied_mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for name in args.sample_names:
        if not (image_dir / name).is_file() or not (mask_dir / name).is_file():
            raise FileNotFoundError(f"Missing paired sample: {name}")

    properties = torch.cuda.get_device_properties(0)
    print(
        f"physical_gpu={args.physical_gpu} device={properties.name} "
        f"dilation={args.dilation_size} samples={len(args.sample_names)}",
        flush=True,
    )
    model = load_model(
        checkpoint_path=Path(args.checkpoint),
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
    print("checkpoint loaded", flush=True)

    started_at = time.monotonic()
    for position, name in enumerate(args.sample_names, start=1):
        seed_sample(sample_seed(args.seed, name))
        sample = load_test_sample(
            image_path=image_dir / name,
            mask_path=mask_dir / name,
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

        atomic_save_png(raw, raw_dir / name, args.dilation_size)
        atomic_save_png(blended, blended_dir / name, args.dilation_size)
        atomic_save_png(
            applied_mask,
            applied_mask_dir / name,
            args.dilation_size,
        )
        elapsed = time.monotonic() - started_at
        print(
            f"dilation={args.dilation_size} position={position}/"
            f"{len(args.sample_names)} file={name} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
