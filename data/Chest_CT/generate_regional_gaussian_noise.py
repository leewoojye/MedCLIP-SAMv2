from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


REGIONS = {
    "bronchus": {"channel": 0, "seed_offset": 0},
    "heart": {"channel": 1, "seed_offset": 1},
    "lung": {"channel": 2, "seed_offset": 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply independent RGB Gaussian noise only inside color-coded "
            "Chest CT regions."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--sigma", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=20250728)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_region_labels(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mask_rgb = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.int16)
    sorted_channels = np.sort(mask_rgb, axis=2)
    strongest = sorted_channels[:, :, 2]
    second = sorted_channels[:, :, 1]
    labels = np.argmax(mask_rgb, axis=2)

    # JPEG compression introduces colored ringing at mask borders. Retain
    # confident colored pixels while rejecting the near-black background.
    valid = (strongest >= 80) & ((strongest - second) >= 30)
    return labels, valid


def add_gaussian_noise(
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    noisy = image_rgb.astype(np.float32, copy=True)
    pixel_count = int(region_mask.sum())
    if pixel_count == 0:
        return image_rgb.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(pixel_count, 3),
    ).astype(np.float32)
    noisy[region_mask] = np.clip(noisy[region_mask] + noise, 0.0, 255.0)
    return noisy.astype(np.uint8)


def atomic_save_jpeg(
    array: np.ndarray,
    output_path: Path,
    quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    Image.fromarray(array, mode="RGB").save(
        temporary_path,
        format="JPEG",
        quality=quality,
        subsampling=2,
    )
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if args.sigma <= 0:
        raise ValueError("--sigma must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")

    root = args.root.resolve()
    image_dir = root / "ChestCT_test_images"
    mask_dir = root / "ChestCT_test_masks"
    output_dirs = {
        name: root / f"{name}_noise"
        for name in REGIONS
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    image_count = 200 if args.max_images is None else min(args.max_images, 200)
    generated_counts = {name: 0 for name in REGIONS}
    skipped_counts = {name: 0 for name in REGIONS}
    mask_pixel_counts = {name: [] for name in REGIONS}

    for index in range(1, image_count + 1):
        image_path = image_dir / f"test_{index}.jpg"
        mask_path = mask_dir / f"test_masks_{index}.jpg"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing image/mask pair for index {index}: "
                f"{image_path}, {mask_path}"
            )

        with Image.open(image_path) as image:
            image_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        labels, valid = decode_region_labels(mask_path)
        if labels.shape != image_rgb.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch for index {index}: "
                f"image={image_rgb.shape[:2]}, mask={labels.shape}"
            )

        for region_name, config in REGIONS.items():
            region_mask = valid & (labels == config["channel"])
            pixel_count = int(region_mask.sum())
            if pixel_count == 0:
                raise ValueError(
                    f"Empty {region_name} mask for index {index}."
                )
            mask_pixel_counts[region_name].append(pixel_count)

            output_path = output_dirs[region_name] / image_path.name
            if output_path.exists() and not args.overwrite:
                skipped_counts[region_name] += 1
                continue

            sample_seed = (
                args.seed
                + index * 10
                + int(config["seed_offset"])
            )
            noisy = add_gaussian_noise(
                image_rgb=image_rgb,
                region_mask=region_mask,
                sigma=args.sigma,
                seed=sample_seed,
            )
            atomic_save_jpeg(
                array=noisy,
                output_path=output_path,
                quality=args.jpeg_quality,
            )
            generated_counts[region_name] += 1

        if index == 1 or index % 25 == 0 or index == image_count:
            print(f"Processed {index}/{image_count}", flush=True)

    manifest = {
        "input_image_dir": str(image_dir),
        "input_mask_dir": str(mask_dir),
        "image_count": image_count,
        "noise": {
            "distribution": "independent RGB Gaussian additive noise",
            "mean": 0.0,
            "sigma": args.sigma,
            "clip_range": [0, 255],
        },
        "mask_decoding": {
            "bronchus": "red / RGB channel 0",
            "heart": "green / RGB channel 1",
            "lung": "blue / RGB channel 2",
            "minimum_dominant_value": 80,
            "minimum_channel_margin": 30,
        },
        "output": {
            "format": "JPEG",
            "quality": args.jpeg_quality,
            "subsampling": "4:2:0",
            "directories": {
                name: str(path)
                for name, path in output_dirs.items()
            },
        },
        "seed": args.seed,
        "generated_counts": generated_counts,
        "skipped_counts": skipped_counts,
        "final_counts": {
            name: len(list(path.glob("*.jpg")))
            for name, path in output_dirs.items()
        },
        "mask_pixel_range": {
            name: [min(values), max(values)]
            for name, values in mask_pixel_counts.items()
        },
    }
    manifest_path = root / "regional_gaussian_noise_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
