from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    binarize_mask,
    center_crop_or_pad,
    load_grayscale_image,
    normalize_to_unit_interval,
    resize_mask_to_shape,
    to_tensor,
)


@dataclass(frozen=True)
class SlicePaths:
    name: str
    image_path: Path
    mask_path: Path


class BrainTumorInpaintingDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        crop_size: int = 224,
        include_target: bool = True,
        mask_mode: str = "provided",
        hide_tumor_in_context: bool = False,
        base_seed: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.crop_size = crop_size
        self.include_target = include_target
        self.mask_mode = mask_mode
        self.hide_tumor_in_context = hide_tumor_in_context
        self.base_seed = base_seed

        image_files = {path.name: path for path in self.image_dir.glob("*.png")}
        mask_files = {path.name: path for path in self.mask_dir.glob("*.png")}
        shared_names = sorted(image_files.keys() & mask_files.keys(), key=_natural_key)
        self.samples = [
            SlicePaths(name=name, image_path=image_files[name], mask_path=mask_files[name])
            for name in shared_names
        ]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise FileNotFoundError(
                f"No paired PNG files found in {self.image_dir} and {self.mask_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        rng = (
            np.random.default_rng(self.base_seed + index)
            if self.base_seed is not None
            else np.random.default_rng()
        )
        image = normalize_to_unit_interval(load_grayscale_image(sample.image_path))
        tumor_mask = resize_mask_to_shape(load_grayscale_image(sample.mask_path), image.shape)
        tumor_mask = binarize_mask(tumor_mask)

        image_crop, meta = center_crop_or_pad(image, self.crop_size)
        tumor_mask_crop, _ = center_crop_or_pad(tumor_mask, self.crop_size)
        mask_crop = self._build_condition_mask(image_crop, tumor_mask_crop, rng)
        baseline = image_crop.copy()
        baseline[mask_crop > 0.5] = 0.0
        if self.hide_tumor_in_context:
            baseline[tumor_mask_crop > 0.5] = 0.0

        target_tensor = (
            to_tensor(image_crop)
            if self.include_target
            else torch.zeros(1, self.crop_size, self.crop_size)
        )

        sample_dict = {
            "name": sample.name,
            "image": to_tensor(image_crop),
            "baseline": to_tensor(baseline),
            "mask": to_tensor(mask_crop),
            "tumor_mask": to_tensor(tumor_mask_crop),
            "loss_mask": to_tensor(mask_crop),
            "target": target_tensor,
            "model_input": torch.cat(
                [to_tensor(baseline), to_tensor(mask_crop), target_tensor], dim=0
            ),
            "meta": meta,
        }
        if not self.include_target:
            sample_dict["orig_image"] = image
            sample_dict["orig_mask"] = tumor_mask
        return sample_dict

    def _build_condition_mask(
        self,
        image_crop: np.ndarray,
        tumor_mask_crop: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.mask_mode == "provided":
            return tumor_mask_crop.astype(np.float32)
        if self.mask_mode == "synthetic_healthy":
            return _sample_shifted_healthy_mask(
                image=image_crop,
                tumor_mask=tumor_mask_crop,
                rng=rng,
            )
        raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")


def _natural_key(value: str) -> tuple:
    stem = Path(value).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _sample_shifted_healthy_mask(
    image: np.ndarray,
    tumor_mask: np.ndarray,
    rng: np.random.Generator,
    min_brain_fraction: float = 0.9,
    max_trials: int = 128,
) -> np.ndarray:
    exclusion = tumor_mask > 0.5
    coords = np.argwhere(exclusion)

    if coords.size == 0:
        return _fallback_random_healthy_mask(image, exclusion, rng)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    template = exclusion[y_min:y_max, x_min:x_max].astype(np.float32)
    transforms = [
        lambda arr: arr,
        np.fliplr,
        np.flipud,
        lambda arr: np.rot90(arr, 1),
        lambda arr: np.rot90(arr, 2),
        lambda arr: np.rot90(arr, 3),
    ]

    brain_mask = image > 0.05
    image_height, image_width = image.shape

    for _ in range(max_trials):
        transform = transforms[int(rng.integers(0, len(transforms)))]
        transformed = transform(template).copy()
        mask_height, mask_width = transformed.shape

        if mask_height >= image_height or mask_width >= image_width:
            continue

        top = int(rng.integers(0, image_height - mask_height + 1))
        left = int(rng.integers(0, image_width - mask_width + 1))
        candidate = np.zeros_like(exclusion, dtype=np.float32)
        candidate[top : top + mask_height, left : left + mask_width] = transformed

        candidate_bool = candidate > 0.5
        if np.any(candidate_bool & exclusion):
            continue
        if candidate_bool.sum() == 0:
            continue

        brain_fraction = brain_mask[candidate_bool].mean()
        if brain_fraction < min_brain_fraction:
            continue

        return candidate.astype(np.float32)

    return _fallback_random_healthy_mask(image, exclusion, rng)


def _fallback_random_healthy_mask(
    image: np.ndarray,
    exclusion: np.ndarray,
    rng: np.random.Generator,
    max_trials: int = 128,
) -> np.ndarray:
    height, width = image.shape
    brain_mask = (image > 0.05) & (~exclusion)
    coords = np.argwhere(brain_mask)
    if coords.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    for _ in range(max_trials):
        center_y, center_x = coords[int(rng.integers(0, len(coords)))]
        radius_y = int(rng.integers(max(6, height // 20), max(8, height // 10)))
        radius_x = int(rng.integers(max(6, width // 20), max(8, width // 10)))

        yy, xx = np.ogrid[:height, :width]
        ellipse = (((yy - center_y) / max(radius_y, 1)) ** 2 + ((xx - center_x) / max(radius_x, 1)) ** 2) <= 1.0
        ellipse &= brain_mask
        if ellipse.sum() >= 32:
            return ellipse.astype(np.float32)

    return np.zeros_like(image, dtype=np.float32)
