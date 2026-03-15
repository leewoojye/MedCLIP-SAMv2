from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_grayscale_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def normalize_to_unit_interval(image: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(image, 0.001))
    hi = float(np.quantile(image, 0.999))
    clipped = np.clip(image, lo, hi)
    denom = float(clipped.max() - clipped.min())
    if denom <= 1e-8:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - clipped.min()) / denom).astype(np.float32)


def resize_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.float32)
    image = Image.fromarray(mask.astype(np.uint8))
    resized = image.resize((shape[1], shape[0]), resample=Image.NEAREST)
    return np.asarray(resized, dtype=np.float32)


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


def center_crop_or_pad(
    array: np.ndarray, target_size: int
) -> tuple[np.ndarray, dict[str, int]]:
    height, width = array.shape
    crop_height = min(height, target_size)
    crop_width = min(width, target_size)
    top = max((height - crop_height) // 2, 0)
    left = max((width - crop_width) // 2, 0)

    cropped = array[top : top + crop_height, left : left + crop_width]
    result = np.zeros((target_size, target_size), dtype=array.dtype)
    pad_top = max((target_size - crop_height) // 2, 0)
    pad_left = max((target_size - crop_width) // 2, 0)
    result[pad_top : pad_top + crop_height, pad_left : pad_left + crop_width] = cropped

    meta = {
        "orig_height": height,
        "orig_width": width,
        "crop_top": top,
        "crop_left": left,
        "crop_height": crop_height,
        "crop_width": crop_width,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "target_size": target_size,
    }
    return result, meta


def restore_from_center_crop_or_pad(
    cropped: np.ndarray,
    meta: dict[str, int],
    background: np.ndarray | None = None,
) -> np.ndarray:
    orig_height = meta["orig_height"]
    orig_width = meta["orig_width"]
    crop_height = meta["crop_height"]
    crop_width = meta["crop_width"]
    crop_top = meta["crop_top"]
    crop_left = meta["crop_left"]
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]

    restored = (
        background.copy()
        if background is not None
        else np.zeros((orig_height, orig_width), dtype=cropped.dtype)
    )
    patch = cropped[pad_top : pad_top + crop_height, pad_left : pad_left + crop_width]
    restored[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width] = (
        patch
    )
    return restored


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).unsqueeze(0).float()


def save_png(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(array, 0.0, 1.0)
    Image.fromarray((image * 255.0).round().astype(np.uint8)).save(path)


def save_debug_montage(
    panels: list[tuple[str, np.ndarray]],
    path: str | Path,
    border: int = 20,
) -> None:
    rendered = []
    for label, array in panels:
        image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
        image = image.convert("L")
        image = ImageOps.expand(image, border=border, fill="white")
        draw = ImageDraw.Draw(image)
        draw.text((8, 2), label, fill="black")
        rendered.append(image)

    canvas = Image.new(
        "L",
        (sum(image.width for image in rendered), max(image.height for image in rendered)),
        color="white",
    )
    cursor = 0
    for image in rendered:
        canvas.paste(image, (cursor, 0))
        cursor += image.width

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(float(sigma), 0.0)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def blend_prediction(
    prediction: np.ndarray, baseline: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    return prediction * mask + baseline * (1.0 - mask)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_step(step: int) -> str:
    width = max(6, int(math.log10(max(step, 1))) + 1)
    return f"{step:0{width}d}"
