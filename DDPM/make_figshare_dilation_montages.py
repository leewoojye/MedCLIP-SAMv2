from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


PANEL_SIZE = 256
LABEL_HEIGHT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual comparison montages for dilation tests."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-names", nargs="+", required=True)
    return parser.parse_args()


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    mask_array = np.asarray(mask.convert("L")) > 0
    base_array = np.asarray(base).copy()
    red = np.zeros_like(base_array)
    red[..., 0] = 255
    base_array[mask_array] = (
        0.55 * base_array[mask_array] + 0.45 * red[mask_array]
    ).astype(np.uint8)
    return Image.fromarray(base_array)


def square_zoom_box(mask: Image.Image, margin: int = 24) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, mask.width, mask.height)
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    side = min(max(width, height) + 2 * margin, mask.width, mask.height)
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    zoom_left = int(round(center_x - side / 2.0))
    zoom_top = int(round(center_y - side / 2.0))
    zoom_left = min(max(zoom_left, 0), mask.width - side)
    zoom_top = min(max(zoom_top, 0), mask.height - side)
    return (zoom_left, zoom_top, zoom_left + side, zoom_top + side)


def panel(image: Image.Image, label: str) -> Image.Image:
    fitted = ImageOps.fit(
        image.convert("RGB"),
        (PANEL_SIZE, PANEL_SIZE),
        method=Image.Resampling.BILINEAR,
    )
    rendered = Image.new(
        "RGB",
        (PANEL_SIZE, PANEL_SIZE + LABEL_HEIGHT),
        color="black",
    )
    rendered.paste(fitted, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(rendered)
    draw.text((6, 6), label, fill="white")
    return rendered


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    test_root = Path(args.test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.sample_names:
        original = load_gray(image_dir / name)
        tumor_mask = load_gray(mask_dir / name)
        dilation_0 = load_gray(test_root / "dilation_0" / "blended" / name)
        dilation_9 = load_gray(test_root / "dilation_9" / "blended" / name)
        dilation_15 = load_gray(test_root / "dilation_15" / "blended" / name)
        applied_15 = load_gray(
            test_root / "dilation_15" / "applied_masks" / name
        )
        zoom_box = square_zoom_box(applied_15)

        full_panels = [
            panel(original, f"{name} original"),
            panel(overlay_mask(original, tumor_mask), "original tumor mask"),
            panel(dilation_0, "dilation 0"),
            panel(dilation_9, "dilation 9x9"),
            panel(dilation_15, "dilation 15x15"),
        ]
        zoom_panels = [
            panel(original.crop(zoom_box), "ROI original"),
            panel(
                overlay_mask(original, tumor_mask).crop(zoom_box),
                "ROI tumor mask",
            ),
            panel(dilation_0.crop(zoom_box), "ROI dilation 0"),
            panel(dilation_9.crop(zoom_box), "ROI dilation 9x9"),
            panel(dilation_15.crop(zoom_box), "ROI dilation 15x15"),
        ]
        canvas = Image.new(
            "RGB",
            (
                PANEL_SIZE * len(full_panels),
                (PANEL_SIZE + LABEL_HEIGHT) * 2,
            ),
            color="black",
        )
        for column, rendered in enumerate(full_panels):
            canvas.paste(rendered, (column * PANEL_SIZE, 0))
        for column, rendered in enumerate(zoom_panels):
            canvas.paste(
                rendered,
                (column * PANEL_SIZE, PANEL_SIZE + LABEL_HEIGHT),
            )
        canvas.save(output_dir / name)
        print(f"Saved montage: {output_dir / name}")


if __name__ == "__main__":
    main()
