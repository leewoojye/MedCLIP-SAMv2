#!/usr/bin/env python3
"""Build a 4-class Chest CT regional-noise CSV for OVR evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHEST_CT_ROOT = PROJECT_ROOT / "data" / "Chest_CT"
DEFAULT_OUTPUT = SCRIPT_DIR / "chest_ct_200x4_classwise_eval.csv"

# Each noise label is paired with the caption describing the structures left
# uncorrupted by that label's regional Gaussian noise.
CLASS_SPECS = (
    (
        "bronchus_noise",
        CHEST_CT_ROOT / "bronchus_noise",
        "Chest CT image showing preserved cardiac contour and normal-appearing bilateral lung parenchyma.",
    ),
    (
        "heart_noise",
        CHEST_CT_ROOT / "heart_noise",
        "Chest CT image showing preserved bilateral lung aeration and patent central airways.",
    ),
    (
        "lung_noise",
        CHEST_CT_ROOT / "lung_noise",
        "Chest CT image showing preserved cardiac contour and patent central tracheobronchial structures.",
    ),
    (
        "normal",
        CHEST_CT_ROOT / "ChestCT_test_images",
        "Chest CT image showing preserved cardiac contour, preserved bilateral lung aeration, and patent central airways.",
    ),
)


def image_id(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("test_") or not stem[5:].isdecimal():
        raise ValueError(f"Expected test_<numeric-id>.jpg file name; got {path.name}")
    return int(stem[5:])


def image_map(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Required directory not found: {directory}")
    paths = list(directory.glob("*.jpg"))
    result = {image_id(path): path.resolve() for path in paths}
    if len(result) != len(paths):
        raise ValueError(f"Duplicate test image IDs found in {directory}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the 800-row Chest CT 4-class OVR evaluation CSV."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Output must stay under {PROJECT_ROOT}: {output}") from error
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing CSV: {output}")

    maps = {label: image_map(directory) for label, directory, _ in CLASS_SPECS}
    reference_ids = set(maps[CLASS_SPECS[0][0]])
    if not reference_ids:
        raise ValueError("Chest CT image directories must not be empty.")
    for label, _, _ in CLASS_SPECS[1:]:
        observed_ids = set(maps[label])
        if observed_ids != reference_ids:
            raise ValueError(
                f"{label} file IDs do not exactly match {CLASS_SPECS[0][0]}: "
                f"missing={len(reference_ids - observed_ids)}, "
                f"unexpected={len(observed_ids - reference_ids)}."
            )

    rows: list[dict[str, str]] = []
    for label, _, caption in CLASS_SPECS:
        for sample_id in sorted(reference_ids):
            rows.append(
                {
                    "image_path": str(maps[label][sample_id]),
                    "class_label": label,
                    "caption": caption,
                    "sample_id": str(sample_id),
                }
            )

    expected_rows = len(reference_ids) * len(CLASS_SPECS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows; built {len(rows)}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_path", "class_label", "caption", "sample_id"),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={output}")
    print(f"rows={len(rows)} samples_per_class={len(reference_ids)}")
    print(" ".join(f"{label}={len(maps[label])}" for label, _, _ in CLASS_SPECS))


if __name__ == "__main__":
    main()
