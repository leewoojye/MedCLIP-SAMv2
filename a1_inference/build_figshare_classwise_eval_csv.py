#!/usr/bin/env python3
"""Build the 4-class Figshare positive/negative CSV for OVR evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FIGSHARE_ROOT = PROJECT_ROOT / "data" / "Figshare" / "test_figshare3064"
POSITIVE_DIR = FIGSHARE_ROOT / "images"
METADATA_PATH = FIGSHARE_ROOT / "figshare_tumor_type_metadata0.json"
NEGATIVE_DIR = (
    PROJECT_ROOT
    / "generated_neg_output"
    / "figshare_test_ddpm_step122000_dilation9"
    / "blended"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "figshare3064x2_classwise_eval.csv"

CLASS_ORDER = ("meningioma", "glioma", "pituitary", "normal")
CAPTIONS = {
    "meningioma": "There is a meningioma tumor.",
    "glioma": "There is a glioma tumor.",
    "pituitary": "There is a pituitary tumor.",
    "normal": "There is no visible tumor.",
}


def numeric_pngs(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Required directory not found: {directory}")
    paths = list(directory.glob("*.png"))
    invalid = [path.name for path in paths if not path.stem.isdecimal()]
    if invalid:
        raise ValueError(f"Expected numeric PNG stems in {directory}; found {invalid[:5]}")
    mapping = {path.stem: path.resolve() for path in paths}
    if len(mapping) != len(paths):
        raise ValueError(f"Duplicate PNG IDs found in {directory}")
    return mapping


def numeric_key(sample_id: str) -> int:
    return int(sample_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 6,128-row Figshare CSV for 4-class OVR AUROC/FPR."
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

    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Figshare metadata must be a dictionary keyed by image ID.")

    positive_by_id = numeric_pngs(POSITIVE_DIR)
    negative_by_id = numeric_pngs(NEGATIVE_DIR)
    metadata_ids = set(metadata)
    positive_ids = set(positive_by_id)
    negative_ids = set(negative_by_id)
    if metadata_ids != positive_ids:
        raise ValueError(
            "Metadata IDs and positive image IDs differ: "
            f"metadata_only={len(metadata_ids - positive_ids)}, "
            f"image_only={len(positive_ids - metadata_ids)}."
        )
    if positive_ids != negative_ids:
        raise ValueError(
            "Positive and dilation=9 negative image IDs differ: "
            f"positive_only={len(positive_ids - negative_ids)}, "
            f"negative_only={len(negative_ids - positive_ids)}."
        )

    rows: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    for sample_id in sorted(positive_ids, key=numeric_key):
        entry = metadata[sample_id]
        if not isinstance(entry, dict) or entry.get("id") != sample_id:
            raise ValueError(f"Invalid metadata entry for image ID {sample_id!r}.")
        label = entry.get("label_name")
        if label not in CLASS_ORDER[:-1]:
            raise ValueError(f"Unexpected tumor label for ID {sample_id}: {label!r}")
        rows.append(
            {
                "image_path": str(positive_by_id[sample_id]),
                "class_label": label,
                "caption": CAPTIONS[label],
                "sample_id": sample_id,
                "sample_origin": "figshare_tumor_original",
            }
        )
        class_counts[label] += 1
        rows.append(
            {
                "image_path": str(negative_by_id[sample_id]),
                "class_label": "normal",
                "caption": CAPTIONS["normal"],
                "sample_id": sample_id,
                "sample_origin": "ddpm_step122000_dilation9_negative",
            }
        )
        class_counts["normal"] += 1

    if set(class_counts) != set(CLASS_ORDER):
        raise ValueError(f"Expected all four classes; observed {dict(class_counts)}")
    if len(rows) != 2 * len(positive_ids):
        raise RuntimeError("CSV row count is not exactly two rows per Figshare image ID.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_path",
                "class_label",
                "caption",
                "sample_id",
                "sample_origin",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote={output}")
    print(f"rows={len(rows)}")
    print(" ".join(f"{label}={class_counts[label]}" for label in CLASS_ORDER))


if __name__ == "__main__":
    main()
