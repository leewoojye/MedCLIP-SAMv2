#!/usr/bin/env python3
"""Create a path-corrected copy of the COD10K 70-subclass evaluation CSV.

The supplied source CSV already has the intended labels, fixed captions and
row order.  This utility intentionally changes only ``image_path``: original
CAM samples remain under ``test/images/original_cam`` and the 2,026 no-object
samples are redirected to ``negative_samples_d9``.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
COD10K_TEST_DIR = PROJECT_ROOT / "data" / "COD10K" / "test"
SOURCE_CSV = COD10K_TEST_DIR / "cod10k_original2026_no_object2026_subclass70_caption_pairs.csv"
OUTPUT_CSV = COD10K_TEST_DIR / "cod10k_original2026_no_object2026_subclass70_caption_pairs_paths_fixed.csv"
ORIGINAL_DIR = COD10K_TEST_DIR / "images" / "original_cam"
NEGATIVE_DIR = PROJECT_ROOT / "data" / "COD10K" / "negative_samples_d9"
NO_OBJECT_LABEL = "no camouflaged animal"


def _inside_project(path: Path, *, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"{description} must stay under {PROJECT_ROOT}: {resolved}") from error
    return resolved


def main() -> None:
    source = _inside_project(SOURCE_CSV, description="Source CSV")
    destination = _inside_project(OUTPUT_CSV, description="Output CSV")
    for directory, description in (
        (ORIGINAL_DIR, "Original-image directory"),
        (NEGATIVE_DIR, "Negative-image directory"),
    ):
        if not _inside_project(directory, description=description).is_dir():
            raise FileNotFoundError(f"{description} does not exist: {directory}")

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Source CSV has no header.")
        required = {"image_path", "file_name", "label_name", "label", "caption"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    rewritten: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    seen_paths: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        file_name = Path((row.get("file_name") or "").strip()).name
        label_name = (row.get("label_name") or "").strip()
        if not file_name or not label_name:
            raise ValueError(f"Row {row_number} has an empty file_name or label_name.")

        is_negative = label_name == NO_OBJECT_LABEL
        # The source manifest predates the current d9 export and records every
        # image as JPG.  The d9 no-object images retain the same stem/ID but
        # are losslessly written as PNG, so only their *path* needs this suffix
        # adjustment; all CSV labels, captions and file_name values stay intact.
        target_name = Path(file_name).with_suffix(".png").name if is_negative else file_name
        target = (NEGATIVE_DIR if is_negative else ORIGINAL_DIR) / target_name
        target = _inside_project(target, description=f"Image on row {row_number}")
        if not target.is_file():
            raise FileNotFoundError(f"Image on row {row_number} is missing: {target}")
        if target in seen_paths:
            raise ValueError(f"CSV would contain a duplicate image path: {target}")
        seen_paths.add(target)

        updated = dict(row)
        updated["image_path"] = str(target)
        rewritten.append(updated)
        class_counts[label_name] += 1
        path_counts["negative" if is_negative else "original"] += 1

    if len(rewritten) != 4052:
        raise ValueError(f"Expected 4,052 rows, found {len(rewritten)}.")
    if len(class_counts) != 70:
        raise ValueError(f"Expected 70 subclasses, found {len(class_counts)}.")
    if path_counts != Counter({"original": 2026, "negative": 2026}):
        raise ValueError(f"Expected 2,026 original and 2,026 negative rows; got {path_counts}.")

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rewritten)

    print(f"Wrote: {destination}")
    print(f"Rows: {len(rewritten)} (original={path_counts['original']}, negative={path_counts['negative']})")
    print(f"Subclasses: {len(class_counts)}")


if __name__ == "__main__":
    main()
