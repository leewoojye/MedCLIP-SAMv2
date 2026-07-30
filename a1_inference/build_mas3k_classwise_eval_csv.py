#!/usr/bin/env python3
"""Create a path-corrected copy of the MAS3K 582+582 evaluation CSV.

The original CSV already contains the intended 34 labels and their fixed
captions.  Only ``image_path`` is rewritten: original images come from
``positive_images`` and d9 no-object images come from ``test`` as PNG files.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MAS3K_DIR = PROJECT_ROOT / "data" / "MAS3K"
SOURCE_CSV = MAS3K_DIR / "mas3k_original582_no_object582_classwise_caption_pairs.csv"
OUTPUT_CSV = MAS3K_DIR / "mas3k_original582_no_object582_classwise_caption_pairs_paths_fixed.csv"
POSITIVE_DIR = MAS3K_DIR / "positive_images"
NEGATIVE_DIR = MAS3K_DIR / "test"


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
        (POSITIVE_DIR, "Positive-image directory"),
        (NEGATIVE_DIR, "Negative-image directory"),
    ):
        if not _inside_project(directory, description=description).is_dir():
            raise FileNotFoundError(f"{description} does not exist: {directory}")

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Source CSV has no header.")
        required = {"image_path", "file_name", "label_name", "label", "caption", "sample_type"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    rewritten: list[dict[str, str]] = []
    group_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    seen_paths: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        file_name = Path((row.get("file_name") or "").strip()).name
        sample_type = (row.get("sample_type") or "").strip()
        if not file_name or not sample_type:
            raise ValueError(f"Row {row_number} has an empty file_name or sample_type.")
        try:
            label = int((row.get("label") or "").strip())
        except ValueError as error:
            raise ValueError(f"Row {row_number} has a non-integer label.") from error

        is_original = sample_type == "original"
        # d9 output keeps the source image stem but is stored as PNG.
        target_name = file_name if is_original else Path(file_name).with_suffix(".png").name
        target = (POSITIVE_DIR if is_original else NEGATIVE_DIR) / target_name
        target = _inside_project(target, description=f"Image on row {row_number}")
        if not target.is_file():
            raise FileNotFoundError(f"Image on row {row_number} is missing: {target}")
        if target in seen_paths:
            raise ValueError(f"CSV would contain a duplicate image path: {target}")
        seen_paths.add(target)

        updated = dict(row)
        updated["image_path"] = str(target)
        rewritten.append(updated)
        group_counts["original" if is_original else "negative"] += 1
        class_counts[label] += 1

    if len(rewritten) != 1164:
        raise ValueError(f"Expected 1,164 rows, found {len(rewritten)}.")
    if set(class_counts) != set(range(34)):
        raise ValueError(f"Expected labels 0..33, found {sorted(class_counts)}.")
    if group_counts != Counter({"original": 582, "negative": 582}):
        raise ValueError(f"Expected 582 original and 582 negative rows; got {group_counts}.")
    if class_counts[33] != 582:
        raise ValueError(f"No-object label 33 must have 582 rows, found {class_counts[33]}.")

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rewritten)

    print(f"Wrote: {destination}")
    print(f"Rows: {len(rewritten)} (original={group_counts['original']}, negative={group_counts['negative']})")
    print(f"Classes: {len(class_counts)}")


if __name__ == "__main__":
    main()
