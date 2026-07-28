"""Build a three-class UDIAT CSV for class-wise AUROC/FPR evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "udiat_classwise_eval.csv"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

CLASSES = (
    (
        "benign",
        PROJECT_ROOT / "data" / "UDIAT" / "Benign",
        "There is a benign tumor.",
    ),
    (
        "malignant",
        PROJECT_ROOT / "data" / "UDIAT" / "Malignant",
        "There is malignant tumor.",
    ),
    (
        "normal",
        PROJECT_ROOT / "data" / "UDIAT2" / "UDIAT음성샘플(closedform crossattention Ver)",
        "There is no visible tumor.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Class directory not found: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing CSV: {args.output}")

    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for class_label, directory, caption in CLASSES:
        files = image_files(directory)
        counts[class_label] = len(files)
        rows.extend(
            {
                "image_path": str(image_path),
                "class_label": class_label,
                "caption": caption,
            }
            for image_path in files
        )

    if counts["benign"] + counts["malignant"] != counts["normal"]:
        raise ValueError(
            "Expected equal positive and normal totals; got "
            f"benign={counts['benign']}, malignant={counts['malignant']}, "
            f"normal={counts['normal']}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "class_label", "caption"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.output}")
    print(f"rows={len(rows)} benign={counts['benign']} malignant={counts['malignant']} normal={counts['normal']}")


if __name__ == "__main__":
    main()
