#!/usr/bin/env python3
"""Evaluate MAS3K with five models and the unchanged Clip4Retrofit OVR core."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cod10k_classwise_eval import (
    MODEL_IDS,
    _encode_model as _encode_base_model,
    _encode_open_clip_model,
)
from csv_classwise_eval import (
    DEFAULT_CACHE_DIR,
    _configure_local_cache,
    _path_in_project,
    resolve_device,
)
from clip4retrofit_ovr_metrics import (
    compute_cosine_similarity_scores,
    compute_ovr_metrics,
    save_evaluation,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = (
    PROJECT_ROOT
    / "data"
    / "MAS3K"
    / "mas3k_original582_no_object582_classwise_caption_pairs_paths_fixed.csv"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "csv_classwise_results" / "mas3k_original582_negative582_classwise"
)
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "model" / "mas3k_global" / "checkpoints" / "global_ind_gmpo_epoch_3.pt"
)
MODEL_KEYS = ("clip", "siglip", "bioclip", "bioclip2", "checkpoint")


@dataclass(frozen=True)
class Mas3kDataset:
    image_paths: tuple[Path, ...]
    labels: np.ndarray
    class_names: tuple[str, ...]
    class_labels: tuple[int, ...]
    captions: tuple[str, ...]


def load_mas3k_csv(csv_path: str | Path) -> Mas3kDataset:
    """Read the path-corrected CSV while preserving labels and captions exactly."""

    source = _path_in_project(csv_path, description="MAS3K CSV")
    if not source.is_file():
        raise FileNotFoundError(f"MAS3K CSV does not exist: {source}")

    image_paths: list[Path] = []
    labels: list[int] = []
    names_by_label: dict[int, str] = {}
    captions_by_label: dict[int, str] = {}
    seen_paths: set[Path] = set()
    required = {"image_path", "label_name", "label", "caption"}
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("MAS3K CSV has no header row.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"MAS3K CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            class_name = (row.get("label_name") or "").strip()
            caption = (row.get("caption") or "").strip()
            try:
                class_label = int((row.get("label") or "").strip())
            except ValueError as error:
                raise ValueError(f"Row {row_number} has a non-integer label.") from error
            if not raw_path or not class_name or not caption:
                raise ValueError(f"Row {row_number} needs image_path, label_name and caption.")
            image_path = _path_in_project(raw_path, description=f"Image on row {row_number}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image on row {row_number} is missing: {image_path}")
            if image_path in seen_paths:
                raise ValueError(f"MAS3K CSV has a duplicate image path: {image_path}")
            seen_paths.add(image_path)
            if names_by_label.setdefault(class_label, class_name) != class_name:
                raise ValueError(f"Label {class_label} has inconsistent class names.")
            if captions_by_label.setdefault(class_label, caption) != caption:
                raise ValueError(f"Label {class_label} has inconsistent captions.")
            image_paths.append(image_path)
            labels.append(class_label)

    class_labels = tuple(sorted(names_by_label))
    if class_labels != tuple(range(34)):
        raise ValueError(f"MAS3K must provide labels 0..33; found {class_labels}.")
    if len(image_paths) != 1164:
        raise ValueError(f"MAS3K must provide 1,164 samples; found {len(image_paths)}.")
    if sum(label == 33 for label in labels) != 582:
        raise ValueError("MAS3K label 33 must contain all 582 no-object samples.")
    if sum(label != 33 for label in labels) != 582:
        raise ValueError("MAS3K labels 0..32 must contain 582 original samples.")
    return Mas3kDataset(
        image_paths=tuple(image_paths),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=tuple(names_by_label[label] for label in class_labels),
        class_labels=class_labels,
        captions=tuple(captions_by_label[label] for label in class_labels),
    )


def _encode_checkpoint(dataset: Mas3kDataset, checkpoint_path: Path, *, batch_size: int, device: str):
    """Encode with the MAS3K OpenAI CLIP ViT-B/32 checkpoint under strict load."""

    try:
        import open_clip
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "open-clip-torch is required for the MAS3K checkpoint. "
            "Run a1_inference/setup_csv_classwise_eval.sh first."
        ) from error
    images, texts, metadata = _encode_open_clip_model(
        open_clip,
        MODEL_IDS["checkpoint"],
        dataset.image_paths,
        dataset.captions,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    return images, texts, {
        "backend": "open_clip_torch",
        "checkpoint_path": str(checkpoint_path),
        "ddp_module_prefix_removed": False,
        **metadata,
    }


def _write_predictions(
    destination: Path,
    dataset: Mas3kDataset,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    y_score: np.ndarray,
) -> None:
    np.savez_compressed(
        destination,
        y_true=dataset.labels,
        y_score=np.asarray(y_score, dtype=np.float64),
        class_names=np.asarray(dataset.class_names, dtype=str),
        class_labels=np.asarray(dataset.class_labels, dtype=np.int64),
        image_paths=np.asarray([str(path) for path in dataset.image_paths], dtype=str),
        captions=np.asarray(dataset.captions, dtype=str),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        text_embeddings=np.asarray(text_embeddings, dtype=np.float32),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAS3K class-wise evaluation using unchanged Clip4Retrofit metrics."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    _configure_local_cache()
    dataset = load_mas3k_csv(args.csv)
    output_dir = _path_in_project(
        args.output_dir or (DEFAULT_OUTPUT_ROOT / args.model),
        description="MAS3K output directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _path_in_project(args.checkpoint, description="MAS3K checkpoint")
    device = resolve_device(args.device)
    print(f"Model: {args.model} ({MODEL_IDS[args.model]})")
    print(f"Samples: {len(dataset.image_paths)}, classes: {len(dataset.class_labels)}")
    print(f"Device: {device}")

    if args.model == "checkpoint":
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"MAS3K checkpoint does not exist: {checkpoint_path}")
        image_embeddings, text_embeddings, metadata = _encode_checkpoint(
            dataset, checkpoint_path, batch_size=args.batch_size, device=device
        )
    else:
        # Reuse the COD10K model-specific embedding adapters.  Dataset labels
        # are not used in embedding, and this MAS3K dataclass exposes the same
        # image_paths/captions interface.
        image_embeddings, text_embeddings, metadata = _encode_base_model(
            args.model,
            dataset,
            checkpoint_path=checkpoint_path,
            batch_size=args.batch_size,
            device=device,
        )
    if image_embeddings.shape[0] != len(dataset.image_paths):
        raise RuntimeError("Model returned a different number of image embeddings.")
    if text_embeddings.shape[0] != len(dataset.class_labels):
        raise RuntimeError("Model returned a different number of text embeddings.")

    y_score = compute_cosine_similarity_scores(image_embeddings, text_embeddings)
    _write_predictions(
        output_dir / "predictions.npz", dataset, image_embeddings, text_embeddings, y_score
    )
    metrics = compute_ovr_metrics(
        dataset.labels,
        y_score,
        class_names=dataset.class_names,
        class_labels=dataset.class_labels,
        target_tpr=args.target_tpr,
    )
    save_evaluation(metrics, output_dir, create_plots=args.plots)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_key": args.model,
                "model_id": MODEL_IDS[args.model],
                "n_samples": len(dataset.image_paths),
                "class_names": list(dataset.class_names),
                "class_labels": list(dataset.class_labels),
                "captions": list(dataset.captions),
                "score_definition": "raw L2-normalised image/text cosine similarity",
                "metric_implementation": "clip4retrofit_ovr_metrics.py (unchanged)",
                "cache_dir": str(DEFAULT_CACHE_DIR),
                **metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for item in metrics.per_class:
        print(
            f"{item.class_name}: AUROC={item.auroc:.6f}, "
            f"FPR@{args.target_tpr:.0%}TPR={item.fpr_at_target_tpr:.6f}"
        )
    print(f"Macro AUROC: {metrics.macro_auroc:.6f}")
    print(f"Macro FPR@{args.target_tpr:.0%}TPR: {metrics.macro_fpr_at_target_tpr:.6f}")


if __name__ == "__main__":
    main()
