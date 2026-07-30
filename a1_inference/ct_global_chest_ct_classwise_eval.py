#!/usr/bin/env python3
"""Evaluate the CT global checkpoint on the existing 800-image Chest CT CSV.

This companion keeps the CSV prompts, raw cosine score creation, and
Clip4Retrofit-style metric functions unchanged.  It only supplies the CT
checkpoint and correct Chest CT result provenance without editing prior files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from checkpoint_classwise_eval import _encode_checkpoint, _write_predictions
from csv_classwise_eval import (
    DEFAULT_CACHE_DIR,
    _configure_local_cache,
    _path_in_project,
    load_csv_dataset,
    resolve_device,
)
from clip4retrofit_ovr_metrics import (
    compute_cosine_similarity_scores,
    compute_ovr_metrics,
    save_evaluation,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CHEST_CT_CSV = SCRIPT_DIR / "chest_ct_200x4_classwise_eval.csv"
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "model" / "ct_global" / "global_ind_gmpo_epoch_3.pt"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "csv_classwise_results" / "chest_ct_200x4" / "ct_global_ind_gmpo_epoch_3"
)
CLASS_ORDER = ("bronchus_noise", "heart_noise", "lung_noise", "normal")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ct_global/global_ind_gmpo_epoch_3.pt with the existing "
            "Chest CT CSV and unchanged OVR AUROC/FPR core."
        )
    )
    parser.add_argument("--csv", default=str(CHEST_CT_CSV))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    _configure_local_cache()
    checkpoint_path = _path_in_project(args.checkpoint, description="Checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    output_dir = _path_in_project(args.output_dir, description="Output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_csv_dataset(args.csv, CLASS_ORDER)
    device = resolve_device(args.device)

    print(f"CSV samples: {len(dataset.image_paths)}")
    print(f"Class order: {', '.join(dataset.class_order)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    image_embeddings, text_embeddings, model_info = _encode_checkpoint(
        checkpoint_path,
        dataset.image_paths,
        dataset.captions,
        batch_size=args.batch_size,
        device=device,
    )
    if image_embeddings.shape[0] != len(dataset.image_paths):
        raise RuntimeError("Checkpoint returned a different number of image embeddings.")
    if text_embeddings.shape[0] != len(dataset.class_order):
        raise RuntimeError("Checkpoint returned a different number of text embeddings.")

    y_score = compute_cosine_similarity_scores(image_embeddings, text_embeddings)
    _write_predictions(
        output_dir / "predictions.npz",
        labels=dataset.labels,
        class_order=dataset.class_order,
        image_paths=dataset.image_paths,
        captions=dataset.captions,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        y_score=y_score,
        checkpoint_path=checkpoint_path,
    )
    metrics = compute_ovr_metrics(
        dataset.labels,
        y_score,
        class_names=dataset.class_order,
        class_labels=dataset.class_order,
        target_tpr=args.target_tpr,
    )
    save_evaluation(metrics, output_dir, create_plots=args.plots)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": 3,
                "checkpoint_name": "chest_ct_global_ind_seed12",
                "n_samples": len(dataset.image_paths),
                "class_order": list(dataset.class_order),
                "captions": list(dataset.captions),
                "score_definition": "raw L2-normalised image/text cosine similarity",
                "metric_implementation": "clip4retrofit_ovr_metrics.py (unchanged)",
                "cache_dir": str(DEFAULT_CACHE_DIR),
                **model_info,
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
