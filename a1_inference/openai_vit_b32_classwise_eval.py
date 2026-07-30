#!/usr/bin/env python3
"""Evaluate OpenAI CLIP ViT-B/32 with the unchanged OVR metric core.

This adapter deliberately does only the upstream inference work:

``CSV -> OpenAI CLIP ViT-B/32 embeddings -> raw cosine scores``.

AUROC and FPR@95%TPR are then calculated by the existing
``clip4retrofit_ovr_metrics.py`` functions.  It supports the already-created
UDIAT, Figshare, Chest CT, COD10K, and MAS3K CSVs without modifying either the
CSV definitions or the original metric implementation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from clip4retrofit_ovr_metrics import (
    compute_cosine_similarity_scores,
    compute_ovr_metrics,
    save_evaluation,
)
from cod10k_classwise_eval import DEFAULT_CSV as DEFAULT_COD10K_CSV
from cod10k_classwise_eval import load_cod10k_csv
from csv_classwise_eval import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLASS_ORDER,
    DEFAULT_CSV as DEFAULT_UDIAT_CSV,
    _configure_local_cache,
    _load_rgb_images,
    _path_in_project,
    _tensor_to_numpy,
    load_csv_dataset,
    resolve_device,
)
from mas3k_classwise_eval import DEFAULT_CSV as DEFAULT_MAS3K_CSV
from mas3k_classwise_eval import load_mas3k_csv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
MODEL_LABEL = "OpenAI CLIP ViT-B/32"
DEFAULT_OUTPUTS = {
    "udiat": SCRIPT_DIR / "csv_classwise_results" / "udiat163x2_openai_vit_b32",
    "figshare": (
        SCRIPT_DIR
        / "csv_classwise_results"
        / "figshare3064x2_dilation9"
        / "openai_vit_b32"
    ),
    "chestct": (
        SCRIPT_DIR / "csv_classwise_results" / "chest_ct_200x4" / "openai_vit_b32"
    ),
    "cod10k": (
        SCRIPT_DIR
        / "csv_classwise_results"
        / "cod10k_original2026_negative_d9_subclass70"
        / "openai_vit_b32"
    ),
    "mas3k": (
        SCRIPT_DIR
        / "csv_classwise_results"
        / "mas3k_original582_negative582_classwise"
        / "openai_vit_b32"
    ),
}
CSV_DATASETS = {
    "udiat": (
        DEFAULT_UDIAT_CSV,
        DEFAULT_CLASS_ORDER,
    ),
    "figshare": (
        SCRIPT_DIR / "figshare3064x2_classwise_eval.csv",
        ("meningioma", "glioma", "pituitary", "normal"),
    ),
    "chestct": (
        SCRIPT_DIR / "chest_ct_200x4_classwise_eval.csv",
        ("bronchus_noise", "heart_noise", "lung_noise", "normal"),
    ),
}


def _import_torch() -> Any:
    import torch

    return torch


def _import_open_clip() -> Any:
    try:
        import open_clip
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "open-clip-torch is required. Run through "
            "a1_inference/run_openai_vit_b32_classwise_eval.sh."
        ) from error
    return open_clip


def encode_openai_vit_b32(
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Return raw OpenAI CLIP feature vectors with its official preprocessing."""

    torch = _import_torch()
    open_clip = _import_open_clip()
    model = None
    try:
        # ``pretrained='openai'`` is the public OpenAI CLIP ViT-B/32 weight
        # release. cache_dir keeps the downloaded artifact inside a1_inference.
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
            device=device,
            cache_dir=str(DEFAULT_CACHE_DIR / "hub"),
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        with torch.inference_mode():
            text_embeddings = model.encode_text(tokenizer(list(captions)).to(device))
            image_batches: list[np.ndarray] = []
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                tensors = [preprocess(image) for image in _load_rgb_images(batch_paths)]
                image_embeddings = model.encode_image(torch.stack(tensors).to(device))
                image_batches.append(_tensor_to_numpy(image_embeddings))

        return (
            np.vstack(image_batches),
            _tensor_to_numpy(text_embeddings),
            {
                "open_clip_module": str(Path(open_clip.__file__).resolve()),
                "image_preprocess": str(preprocess),
            },
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_predictions(
    destination: Path,
    *,
    labels: np.ndarray,
    class_names: Sequence[str],
    class_labels: np.ndarray,
    image_paths: Sequence[Path],
    captions: Sequence[str],
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    scores: np.ndarray,
) -> None:
    np.savez_compressed(
        destination,
        y_true=labels,
        y_score=np.asarray(scores, dtype=np.float64),
        class_names=np.asarray(class_names, dtype=str),
        class_labels=class_labels,
        image_paths=np.asarray([str(path) for path in image_paths], dtype=str),
        captions=np.asarray(captions, dtype=str),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        text_embeddings=np.asarray(text_embeddings, dtype=np.float32),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OpenAI CLIP ViT-B/32 on a prepared evaluation CSV "
            "using the unchanged Clip4Retrofit-style OVR AUROC/FPR core."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("udiat", "figshare", "chestct", "cod10k", "mas3k"),
        required=True,
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Prepared evaluation CSV (defaults to the selected dataset CSV).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Result directory under this project (dataset-specific default).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write the existing ROC visualisations as well as numerical artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    _configure_local_cache()
    if args.dataset in CSV_DATASETS:
        default_csv, class_order = CSV_DATASETS[args.dataset]
        dataset = load_csv_dataset(
            args.csv or default_csv,
            class_order,
        )
        labels = dataset.labels
        class_names = dataset.class_order
        class_labels = np.asarray(dataset.class_order, dtype=str)
        image_paths = dataset.image_paths
        captions = dataset.captions
    elif args.dataset == "cod10k":
        dataset = load_cod10k_csv(args.csv or DEFAULT_COD10K_CSV)
        labels = dataset.labels
        class_names = dataset.class_names
        class_labels = np.asarray(dataset.class_labels, dtype=np.int64)
        image_paths = dataset.image_paths
        captions = dataset.captions
    else:  # MAS3K has integer labels and a 34-class CSV schema.
        dataset = load_mas3k_csv(args.csv or DEFAULT_MAS3K_CSV)
        labels = dataset.labels
        class_names = dataset.class_names
        class_labels = np.asarray(dataset.class_labels, dtype=np.int64)
        image_paths = dataset.image_paths
        captions = dataset.captions

    output_dir = _path_in_project(
        args.output_dir or DEFAULT_OUTPUTS[args.dataset],
        description="Output directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Model: {MODEL_LABEL} ({MODEL_NAME}, pretrained={PRETRAINED})")
    print(f"Dataset: {args.dataset}; samples={len(image_paths)}; classes={len(class_names)}")
    print(f"Device: {device}; cache: {DEFAULT_CACHE_DIR}")

    image_embeddings, text_embeddings, backend_metadata = encode_openai_vit_b32(
        image_paths,
        captions,
        batch_size=args.batch_size,
        device=device,
    )
    if image_embeddings.shape[0] != len(image_paths):
        raise RuntimeError("Model returned a different number of image embeddings.")
    if text_embeddings.shape[0] != len(class_names):
        raise RuntimeError("Model returned a different number of text embeddings.")

    # The scoring and OVR metric implementations below are intentionally the
    # pre-existing shared core, without CLIP logit scaling or softmax.
    scores = compute_cosine_similarity_scores(image_embeddings, text_embeddings)
    _write_predictions(
        output_dir / "predictions.npz",
        labels=labels,
        class_names=class_names,
        class_labels=class_labels,
        image_paths=image_paths,
        captions=captions,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        scores=scores,
    )
    metrics = compute_ovr_metrics(
        labels,
        scores,
        class_names=class_names,
        class_labels=class_labels,
        target_tpr=args.target_tpr,
    )
    save_evaluation(metrics, output_dir, create_plots=args.plots)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model_key": "openai_vit_b32",
        "model_label": MODEL_LABEL,
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "backend": "open_clip_torch",
        "device": device,
        "n_samples": len(image_paths),
        "class_names": list(class_names),
        "class_labels": class_labels.tolist(),
        "captions": list(captions),
        "score_definition": "raw L2-normalised image/text cosine similarity",
        "metric_implementation": "clip4retrofit_ovr_metrics.py (unchanged)",
        "cache_dir": str(DEFAULT_CACHE_DIR),
        **backend_metadata,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
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
