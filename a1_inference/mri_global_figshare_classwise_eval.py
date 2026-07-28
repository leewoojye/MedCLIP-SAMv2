#!/usr/bin/env python3
"""Evaluate the DDP-formatted MRI global checkpoint on Figshare 6,128 images.

This file intentionally leaves the prior UDIAT checkpoint evaluator unchanged.
It differs only by normalising this checkpoint's required ``module.`` DDP key
prefix before strict state-dict loading.  CSV prompts, raw cosine scoring, and
the Clip4Retrofit-style OVR metric core are shared unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from checkpoint_classwise_eval import (
    BIOMEDCLIP_MODEL_ID,
    _import_torch,
    _load_checkpoint_state,
    _write_predictions,
)
from csv_classwise_eval import (
    DEFAULT_CACHE_DIR,
    _configure_local_cache,
    _load_rgb_images,
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
FIGSHARE_CSV = SCRIPT_DIR / "figshare3064x2_classwise_eval.csv"
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "model" / "mri_global" / "global_ind_gmpo_epoch_3.pt"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "csv_classwise_results"
    / "figshare3064x2_dilation9"
    / "mri_global_ind_gmpo_epoch_3"
)
CLASS_ORDER = ("meningioma", "glioma", "pituitary", "normal")


def _normalise_ddp_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip the uniform DDP ``module.`` prefix and reject mixed key formats."""

    keys = tuple(state_dict)
    if not keys:
        raise ValueError("Checkpoint state_dict is empty.")
    prefixed = [key.startswith("module.") for key in keys]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Checkpoint mixes DDP-prefixed and unprefixed state_dict keys.")
    if not all(prefixed):
        return state_dict

    normalised = {key[len("module.") :]: value for key, value in state_dict.items()}
    if len(normalised) != len(state_dict):
        raise ValueError("DDP prefix removal produced duplicate state_dict keys.")
    return normalised


def _encode_mri_global_checkpoint(
    checkpoint_path: Path,
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import open_clip
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "open-clip-torch is required. Run a1_inference/setup_csv_classwise_eval.sh."
        ) from error

    torch = _import_torch()
    model = None
    try:
        model, preprocess = open_clip.create_model_from_pretrained(
            BIOMEDCLIP_MODEL_ID,
            device=device,
            cache_dir=str(DEFAULT_CACHE_DIR / "hub"),
        )
        checkpoint_state = _normalise_ddp_state_dict(
            _load_checkpoint_state(checkpoint_path)
        )
        incompatible = model.load_state_dict(checkpoint_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Strict checkpoint load reported incompatible keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}."
            )
        del checkpoint_state
        gc.collect()

        tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL_ID)
        model.eval()
        with torch.inference_mode():
            try:
                tokens = tokenizer(list(captions), context_length=256)
            except TypeError:
                tokens = tokenizer(list(captions))
            text_embeddings = model.encode_text(tokens.to(device))

            image_batches: list[np.ndarray] = []
            for start in range(0, len(image_paths), batch_size):
                tensors = [
                    preprocess(image)
                    for image in _load_rgb_images(image_paths[start : start + batch_size])
                ]
                image_embeddings = model.encode_image(torch.stack(tensors).to(device))
                image_batches.append(
                    image_embeddings.detach().float().cpu().numpy()
                )
        model_info = {
            "base_architecture": BIOMEDCLIP_MODEL_ID,
            "checkpoint_state_key_count": len(model.state_dict()),
            "state_dict_key_normalisation": "removed uniform 'module.' DDP prefix",
            "image_preprocess": str(preprocess),
        }
        return (
            np.vstack(image_batches),
            text_embeddings.detach().float().cpu().numpy(),
            model_info,
        )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate mri_global/global_ind_gmpo_epoch_3.pt on the Figshare "
            "3-tumor-plus-normal CSV using the unchanged OVR metric core."
        )
    )
    parser.add_argument("--csv", default=str(FIGSHARE_CSV))
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
    image_embeddings, text_embeddings, model_info = _encode_mri_global_checkpoint(
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

    # Reuse the exact score and metric functions used for all previous models.
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
                "checkpoint_name": "brain_mri_gmpo_global_ind_seed0",
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
