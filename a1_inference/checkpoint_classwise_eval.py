#!/usr/bin/env python3
"""Evaluate a compatible OpenCLIP checkpoint with the unchanged OVR metrics.

This is a checkpoint-specific companion to ``csv_classwise_eval.py``.  It
preserves the UDIAT CSV, captions, preprocessing, raw cosine score creation,
and Clip4Retrofit-style AUROC/FPR calculation used for the four base models.
Only the BioMedCLIP weights are replaced with the supplied checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from csv_classwise_eval import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLASS_ORDER,
    DEFAULT_CSV,
    PROJECT_ROOT,
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
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "model" / "buscot-upstream" / "global_ind_gmpo" / "epoch_3.pt"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "csv_classwise_results" / "global_ind_gmpo_epoch_3"
BIOMEDCLIP_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _import_torch() -> Any:
    import torch

    return torch


def _load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    """Load the trainer checkpoint and return its strictly compatible state dict.

    The supplied checkpoint uses an older NumPy scalar pickle global, which
    cannot be aliases-safe-loaded by the project's NumPy 1.26 runtime.  Its
    archive metadata was checked before use and contains only standard tensor,
    NumPy scalar, and dictionary globals.  ``mmap`` avoids materialising the
    unrelated optimizer tensors in RAM while model weights are copied to GPU.
    """

    torch = _import_torch()
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except (TypeError, RuntimeError):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Checkpoint must contain a dictionary named 'state_dict'.")
    state_dict = payload["state_dict"]
    if not all(isinstance(key, str) for key in state_dict):
        raise ValueError("Checkpoint state_dict keys must be strings.")
    return state_dict


def _encode_checkpoint(
    checkpoint_path: Path,
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the checkpoint to the exact BioMedCLIP architecture/tokenizer."""

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
        checkpoint_state = _load_checkpoint_state(checkpoint_path)
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
                image_tensors = [
                    preprocess(image)
                    for image in _load_rgb_images(image_paths[start : start + batch_size])
                ]
                image_embeddings = model.encode_image(
                    torch.stack(image_tensors).to(device)
                )
                image_batches.append(
                    image_embeddings.detach().float().cpu().numpy()
                )

        model_info = {
            "base_architecture": BIOMEDCLIP_MODEL_ID,
            "image_preprocess": str(preprocess),
            "checkpoint_state_key_count": len(model.state_dict()),
        }
        return np.vstack(image_batches), text_embeddings.detach().float().cpu().numpy(), model_info
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
    class_order: Sequence[str],
    image_paths: Sequence[Path],
    captions: Sequence[str],
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    y_score: np.ndarray,
    checkpoint_path: Path,
) -> None:
    np.savez_compressed(
        destination,
        y_true=labels,
        y_score=np.asarray(y_score, dtype=np.float64),
        class_names=np.asarray(class_order, dtype=str),
        class_labels=np.asarray(class_order, dtype=str),
        image_paths=np.asarray([str(path) for path in image_paths], dtype=str),
        captions=np.asarray(captions, dtype=str),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        text_embeddings=np.asarray(text_embeddings, dtype=np.float32),
        checkpoint_path=np.asarray(str(checkpoint_path), dtype=str),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the UDIAT CSV with epoch_3.pt, then call the unchanged "
            "Clip4Retrofit-style AUROC/FPR metric implementation."
        )
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--class-order", nargs="+", default=DEFAULT_CLASS_ORDER)
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
    dataset = load_csv_dataset(args.csv, args.class_order)
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

    # Keep the scoring and metric definitions exactly shared with the prior runs.
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
                "checkpoint_name": "global_ind_gmpo_seed12",
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
