#!/usr/bin/env python3
"""Run CSV-backed zero-shot evaluation without changing the metric protocol.

This is an inference adapter for :mod:`clip4retrofit_ovr_metrics`, not a new
metric implementation.  It performs the missing upstream steps:

``CSV -> image/text embeddings -> raw cosine-similarity scores -> existing OVR``

The resulting ``predictions.npz`` is compatible with the existing NPZ-only
metric CLI.  ``compute_cosine_similarity_scores`` and ``compute_ovr_metrics``
remain the sole implementations of the Clip4Retrofit-style scoring and ROC
calculation.

All model downloads are forced into ``a1_inference/.hf_cache`` and all output
artifacts into ``a1_inference/csv_classwise_results`` by default.  Input image
paths and all writable paths are restricted to this MedCLIP-SAMv2 project.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

try:  # Direct-script execution from a1_inference.
    from clip4retrofit_ovr_metrics import (
        compute_cosine_similarity_scores,
        compute_ovr_metrics,
        save_evaluation,
    )
except ModuleNotFoundError:  # Module execution from the repository root.
    from a1_inference.clip4retrofit_ovr_metrics import (
        compute_cosine_similarity_scores,
        compute_ovr_metrics,
        save_evaluation,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CSV = SCRIPT_DIR / "udiat_classwise_eval.csv"
DEFAULT_CACHE_DIR = SCRIPT_DIR / ".hf_cache"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "csv_classwise_results"
DEFAULT_CLASS_ORDER = ("benign", "malignant", "normal")


@dataclass(frozen=True)
class CsvDataset:
    """Validated single-label dataset and one text prompt per output class."""

    image_paths: tuple[Path, ...]
    labels: np.ndarray
    captions: tuple[str, ...]
    class_order: tuple[str, ...]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    backend: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "openai_clip": ModelSpec(
        key="openai_clip",
        model_id="openai/clip-vit-large-patch14",
        backend="transformers",
    ),
    "siglip": ModelSpec(
        key="siglip",
        model_id="google/siglip-base-patch16-224",
        backend="transformers",
    ),
    "medsiglip": ModelSpec(
        key="medsiglip",
        model_id="google/medsiglip-448",
        backend="transformers",
    ),
    "biomedclip": ModelSpec(
        key="biomedclip",
        model_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        backend="open_clip",
    ),
}


def _path_in_project(value: str | Path, *, description: str) -> Path:
    """Resolve a path and reject anything outside this project directory."""

    candidate = Path(value).expanduser().resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(
            f"{description} must stay under {PROJECT_ROOT}; got {candidate}."
        ) from error
    return candidate


def _configure_local_cache() -> Path:
    """Force all Hugging Face / Transformers cache writes into a1_inference."""

    cache_dir = _path_in_project(DEFAULT_CACHE_DIR, description="Model cache")
    hub_dir = cache_dir / "hub"
    transformers_dir = cache_dir / "transformers"
    for directory in (cache_dir, hub_dir, transformers_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Assignment (rather than setdefault) intentionally prevents a global
    # cache setting from moving model files outside a1_inference.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    return cache_dir


def load_csv_dataset(csv_path: str | Path, class_order: Sequence[str]) -> CsvDataset:
    """Read the required CSV columns and derive exactly one text prompt/class."""

    source = _path_in_project(csv_path, description="CSV input")
    if not source.is_file():
        raise FileNotFoundError(f"CSV input does not exist: {source}")

    ordered_labels = tuple(class_order)
    if len(ordered_labels) < 2 or len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("--class-order must contain at least two unique labels.")

    image_paths: list[Path] = []
    labels: list[str] = []
    captions_by_label: dict[str, str] = {}
    required_columns = {"image_path", "class_label", "caption"}

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            label = (row.get("class_label") or "").strip()
            caption = (row.get("caption") or "").strip()
            if not raw_path or not label or not caption:
                raise ValueError(
                    f"CSV row {row_number} needs non-empty image_path, class_label, "
                    "and caption."
                )
            if label not in ordered_labels:
                raise ValueError(
                    f"CSV row {row_number} uses unknown label {label!r}; expected "
                    f"one of {ordered_labels}."
                )

            image_path = _path_in_project(raw_path, description=f"Image on row {row_number}")
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image on CSV row {row_number} does not exist: {image_path}"
                )
            existing_caption = captions_by_label.setdefault(label, caption)
            if existing_caption != caption:
                raise ValueError(
                    f"Label {label!r} has more than one caption. This evaluator "
                    "requires one fixed text embedding per class."
                )
            image_paths.append(image_path)
            labels.append(label)

    if not image_paths:
        raise ValueError("CSV contains no samples.")
    missing_captions = [label for label in ordered_labels if label not in captions_by_label]
    if missing_captions:
        raise ValueError(
            "CSV needs at least one row for every class. Missing: "
            f"{missing_captions}"
        )
    missing_positives = [label for label in ordered_labels if label not in labels]
    if missing_positives:
        raise ValueError(f"CSV has no positive samples for: {missing_positives}")

    return CsvDataset(
        image_paths=tuple(image_paths),
        labels=np.asarray(labels, dtype=str),
        captions=tuple(captions_by_label[label] for label in ordered_labels),
        class_order=ordered_labels,
    )


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required. Run this script with the project evaluation "
            "environment (for example, bioclip2/.venv/bin/python)."
        ) from error
    return torch


def resolve_device(requested: str) -> str:
    torch = _import_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return requested


def _load_rgb_images(paths: Sequence[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _move_tensor_inputs(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        name: value.to(device) if hasattr(value, "to") else value
        for name, value in inputs.items()
    }


def _call_feature_method(method: Callable[..., Any], inputs: dict[str, Any]) -> Any:
    """Call model feature methods while passing only supported processor fields."""

    parameters = inspect.signature(method).parameters.values()
    accepts_kwargs = any(parameter.kind == parameter.VAR_KEYWORD for parameter in parameters)
    if accepts_kwargs:
        usable = inputs
    else:
        allowed = {parameter.name for parameter in parameters}
        usable = {name: value for name, value in inputs.items() if name in allowed}
    output = method(**usable)
    if hasattr(output, "pooler_output"):
        output = output.pooler_output
    if isinstance(output, tuple):
        output = output[0]
    if not hasattr(output, "detach"):
        raise TypeError(
            f"{method.__qualname__} did not return a tensor-like embedding."
        )
    return output


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def encode_with_transformers(
    model_id: str,
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalised feature vectors from a standard HF CLIP/SigLIP model."""

    try:
        from transformers import AutoModel, AutoProcessor
    except ModuleNotFoundError as error:
        raise RuntimeError("transformers is required for this model backend.") from error

    torch = _import_torch()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    try:
        with torch.inference_mode():
            text_inputs = processor(
                text=list(captions),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            text_features = _call_feature_method(
                model.get_text_features,
                _move_tensor_inputs(dict(text_inputs), device),
            )

            image_batches: list[np.ndarray] = []
            for start in range(0, len(image_paths), batch_size):
                image_inputs = processor(
                    images=_load_rgb_images(image_paths[start : start + batch_size]),
                    return_tensors="pt",
                )
                image_features = _call_feature_method(
                    model.get_image_features,
                    _move_tensor_inputs(dict(image_inputs), device),
                )
                image_batches.append(_tensor_to_numpy(image_features))
        return np.vstack(image_batches), _tensor_to_numpy(text_features)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def encode_with_open_clip(
    model_id: str,
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return BioMedCLIP feature vectors using its official OpenCLIP interface."""

    try:
        import open_clip
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "BioMedCLIP needs open-clip-torch. Install the isolated dependency "
            "listed in a1_inference/requirements-csv-eval.txt."
        ) from error

    torch = _import_torch()
    hf_name = f"hf-hub:{model_id}"
    model, preprocess = open_clip.create_model_from_pretrained(
        hf_name,
        device=device,
        cache_dir=str(DEFAULT_CACHE_DIR / "hub"),
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(hf_name)
    try:
        with torch.inference_mode():
            try:
                tokens = tokenizer(list(captions), context_length=256)
            except TypeError:
                tokens = tokenizer(list(captions))
            text_features = model.encode_text(tokens.to(device))

            image_batches: list[np.ndarray] = []
            for start in range(0, len(image_paths), batch_size):
                image_tensors = [
                    preprocess(image)
                    for image in _load_rgb_images(image_paths[start : start + batch_size])
                ]
                image_features = model.encode_image(torch.stack(image_tensors).to(device))
                image_batches.append(_tensor_to_numpy(image_features))
        return np.vstack(image_batches), _tensor_to_numpy(text_features)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_predictions_npz(
    destination: Path,
    *,
    dataset: CsvDataset,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    y_score: np.ndarray,
) -> None:
    np.savez_compressed(
        destination,
        y_true=dataset.labels,
        y_score=np.asarray(y_score, dtype=np.float64),
        class_names=np.asarray(dataset.class_order, dtype=str),
        class_labels=np.asarray(dataset.class_order, dtype=str),
        image_paths=np.asarray([str(path) for path in dataset.image_paths], dtype=str),
        captions=np.asarray(dataset.captions, dtype=str),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        text_embeddings=np.asarray(text_embeddings, dtype=np.float32),
    )


def _write_metadata(
    destination: Path, *, model: ModelSpec, dataset: CsvDataset, device: str) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_key": model.key,
        "model_id": model.model_id,
        "backend": model.backend,
        "device": device,
        "n_samples": len(dataset.image_paths),
        "class_order": list(dataset.class_order),
        "captions": list(dataset.captions),
        "score_definition": "raw L2-normalised image/text cosine similarity",
        "metric_implementation": "clip4retrofit_ovr_metrics.py (unchanged)",
        "cache_dir": str(DEFAULT_CACHE_DIR),
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def evaluate_one_model(
    model: ModelSpec,
    dataset: CsvDataset,
    *,
    output_root: Path,
    batch_size: int,
    device: str,
    target_tpr: float,
    create_plots: bool,
) -> None:
    """Embed one model, then hand unmodified core metric functions its scores."""

    model_output = output_root / model.key
    model_output.mkdir(parents=True, exist_ok=True)
    if model.backend == "transformers":
        image_embeddings, text_embeddings = encode_with_transformers(
            model.model_id,
            dataset.image_paths,
            dataset.captions,
            batch_size=batch_size,
            device=device,
        )
    elif model.backend == "open_clip":
        image_embeddings, text_embeddings = encode_with_open_clip(
            model.model_id,
            dataset.image_paths,
            dataset.captions,
            batch_size=batch_size,
            device=device,
        )
    else:  # Defensive guard in case a model spec is added incorrectly.
        raise RuntimeError(f"Unsupported model backend: {model.backend}")

    if image_embeddings.shape[0] != len(dataset.image_paths):
        raise RuntimeError("Model returned a different number of image embeddings.")
    if text_embeddings.shape[0] != len(dataset.class_order):
        raise RuntimeError("Model returned a different number of text embeddings.")

    # This is the existing core cosine implementation. Do not replace it with
    # a model-specific logit scale, sigmoid, or softmax.
    y_score = compute_cosine_similarity_scores(image_embeddings, text_embeddings)
    _write_predictions_npz(
        model_output / "predictions.npz",
        dataset=dataset,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        y_score=y_score,
    )

    # This is the existing, unchanged Clip4Retrofit-style OVR AUROC/FPR logic.
    metrics = compute_ovr_metrics(
        dataset.labels,
        y_score,
        class_names=dataset.class_order,
        class_labels=dataset.class_order,
        target_tpr=target_tpr,
    )
    save_evaluation(metrics, model_output, create_plots=create_plots)
    _write_metadata(model_output / "run_metadata.json", model=model, dataset=dataset, device=device)
    # A previous invocation may have failed before a later retry succeeds.
    # Do not leave that stale failure marker beside valid, newly written scores.
    (model_output / "failure.json").unlink(missing_ok=True)

    print(f"[{model.key}] samples={metrics.n_samples}, classes={metrics.n_classes}")
    for item in metrics.per_class:
        print(
            f"[{model.key}] {item.class_name}: AUROC={item.auroc:.6f}, "
            f"FPR@{target_tpr:.0%}TPR={item.fpr_at_target_tpr:.6f}"
        )
    print(f"[{model.key}] macro AUROC={metrics.macro_auroc:.6f}")
    print(f"[{model.key}] macro FPR@{target_tpr:.0%}TPR={metrics.macro_fpr_at_target_tpr:.6f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate zero-shot image/text cosine scores from CSV, save a compatible "
            "NPZ, and use the existing Clip4Retrofit OVR AUROC/FPR implementation."
        )
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="CSV with image_path,class_label,caption.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Evaluation artifacts directory (must stay inside this project).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=tuple(MODEL_SPECS),
        help="Models to evaluate sequentially (default: all four).",
    )
    parser.add_argument(
        "--class-order",
        nargs="+",
        default=DEFAULT_CLASS_ORDER,
        help="Score-column order and required labels (default: benign malignant normal).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save existing ROC visualisations alongside numerical artifacts.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first model failure instead of processing later models.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    _configure_local_cache()
    output_root = _path_in_project(args.output_dir, description="Output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_csv_dataset(args.csv, args.class_order)
    device = resolve_device(args.device)
    print(f"CSV samples: {len(dataset.image_paths)}")
    print(f"Class order: {', '.join(dataset.class_order)}")
    print(f"Device: {device}")
    print(f"Model cache: {DEFAULT_CACHE_DIR}")

    failed_models: list[str] = []
    for model_key in args.models:
        model = MODEL_SPECS[model_key]
        try:
            evaluate_one_model(
                model,
                dataset,
                output_root=output_root,
                batch_size=args.batch_size,
                device=device,
                target_tpr=args.target_tpr,
                create_plots=args.plots,
            )
        except Exception as error:  # Keep other requested models evaluable.
            failed_models.append(model_key)
            failure_dir = output_root / model_key
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "model_key": model_key,
                "model_id": model.model_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            (failure_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[{model_key}] FAILED: {error}", file=sys.stderr)
            if args.fail_fast:
                break

    if failed_models:
        raise SystemExit(
            "Evaluation failed for: " + ", ".join(failed_models) + ". See each failure.json."
        )


if __name__ == "__main__":
    main()
