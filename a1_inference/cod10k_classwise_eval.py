#!/usr/bin/env python3
"""Evaluate the path-corrected COD10K CSV with five CLIP-family models.

This is an inference adapter only.  It keeps the score and OVR metric logic in
``clip4retrofit_ovr_metrics.py`` unchanged: raw L2-normalised image/text
cosines, per-class one-vs-rest ROC/AUROC, first FPR at TPR >= 0.95, and an
unweighted macro mean.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from checkpoint_classwise_eval import _load_checkpoint_state
from csv_classwise_eval import (
    DEFAULT_CACHE_DIR,
    _configure_local_cache,
    _load_rgb_images,
    _path_in_project,
    _tensor_to_numpy,
    encode_with_transformers,
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
    / "COD10K"
    / "test"
    / "cod10k_original2026_no_object2026_subclass70_caption_pairs_paths_fixed.csv"
)
DEFAULT_OUTPUT_ROOT = (
    SCRIPT_DIR / "csv_classwise_results" / "cod10k_original2026_negative_d9_subclass70"
)
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR
    / "model"
    / "cod10k_global"
    / "checkpoints"
    / "gmpo_global_ind_explicitneg_2886_seed0_epoch_3.pt"
)


@dataclass(frozen=True)
class Cod10kDataset:
    image_paths: tuple[Path, ...]
    labels: np.ndarray
    class_names: tuple[str, ...]
    class_labels: tuple[int, ...]
    captions: tuple[str, ...]


MODEL_IDS = {
    "clip": "openai/clip-vit-large-patch14",
    "siglip": "google/siglip-base-patch16-224",
    "bioclip": "hf-hub:imageomics/bioclip",
    "bioclip2": "hf-hub:imageomics/bioclip-2",
    "checkpoint": "ViT-B-32 (OpenAI CLIP architecture)",
}
LOCAL_OPEN_CLIP_SOURCES = {
    "bioclip": PROJECT_ROOT / "bioclip" / "src",
    "bioclip2": PROJECT_ROOT / "bioclip2" / "src",
}


def _import_torch() -> Any:
    import torch

    return torch


def load_cod10k_csv(csv_path: str | Path) -> Cod10kDataset:
    """Load the supplied CSV without changing its label/caption definitions."""

    source = _path_in_project(csv_path, description="COD10K CSV")
    if not source.is_file():
        raise FileNotFoundError(f"COD10K CSV does not exist: {source}")

    image_paths: list[Path] = []
    labels: list[int] = []
    names_by_label: dict[int, str] = {}
    captions_by_label: dict[int, str] = {}
    seen_paths: set[Path] = set()
    required_columns = {"image_path", "label_name", "label", "caption"}

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("COD10K CSV has no header row.")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(f"COD10K CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            class_name = (row.get("label_name") or "").strip()
            caption = (row.get("caption") or "").strip()
            raw_label = (row.get("label") or "").strip()
            if not raw_path or not class_name or not caption or not raw_label:
                raise ValueError(
                    f"Row {row_number} needs image_path, label_name, label, and caption."
                )
            try:
                class_label = int(raw_label)
            except ValueError as error:
                raise ValueError(f"Row {row_number} has a non-integer label: {raw_label!r}") from error
            image_path = _path_in_project(raw_path, description=f"Image on row {row_number}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Image on row {row_number} is missing: {image_path}")
            if image_path in seen_paths:
                raise ValueError(f"COD10K CSV has a duplicate image path: {image_path}")
            seen_paths.add(image_path)

            previous_name = names_by_label.setdefault(class_label, class_name)
            previous_caption = captions_by_label.setdefault(class_label, caption)
            if previous_name != class_name:
                raise ValueError(f"Label {class_label} has inconsistent class names.")
            if previous_caption != caption:
                raise ValueError(f"Label {class_label} has inconsistent captions.")
            image_paths.append(image_path)
            labels.append(class_label)

    class_labels = tuple(sorted(names_by_label))
    if class_labels != tuple(range(70)):
        raise ValueError(f"COD10K must provide exactly labels 0..69; found {class_labels}.")
    if len(image_paths) != 4052:
        raise ValueError(f"COD10K must provide 4,052 samples; found {len(image_paths)}.")
    if sum(label == 69 for label in labels) != 2026:
        raise ValueError("COD10K label 69 must contain all 2,026 no-object samples.")
    if sum(label != 69 for label in labels) != 2026:
        raise ValueError("COD10K labels 0..68 must contain 2,026 original samples.")

    return Cod10kDataset(
        image_paths=tuple(image_paths),
        labels=np.asarray(labels, dtype=np.int64),
        class_names=tuple(names_by_label[label] for label in class_labels),
        class_labels=class_labels,
        captions=tuple(captions_by_label[label] for label in class_labels),
    )


def _import_local_open_clip(model_key: str) -> Any:
    source = LOCAL_OPEN_CLIP_SOURCES[model_key]
    if not source.is_dir():
        raise FileNotFoundError(f"Local {model_key} OpenCLIP source is missing: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    module = importlib.import_module("open_clip")
    module_file = Path(getattr(module, "__file__", "")).resolve()
    if not module_file.is_relative_to(source.resolve()):
        raise RuntimeError(
            f"Expected {model_key} OpenCLIP from {source}, but imported {module_file}."
        )
    return module


def _first_tensor(value: Any) -> Any:
    return value[0] if isinstance(value, tuple) else value


def _encode_open_clip_model(
    open_clip: Any,
    model_id: str,
    image_paths: Sequence[Path],
    captions: Sequence[str],
    *,
    batch_size: int,
    device: str,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode with an official OpenCLIP model, optionally swapping exact weights."""

    torch = _import_torch()
    model = None
    try:
        if checkpoint_path is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_id,
                device=device,
                cache_dir=str(DEFAULT_CACHE_DIR / "hub"),
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device=device,
                cache_dir=str(DEFAULT_CACHE_DIR / "hub"),
            )
            checkpoint_state = _load_checkpoint_state(checkpoint_path)
            if checkpoint_state and all(key.startswith("module.") for key in checkpoint_state):
                checkpoint_state = {
                    key.removeprefix("module."): value
                    for key, value in checkpoint_state.items()
                }
            incompatible = model.load_state_dict(checkpoint_state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Strict checkpoint load reported incompatible keys: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}."
                )
            del checkpoint_state
            gc.collect()

        model.eval()
        tokenizer_name = "ViT-B-32" if checkpoint_path is not None else model_id
        tokenizer = open_clip.get_tokenizer(tokenizer_name)
        with torch.inference_mode():
            text_embeddings = _first_tensor(model.encode_text(tokenizer(list(captions)).to(device)))
            image_batches: list[np.ndarray] = []
            for start in range(0, len(image_paths), batch_size):
                images = _load_rgb_images(image_paths[start : start + batch_size])
                image_tensors = torch.stack([preprocess(image) for image in images]).to(device)
                image_embeddings = _first_tensor(model.encode_image(image_tensors))
                image_batches.append(_tensor_to_numpy(image_embeddings))
        metadata = {
            "model_id": model_id,
            "open_clip_module": str(Path(open_clip.__file__).resolve()),
            "image_preprocess": str(preprocess),
            "checkpoint_state_key_count": len(model.state_dict()),
        }
        return np.vstack(image_batches), _tensor_to_numpy(text_embeddings), metadata
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _encode_model(
    model_key: str,
    dataset: Cod10kDataset,
    *,
    checkpoint_path: Path,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if model_key in {"clip", "siglip"}:
        embeddings = encode_with_transformers(
            MODEL_IDS[model_key],
            dataset.image_paths,
            dataset.captions,
            batch_size=batch_size,
            device=device,
        )
        return (*embeddings, {"model_id": MODEL_IDS[model_key], "backend": "transformers"})
    if model_key in {"bioclip", "bioclip2"}:
        open_clip = _import_local_open_clip(model_key)
        images, texts, metadata = _encode_open_clip_model(
            open_clip,
            MODEL_IDS[model_key],
            dataset.image_paths,
            dataset.captions,
            batch_size=batch_size,
            device=device,
        )
        return images, texts, {"backend": f"{model_key}_local_open_clip", **metadata}
    if model_key == "checkpoint":
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"COD10K checkpoint does not exist: {checkpoint_path}")
        try:
            import open_clip
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "open-clip-torch is required for the COD10K checkpoint. "
                "Run a1_inference/setup_csv_classwise_eval.sh first."
            ) from error
        images, texts, metadata = _encode_open_clip_model(
            open_clip,
            MODEL_IDS[model_key],
            dataset.image_paths,
            dataset.captions,
            batch_size=batch_size,
            device=device,
            checkpoint_path=checkpoint_path,
        )
        return images, texts, {
            "backend": "open_clip_torch",
            "checkpoint_path": str(checkpoint_path),
            "ddp_module_prefix_removed": True,
            **metadata,
        }
    raise ValueError(f"Unknown model key: {model_key}")


def _write_predictions(
    path: Path,
    dataset: Cod10kDataset,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    y_score: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        y_true=dataset.labels,
        y_score=np.asarray(y_score, dtype=np.float64),
        class_names=np.asarray(dataset.class_names, dtype=str),
        class_labels=np.asarray(dataset.class_labels, dtype=np.int64),
        image_paths=np.asarray([str(item) for item in dataset.image_paths], dtype=str),
        captions=np.asarray(dataset.captions, dtype=str),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        text_embeddings=np.asarray(text_embeddings, dtype=np.float32),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="COD10K 70-subclass OVR evaluation using the unchanged core metrics."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
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
    dataset = load_cod10k_csv(args.csv)
    checkpoint_path = _path_in_project(args.checkpoint, description="COD10K checkpoint")
    output_dir = _path_in_project(
        args.output_dir or (DEFAULT_OUTPUT_ROOT / args.model),
        description="COD10K output directory",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Model: {args.model} ({MODEL_IDS[args.model]})")
    print(f"Samples: {len(dataset.image_paths)}, classes: {len(dataset.class_labels)}")
    print(f"Device: {device}")

    image_embeddings, text_embeddings, metadata = _encode_model(
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

    # These calls are deliberately the pre-existing, shared metric core.
    y_score = compute_cosine_similarity_scores(image_embeddings, text_embeddings)
    _write_predictions(
        output_dir / "predictions.npz",
        dataset,
        image_embeddings,
        text_embeddings,
        y_score,
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
