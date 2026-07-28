"""Clip4Retrofit-style one-vs-rest ROC-AUC evaluation.

The Clip4Retrofit paper evaluates zero-shot image labels as follows:

1. Semantic classes present in an image's segmentation mask are treated as
   image-level labels.
2. An image embedding is compared with every class/query embedding using
   cosine similarity.
3. A separate binary ROC curve and its AUC are computed for every class,
   treating that class as positive and every other image as negative.

This module implements that protocol for both single-label multiclass targets
(``y_true.shape == (n_samples,)``) and image-level multi-label targets
(``y_true.shape == (n_samples, n_classes)``).

The paper does not report FPR@95%TPR.  As an explicitly separate extension,
this module reports the first attainable ROC point whose TPR is at least 0.95
for every class, then takes the unweighted arithmetic mean across classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import auc, roc_curve


ArrayLike = Sequence[Any] | np.ndarray


@dataclass(frozen=True)
class ClassROC:
    """One-vs-rest ROC result for one class."""

    class_index: int
    class_label: Any
    class_name: str
    positive_count: int
    negative_count: int
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float
    target_tpr: float
    fpr_at_target_tpr: float
    threshold_at_target_tpr: float


@dataclass(frozen=True)
class OVRMetrics:
    """All per-class curves plus unweighted macro metrics."""

    per_class: tuple[ClassROC, ...]
    macro_auroc: float
    macro_fpr_at_target_tpr: float
    target_tpr: float
    n_samples: int
    n_classes: int


def compute_cosine_similarity_scores(
    image_embeddings: ArrayLike,
    class_embeddings: ArrayLike,
) -> np.ndarray:
    """Return the paper's image-to-query cosine similarity score matrix.

    Args:
        image_embeddings: Shape ``(n_samples, embedding_dim)``.
        class_embeddings: Shape ``(n_classes, embedding_dim)``.  Each row is
            the text/query embedding for the corresponding output class.

    Returns:
        ``float64`` array of shape ``(n_samples, n_classes)``.

    Raises:
        ValueError: If shapes are incompatible, values are non-finite, or an
            embedding has zero norm.
    """

    images = np.asarray(image_embeddings, dtype=np.float64)
    classes = np.asarray(class_embeddings, dtype=np.float64)

    if images.ndim != 2 or classes.ndim != 2:
        raise ValueError("image_embeddings and class_embeddings must both be 2-D.")
    if images.shape[1] != classes.shape[1]:
        raise ValueError(
            "Embedding dimensions differ: "
            f"{images.shape[1]} for images vs {classes.shape[1]} for classes."
        )
    if not np.isfinite(images).all() or not np.isfinite(classes).all():
        raise ValueError("Embeddings must contain only finite values.")

    image_norms = np.linalg.norm(images, axis=1, keepdims=True)
    class_norms = np.linalg.norm(classes, axis=1, keepdims=True)
    if np.any(image_norms == 0.0):
        raise ValueError("image_embeddings contains a zero-norm row.")
    if np.any(class_norms == 0.0):
        raise ValueError("class_embeddings contains a zero-norm row.")

    return (images / image_norms) @ (classes / class_norms).T


def label_sets_to_multihot(
    labels_per_image: Iterable[Iterable[Any]],
    class_labels: Sequence[Any],
) -> np.ndarray:
    """Convert per-image semantic-mask class sets to image-level multi-hot labels.

    This is the target construction described for Cityscapes and Mapillary in
    Clip4Retrofit: every semantic class occurring in an image mask is treated
    as a label for that whole image.
    """

    labels = list(class_labels)
    if not labels:
        raise ValueError("class_labels must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError("class_labels must be unique.")

    label_to_index = {label: index for index, label in enumerate(labels)}
    rows = list(labels_per_image)
    targets = np.zeros((len(rows), len(labels)), dtype=np.uint8)

    for row_index, present_labels in enumerate(rows):
        for label in set(present_labels):
            if label not in label_to_index:
                raise ValueError(
                    f"Unknown class label {label!r} in sample {row_index}."
                )
            targets[row_index, label_to_index[label]] = 1
    return targets


def compute_ovr_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    class_names: Sequence[str] | None = None,
    class_labels: Sequence[Any] | None = None,
    target_tpr: float = 0.95,
    drop_intermediate: bool = True,
) -> OVRMetrics:
    """Compute a separate one-vs-rest ROC-AUC for every class.

    ``y_score[:, class_index]`` must contain that class's confidence score.
    For Clip4Retrofit reproduction, pass raw cosine similarities returned by
    :func:`compute_cosine_similarity_scores`; softmax is neither needed nor
    applied.

    Args:
        y_true: Either integer/string class labels with shape ``(N,)``, or a
            binary multi-hot matrix with shape ``(N, C)``.
        y_score: Per-class confidence scores with shape ``(N, C)``.
        class_names: Optional display names in score-column order.
        class_labels: Optional raw label values in score-column order.  This is
            useful for a 1-D ``y_true`` whose values are not ``0..C-1``.
        target_tpr: Target used only for the added FPR metric.
        drop_intermediate: Forwarded unchanged to ``sklearn.metrics.roc_curve``.
            The default matches scikit-learn's standard ROC construction.

    Returns:
        Per-class curves/AUC values, macro AUROC, and macro FPR@target TPR.

    Raises:
        ValueError: For malformed inputs or when any class lacks either positive
            or negative examples.  ROC-AUC is undefined in that situation.
    """

    scores = np.asarray(y_score, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"y_score must have shape (N, C); got {scores.shape}.")
    if scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("y_score must contain samples and at least two classes.")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values.")
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in the interval (0, 1].")

    n_samples, n_classes = scores.shape
    binary_targets, labels = _prepare_targets(
        y_true=y_true,
        n_samples=n_samples,
        n_classes=n_classes,
        class_labels=class_labels,
    )
    names = _prepare_class_names(class_names, labels)

    class_results: list[ClassROC] = []
    for class_index in range(n_classes):
        binary = binary_targets[:, class_index]
        positive_count = int(binary.sum())
        negative_count = int(binary.size - positive_count)
        if positive_count == 0 or negative_count == 0:
            raise ValueError(
                f"Class {names[class_index]!r} needs at least one positive and "
                f"one negative sample; got {positive_count} positive and "
                f"{negative_count} negative."
            )

        fpr, tpr, thresholds = roc_curve(
            binary,
            scores[:, class_index],
            pos_label=1,
            drop_intermediate=drop_intermediate,
        )
        class_auroc = float(auc(fpr, tpr))
        target_index = _first_index_at_or_above(tpr, target_tpr)

        class_results.append(
            ClassROC(
                class_index=class_index,
                class_label=labels[class_index],
                class_name=names[class_index],
                positive_count=positive_count,
                negative_count=negative_count,
                fpr=fpr,
                tpr=tpr,
                thresholds=thresholds,
                auroc=class_auroc,
                target_tpr=target_tpr,
                fpr_at_target_tpr=float(fpr[target_index]),
                threshold_at_target_tpr=float(thresholds[target_index]),
            )
        )

    macro_auroc = float(np.mean([item.auroc for item in class_results]))
    macro_fpr = float(
        np.mean([item.fpr_at_target_tpr for item in class_results])
    )
    return OVRMetrics(
        per_class=tuple(class_results),
        macro_auroc=macro_auroc,
        macro_fpr_at_target_tpr=macro_fpr,
        target_tpr=target_tpr,
        n_samples=n_samples,
        n_classes=n_classes,
    )


def save_evaluation(
    result: OVRMetrics,
    output_dir: str | Path,
    *,
    create_plots: bool = True,
    dpi: int = 180,
) -> None:
    """Save metrics, full ROC points, and paper-style per-class ROC plots."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _save_summary_json(result, destination / "metrics.json")
    _save_per_class_csv(result, destination / "per_class_metrics.csv")
    _save_curve_arrays(result, destination / "roc_curves.npz")
    if create_plots:
        _save_combined_plot(result, destination / "ovr_roc_curves.png", dpi=dpi)
        _save_subplot_plot(
            result,
            destination / "ovr_roc_subplots.png",
            dpi=dpi,
        )


def _prepare_targets(
    *,
    y_true: ArrayLike,
    n_samples: int,
    n_classes: int,
    class_labels: Sequence[Any] | None,
) -> tuple[np.ndarray, tuple[Any, ...]]:
    targets = np.asarray(y_true)
    if targets.ndim == 1:
        if targets.shape[0] != n_samples:
            raise ValueError(
                f"y_true has {targets.shape[0]} samples; y_score has {n_samples}."
            )
        labels = _resolve_single_label_columns(targets, n_classes, class_labels)
        binary = np.column_stack([targets == label for label in labels]).astype(
            np.uint8
        )
        return binary, labels

    if targets.ndim == 2:
        if targets.shape != (n_samples, n_classes):
            raise ValueError(
                "Multi-label y_true must have the same (N, C) shape as y_score; "
                f"got {targets.shape} and {(n_samples, n_classes)}."
            )
        if not np.isin(targets, (0, 1, False, True)).all():
            raise ValueError("Two-dimensional y_true must be binary/multi-hot.")
        if class_labels is None:
            labels = tuple(range(n_classes))
        else:
            labels = tuple(class_labels)
            if len(labels) != n_classes:
                raise ValueError(
                    f"class_labels has {len(labels)} values; expected {n_classes}."
                )
            if len(set(labels)) != len(labels):
                raise ValueError("class_labels must be unique.")
        return targets.astype(np.uint8, copy=False), labels

    raise ValueError("y_true must be a 1-D class vector or a 2-D multi-hot matrix.")


def _resolve_single_label_columns(
    targets: np.ndarray,
    n_classes: int,
    class_labels: Sequence[Any] | None,
) -> tuple[Any, ...]:
    if class_labels is not None:
        labels = tuple(class_labels)
        if len(labels) != n_classes:
            raise ValueError(
                f"class_labels has {len(labels)} values; expected {n_classes}."
            )
        if len(set(labels)) != len(labels):
            raise ValueError("class_labels must be unique.")
        unknown = set(targets.tolist()) - set(labels)
        if unknown:
            raise ValueError(f"y_true contains labels not in class_labels: {unknown}.")
        return labels

    unique = np.unique(targets)
    if np.issubdtype(targets.dtype, np.integer):
        integer_labels = tuple(range(n_classes))
        if set(unique.tolist()).issubset(set(integer_labels)):
            return integer_labels
    if unique.size == n_classes:
        return tuple(unique.tolist())
    raise ValueError(
        "Cannot infer score-column labels from y_true. Pass class_labels in the "
        "same order as y_score columns."
    )


def _prepare_class_names(
    class_names: Sequence[str] | None,
    class_labels: Sequence[Any],
) -> tuple[str, ...]:
    if class_names is None:
        return tuple(str(label) for label in class_labels)
    names = tuple(str(name) for name in class_names)
    if len(names) != len(class_labels):
        raise ValueError(
            f"class_names has {len(names)} values; expected {len(class_labels)}."
        )
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique.")
    return names


def _first_index_at_or_above(tpr: np.ndarray, target_tpr: float) -> int:
    eligible = np.flatnonzero(tpr >= target_tpr)
    if eligible.size == 0:
        raise RuntimeError(
            f"ROC curve never reaches target TPR {target_tpr:.6f}."
        )
    return int(eligible[0])


def _save_summary_json(result: OVRMetrics, path: Path) -> None:
    payload = {
        "protocol": {
            "name": "Clip4Retrofit-style per-class one-vs-rest ROC-AUC",
            "score": "raw per-class confidence; cosine similarity for paper reproduction",
            "auroc": "sklearn roc_curve followed by trapezoidal sklearn auc",
            "macro_auroc": "unweighted arithmetic mean of per-class AUROCs",
            "fpr_at_target_tpr": (
                "first attainable ROC point with TPR >= target_tpr"
            ),
            "macro_fpr_at_target_tpr": (
                "unweighted arithmetic mean of per-class FPR@targetTPR"
            ),
        },
        "n_samples": result.n_samples,
        "n_classes": result.n_classes,
        "target_tpr": result.target_tpr,
        "macro_auroc": result.macro_auroc,
        "macro_fpr_at_target_tpr": result.macro_fpr_at_target_tpr,
        "per_class": [
            {
                "class_index": item.class_index,
                "class_label": _json_scalar(item.class_label),
                "class_name": item.class_name,
                "positive_count": item.positive_count,
                "negative_count": item.negative_count,
                "auroc": item.auroc,
                "fpr_at_target_tpr": item.fpr_at_target_tpr,
                "threshold_at_target_tpr": item.threshold_at_target_tpr,
            }
            for item in result.per_class
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_per_class_csv(result: OVRMetrics, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class_index",
                "class_label",
                "class_name",
                "positive_count",
                "negative_count",
                "auroc",
                f"fpr_at_tpr_{result.target_tpr:g}",
                f"threshold_at_tpr_{result.target_tpr:g}",
            ]
        )
        for item in result.per_class:
            writer.writerow(
                [
                    item.class_index,
                    item.class_label,
                    item.class_name,
                    item.positive_count,
                    item.negative_count,
                    f"{item.auroc:.12g}",
                    f"{item.fpr_at_target_tpr:.12g}",
                    f"{item.threshold_at_target_tpr:.12g}",
                ]
            )


def _save_curve_arrays(result: OVRMetrics, path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for item in result.per_class:
        prefix = f"class_{item.class_index}"
        arrays[f"{prefix}_fpr"] = item.fpr
        arrays[f"{prefix}_tpr"] = item.tpr
        arrays[f"{prefix}_thresholds"] = item.thresholds
    np.savez_compressed(path, **arrays)


def _save_combined_plot(result: OVRMetrics, path: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    for item in result.per_class:
        axis.plot(
            item.fpr,
            item.tpr,
            linewidth=2,
            label=f"{item.class_name} (AUC={item.auroc:.4f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=(
            "One-vs-Rest ROC Curves\n"
            f"Macro AUROC={result.macro_auroc:.4f}, "
            f"Macro FPR@{result.target_tpr:.0%}TPR="
            f"{result.macro_fpr_at_target_tpr:.4f}"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize="small")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _save_subplot_plot(result: OVRMetrics, path: Path, *, dpi: int) -> None:
    """Save one subplot per class, matching the paper's per-class presentation."""

    import matplotlib.pyplot as plt

    ncols = max(1, min(4, math.ceil(math.sqrt(result.n_classes))))
    nrows = math.ceil(result.n_classes / ncols)
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for axis, item in zip(axes.flat, result.per_class):
        axis.plot(
            item.fpr,
            item.tpr,
            linewidth=2,
            label=f"ROC (AUC={item.auroc:.4f})",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        axis.scatter(
            [item.fpr_at_target_tpr],
            [item.tpr[_first_index_at_or_above(item.tpr, item.target_tpr)]],
            s=28,
            zorder=3,
            label=(
                f"FPR@{item.target_tpr:.0%}TPR="
                f"{item.fpr_at_target_tpr:.4f}"
            ),
        )
        axis.set(
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f'ROC curve for "{item.class_name}" class',
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize="small")

    for axis in axes.flat[result.n_classes :]:
        axis.set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Clip4Retrofit-style per-class one-vs-rest ROC-AUC and "
            "macro FPR@95%TPR from an NPZ file."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "NPZ containing y_true and y_score. y_true may be (N,) labels or "
            "(N,C) multi-hot targets; y_score must be (N,C)."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Optional display names in the same order as y_score columns.",
    )
    parser.add_argument("--target-tpr", type=float, default=0.95)
    parser.add_argument(
        "--drop-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use scikit-learn's standard intermediate ROC point pruning.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    with np.load(input_path, allow_pickle=False) as payload:
        if "y_true" not in payload or "y_score" not in payload:
            raise KeyError("Input NPZ must contain y_true and y_score arrays.")
        y_true = payload["y_true"]
        y_score = payload["y_score"]
        stored_names = (
            [str(value) for value in payload["class_names"].tolist()]
            if "class_names" in payload
            else None
        )
        stored_labels = (
            payload["class_labels"].tolist()
            if "class_labels" in payload
            else None
        )

    class_names = args.class_names if args.class_names else stored_names
    result = compute_ovr_metrics(
        y_true,
        y_score,
        class_names=class_names,
        class_labels=stored_labels,
        target_tpr=args.target_tpr,
        drop_intermediate=args.drop_intermediate,
    )
    save_evaluation(
        result,
        args.output_dir,
        create_plots=args.plots,
        dpi=args.dpi,
    )

    print(f"Samples: {result.n_samples}")
    print(f"Classes: {result.n_classes}")
    for item in result.per_class:
        print(
            f"{item.class_name}: AUROC={item.auroc:.6f}, "
            f"FPR@{result.target_tpr:.0%}TPR={item.fpr_at_target_tpr:.6f}"
        )
    print(f"Macro AUROC: {result.macro_auroc:.6f}")
    print(
        f"Macro FPR@{result.target_tpr:.0%}TPR: "
        f"{result.macro_fpr_at_target_tpr:.6f}"
    )


if __name__ == "__main__":
    main()
