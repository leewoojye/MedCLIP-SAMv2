# Clip4Retrofit-style multiclass ROC-AUC

`clip4retrofit_ovr_metrics.py` implements the per-class ROC-AUC protocol used
in **Clip4Retrofit: Enabling Real-Time Image Labeling on Edge Devices via
Cross-Architecture CLIP Distillation** (CVPRW 2025).

The paper's protocol is:

1. Convert every semantic class present in an image mask into an image-level
   label. An image may therefore be positive for more than one class.
2. Compare each image embedding with every class/query text embedding using
   cosine similarity.
3. For each class independently, treat images containing that class as
   positive and all remaining images as negative.
4. Construct one binary ROC curve per class and compute its trapezoidal AUC.

The module supports both ordinary single-label multiclass targets `(N,)` and
the paper's image-level multi-label targets `(N, C)`.

## Added metrics

The paper reports per-class AUROC. This implementation additionally reports:

- **Macro AUROC**: unweighted mean of the per-class one-vs-rest AUROCs.
- **Per-class FPR@95%TPR**: FPR at the first attainable ROC point with
  `TPR >= 0.95`.
- **Macro FPR@95%TPR**: unweighted mean of those per-class FPR values.

A class must have at least one positive and one negative sample. The evaluator
raises an error instead of silently dropping an undefined class.

## Python API

```python
import numpy as np

from a1_inference.clip4retrofit_ovr_metrics import (
    compute_cosine_similarity_scores,
    compute_ovr_metrics,
    label_sets_to_multihot,
    save_evaluation,
)

class_names = ["bridge", "truck", "bicycle", "guard rail"]

# For semantic segmentation datasets, collect the class IDs/names present in
# each image mask.
labels_per_image = [
    {"bridge", "truck"},
    {"bicycle"},
    {"truck", "guard rail"},
]
y_true = label_sets_to_multihot(labels_per_image, class_names)

# image_embeddings: (N, D), text_embeddings: (C, D)
y_score = compute_cosine_similarity_scores(
    image_embeddings,
    text_embeddings,
)

result = compute_ovr_metrics(
    y_true,
    y_score,
    class_names=class_names,
    class_labels=class_names,
)

print(result.macro_auroc)
print(result.macro_fpr_at_target_tpr)
save_evaluation(result, "a1_inference/metrics_output")
```

For an ordinary single-label dataset:

```python
result = compute_ovr_metrics(
    y_true=integer_labels,       # shape (N,), values 0..C-1
    y_score=class_scores,        # shape (N,C)
    class_names=class_names,
)
```

Pass raw cosine similarities as `y_score`; ROC-AUC only depends on score
ordering, and the paper compares image and query embeddings using cosine
similarity without requiring softmax.

## CLI

Prepare an NPZ:

```python
np.savez(
    "a1_inference/predictions.npz",
    y_true=y_true,
    y_score=y_score,
    class_names=np.asarray(class_names),
)
```

Run:

```bash
python -m a1_inference.clip4retrofit_ovr_metrics \
  --input a1_inference/predictions.npz \
  --output-dir a1_inference/metrics_output
```

Outputs:

- `metrics.json`: macro and per-class values plus the exact metric definitions
- `per_class_metrics.csv`: compact per-class table
- `roc_curves.npz`: full FPR, TPR, and threshold arrays
- `ovr_roc_curves.png`: all one-vs-rest curves in one plot
- `ovr_roc_subplots.png`: one subplot per class, matching the paper's
  per-class ROC presentation

Run the tests with:

```bash
python -m unittest a1_inference.test_clip4retrofit_ovr_metrics
```

## Direct CSV inference for UDIAT

`csv_classwise_eval.py` supplies the upstream inference stage while leaving
`clip4retrofit_ovr_metrics.py` unchanged:

```text
udiat_classwise_eval.csv
  -> model image/text embeddings
  -> existing compute_cosine_similarity_scores
  -> predictions.npz + existing compute_ovr_metrics / save_evaluation
```

It uses one fixed caption embedding for each score column, in this order:
`benign`, `malignant`, `normal`.  It does not use a model logit scale, sigmoid,
or softmax; the saved `y_score` is the existing raw cosine-similarity matrix.

Install the isolated BioMedCLIP dependency once.  This does not alter the
existing virtual environment; it writes only to `a1_inference/pydeps`:

```bash
bash a1_inference/setup_csv_classwise_eval.sh
```

Evaluate the four models sequentially (one model is kept in GPU memory at a
time):

```bash
bash a1_inference/run_csv_classwise_eval.sh \
  --models openai_clip siglip medsiglip biomedclip \
  --device cuda:0
```

For a smaller initial run, replace `--models ...` with one model key, such as
`--models openai_clip`.  Outputs are placed in
`a1_inference/csv_classwise_results/<model_key>/`:

- `predictions.npz`: labels, raw cosine scores, source paths, captions, and
  the underlying image/text embeddings; directly compatible with the existing
  NPZ metric CLI.
- `metrics.json`, `per_class_metrics.csv`, `roc_curves.npz`, and ROC plots:
  produced by the existing metric module.
- `run_metadata.json`: model, prompt, device, and cache provenance.

The runner forces Hugging Face, Transformers, and OpenCLIP downloads into
`a1_inference/.hf_cache`; it also rejects CSV image paths and output paths
outside this repository.  The `google/medsiglip-448` repository is gated, so
its access terms must be accepted and the account logged in before that one
model can download.  To keep that login state within `a1_inference` too, run
the following from this repository after accepting the model terms:

```bash
HF_HOME="$PWD/a1_inference/.hf_cache" hf auth login
```

The other requested model runs do not depend on that access.

The wrapper defaults `CUDA_VISIBLE_DEVICES` to the compatible A6000 physical
index set `1,2,3`, avoiding the unsupported Blackwell GPU at physical index
`0` in the current PyTorch build.  An explicitly supplied
`CUDA_VISIBLE_DEVICES` value is left unchanged.
