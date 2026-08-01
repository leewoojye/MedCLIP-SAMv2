# Class-wise AUROC and FPR@95%TPR Evaluation

This repository provides a CSV-based, zero-shot evaluation pipeline for
class-wise one-vs-rest (OVR) AUROC and macro FPR@95%TPR. The metric core is
implemented in `a1_inference/clip4retrofit_ovr_metrics.py`; the CSV evaluators
only add the upstream inference steps:

```text
CSV -> image and text embeddings -> raw cosine-similarity scores -> OVR metrics
```

The pipeline keeps one fixed text caption per class, encodes all images and
captions with the selected vision-language model, and uses raw L2-normalised
image-text cosine similarity as the score. It does not apply CLIP logit
scaling, sigmoid, or softmax before ROC calculation.

## Metric definition

For every class `k`, the evaluator treats samples from class `k` as positive
and all other samples as negative. It then constructs one binary ROC curve
from the `k`-th cosine-similarity score column and reports:

- **Per-class AUROC:** trapezoidal area under that ROC curve.
- **Macro AUROC:** the unweighted arithmetic mean of all per-class AUROCs.
- **Per-class FPR@95%TPR:** the FPR at the first ROC point whose TPR is at
  least 0.95. No interpolation is applied.
- **Macro FPR@95%TPR:** the unweighted arithmetic mean of the per-class
  FPR@95%TPR values.

The implementation follows the one-vs-rest ROC-AUC procedure used by
Clip4Retrofit. FPR@95%TPR is an additional metric in this repository. Every
class must have at least one positive and one negative sample; undefined
classes raise an error rather than being silently removed from the macro mean.

## Evaluation environment

The standard runners use `bioclip2/.venv/bin/python` and keep the additional
OpenCLIP dependency in `a1_inference/pydeps`. From the repository root, install
the evaluator dependency once:

```bash
bash a1_inference/setup_csv_classwise_eval.sh
```

All Hugging Face and OpenCLIP downloads are cached under
`a1_inference/.hf_cache`. The generic CSV runner supports these model keys:

| Model key | Model |
| --- | --- |
| `openai_clip` | `openai/clip-vit-large-patch14` |
| `siglip` | `google/siglip-base-patch16-224` |
| `medsiglip` | `google/medsiglip-448` |
| `biomedclip` | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |

`google/medsiglip-448` is a gated Hugging Face model. Accept its access terms
on Hugging Face and authenticate before including `medsiglip` in a run:

```bash
HF_HOME="$PWD/a1_inference/.hf_cache" hf auth login
```

The runner evaluates requested models sequentially, so only one model is kept
in GPU memory at a time. Unless overridden, the generic runner exposes the
compatible physical GPUs `1,2,3` through `CUDA_VISIBLE_DEVICES`; therefore
`--device cuda:0` refers to the first visible GPU, not necessarily physical GPU
0. To choose visible devices explicitly, set `CUDA_VISIBLE_DEVICES` before the
command.

## CSV requirements

The generic evaluator accepts a CSV with exactly these required columns:

```csv
image_path,class_label,caption
/absolute/path/inside/MedCLIP-SAMv2/data/example/a.png,class_a,A fixed caption for class A.
/absolute/path/inside/MedCLIP-SAMv2/data/example/b.png,class_b,A fixed caption for class B.
```

- `image_path` must exist inside this repository.
- `class_label` must appear in `--class-order`.
- Each class must occur at least once.
- All rows sharing a `class_label` must use exactly the same `caption`, because
  one text embedding is created per class.
- The order passed to `--class-order` fixes both the cosine-score columns and
  the reported class order.

Prepared CSVs and dataset-specific wrappers are available in `a1_inference/`.
They cover UDIAT, Figshare brain MRI, Chest CT, COD10K, and MAS3K.

## Run a CSV evaluation

The following command evaluates the prepared UDIAT CSV with all four generic
model keys on the first visible GPU:

```bash
bash a1_inference/run_csv_classwise_eval.sh \
  --csv a1_inference/udiat_classwise_eval.csv \
  --output-dir a1_inference/csv_classwise_results/udiat_example \
  --class-order benign malignant normal \
  --models openai_clip siglip medsiglip biomedclip \
  --device cuda:0
```

For a custom CSV, replace the CSV path, output directory, class order, and
model selection as needed. For example, a small single-model smoke test is:

```bash
bash a1_inference/run_csv_classwise_eval.sh \
  --csv a1_inference/udiat_classwise_eval.csv \
  --output-dir a1_inference/csv_classwise_results/udiat_openai_clip_smoke \
  --class-order benign malignant normal \
  --models openai_clip \
  --batch-size 16 \
  --device cuda:0
```

Useful options are:

- `--target-tpr 0.95` sets the FPR operating point (default: `0.95`).
- `--no-plots` skips ROC PNG creation.
- `--fail-fast` stops after the first model error; by default the runner writes
  a `failure.json` for a failed model and continues with the remaining models.
- `--batch-size N` controls image-encoding batch size. Lower it if GPU memory
  is insufficient.

Dataset-specific wrappers encode the corresponding CSV, class order, and
default output location. Examples include:

```bash
bash a1_inference/run_figshare_classwise_eval.sh --device cuda:0
bash a1_inference/run_chest_ct_classwise_eval.sh --device cuda:0
```

For the separate OpenAI CLIP ViT-B/32 baseline, use the dedicated runner:

```bash
bash a1_inference/run_openai_vit_b32_classwise_eval.sh \
  --dataset figshare \
  --device cuda:0
```

Supported `--dataset` values for this baseline are `udiat`, `figshare`,
`chestct`, `cod10k`, and `mas3k`. Dataset- and checkpoint-specific scripts
such as `checkpoint_classwise_eval.py`, `mri_global_figshare_classwise_eval.py`,
`ct_global_chest_ct_classwise_eval.py`, `cod10k_classwise_eval.py`, and
`mas3k_classwise_eval.py` reuse the same metric core after their model-specific
inference step.

## Evaluate precomputed scores only

If image and text embeddings have already been converted to an NPZ containing
`y_true` and `y_score`, calculate the OVR metrics without running inference:

```bash
bioclip2/.venv/bin/python a1_inference/clip4retrofit_ovr_metrics.py \
  --input a1_inference/predictions.npz \
  --output-dir a1_inference/metrics_output \
  --target-tpr 0.95
```

`y_true` may be a single-label vector of shape `(N,)` or a multi-hot matrix of
shape `(N, C)`. `y_score` must have shape `(N, C)`, with columns aligned to
`class_names` when supplied. This supports the image-level multi-label OVR
setting as well as ordinary single-label multiclass evaluation.

## Outputs

For each evaluated model, the result directory contains:

- `predictions.npz`: labels, raw cosine scores, source image paths, captions,
  and image/text embeddings.
- `metrics.json`: metric definitions, macro values, and per-class values.
- `per_class_metrics.csv`: class label, positive/negative counts, AUROC, and
  FPR@95%TPR.
- `roc_curves.npz`: complete FPR, TPR, and threshold arrays.
- `ovr_roc_curves.png` and `ovr_roc_subplots.png`: ROC visualisations, unless
  `--no-plots` was used.
- `run_metadata.json`: model ID, backend, device, captions, class order, cache
  location, and score definition.

For generic CSV runs, these artifacts are written to
`<output-dir>/<model_key>/`. Keep CSV inputs, output directories, and model
caches inside this repository; the evaluators enforce this path boundary.

## Validation

Run the metric unit tests with:

```bash
bioclip2/.venv/bin/python -m unittest a1_inference.test_clip4retrofit_ovr_metrics
```

## Negative-sample generation

### Brain MRI: conditional DDPM inpainting

`DDPM/` contains the 2D conditional DDPM implementation for replacing a
masked brain-tumour region with an inpainted healthy-looking sample. The
Figshare runner uses the test images and their corresponding masks, a local
DDPM checkpoint, EMA weights, a 1,000-step linear schedule, and Gaussian
post-processing with `sigma=1.075`. The maintained three-GPU configuration
uses GPUs `0,1,2`, a mask-dilation size of `9`, crop size `224`, and context
margin `16`.

After placing the input images, masks, checkpoint, and DDPM virtual
environment at the paths configured in the runner, launch it from the
repository root as follows:

```bash
mkdir -p DDPM/logs
nohup bash DDPM/run_figshare_test_dilation9_3gpu.sh \
  > DDPM/logs/figshare_dilation9_launcher.log 2>&1 &
```

The runner gives each GPU a deterministic disjoint shard and writes a separate
worker log. It skips outputs that already exist, so an interrupted job can be
resumed with the same command. Images, masks, checkpoints, environments, and
generated samples are intentionally ignored by Git and must be provided
locally.

### Breast ultrasound: image-to-prompt Stable Diffusion inpainting

`experiments/bus_cot_image2prompt_trial/generate_sd_ipc_batch.py` implements
the training-free image-to-prompt (SD-IPC) procedure. It encodes one normal
ultrasound reference image with OpenAI CLIP ViT-L/14 and converts the visual
feature into Stable Diffusion's native text-hidden space using the closed-form
mapping described in arXiv:2305.12716. The Stable Diffusion inpainting model
architecture is not changed. For each source image, the original BUS-CoT mask
is dilated (default `9` pixels) before inpainting.

Set `ROOT` to this repository, then start one disjoint shard per GPU. The
example below keeps all downloaded model assets and auxiliary dependencies
inside the experiment directory:

```bash
ROOT="$PWD"
EXP="$ROOT/experiments/bus_cot_image2prompt_trial"
export HF_HOME="$EXP/.hf_cache"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$EXP/pydeps${PYTHONPATH:+:$PYTHONPATH}"

for SHARD in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$SHARD" "$ROOT/bioclip2/.venv/bin/python" \
    "$EXP/generate_sd_ipc_batch.py" \
    --source-dir "$ROOT/data/BUS-COT/images_matching_BUS_COT_test_ids" \
    --mask-dir "$ROOT/data/BUS-COT/masks" \
    --normal-image /path/to/one_normal_ultrasound.png \
    --output-dir "$EXP/outputs_bus_cot_all_original_masks_dilation9" \
    --run-label production --shard-index "$SHARD" --num-shards 3 \
    --batch-size 2 --steps 50 --guidance-scale 5.0 \
    --mask-dilation-px 9 --svd-threshold 0.3 --seed 20260728 \
    > "$EXP/gpu${SHARD}_production.log" 2>&1 &
done
```

Each shard records its settings and generated files in a manifest. Re-running
the same command resumes safely because files already present in the output
directory are skipped.
