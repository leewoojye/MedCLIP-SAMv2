"""Resumable multi-GPU batch generation using the validated SD-IPC settings.

Each worker receives a disjoint, stable shard of BUS-CoT images.  It preserves
the SD-IPC closed-form image-to-prompt conversion and uses original BUS-CoT
lesion masks dilated by the requested number of pixels.  Generated image names
remain identical to their sources, so interrupted runs safely resume by
skipping already written outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image

from generate_sd_ipc_trial import (
    DEFAULT_NORMAL_IMAGE,
    SD_MODEL_ID,
    dilate_mask,
    image_to_sd_ipc_token,
    native_empty_prompt_embedding,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "BUS-COT" / "images_matching_BUS_COT_test_ids"
DEFAULT_MASK_DIR = PROJECT_ROOT / "data" / "BUS-COT" / "masks"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs_bus_cot_all_original_masks_dilation9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--normal-image", type=Path, default=DEFAULT_NORMAL_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-label",
        default="run",
        help="Label included in this worker's manifest filename.",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--mask-dilation-px", type=int, default=9)
    parser.add_argument("--svd-threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def load_items(source_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    supported_suffixes = {".png", ".jpg", ".jpeg"}
    images = sorted(
        path for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_suffixes
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in source directory: {source_dir}")

    def resolve_mask(image: Path) -> Path | None:
        # BUS-CoT JPEG exports are named like ``002044@0.png.jpg`` while their
        # original masks retain the intermediate ``.png`` filename.
        for candidate in (mask_dir / image.name, mask_dir / image.stem):
            if candidate.is_file():
                return candidate
        return None

    resolved = [(image, resolve_mask(image)) for image in images]
    missing_masks = [image.name for image, mask in resolved if mask is None]
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        raise FileNotFoundError(f"Missing {len(missing_masks)} source masks; first: {preview}")
    return [(image, mask) for image, mask in resolved if mask is not None]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batch generation.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0, num-shards).")
    if args.batch_size < 1 or args.steps < 1 or args.mask_dilation_px < 0:
        raise ValueError("batch-size and steps must be positive; dilation must be non-negative.")
    for path, label in ((args.source_dir, "source directory"), (args.mask_dir, "mask directory")):
        if not path.is_dir():
            raise NotADirectoryError(f"{label} not found: {path}")
    if not args.normal_image.is_file():
        raise FileNotFoundError(f"Normal reference not found: {args.normal_image}")

    all_items = load_items(args.source_dir, args.mask_dir)
    shard_items = [(index, *all_items[index]) for index in range(args.shard_index, len(all_items), args.num_shards)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dilated_mask_dir = args.output_dir / "dilated_masks"
    dilated_mask_dir.mkdir(exist_ok=True)

    pending = [item for item in shard_items if not (args.output_dir / item[1].name).is_file()]
    print(
        f"shard={args.shard_index}/{args.num_shards} total={len(shard_items)} "
        f"pending={len(pending)} device={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    if not pending:
        return

    device = torch.device("cuda")
    image_token, converter_metadata = image_to_sd_ipc_token(
        args.normal_image, device, args.svd_threshold
    )
    pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    uncond = native_empty_prompt_embedding(pipe, device).to(dtype=pipe.unet.dtype)
    image_token = image_token.to(dtype=pipe.unet.dtype)

    generated_records = []
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        source_images: list[Image.Image] = []
        masks: list[Image.Image] = []
        for _, source_path, source_mask_path in batch:
            with Image.open(source_path) as source_file:
                source = source_file.convert("RGB")
            with Image.open(source_mask_path) as source_mask_file:
                mask = source_mask_file.convert("L")
            if mask.size != source.size:
                raise ValueError(f"Mask/image size mismatch for {source_path.name}: {mask.size} vs {source.size}")
            mask = dilate_mask(mask, args.mask_dilation_px)
            dilated_mask_path = dilated_mask_dir / f"{source_path.name}.png"
            mask.save(dilated_mask_path)
            source_images.append(source)
            masks.append(mask)

        batch_count = len(batch)
        pseudo_prompt = torch.cat(
            (uncond[:, :1], image_token.unsqueeze(1).repeat(1, 76, 1)), dim=1
        ).repeat(batch_count, 1, 1)
        negative_prompt = uncond.repeat(batch_count, 1, 1)
        generators = [
            torch.Generator(device=device).manual_seed(args.seed + global_index)
            for global_index, _, _ in batch
        ]
        generated = pipe(
            prompt_embeds=pseudo_prompt,
            negative_prompt_embeds=negative_prompt,
            image=[image.resize((512, 512), Image.Resampling.LANCZOS) for image in source_images],
            mask_image=[mask.resize((512, 512), Image.Resampling.NEAREST) for mask in masks],
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
        ).images
        for (global_index, source_path, source_mask_path), source, output in zip(batch, source_images, generated):
            output_path = args.output_dir / source_path.name
            output.resize(source.size, Image.Resampling.LANCZOS).save(output_path)
            generated_records.append({
                "index": global_index,
                "source": str(source_path),
                "source_mask": str(source_mask_path),
                "dilated_mask": str(dilated_mask_dir / f"{source_path.name}.png"),
                "output": str(output_path),
                "seed": args.seed + global_index,
            })
        completed = min(start + batch_count, len(pending))
        if completed == len(pending) or completed % 10 == 0:
            print(f"shard={args.shard_index} completed={completed}/{len(pending)}", flush=True)

    manifest = {
        "method": "training-free SD-IPC closed-form CLIP image-to-prompt conversion",
        "paper": "arXiv:2305.12716",
        "stable_diffusion_model": SD_MODEL_ID,
        "scheduler": "DDIM",
        "normal_reference": str(args.normal_image),
        "source_dir": str(args.source_dir),
        "source_mask_dir": str(args.mask_dir),
        "mask_dilation_px": args.mask_dilation_px,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "converter": converter_metadata,
        "generated_records": generated_records,
    }
    (args.output_dir / f"manifest_{args.run_label}_shard_{args.shard_index}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"shard={args.shard_index} finished generated={len(generated_records)}", flush=True)


if __name__ == "__main__":
    main()
