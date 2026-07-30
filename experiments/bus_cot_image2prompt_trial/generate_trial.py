"""Small BUS-CoT image-to-prompt negative-sample trial.

This is a batched, reproducible runner for the existing project's
``zero_shot_translation/vector_injection_image.py`` conditioning idea.  A
normal BUSI image is encoded by BiomedCLIP's vision tower and its 197x768
hidden-state sequence is used as Stable Diffusion's cross-attention condition.

BUS-CoT image-only exports in this workspace do not contain lesion masks.
The three mask ellipses below were manually selected from visually inspected,
clearly visible lesions.  They are deliberately kept in this new experiment
file, rather than changing source data or an existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFilter
from transformers import AutoModel, AutoProcessor, AutoTokenizer


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "BUS-COT" / "images_matching_BUS_COT_test_ids"
DEFAULT_NORMAL_IMAGE = EXPERIMENT_DIR / "assets" / "busi_normal_1.png"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

# (centre_x, centre_y, width, height), each expressed relative to source size.
# These are trial masks only, not BUS-CoT ground-truth annotations.
TRIAL_ELLIPSES = {
    "000015@0.png": (0.510, 0.340, 0.170, 0.150),
    "000019@0.png": (0.535, 0.180, 0.200, 0.130),
    "000020@0.png": (0.290, 0.185, 0.290, 0.210),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-image", type=Path, default=DEFAULT_NORMAL_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sd-model",
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--mask-dilation-px", type=int, default=5)
    return parser.parse_args()


def create_trial_mask(size: tuple[int, int], ellipse: tuple[float, float, float, float], dilation_px: int) -> Image.Image:
    width, height = size
    centre_x, centre_y, ellipse_width, ellipse_height = ellipse
    left = int(round((centre_x - ellipse_width / 2) * width))
    top = int(round((centre_y - ellipse_height / 2) * height))
    right = int(round((centre_x + ellipse_width / 2) * width))
    bottom = int(round((centre_y + ellipse_height / 2) * height))
    mask = Image.new("L", size, color=0)
    ImageDraw.Draw(mask).ellipse((left, top, right, bottom), fill=255)
    if dilation_px:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    return mask


def load_biomedclip(device: torch.device):
    model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    if hasattr(model.config, "text_config") and not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False
    model.eval()
    return tokenizer, processor, model


def vision_hidden(processor, model, image_path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(image_path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.inference_mode():
        return model.vision_model(
            pixel_values=inputs["pixel_values"], output_hidden_states=True
        )[0]


def unconditional_hidden(tokenizer, model, length: int, device: torch.device) -> torch.Tensor:
    inputs = tokenizer([""], padding=True, return_tensors="pt")
    vocab_size = model.config.text_config.vocab_size
    inputs["input_ids"] = inputs["input_ids"].clamp(max=vocab_size - 1)
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    sequence_length = inputs["input_ids"].shape[1]
    inputs["position_ids"] = torch.arange(sequence_length).unsqueeze(0)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        hidden = model.text_model(**inputs, output_hidden_states=True)[0]
    if hidden.shape[1] < length:
        hidden = functional.pad(hidden, (0, 0, 0, length - hidden.shape[1]))
    return hidden[:, :length, :]


def make_comparison(source: Image.Image, mask: Image.Image, generated: Image.Image) -> Image.Image:
    source = source.convert("RGB")
    generated = generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    overlay = source.copy()
    red = Image.new("RGB", source.size, (255, 0, 0))
    overlay = Image.composite(red, overlay, mask.point(lambda value: 100 if value else 0))
    comparison = Image.new("RGB", (source.width * 3, source.height))
    comparison.paste(source, (0, 0))
    comparison.paste(overlay, (source.width, 0))
    comparison.paste(generated, (source.width * 2, 0))
    return comparison


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This trial is intentionally GPU-only; no CUDA device is available.")
    if not args.normal_image.is_file():
        raise FileNotFoundError(f"Normal BUSI reference not found: {args.normal_image}")
    if args.steps < 1 or args.mask_dilation_px < 0:
        raise ValueError("steps must be positive and mask-dilation-px non-negative.")

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "trial_masks"
    comparison_dir = args.output_dir / "comparisons"
    mask_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)

    tokenizer, processor, biomedclip = load_biomedclip(device)
    prompt_embeds = vision_hidden(processor, biomedclip, args.normal_image, device)
    negative_prompt_embeds = unconditional_hidden(tokenizer, biomedclip, prompt_embeds.shape[1], device)
    del biomedclip, processor, tokenizer
    torch.cuda.empty_cache()

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model, torch_dtype=torch.float16
    ).to(device)
    pipe.safety_checker = None
    prompt_embeds = prompt_embeds.to(dtype=pipe.unet.dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=pipe.unet.dtype)

    records = []
    for index, (filename, ellipse) in enumerate(TRIAL_ELLIPSES.items()):
        source_path = SOURCE_DIR / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Selected BUS-CoT source missing: {source_path}")
        with Image.open(source_path) as source_file:
            source = source_file.convert("RGB")
        mask = create_trial_mask(source.size, ellipse, args.mask_dilation_px)
        mask_path = mask_dir / filename
        mask.save(mask_path)

        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=source.resize((512, 512), Image.Resampling.LANCZOS),
            mask_image=mask.resize((512, 512), Image.Resampling.NEAREST),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0].resize(source.size, Image.Resampling.LANCZOS)
        output_path = args.output_dir / filename
        result.save(output_path)
        make_comparison(source, mask, result).save(comparison_dir / filename)
        records.append({
            "source": str(source_path),
            "output": str(output_path),
            "mask": str(mask_path),
            "ellipse_relative_xywh": ellipse,
            "seed": args.seed + index,
        })
        print(f"generated: {output_path}", flush=True)

    (args.output_dir / "trial_manifest.json").write_text(
        json.dumps(
            {
                "method": "BiomedCLIP normal-image hidden-state injection into Stable Diffusion inpainting",
                "normal_reference": str(args.normal_image),
                "manual_trial_masks": True,
                "mask_dilation_px": args.mask_dilation_px,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
