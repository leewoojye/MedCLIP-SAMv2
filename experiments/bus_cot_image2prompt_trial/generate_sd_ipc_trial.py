"""Training-free SD-IPC trial for three BUS-CoT ultrasound images.

This follows Ding et al. (arXiv:2305.12716) without changing the Stable
Diffusion inpainting architecture. A normal reference image is encoded by the
same OpenAI CLIP ViT-L/14 family used by SD 1.x. The closed-form converter is
W_map = pinv(W_text) @ W_visual, with singular values below 0.3 discarded.
The image-derived token fills positions 1..76 of an SD CLIP pseudo-prompt;
the SD CLIP start-token embedding remains at position 0.

The BUS-CoT export used here has no lesion masks. The three ellipses are
therefore manually selected trial masks, not dataset ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw, ImageFilter
from transformers import CLIPImageProcessor, CLIPModel


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "BUS-COT" / "images_matching_BUS_COT_test_ids"
DEFAULT_NORMAL_IMAGE = EXPERIMENT_DIR / "assets" / "busi_normal_1.png"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs_sd_ipc_closed_form"
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"

# (centre_x, centre_y, width, height), expressed relative to source size.
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
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory of source masks named identically to the selected images.",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--mask-dilation-px", type=int, default=5)
    parser.add_argument("--svd-threshold", type=float, default=0.3)
    return parser.parse_args()


def create_trial_mask(
    size: tuple[int, int], ellipse: tuple[float, float, float, float], dilation_px: int
) -> Image.Image:
    width, height = size
    centre_x, centre_y, ellipse_width, ellipse_height = ellipse
    bounds = (
        int(round((centre_x - ellipse_width / 2) * width)),
        int(round((centre_y - ellipse_height / 2) * height)),
        int(round((centre_x + ellipse_width / 2) * width)),
        int(round((centre_y + ellipse_height / 2) * height)),
    )
    mask = Image.new("L", size, color=0)
    ImageDraw.Draw(mask).ellipse(bounds, fill=255)
    return dilate_mask(mask, dilation_px)


def dilate_mask(mask: Image.Image, dilation_px: int) -> Image.Image:
    """Apply square-kernel dilation to a binary/greyscale lesion mask."""
    mask = mask.convert("L")
    if dilation_px:
        mask = mask.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    return mask


def truncated_pseudoinverse(matrix: torch.Tensor, threshold: float) -> tuple[torch.Tensor, int]:
    """Moore-Penrose inverse with the SD-IPC paper's singular-value cutoff."""
    u, singular_values, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    keep = singular_values >= threshold
    inverse_values = torch.where(keep, singular_values.reciprocal(), torch.zeros_like(singular_values))
    return (vh.transpose(-2, -1) * inverse_values) @ u.transpose(-2, -1), int(keep.sum().item())


def image_to_sd_ipc_token(
    normal_image: Path, device: torch.device, svd_threshold: float
) -> tuple[torch.Tensor, dict[str, int | float | str]]:
    """Return the paper's image-derived SD CLIP text-token embedding."""
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    with Image.open(normal_image) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)
    with torch.inference_mode():
        # ViT-L/14 CLS output is 1024-D; SD CLIP hidden space is 768-D.
        vision_cls = clip_model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        text_inverse, kept_singular_values = truncated_pseudoinverse(
            clip_model.text_projection.weight.detach(), svd_threshold
        )
        # HF Linear(x, W) is x @ W.T, so the paper's map is pinv(W_text) @ W_visual.
        projector = text_inverse @ clip_model.visual_projection.weight.detach().float()
        converted_token = functional.linear(vision_cls.float(), projector)
    metadata: dict[str, int | float | str] = {
        "clip_model": CLIP_MODEL_ID,
        "clip_vision_hidden_size": int(vision_cls.shape[-1]),
        "sd_clip_hidden_size": int(converted_token.shape[-1]),
        "kept_singular_values": kept_singular_values,
        "svd_threshold": svd_threshold,
        "converted_token_l2_norm": float(converted_token.norm(dim=-1).item()),
    }
    del clip_model, processor
    torch.cuda.empty_cache()
    return converted_token, metadata


def native_empty_prompt_embedding(pipe: StableDiffusionInpaintPipeline, device: torch.device) -> torch.Tensor:
    tokens = pipe.tokenizer(
        [""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.inference_mode():
        return pipe.text_encoder(tokens.input_ids.to(device))[0]


def make_comparison(source: Image.Image, mask: Image.Image, generated: Image.Image) -> Image.Image:
    source = source.convert("RGB")
    generated = generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    red_mask = Image.composite(
        Image.new("RGB", source.size, (255, 0, 0)), source,
        mask.point(lambda value: 100 if value else 0),
    )
    comparison = Image.new("RGB", (source.width * 3, source.height))
    comparison.paste(source, (0, 0))
    comparison.paste(red_mask, (source.width, 0))
    comparison.paste(generated, (source.width * 2, 0))
    return comparison


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU.")
    if not args.normal_image.is_file():
        raise FileNotFoundError(f"Normal BUSI reference not found: {args.normal_image}")
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {args.mask_dir}")
    if args.steps < 1 or args.mask_dilation_px < 0 or args.svd_threshold < 0:
        raise ValueError("steps must be positive; dilation and SVD threshold must be non-negative.")

    device = torch.device("cuda")
    image_token, converter_metadata = image_to_sd_ipc_token(args.normal_image, device, args.svd_threshold)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    uncond = native_empty_prompt_embedding(pipe, device)
    pseudo_prompt = torch.cat((uncond[:, :1], image_token.unsqueeze(1).repeat(1, 76, 1)), dim=1)
    if pseudo_prompt.shape != uncond.shape:
        raise RuntimeError(f"Unexpected SD-IPC prompt shape: {pseudo_prompt.shape}; expected {uncond.shape}")
    pseudo_prompt = pseudo_prompt.to(dtype=pipe.unet.dtype)
    uncond = uncond.to(dtype=pipe.unet.dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "trial_masks"
    comparison_dir = args.output_dir / "comparisons"
    mask_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    records = []
    for index, (filename, ellipse) in enumerate(TRIAL_ELLIPSES.items()):
        source_path = SOURCE_DIR / filename
        if not source_path.is_file():
            raise FileNotFoundError(f"Selected BUS-CoT source missing: {source_path}")
        with Image.open(source_path) as source_file:
            source = source_file.convert("RGB")
        source_mask_path = None
        if args.mask_dir is None:
            mask = create_trial_mask(source.size, ellipse, args.mask_dilation_px)
        else:
            source_mask_path = args.mask_dir / filename
            if not source_mask_path.is_file():
                raise FileNotFoundError(f"Source mask missing: {source_mask_path}")
            with Image.open(source_mask_path) as source_mask_file:
                mask = source_mask_file.convert("L")
            if mask.size != source.size:
                raise ValueError(
                    f"Mask/image size mismatch for {filename}: {mask.size} vs {source.size}"
                )
            mask = dilate_mask(mask, args.mask_dilation_px)
        mask_path = mask_dir / filename
        mask.save(mask_path)
        result = pipe(
            prompt_embeds=pseudo_prompt, negative_prompt_embeds=uncond,
            image=source.resize((512, 512), Image.Resampling.LANCZOS),
            mask_image=mask.resize((512, 512), Image.Resampling.NEAREST),
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            generator=torch.Generator(device=device).manual_seed(args.seed + index),
        ).images[0].resize(source.size, Image.Resampling.LANCZOS)
        output_path = args.output_dir / filename
        result.save(output_path)
        make_comparison(source, mask, result).save(comparison_dir / filename)
        records.append({
            "source": str(source_path), "output": str(output_path), "mask": str(mask_path),
            "source_mask": str(source_mask_path) if source_mask_path else None,
            "ellipse_relative_xywh": None if source_mask_path else ellipse,
            "seed": args.seed + index,
        })
        print(f"generated: {output_path}", flush=True)

    (args.output_dir / "trial_manifest.json").write_text(
        json.dumps({
            "method": "training-free SD-IPC closed-form CLIP image-to-prompt conversion",
            "paper": "arXiv:2305.12716", "stable_diffusion_model": SD_MODEL_ID,
            "scheduler": "DDIM", "normal_reference": str(args.normal_image),
            "manual_trial_masks": args.mask_dir is None,
            "mask_source": str(args.mask_dir) if args.mask_dir else "manual trial ellipses",
            "mask_dilation_px": args.mask_dilation_px,
            "converter": converter_metadata, "records": records,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
