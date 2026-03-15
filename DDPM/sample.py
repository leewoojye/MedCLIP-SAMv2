from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import BrainTumorInpaintingDataset
from .model import PaperModelConfig, build_paper_diffusion, build_paper_unet
from .utils import (
    blend_prediction,
    gaussian_blur_2d,
    restore_from_center_crop_or_pad,
    save_png,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample paper-style 2D DDPM inpainting results for PNG slices."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--blur-sigma", type=float, default=1.075)
    parser.add_argument("--blend-known-region", action="store_true")
    parser.add_argument("--restore-original-size", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BrainTumorInpaintingDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        crop_size=args.crop_size,
        include_target=False,
        max_samples=args.max_samples,
    )

    model = build_paper_unet(
        PaperModelConfig(image_size=args.crop_size, learn_sigma=args.learn_sigma)
    ).to(device)
    diffusion = build_paper_diffusion(
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["ema"]["shadow"] if args.use_ema else checkpoint["model"]
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    blended_dir = output_dir / "blended"
    raw_dir.mkdir(parents=True, exist_ok=True)
    blended_dir.mkdir(parents=True, exist_ok=True)

    for sample in dataset:
        conditioned = sample["model_input"].unsqueeze(0).to(device)
        with torch.no_grad():
            generated, _, _ = diffusion.p_sample_loop_known(
                model=model,
                shape=(
                    1,
                    conditioned.shape[1],
                    conditioned.shape[2],
                    conditioned.shape[3],
                ),
                img=conditioned,
                clip_denoised=True,
                progress=False,
            )

        prediction = generated.squeeze(0).squeeze(0).cpu().numpy()
        if args.blur_sigma > 0:
            prediction = gaussian_blur_2d(prediction, args.blur_sigma)

        baseline = sample["baseline"].squeeze(0).numpy()
        mask = sample["mask"].squeeze(0).numpy()
        blended = (
            blend_prediction(prediction, baseline, mask)
            if args.blend_known_region
            else prediction
        )

        if args.restore_original_size:
            prediction = restore_from_center_crop_or_pad(
                prediction,
                sample["meta"],
                background=sample["orig_image"],
            )
            blended = restore_from_center_crop_or_pad(
                blended,
                sample["meta"],
                background=sample["orig_image"],
            )

        save_png(prediction, raw_dir / sample["name"])
        save_png(blended, blended_dir / sample["name"])

    print(f"Saved samples to {output_dir}")


if __name__ == "__main__":
    main()
