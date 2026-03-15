"""
Brain Tumor MRI Dataset for Basic Fine-Tuning with Latent Drifting.

Expected directory layout
--------------------------
data/
    train/
        tumor/          <- positive: brain MRI slices with tumor
            img_001.png
            ...
        healthy/        <- negative: healthy brain MRI slices
            img_001.png
            ...
    val/   (same structure)
    test/  (same structure)

Images can be .png, .jpg, or .jpeg (any colour-mode; converted to RGB).
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from prompts import get_prompt

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


class BrainTumorMRIDataset(Dataset):
    """
    Loads paired (tumor, healthy) brain MRI slices together with a text prompt.

    Returns dict with keys:
        pixel_values : Float tensor [3, H, W] normalised to [-1, 1]
        prompt       : str   – text caption for this image
        label        : str   – "tumor" | "healthy"
        path         : str   – absolute path to the source file
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        use_diverse_prompts: bool = True,
        use_patient_info: bool = False,
        augment: bool = True,
        label_filter: Optional[str] = None,   # None → load both classes
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.use_diverse_prompts = use_diverse_prompts
        self.use_patient_info = use_patient_info
        self.augment = augment
        self.label_filter = label_filter

        self.samples: List[Dict] = []
        self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {data_dir}. "
                "Please organise images as data/[split]/tumor/ and data/[split]/healthy/."
            )

        print(
            f"[BrainTumorMRIDataset] {data_dir}: "
            f"{sum(1 for s in self.samples if s['label']=='tumor')} tumor, "
            f"{sum(1 for s in self.samples if s['label']=='healthy')} healthy"
        )

        # Deterministic resize + crop
        self.base_transform = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # → [-1, 1]
        ])

        # Light augmentation (horizontal flip – preserves brain anatomy)
        self.aug_transform = T.RandomHorizontalFlip(p=0.5)

    def _load_samples(self):
        labels = ["tumor", "healthy"]
        if self.label_filter is not None:
            labels = [self.label_filter]

        for label in labels:
            label_dir = self.data_dir / label
            if not label_dir.exists():
                continue
            for path in sorted(label_dir.iterdir()):
                if path.suffix.lower() in SUPPORTED_EXTS:
                    self.samples.append({"path": path, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.augment:
            image = self.aug_transform(image)

        pixel_values = self.base_transform(image)

        prompt = get_prompt(
            sample["label"],
            use_diverse=self.use_diverse_prompts,
        )

        return {
            "pixel_values": pixel_values,
            "prompt": prompt,
            "label": sample["label"],
            "path": str(sample["path"]),
        }


class BrainTumorInferenceDataset(Dataset):
    """
    Lightweight dataset for inference / evaluation.

    Returns dict:
        image      : PIL.Image (RGB)
        label      : str
        path       : str
        patient_info : Optional dict (parsed from filename if possible)
    """

    def __init__(
        self,
        data_dir: str,
        label: str = "tumor",
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.label = label

        label_dir = self.data_dir / label
        if not label_dir.exists():
            # Fallback: treat data_dir itself as the image folder
            label_dir = self.data_dir

        paths = [
            p for p in sorted(label_dir.iterdir())
            if p.suffix.lower() in SUPPORTED_EXTS
        ]

        if max_samples is not None:
            paths = paths[:max_samples]

        self.paths = paths
        print(f"[InferenceDataset] {label}: {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB").resize((512, 512), Image.BICUBIC)
        return {
            "image": image,
            "label": self.label,
            "path": str(path),
        }
