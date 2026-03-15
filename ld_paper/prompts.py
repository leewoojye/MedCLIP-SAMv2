"""
Prompt templates for Brain Tumor (positive) and Healthy (negative) brain MRI.

Paper [Table 3 & 4] reports best results with:
  - Diverse prompts
  - With Patient Information (PI) when metadata is available

We adapt the original AD/CN prompts of LD to brain tumor / healthy setting.
"""

import random
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# 21 diverse prompts per class  (mirror Table 3 / Section 4.1)
# ---------------------------------------------------------------------------

TUMOR_PROMPTS_DIVERSE = [
    "a brain MRI scan showing a brain tumor",
    "brain MRI of a patient diagnosed with brain tumor",
    "a magnetic resonance image of a brain with tumor",
    "brain MRI scan with visible tumor lesion",
    "a brain scan showing tumor presence",
    "MRI scan of a brain with tumor lesion",
    "a brain MRI showing benign brain tumor",
    "brain magnetic resonance imaging with tumor pathology",
    "a T1-weighted brain MRI with brain tumor",
    "brain MRI of a patient with brain tumor diagnosis",
    "an axial brain MRI showing brain tumor",
    "a coronal brain MRI with visible tumor lesion",
    "brain MRI scan with intracranial tumor",
    "a brain MRI of a patient with an intracranial mass",
    "MRI scan showing brain tumor pathology",
    "a brain MRI with abnormal mass",
    "brain MRI with tumor visible in scan",
    "a medical brain MRI image showing tumor",
    "brain MRI with hyperintense tumor region",
    "a brain magnetic resonance image with tumor lesion",
    "brain MRI scan with abnormal tumor growth",
]

HEALTHY_PROMPTS_DIVERSE = [
    "a healthy brain MRI scan",
    "brain MRI of a healthy patient",
    "a normal brain MRI without tumor",
    "healthy brain MRI scan with no pathology",
    "a brain MRI showing no signs of tumor",
    "MRI scan of a normal healthy brain",
    "a brain MRI of a cognitively normal patient",
    "normal brain magnetic resonance imaging",
    "a T1-weighted brain MRI of a healthy brain",
    "brain MRI of a patient with no abnormality",
    "an axial brain MRI with normal appearance",
    "a coronal brain MRI with healthy brain tissue",
    "brain MRI scan showing normal brain structure",
    "a brain MRI with no tumor or lesion",
    "MRI scan showing a healthy brain",
    "a brain MRI of a patient with no diagnosis",
    "healthy brain MRI with normal signal intensity",
    "a medical brain MRI image of a healthy brain",
    "brain MRI with normal brain tissue",
    "a brain magnetic resonance image without abnormality",
    "brain MRI scan with no signs of disease",
]

TUMOR_PROMPTS_SIMPLE = [
    "a brain MRI with brain tumor",
]

HEALTHY_PROMPTS_SIMPLE = [
    "a healthy brain MRI",
]


def get_prompt(
    label: str,
    use_diverse: bool = True,
    patient_info: Optional[Dict] = None,
) -> str:
    """
    Generate a text prompt for the given label.

    Args:
        label        : "tumor" or "healthy"
        use_diverse  : Use diverse (21-variant) vs. simple prompts.
        patient_info : Optional dict with keys "age" (int) and "sex" (str).
                       When provided, formats a patient-information prompt.

    Returns:
        Prompt string.
    """
    assert label in ("tumor", "healthy"), f"Unknown label: {label}"

    if patient_info and "age" in patient_info and "sex" in patient_info:
        # Patient-information prompt (PI)  [Tables 3 & 4 — best performance]
        age = patient_info["age"]
        sex = patient_info["sex"]
        condition = "brain tumor" if label == "tumor" else "no tumor"
        return (
            f"A brain MRI of a {age} year old {sex} with {condition}"
        )

    if use_diverse:
        pool = TUMOR_PROMPTS_DIVERSE if label == "tumor" else HEALTHY_PROMPTS_DIVERSE
    else:
        pool = TUMOR_PROMPTS_SIMPLE if label == "tumor" else HEALTHY_PROMPTS_SIMPLE

    return random.choice(pool)


def get_manipulation_prompt_pair(
    source_label: str = "tumor",
    target_label: str = "healthy",
    use_diverse: bool = True,
    patient_info: Optional[Dict] = None,
) -> tuple:
    """Return (source_prompt, target_prompt) for Pix2Pix Zero + LD manipulation."""
    src = get_prompt(source_label, use_diverse, patient_info)
    tgt = get_prompt(target_label, use_diverse, patient_info)
    return src, tgt
