import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import os

# 1. 파이프라인 로드 (GPU 권장)
# model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# model_id = "runwayml/stable-diffusion-inpainting" # 이전 인페인팅 전용 모델
model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting" # 이전 인페인팅 전용 모델
# SD2 모델은 공개 버전이 제한적이므로 Runway ML 사용
# model_id = "stabilityai/stable-diffusion-2-inpainting"  # 대체 옵션
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to("cuda")

# LoRA 가중치 로드
lora_model_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/finetune4neg/finetuned_output"
pipe.load_lora_weights(lora_model_dir, weight_name="pytorch_lora_weights.safetensors")
print(f"Loaded LoRA weights from {lora_model_dir}")

def resize_and_pad_square(img: Image.Image, size: int, resample: int, mode: str = "RGB") -> Image.Image:
    """비율 유지하면서 타겟 크기에 맞춰 리사이즈 + 중앙 정렬 패딩."""
    img = img.convert(mode)
    w, h = img.size
    # 긴 변을 기준으로 스케일하여 aspect ratio 유지
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), resample=resample)
    canvas = Image.new(mode, (size, size))
    # 중앙 정렬 패딩
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(img_resized, offset)
    return canvas

def generate_healthy_ultrasound(image_path, mask_path, save_path):
    # 2. 이미지 및 마스크 로드 + 리사이즈/패딩
    # INPAINT_SIZE 환경변수로 타겟 크기 제어 (기본 768). 용량 부족 시 512/640으로 조정 가능.
    target_size = int(os.getenv("INPAINT_SIZE", "768"))

    # SD 모델은 RGB를 기대하므로 Grayscale 초음파 이미지를 RGB로 변환
    init_image = resize_and_pad_square(Image.open(image_path), target_size, Image.BICUBIC, mode="RGB")
    # 마스크는 최근접 보간으로 이진값 보존
    mask_image = resize_and_pad_square(Image.open(mask_path), target_size, Image.NEAREST, mode="RGB")

    # 3. 프롬프트 설정 (초음파 조직 특성을 강조)
    # 종양(Tumor, Cyst)을 언급하지 않고 '건강한 조직'임을 명시
    prompt = "healthy ultrasound texture, homogeneous tissue, medical sonography, high quality"
    negative_prompt = "tumor, cyst, lesion, dark spot, abnormal growth, cancer, artifact, text, watermark"

    # 4. 인페인팅 실행
    # num_inference_steps: 생성 단계 (보통 30~50)
    # guidance_scale: 프롬프트 준수 정도
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=70,
        guidance_scale=9
    ).images[0]

    # 5. 결과 저장
    if os.path.isdir(save_path):
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_path, filename)
    
    output.save(save_path)
    print(f"Result saved at: {save_path}")

    # 마스킹 이미지도 함께 저장
    mask_save_path = os.path.splitext(save_path)[0] + "_mask.png"
    mask_image.save(mask_save_path)
    print(f"Mask saved at: {mask_save_path}")

def generate_healthy_brain(image_path, mask_path, save_path):
    # 2. 이미지 및 마스크 로드 + 리사이즈/패딩
    # INPAINT_SIZE 환경변수로 타겟 크기 제어 (기본 768). 용량 부족 시 512/640으로 조정 가능.
    target_size = int(os.getenv("INPAINT_SIZE", "768"))

    # SD 모델은 RGB를 기대하므로 Grayscale 이미지를 RGB로 변환
    init_image = resize_and_pad_square(Image.open(image_path), target_size, Image.BICUBIC, mode="RGB")
    # 마스크는 최근접 보간으로 이진값 보존
    mask_image = resize_and_pad_square(Image.open(mask_path), target_size, Image.NEAREST, mode="RGB")

    # 3. 프롬프트 설정 (Brain MRI 특성 강조)
    # 종양(Tumor, Mass)을 언급하지 않고 '건강한 조직'임을 명시
    prompt = "healthy brain MRI, homogeneous brain tissue, T1 weighted, medical imaging, high quality"
    negative_prompt = "tumor, mass, lesion, cyst, edema, abnormal growth, cancer, glioma, meningioma, pituitary"

    # 4. 인페인팅 실행
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=70,
        guidance_scale=9
    ).images[0]

    # 5. 결과 저장
    if os.path.isdir(save_path):
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_path, filename)
    
    output.save(save_path)
    print(f"Result saved at: {save_path}")

    # 마스킹 이미지도 함께 저장
    mask_save_path = os.path.splitext(save_path)[0] + "_mask.png"
    mask_image.save(mask_save_path)
    print(f"Mask saved at: {mask_save_path}")

# 사용 예시
image_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/data/breast_tumors/test_images/000004.png"
mask_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/data/breast_tumors/test_masks/000004.png"
save_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/generated_neg_output"
generate_healthy_ultrasound(image_path, mask_path, save_path)