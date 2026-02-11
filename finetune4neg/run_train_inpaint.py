import os
import subprocess
import sys
from accelerate.utils import write_basic_config

def run_training_inpaint():
    # 1. 버전이 맞지 않는 기존 스크립트 삭제 (자동 삭제 비활성화)
    script_name = "train_dreambooth_inpaint_lora.py"
    if not os.path.exists(script_name):
        # script_name이 없을 때만 다운로드
        print("⬇️ 인페인팅 전용 LoRA 학습 스크립트를 다운로드합니다...")
        download_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora.py"
        subprocess.run(["wget", download_url], check=True)
    else:
        print(f"✅ {script_name}이(가) 이미 존재하므로 다운로드를 건너뜁니다.")

    # 3. accelerate 설정 (기본값 생성)
    print("⚙️ Accelerate 설정을 확인합니다...")
    write_basic_config()

    # === [설정 영역] ===
    # 타임스탬프를 포함한 폴더명 생성
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    DATA_DIR = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT/normal"
    OUTPUT_DIR = f"/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/finetune4neg/finetuned_output/inpaint_output_{current_time}"
    
    # 디렉토리가 없으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # =================

    print(f"\n🚀 인페인팅 전용 학습을 시작합니다!")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"저장 경로: {OUTPUT_DIR}")
    print(f"모델: stable-diffusion-v1-5/stable-diffusion-inpainting")

    # 4. 학습 실행 (명령어를 리스트 형태로 전달)
    command = [
        "accelerate", "launch", script_name,
        "--pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-inpainting",
        f"--instance_data_dir={DATA_DIR}",
        f"--output_dir={OUTPUT_DIR}",
        "--instance_prompt=a photo of sks ultrasound",
        "--resolution=512",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=1",
        "--checkpointing_steps=500",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=1000",
        "--seed=0",
        # 인페인팅 전용 추가 파라미터가 필요하다면 여기에 추가
        # 보통 인페인팅 학습은 마스크 이미지가 필요할 수 있으나, 
        # Dreambooth Inpainting 스크립트는 내부적으로 마스크를 생성하거나 처리할 수 있음.
    ]

    try:
        # 학습 프로세스 실행
        subprocess.run(command, check=True)
        print("\n✅ 인페인팅 학습이 성공적으로 완료되었습니다!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_training_inpaint()
