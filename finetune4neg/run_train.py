import os
import subprocess
import sys
from accelerate.utils import write_basic_config

def run_training():
    # 1. 버전이 맞지 않는 기존 스크립트 삭제
    script_name = "train_dreambooth_lora.py"
    if os.path.exists(script_name):
        os.remove(script_name)
        print(f"🗑️ 호환되지 않는 {script_name}를 삭제했습니다.")

    # 2. 현재 설치된 diffusers 버전(0.36.0)에 맞는 스크립트 다운로드
    # wget 대신 파이썬 내부에서 명령 실행
    print("⬇️ 호환되는 버전(v0.36.0)의 학습 스크립트를 다운로드합니다...")
    download_url = "https://raw.githubusercontent.com/huggingface/diffusers/v0.36.0/examples/dreambooth/train_dreambooth_lora.py"
    subprocess.run(["wget", download_url], check=True)

    # 3. accelerate 설정 (기본값 생성)
    print("⚙️ Accelerate 설정을 확인합니다...")
    write_basic_config()

    # === [설정 영역] ===
    DATA_DIR = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT/normal"
    OUTPUT_DIR = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/finetune4neg/sdv5_impainting"
    
    # 디렉토리가 없으면 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # =================

    print(f"\n🚀 학습을 시작합니다!")
    print(f"데이터 경로: {DATA_DIR}")
    print(f"저장 경로: {OUTPUT_DIR}")

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
        "--seed=0"
    ]

    try:
        # 학습 프로세스 실행
        subprocess.run(command, check=True)
        print("\n✅ 학습이 성공적으로 완료되었습니다!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 학습 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    run_training()