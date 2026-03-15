"""
IXI-T1 NIfTI → PNG 변환 + 데이터셋 폴더 구조 자동 정리 스크립트

출력 구조:
  <output_dir>/
    train/healthy/   ← 학습용 정상 MRI 슬라이스
    val/healthy/     ← 검증용 (선택)
    test/healthy/    ← 평가 기준 이미지 (real_images_dir)

사용법:
  python prepare_ixi_healthy.py \
      --ixi_dir /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/IXI-T1 \
      --output_dir ./data \
      --n_train 200 \
      --n_val 0 \
      --n_test 300 \
      --slices_per_volume 5
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def load_volume(nii_path: Path) -> np.ndarray:
    """NIfTI 로드 후 (H, W, D) float32 반환."""
    img = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    # 3D인지 확인
    if data.ndim == 4:
        data = data[..., 0]
    return data


def select_axial_slices(volume: np.ndarray, n_slices: int) -> list[int]:
    """
    뇌 조직이 가장 많은 중앙 구간에서 균등 간격으로 슬라이스 인덱스 선택.
    axial 축 = axis 2 (z).
    """
    depth = volume.shape[2]
    z_start = int(depth * 0.25)
    z_end   = int(depth * 0.75)
    indices = np.linspace(z_start, z_end - 1, n_slices, dtype=int).tolist()
    return indices


def normalize_slice(sl: np.ndarray) -> np.ndarray:
    """99th percentile 클리핑 후 0-255 uint8 정규화."""
    p99 = np.percentile(sl, 99)
    if p99 < 1e-6:
        return np.zeros(sl.shape, dtype=np.uint8)
    sl = np.clip(sl, 0, p99) / p99
    return (sl * 255).astype(np.uint8)


def volume_to_pngs(
    nii_path: Path,
    out_dir: Path,
    n_slices: int,
    size: int = 512,
) -> list[Path]:
    """단일 NIfTI 볼륨을 PNG 슬라이스들로 변환 후 저장된 경로 목록 반환."""
    try:
        volume = load_volume(nii_path)
    except Exception as e:
        print(f"  [SKIP] {nii_path.name}: {e}")
        return []

    indices = select_axial_slices(volume, n_slices)
    saved = []
    stem = nii_path.name.replace(".nii.gz", "").replace(".nii", "")

    for i, z in enumerate(indices):
        sl = volume[:, :, z]
        sl_norm = normalize_slice(sl)

        # PIL 이미지: 회전 보정 (IXI는 종종 90도 회전 필요)
        img = Image.fromarray(sl_norm).rotate(90, expand=True)
        img = img.resize((size, size), Image.LANCZOS)
        # RGB 변환 (SD는 3채널 필요)
        img = img.convert("RGB")

        out_path = out_dir / f"{stem}_z{z:03d}.png"
        img.save(out_path)
        saved.append(out_path)

    return saved


def main():
    parser = argparse.ArgumentParser(description="IXI-T1 → PNG 변환 및 split 정리")
    parser.add_argument("--ixi_dir",      required=True,  help="IXI-T1/ 폴더 경로")
    parser.add_argument("--output_dir",   required=True,  help="출력 루트 폴더 (data/)")
    parser.add_argument("--n_train",      type=int, default=200,  help="학습용 볼륨 수")
    parser.add_argument("--n_val",        type=int, default=0,    help="검증용 볼륨 수 (0=skip)")
    parser.add_argument("--n_test",       type=int, default=300,  help="테스트용 볼륨 수")
    parser.add_argument("--slices_per_volume", type=int, default=5,
                        help="볼륨당 추출할 슬라이스 수")
    parser.add_argument("--size",         type=int, default=512,  help="출력 이미지 크기")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    ixi_dir = Path(args.ixi_dir)
    out_dir = Path(args.output_dir)

    # NIfTI 파일 목록
    nii_files = sorted(ixi_dir.glob("*.nii.gz"))
    if not nii_files:
        nii_files = sorted(ixi_dir.glob("*.nii"))
    print(f"발견된 볼륨 수: {len(nii_files)}")

    need = args.n_train + args.n_val + args.n_test
    if need > len(nii_files):
        print(f"[경고] 요청 볼륨 수({need})가 파일 수({len(nii_files)})를 초과합니다.")
        print("       n_train/n_val/n_test를 줄이거나 slices_per_volume을 늘리세요.")

    random.shuffle(nii_files)

    splits: dict[str, list] = {}
    cursor = 0
    splits["train"] = nii_files[cursor : cursor + args.n_train]; cursor += args.n_train
    if args.n_val > 0:
        splits["val"]   = nii_files[cursor : cursor + args.n_val];   cursor += args.n_val
    splits["test"]  = nii_files[cursor : cursor + args.n_test]

    total_saved = 0
    for split_name, files in splits.items():
        dest = out_dir / split_name / "healthy"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split_name}] {len(files)}개 볼륨 → {dest}")

        for nii_path in files:
            saved = volume_to_pngs(nii_path, dest, args.slices_per_volume, args.size)
            total_saved += len(saved)
            print(f"  {nii_path.name}: {len(saved)}장 저장")

    print(f"\n완료! 총 {total_saved}장의 PNG 저장")
    print(f"\n폴더 구조:")
    for split_name in splits:
        dest = out_dir / split_name / "healthy"
        n = len(list(dest.glob("*.png")))
        print(f"  {dest}  ({n}장)")


if __name__ == "__main__":
    main()
