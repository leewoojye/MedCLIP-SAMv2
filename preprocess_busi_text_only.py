import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import easyocr  # type: ignore
    EASYOCR_IMPORT_ERROR = None
except Exception as exc:
    easyocr = None
    EASYOCR_IMPORT_ERROR = exc

try:
    from simple_lama_inpainting import SimpleLama  # type: ignore
    LAMA_IMPORT_ERROR = None
except Exception as exc:
    SimpleLama = None
    LAMA_IMPORT_ERROR = exc


SOURCE_DIR = Path("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_TEST/test_images")
OUTPUT_ROOT = Path("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_processed_text_only")
PILOT_FILES = ["benign_benign (1).png", "benign_benign (5).png", "benign_benign (6).png"]


def collect_inputs(run_all: bool) -> list[Path]:
    if run_all:
        return sorted(SOURCE_DIR.glob("*.png"))
    return [SOURCE_DIR / name for name in PILOT_FILES]


def build_easyocr_reader():
    if easyocr is None:
        return None
    try:
        return easyocr.Reader(["en"], gpu=True)
    except Exception:
        try:
            return easyocr.Reader(["en"], gpu=False)
        except Exception:
            return None


def build_lama():
    if SimpleLama is None:
        return None
    try:
        return SimpleLama()
    except Exception:
        return None


def get_text_fallback_mask(img_gray: np.ndarray) -> np.ndarray:
    tophat = cv2.morphologyEx(
        img_gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    )
    binary = cv2.adaptiveThreshold(
        tophat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -6
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)


def generate_text_only_mask(img_bgr: np.ndarray, ocr_reader) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if ocr_reader is not None:
        ocr_results = ocr_reader.readtext(img_bgr)
        for (bbox, _text, _prob) in ocr_results:
            tl = (int(bbox[0][0]), int(bbox[0][1]))
            br = (int(bbox[2][0]), int(bbox[2][1]))
            cv2.rectangle(mask, tl, br, 255, -1)
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = cv2.bitwise_or(mask, get_text_fallback_mask(gray))

    # Apply yellow marker removal for all images.
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(mask, yellow_mask)

    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)


def inpaint_with_lama_or_fallback(img_bgr: np.ndarray, mask: np.ndarray, lama_model) -> np.ndarray:
    if lama_model is not None:
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert("L")
        result_pil = lama_model(img_pil, mask_pil)
        return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    return cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)


def run_pipeline(run_all: bool) -> None:
    selected = collect_inputs(run_all)
    missing = [p.name for p in selected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {SOURCE_DIR}: {missing}")

    mode = "all images" if run_all else f"pilot images ({', '.join(PILOT_FILES)})"
    print(f"[INFO] Processing {mode}")
    print(f"[INFO] Input dir: {SOURCE_DIR}")
    print(f"[INFO] Output dir: {OUTPUT_ROOT}")

    shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    (OUTPUT_ROOT / "test_images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "debug_masks").mkdir(parents=True, exist_ok=True)

    ocr_reader = build_easyocr_reader()
    if ocr_reader is None:
        msg = "not installed" if EASYOCR_IMPORT_ERROR else "runtime init failed"
        print(f"[WARN] EasyOCR unavailable ({msg}). Using fallback text detector.")
        if EASYOCR_IMPORT_ERROR:
            print(f"[WARN] EasyOCR import error: {EASYOCR_IMPORT_ERROR}")
    else:
        print("[INFO] EasyOCR initialized.")

    lama_model = build_lama()
    if lama_model is None:
        msg = "not installed" if LAMA_IMPORT_ERROR else "runtime init failed"
        print(f"[WARN] SimpleLaMa unavailable ({msg}). Using OpenCV inpaint fallback.")
        if LAMA_IMPORT_ERROR:
            print(f"[WARN] SimpleLaMa import error: {LAMA_IMPORT_ERROR}")
    else:
        print("[INFO] SimpleLaMa initialized.")

    for img_path in tqdm(selected, desc="Preprocessing"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Failed to read image: {img_path.name}")
            continue

        mask = generate_text_only_mask(img_bgr, ocr_reader)
        cleaned = inpaint_with_lama_or_fallback(img_bgr, mask, lama_model)

        cv2.imwrite(str(OUTPUT_ROOT / "debug_masks" / img_path.name), mask)
        cv2.imwrite(str(OUTPUT_ROOT / "test_images" / img_path.name), cleaned)

    print("[INFO] Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BUSI text-only artifact removal")
    parser.add_argument("--all", action="store_true", help="Process all BUSI_TEST/test_images")
    args = parser.parse_args()
    run_pipeline(run_all=args.all)
