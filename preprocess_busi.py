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
OUTPUT_ROOT = Path("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_processed")
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


def border_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    top_h = int(0.28 * h)
    side_w = int(0.18 * w)
    bottom_h = int(0.10 * h)
    mask[:top_h, :] = 255
    mask[:, :side_w] = 255
    mask[:, w - side_w :] = 255
    mask[h - bottom_h :, :] = 255

    # Protect probable lesion center to avoid inpainting core tissue.
    center_guard = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    ax, by = int(0.28 * w), int(0.24 * h)
    cv2.ellipse(center_guard, (cx, cy), (ax, by), 0, 0, 360, 255, -1)
    return cv2.bitwise_and(mask, cv2.bitwise_not(center_guard))


def get_text_fallback_mask(img_gray: np.ndarray) -> np.ndarray:
    tophat = cv2.morphologyEx(
        img_gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    )
    bhat = cv2.morphologyEx(
        img_gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    )
    enhanced = cv2.max(tophat, bhat)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -6
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    mask = np.zeros_like(img_gray)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h_img, w_img = img_gray.shape
    border_x = int(0.18 * w_img)
    border_y = int(0.18 * h_img)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if h <= 0 or w <= 0:
            continue
        ar = w / float(h)
        near_border = (
            x < border_x
            or y < border_y
            or (x + w) > (w_img - border_x)
            or (y + h) > (h_img - border_y)
        )
        if near_border and (12 <= area <= 2200 and 0.25 <= ar <= 10.0 and h <= 52):
            mask[labels == i] = 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def get_linear_marker_mask(img_gray: np.ndarray) -> np.ndarray:
    bright_thr = max(205, int(np.percentile(img_gray, 99.2)))
    _, bright = cv2.threshold(img_gray, bright_thr, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(bright, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=14, maxLineGap=8)

    raw = np.zeros_like(img_gray)
    if lines is not None:
        for item in lines:
            x1, y1, x2, y2 = item[0]
            dx = x2 - x1
            dy = y2 - y1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < 16:
                continue
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle > 90:
                angle = 180 - angle
            if angle <= 12 or angle >= 78:
                cv2.line(raw, (x1, y1), (x2, y2), 255, 2)

    refined = np.zeros_like(img_gray)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    for i in range(1, num_labels):
        _x, _y, w, h, area = stats[i]
        if h <= 0 or w <= 0:
            continue
        ar = w / float(h)
        fill = area / float(w * h)
        if area > 2400:
            continue
        if (ar > 3.5 or ar < 0.28) and fill > 0.08:
            refined[labels == i] = 255
        elif 25 <= area <= 450 and 0.5 <= ar <= 2.0:
            refined[labels == i] = 255

    # Pick up dotted/segmented line traces.
    dotted_h = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    )
    dotted_h = cv2.morphologyEx(
        dotted_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    )
    dotted_v = cv2.morphologyEx(
        bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    )
    dotted_v = cv2.morphologyEx(
        dotted_v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    )
    refined = cv2.bitwise_or(refined, dotted_h)
    refined = cv2.bitwise_or(refined, dotted_v)
    return refined


def enforce_mask_ratio(mask: np.ndarray, max_ratio: float = 0.03) -> np.ndarray:
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if ratio <= max_ratio:
        return mask

    refined = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        _x, _y, w, h, area = stats[i]
        if h <= 0 or w <= 0:
            continue
        ar = w / float(h)
        if area <= 1200 and (ar > 4.0 or ar < 0.25):
            refined[labels == i] = 255
    return refined


def generate_artifact_mask(img_bgr: np.ndarray, ocr_reader) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    if ocr_reader is not None:
        ocr_results = ocr_reader.readtext(img_bgr)
        for (bbox, _text, _prob) in ocr_results:
            tl = (int(bbox[0][0]), int(bbox[0][1]))
            br = (int(bbox[2][0]), int(bbox[2][1]))
            cv2.rectangle(final_mask, tl, br, 255, -1)
    else:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        final_mask = cv2.bitwise_or(final_mask, get_text_fallback_mask(img_gray))

    # Yellow marker mask (cross, dotted lines, boxes)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))
    final_mask = cv2.bitwise_or(final_mask, yellow_mask)

    # Additional geometric marker mask for non-yellow lines/crosses.
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    line_mask = get_linear_marker_mask(img_gray)
    final_mask = cv2.bitwise_or(final_mask, line_mask)

    # Dilate exactly like the sample script
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=2)
    return final_mask


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

        mask = generate_artifact_mask(img_bgr, ocr_reader)
        cleaned = inpaint_with_lama_or_fallback(img_bgr, mask, lama_model)

        cv2.imwrite(str(OUTPUT_ROOT / "debug_masks" / img_path.name), mask)
        cv2.imwrite(str(OUTPUT_ROOT / "test_images" / img_path.name), cleaned)

    print("[INFO] Completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BUSI marker-removal preprocessing")
    parser.add_argument("--all", action="store_true", help="Process all BUSI_TEST/test_images")
    args = parser.parse_args()
    run_pipeline(run_all=args.all)
