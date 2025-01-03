from typing import Optional

import re
from pathlib import Path

import cv2
import pytesseract
from pytesseract import Output


def preprocess_image(image, img_debug_dir: Optional[Path] = None):
    # Scaling
    height, width = image.shape[:2]
    scale_factor = 2 if min(height, width) < 1000 else 1
    resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Binary
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    if img_debug_dir:
        cv2.imwrite(img_debug_dir / f"gray/{image_name}", gray)
        cv2.imwrite(img_debug_dir / f"binary/{image_name}", binary)

    return binary


def run_ocr(img):
    custom_config = r"--psm 6 --oem 3"  # PSM 6 is for text blocks with uniform alignment
    return pytesseract.image_to_string(img, config=custom_config)


def extract_collector_number(ocr_result):
    """
    Generalize collector number to match C, U, R, or M. Sometimes
    OCR picks up 'U0303' instead of 'U 0303' so we need to handle this.
    """
    pat = r"[CUMR] ?\d{3,4}"
    match = re.search(pat, ocr_result)
    return match.group(0).replace(" ", "")[1:] if match else None


def extract_set_abbr(ocr_result):
    pat = r"[A-Z]{3,4} [«•⭑\-\+] EN"
    match = re.search(pat, ocr_result)
    return match.group(0).split(" ")[0] if match else None


if __name__ == "__main__":

    IMAGE_DIR = Path("data/images")
    IMG_DEBUG_DIR = IMAGE_DIR / "debug"
    # Ensure that debug dirs exist
    (IMG_DEBUG_DIR / "gray").mkdir(parents=True, exist_ok=True)
    (IMG_DEBUG_DIR / "binary").mkdir(parents=True, exist_ok=True)

    image_name = "scryfall/ltr-95-mirkwood-bats.jpg"
    image_name = "scryfall/ltr-303-faramir-field-commander.jpg"
    image_name = "scryfall/ltr-203-flame-of-anor.jpg"
    image_name = "scryfall/ltr-328-saruman-of-many-colors.jpg"

    image_name = "wild/mines_of_moria.jpg"
    image_name = "wild/swamp.jpg"

    image_path = IMAGE_DIR / image_name
    image = cv2.imread(image_path)

    img = preprocess_image(image, IMG_DEBUG_DIR)
    ocr_result = run_ocr(img)
    collector_num = extract_collector_number(ocr_result)
    set_abbr = extract_set_abbr(ocr_result)

    print("Collector Number:", collector_num)
    print("Set Info:", set_abbr)

    if not collector_num or not set_abbr:
        print("\nIssue parsing collector num and set. Full OCR result:")
        print(ocr_result)
