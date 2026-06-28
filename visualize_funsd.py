"""FUNSD 전체 페이지에서 각 모델의 BBox 탐지 영역을 시각화."""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = "outputs/bbox_funsd"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5
ds = load_dataset("nielsr/FUNSD", split="test")


def draw_easyocr_boxes(image: Image.Image, idx: int):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True)
    img_array = np.array(image)
    results = reader.readtext(img_array)

    draw = ImageDraw.Draw(image)
    for bbox, text, conf in results:
        pts = [(int(p[0]), int(p[1])) for p in bbox]
        draw.polygon(pts, outline="red", width=2)

    image.save(f"{OUTPUT_DIR}/funsd_{idx:03d}_easyocr.png")
    print(f"  EasyOCR: {len(results)} boxes detected")
    return results


def draw_paddleocr_boxes(image: Image.Image, idx: int):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    img_array = np.array(image)
    results = ocr.ocr(img_array, cls=True)

    draw = ImageDraw.Draw(image)
    count = 0
    if results and results[0]:
        for line in results[0]:
            bbox = line[0]
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            draw.polygon(pts, outline="blue", width=2)
            count += 1

    image.save(f"{OUTPUT_DIR}/funsd_{idx:03d}_paddleocr.png")
    print(f"  PaddleOCR: {count} boxes detected")
    return results


def draw_unlimited_ocr_raw(image: Image.Image, idx: int):
    """Unlimited-OCR는 전체 이미지를 한번에 읽음 - 전체 영역 표시."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.rectangle([0, 0, w - 1, h - 1], outline="green", width=3)
    draw.text((5, 5), "Unlimited-OCR: full page (no per-word bbox)", fill="green")
    image.save(f"{OUTPUT_DIR}/funsd_{idx:03d}_unlimited_ocr.png")
    print(f"  Unlimited-OCR: full page (no individual bbox)")


def draw_gt_boxes(image: Image.Image, words, bboxes, idx: int):
    """GT bbox를 그려서 원본 참조용으로 저장. FUNSD bbox는 0-1000 정규화 좌표."""
    w, h = image.size
    sx, sy = w / 1000.0, h / 1000.0
    draw = ImageDraw.Draw(image)
    for word, bbox in zip(words, bboxes):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline="orange", width=1)
    image.save(f"{OUTPUT_DIR}/funsd_{idx:03d}_gt.png")
    print(f"  GT: {len(words)} word boxes")


if __name__ == "__main__":
    print(f"Generating BBox visualizations for {NUM_SAMPLES} FUNSD pages...\n")

    for i in range(NUM_SAMPLES):
        item = ds[i]
        image = item["image"].convert("RGB")
        print(f"[Sample {i}] size={image.size}")

        draw_gt_boxes(image.copy(), item["words"], item["bboxes"], i)
        draw_easyocr_boxes(image.copy(), i)
        draw_paddleocr_boxes(image.copy(), i)
        draw_unlimited_ocr_raw(image.copy(), i)
        print()

    print(f"Done. Images saved to {OUTPUT_DIR}/")
