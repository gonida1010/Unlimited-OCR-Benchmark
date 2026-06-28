"""각 모델의 BBox 탐지 영역을 시각화하는 스크립트."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

OUTPUT_DIR = "outputs/bbox_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5

ds = load_dataset("Teklia/IAM-line", split="test")


def draw_easyocr_boxes(image: Image.Image, idx: int):
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True)
    img_array = np.array(image.convert("RGB"))
    results = reader.readtext(img_array)

    draw = ImageDraw.Draw(image)
    for bbox, text, conf in results:
        pts = [(int(p[0]), int(p[1])) for p in bbox]
        draw.polygon(pts, outline="red", width=2)
        draw.text(pts[0], f"{text} ({conf:.2f})", fill="red")

    image.save(f"{OUTPUT_DIR}/iam_{idx:03d}_easyocr.png")
    return results


def draw_paddleocr_boxes(image: Image.Image, idx: int):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    img_array = np.array(image.convert("RGB"))
    results = ocr.ocr(img_array, cls=True)

    draw = ImageDraw.Draw(image)
    if results and results[0]:
        for line in results[0]:
            bbox, (text, conf) = line
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            draw.polygon(pts, outline="blue", width=2)
            draw.text(pts[0], f"{text} ({conf:.2f})", fill="blue")

    image.save(f"{OUTPUT_DIR}/iam_{idx:03d}_paddleocr.png")
    return results


def draw_unlimited_ocr_boxes(image: Image.Image, idx: int):
    """Unlimited-OCR는 BBox를 반환하지 않음 - 전체 이미지를 하나로 처리.
    모델 출력의 <|det|> 태그에서 좌표를 추출하여 표시."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    import tempfile

    tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "baidu/Unlimited-OCR",
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    rgb = image.convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        rgb.save(tmp, format="PNG")
        tmp_path = tmp.name

    raw_text = model.infer(
        tokenizer,
        prompt="<image>Extract the text in the image.",
        image_file=tmp_path,
        output_path=tempfile.gettempdir(),
        base_size=1024,
        image_size=640,
        crop_mode=True,
        eval_mode=True,
        max_length=4096,
        save_results=False,
    )
    os.unlink(tmp_path)

    draw = ImageDraw.Draw(rgb)
    w, h = rgb.size

    import re
    det_pattern = re.compile(r'<\|det\|>\w+\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]')
    matches = det_pattern.findall(raw_text or "")

    if matches:
        for m in matches:
            x1 = int(m[0]) * w // 999
            y1 = int(m[1]) * h // 999
            x2 = int(m[2]) * w // 999
            y2 = int(m[3]) * h // 999
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
    else:
        draw.rectangle([0, 0, w - 1, h - 1], outline="green", width=2)
        draw.text((5, 5), "[no det box - full image]", fill="green")

    clean_text = re.sub(r'<\|[^|]*\|>[^<]*(<\|/[^|]*\|>)?', '', raw_text or "").strip()
    draw.text((5, h - 15), clean_text[:80] if clean_text else "[empty]", fill="green")

    rgb.save(f"{OUTPUT_DIR}/iam_{idx:03d}_unlimited_ocr.png")

    del model, tokenizer
    torch.cuda.empty_cache()
    return raw_text


if __name__ == "__main__":
    print(f"Generating BBox visualizations for {NUM_SAMPLES} samples...")

    print("\n--- Unlimited-OCR (loading model once is slow, ~30s) ---")
    for i in range(NUM_SAMPLES):
        item = ds[i]
        image = item["image"].convert("RGB")
        gt = item["text"]
        raw = draw_unlimited_ocr_boxes(image.copy(), i)
        print(f"  [{i}] GT: {gt}")
        print(f"       PR: {raw}")

    print("\n--- EasyOCR ---")
    for i in range(NUM_SAMPLES):
        item = ds[i]
        image = item["image"].convert("RGB")
        gt = item["text"]
        results = draw_easyocr_boxes(image.copy(), i)
        texts = [t for _, t, _ in results]
        print(f"  [{i}] GT: {gt}")
        print(f"       PR: {' '.join(texts)}")
        print(f"       Boxes: {len(results)}개")

    print("\n--- PaddleOCR ---")
    for i in range(NUM_SAMPLES):
        item = ds[i]
        image = item["image"].convert("RGB")
        gt = item["text"]
        results = draw_paddleocr_boxes(image.copy(), i)
        if results and results[0]:
            texts = [line[1][0] for line in results[0]]
            print(f"  [{i}] GT: {gt}")
            print(f"       PR: {' '.join(texts)}")
            print(f"       Boxes: {len(results[0])}개")
        else:
            print(f"  [{i}] GT: {gt}")
            print(f"       PR: [no detection]")

    print(f"\nBBox 이미지 저장 완료: {OUTPUT_DIR}/")
