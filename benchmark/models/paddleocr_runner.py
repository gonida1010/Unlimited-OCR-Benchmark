import time

import numpy as np
from PIL import Image

from benchmark.models.base import BaseOCRModel, OCRResult


class PaddleOCRModel(BaseOCRModel):
    name = "paddleocr"
    description = "PaddleOCR v3 (detection + recognition, CPU/GPU)"

    def __init__(self, lang="en"):
        self.lang = lang
        self.ocr = None

    def load(self) -> None:
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(lang=self.lang)

    def predict(self, image: Image.Image) -> OCRResult:
        img_array = np.array(image)

        t0 = time.perf_counter()
        results = self.ocr.predict(img_array)
        elapsed = time.perf_counter() - t0

        lines = []
        for res in results:
            if hasattr(res, "rec_texts") and res.rec_texts:
                lines.extend(res.rec_texts)

        text = " ".join(lines)
        return OCRResult(text=text, inference_time_s=elapsed)

    def unload(self) -> None:
        del self.ocr
        self.ocr = None
