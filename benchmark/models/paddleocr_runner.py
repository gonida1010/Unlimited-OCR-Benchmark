import time

import numpy as np
from PIL import Image

from benchmark.models.base import BaseOCRModel, OCRResult


class PaddleOCRModel(BaseOCRModel):
    name = "paddleocr"
    description = "PaddleOCR v2 (detection + recognition, CPU/GPU)"

    def __init__(self, lang="en"):
        self.lang = lang
        self.ocr = None

    def load(self) -> None:
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)

    def predict(self, image: Image.Image) -> OCRResult:
        img_array = np.array(image)

        t0 = time.perf_counter()
        results = self.ocr.ocr(img_array, cls=True)
        elapsed = time.perf_counter() - t0

        lines = []
        if results and results[0]:
            for line in results[0]:
                lines.append(line[1][0])

        text = " ".join(lines)
        return OCRResult(text=text, inference_time_s=elapsed)

    def unload(self) -> None:
        del self.ocr
        self.ocr = None
