import time

import numpy as np
from PIL import Image

from benchmark.models.base import BaseOCRModel, OCRResult


class EasyOCRModel(BaseOCRModel):
    name = "easyocr"
    description = "EasyOCR (English, GPU-accelerated)"

    def __init__(self, languages=None, gpu=True):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = None

    def load(self) -> None:
        import easyocr

        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def predict(self, image: Image.Image, sample_id: str = "") -> OCRResult:
        img_array = np.array(image)

        t0 = time.perf_counter()
        results = self.reader.readtext(img_array)
        elapsed = time.perf_counter() - t0

        lines = [text for (_, text, _) in results]
        confidences = [conf for (_, _, conf) in results]

        text = " ".join(lines)
        return OCRResult(
            text=text,
            inference_time_s=elapsed,
            metadata={"confidences": confidences},
        )

    def unload(self) -> None:
        del self.reader
        self.reader = None
