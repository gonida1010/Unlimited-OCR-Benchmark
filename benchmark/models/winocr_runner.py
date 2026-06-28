import time
import io

from PIL import Image

from benchmark.models.base import BaseOCRModel, OCRResult


class WinOCRModel(BaseOCRModel):
    name = "winocr"
    description = "Windows Runtime OCR (CPU only, Windows 10+)"

    def __init__(self, lang="en"):
        self.lang = lang
        self._winocr = None

    def load(self) -> None:
        import winocr

        self._winocr = winocr

    def predict(self, image: Image.Image) -> OCRResult:
        import asyncio

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        t0 = time.perf_counter()
        result = asyncio.run(
            self._winocr.recognize_pil(image, lang=self.lang)
        )
        elapsed = time.perf_counter() - t0

        text = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", str(result))
        return OCRResult(text=text, inference_time_s=elapsed)

    def unload(self) -> None:
        self._winocr = None
