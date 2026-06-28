import os
import re
import time
import tempfile

from PIL import Image

from benchmark.models.base import BaseOCRModel, OCRResult

DET_TAG_PATTERN = re.compile(r'<\|[^|]*\|>[^<]*(?:<\|/[^|]*\|>)?')


class UnlimitedOCRModel(BaseOCRModel):
    name = "unlimited_ocr"
    description = "Baidu Unlimited-OCR 3B (BF16, CUDA required)"

    def __init__(self, image_size=640, base_size=1024, crop_mode=True):
        self.model_name = "baidu/Unlimited-OCR"
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()

    def predict(self, image: Image.Image) -> OCRResult:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        try:
            t0 = time.perf_counter()
            raw = self.model.infer(
                self.tokenizer,
                prompt="<image>document parsing.",
                image_file=tmp_path,
                output_path=tempfile.gettempdir(),
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                eval_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=False,
            )
            elapsed = time.perf_counter() - t0
        finally:
            os.unlink(tmp_path)

        text = DET_TAG_PATTERN.sub('', raw or '').strip()
        return OCRResult(text=text, inference_time_s=elapsed)

    def unload(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
