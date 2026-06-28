import os
import time
import tempfile
import shutil
from pathlib import Path

from PIL import Image

from benchmark.models.base import BaseOCRModel, OCRResult


class UnlimitedOCRModel(BaseOCRModel):
    name = "unlimited_ocr"
    description = "Baidu Unlimited-OCR 3B (BF16, CUDA required)"

    def __init__(self, image_size=640, base_size=1024, crop_mode=True,
                 save_visuals_dir: Path | None = None):
        self.model_name = "baidu/Unlimited-OCR"
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.save_visuals_dir = save_visuals_dir
        self.model = None
        self.tokenizer = None
        self._sample_counter = 0

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

        if self.save_visuals_dir:
            self.save_visuals_dir.mkdir(parents=True, exist_ok=True)

    def predict(self, image: Image.Image, sample_id: str = "") -> OCRResult:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        output_dir = tempfile.mkdtemp(prefix="uocr_")
        try:
            t0 = time.perf_counter()
            self.model.infer(
                self.tokenizer,
                prompt="<image>document parsing.",
                image_file=tmp_path,
                output_path=output_dir,
                base_size=self.base_size,
                image_size=self.image_size,
                crop_mode=self.crop_mode,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )
            elapsed = time.perf_counter() - t0

            result_path = os.path.join(output_dir, "result.md")
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            else:
                text = ""

            if self.save_visuals_dir:
                tag = sample_id or f"sample_{self._sample_counter:04d}"
                bbox_src = os.path.join(output_dir, "result_with_boxes.jpg")
                if os.path.exists(bbox_src):
                    shutil.copy2(bbox_src, self.save_visuals_dir / f"{tag}_bbox.jpg")
                md_src = os.path.join(output_dir, "result.md")
                if os.path.exists(md_src):
                    shutil.copy2(md_src, self.save_visuals_dir / f"{tag}_result.md")
                self._sample_counter += 1
        finally:
            os.unlink(tmp_path)
            shutil.rmtree(output_dir, ignore_errors=True)

        return OCRResult(text=text, inference_time_s=elapsed)

    def unload(self) -> None:
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
