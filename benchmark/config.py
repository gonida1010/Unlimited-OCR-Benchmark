from pathlib import Path
from dataclasses import dataclass, field

ROOT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"

DATASET_HF_IDS = {
    "iiit5k": "MiXaiLL76/IIIT5K_OCR",
    "icdar2015": "MiXaiLL76/ICDAR2015_OCR",
    "iam": "Teklia/IAM-line",
}

MODEL_NAMES = ["unlimited_ocr", "paddleocr", "easyocr"]
DATASET_NAMES = ["funsd"]


@dataclass
class BenchmarkConfig:
    models: list[str] = field(default_factory=lambda: MODEL_NAMES.copy())
    datasets: list[str] = field(default_factory=lambda: DATASET_NAMES.copy())
    max_samples: int | None = 500
    output_dir: Path = field(default_factory=lambda: RESULTS_DIR)
    seed: int = 42
    unlimited_ocr_image_size: int = 640
    unlimited_ocr_base_size: int = 1024
    unlimited_ocr_crop_mode: bool = True
