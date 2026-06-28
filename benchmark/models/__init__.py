from benchmark.models.unlimited_ocr import UnlimitedOCRModel
from benchmark.models.paddleocr_runner import PaddleOCRModel
from benchmark.models.easyocr_runner import EasyOCRModel

MODEL_REGISTRY: dict[str, type] = {
    "unlimited_ocr": UnlimitedOCRModel,
    "paddleocr": PaddleOCRModel,
    "easyocr": EasyOCRModel,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
