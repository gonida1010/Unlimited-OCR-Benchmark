from PIL import Image


def preprocess_for_benchmark(image: Image.Image, dataset_name: str) -> Image.Image:
    """Shared preprocessing applied before any model sees the image.

    Intentionally minimal to ensure fair comparison — each model's
    internal resize/normalize pipeline handles the rest.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image
