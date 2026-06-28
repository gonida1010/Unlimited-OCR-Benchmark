from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


@dataclass
class OCRResult:
    text: str
    inference_time_s: float
    metadata: dict | None = None


class BaseOCRModel(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def predict(self, image: Image.Image) -> OCRResult: ...

    @abstractmethod
    def unload(self) -> None: ...

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
