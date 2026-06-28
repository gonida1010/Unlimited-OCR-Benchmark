from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator
from PIL import Image


@dataclass
class OCRSample:
    image: Image.Image
    ground_truth: str
    sample_id: str
    dataset_name: str
    metadata: dict | None = None


class BaseDataset(ABC):

    def __init__(self, split: str = "test", max_samples: int | None = None):
        self.split = split
        self.max_samples = max_samples
        self._data = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[OCRSample]: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> OCRSample: ...
