from benchmark.datasets.iiit5k import IIIT5KDataset
from benchmark.datasets.icdar2015 import ICDAR2015Dataset
from benchmark.datasets.iam import IAMDataset
from benchmark.datasets.funsd import FUNSDDataset

DATASET_REGISTRY: dict[str, type] = {
    "iiit5k": IIIT5KDataset,
    "icdar2015": ICDAR2015Dataset,
    "iam": IAMDataset,
    "funsd": FUNSDDataset,
}


def load_dataset_by_name(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
