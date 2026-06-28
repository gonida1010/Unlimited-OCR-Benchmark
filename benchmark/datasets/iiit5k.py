from datasets import load_dataset as hf_load_dataset

from benchmark.datasets.base import BaseDataset, OCRSample


class IIIT5KDataset(BaseDataset):
    name = "iiit5k"
    description = "IIIT 5K-word scene text recognition (3000 test images)"

    def load(self) -> None:
        ds = hf_load_dataset("MiXaiLL76/IIIT5K_OCR", split=self.split)
        if self.max_samples is not None:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        self._data = ds

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for idx in range(len(self._data)):
            yield self[idx]

    def __getitem__(self, idx: int) -> OCRSample:
        item = self._data[idx]
        image = item["image"].convert("RGB")
        return OCRSample(
            image=image,
            ground_truth=item["text"],
            sample_id=f"iiit5k_{idx:05d}",
            dataset_name=self.name,
        )
