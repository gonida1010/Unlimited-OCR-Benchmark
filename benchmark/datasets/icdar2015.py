from datasets import load_dataset as hf_load_dataset

from benchmark.datasets.base import BaseDataset, OCRSample


class ICDAR2015Dataset(BaseDataset):
    name = "icdar2015"
    description = "ICDAR2015 scene text from camera images (2077 test crops)"

    def load(self) -> None:
        ds = hf_load_dataset("MiXaiLL76/ICDAR2015_OCR", split=self.split)
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
            sample_id=f"icdar2015_{idx:05d}",
            dataset_name=self.name,
        )
