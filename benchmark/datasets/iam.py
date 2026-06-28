from datasets import load_dataset as hf_load_dataset

from benchmark.datasets.base import BaseDataset, OCRSample


class IAMDataset(BaseDataset):
    name = "iam"
    description = "IAM Handwriting Database - line-level transcriptions (2915 test lines)"

    def __init__(self, split: str = "test", max_samples: int | None = None):
        super().__init__(split=split, max_samples=max_samples)

    def load(self) -> None:
        ds = hf_load_dataset("Teklia/IAM-line", split=self.split)
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
            sample_id=f"iam_{idx:05d}",
            dataset_name=self.name,
        )
