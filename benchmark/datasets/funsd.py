from datasets import load_dataset as hf_load_dataset

from benchmark.datasets.base import BaseDataset, OCRSample


class FUNSDDataset(BaseDataset):
    name = "funsd"
    description = "FUNSD scanned forms - full page (50 test pages)"

    def load(self) -> None:
        ds = hf_load_dataset("nielsr/FUNSD", split=self.split)
        if self.max_samples is not None:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        self._data = ds

    def _build_reading_order_text(self, words: list[str], bboxes: list[list[int]]) -> str:
        pairs = list(zip(words, bboxes))
        if not pairs:
            return ""

        heights = [b[3] - b[1] for _, b in pairs]
        avg_height = sum(heights) / len(heights) if heights else 20
        line_threshold = avg_height * 0.5

        pairs.sort(key=lambda p: (p[1][1], p[1][0]))

        lines: list[list[tuple]] = []
        current_line: list[tuple] = [pairs[0]]

        for word, bbox in pairs[1:]:
            prev_y = current_line[-1][1][1]
            if abs(bbox[1] - prev_y) <= line_threshold:
                current_line.append((word, bbox))
            else:
                lines.append(current_line)
                current_line = [(word, bbox)]
        lines.append(current_line)

        text_lines = []
        for line in lines:
            line.sort(key=lambda p: p[1][0])
            text_lines.append(" ".join(w for w, _ in line))

        return "\n".join(text_lines)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for idx in range(len(self._data)):
            yield self[idx]

    def __getitem__(self, idx: int) -> OCRSample:
        item = self._data[idx]
        image = item["image"].convert("RGB")
        gt = self._build_reading_order_text(item["words"], item["bboxes"])
        return OCRSample(
            image=image,
            ground_truth=gt,
            sample_id=f"funsd_{idx:03d}",
            dataset_name=self.name,
        )
