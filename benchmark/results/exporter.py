import csv
import json
from pathlib import Path
from dataclasses import asdict

from benchmark.evaluation.evaluator import Evaluator, AggregatedResult, SampleResult


class ResultExporter:

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_existing_samples(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _load_existing_aggregated(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _load_existing_json(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def export_sample_results_csv(
        self, evaluator: Evaluator, filename: str = "sample_results.csv"
    ) -> Path:
        path = self.output_dir / filename
        if not evaluator.sample_results:
            return path

        existing = self._load_existing_samples(path)
        new_pairs = {(r.model_name, r.dataset_name) for r in evaluator.sample_results}
        kept = [
            row for row in existing
            if (row["model_name"], row["dataset_name"]) not in new_pairs
        ]

        fields = list(asdict(evaluator.sample_results[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in kept:
                writer.writerow(row)
            for r in evaluator.sample_results:
                writer.writerow(asdict(r))
        return path

    def export_aggregated_csv(
        self, results: list[AggregatedResult], filename: str = "aggregated_results.csv"
    ) -> Path:
        path = self.output_dir / filename
        if not results:
            return path

        existing = self._load_existing_aggregated(path)
        new_pairs = {(r.model_name, r.dataset_name) for r in results}
        kept = [
            row for row in existing
            if (row["model_name"], row["dataset_name"]) not in new_pairs
        ]

        fields = list(asdict(results[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in kept:
                writer.writerow(row)
            for r in results:
                writer.writerow(asdict(r))
        return path

    def export_aggregated_json(
        self, results: list[AggregatedResult], filename: str = "aggregated_results.json"
    ) -> Path:
        path = self.output_dir / filename
        new_data = [asdict(r) for r in results]
        new_pairs = {(r.model_name, r.dataset_name) for r in results}

        existing = self._load_existing_json(path)
        kept = [
            row for row in existing
            if (row["model_name"], row["dataset_name"]) not in new_pairs
        ]

        merged = kept + new_data
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        return path
