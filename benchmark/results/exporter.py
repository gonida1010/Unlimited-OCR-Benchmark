import csv
import json
from pathlib import Path
from dataclasses import asdict

from benchmark.evaluation.evaluator import Evaluator, AggregatedResult


class ResultExporter:

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_sample_results_csv(
        self, evaluator: Evaluator, filename: str = "sample_results.csv"
    ) -> Path:
        path = self.output_dir / filename
        if not evaluator.sample_results:
            return path
        fields = list(asdict(evaluator.sample_results[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in evaluator.sample_results:
                writer.writerow(asdict(r))
        return path

    def export_aggregated_csv(
        self, results: list[AggregatedResult], filename: str = "aggregated_results.csv"
    ) -> Path:
        path = self.output_dir / filename
        if not results:
            return path
        fields = list(asdict(results[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        return path

    def export_aggregated_json(
        self, results: list[AggregatedResult], filename: str = "aggregated_results.json"
    ) -> Path:
        path = self.output_dir / filename
        data = [asdict(r) for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path
