import statistics
from dataclasses import dataclass

from benchmark.evaluation.metrics import character_error_rate, word_error_rate, exact_match
from benchmark.evaluation.postprocessing import NORMALIZATION_REGISTRY


@dataclass
class SampleResult:
    sample_id: str
    dataset_name: str
    model_name: str
    ground_truth_raw: str
    prediction_raw: str
    ground_truth_normalized: str
    prediction_normalized: str
    cer: float
    wer: float
    is_exact_match: bool
    inference_time_s: float


@dataclass
class AggregatedResult:
    model_name: str
    dataset_name: str
    num_samples: int
    mean_cer: float
    median_cer: float
    std_cer: float
    mean_wer: float
    median_wer: float
    std_wer: float
    exact_match_accuracy: float
    mean_inference_time_s: float
    total_inference_time_s: float


class Evaluator:

    def __init__(self):
        self.sample_results: list[SampleResult] = []

    def add_result(
        self,
        sample_id: str,
        dataset_name: str,
        model_name: str,
        ground_truth: str,
        prediction: str,
        inference_time_s: float,
    ) -> SampleResult:
        normalize_fn = NORMALIZATION_REGISTRY.get(dataset_name)
        gt_norm = normalize_fn(ground_truth) if normalize_fn else ground_truth
        pred_norm = normalize_fn(prediction) if normalize_fn else prediction

        result = SampleResult(
            sample_id=sample_id,
            dataset_name=dataset_name,
            model_name=model_name,
            ground_truth_raw=ground_truth,
            prediction_raw=prediction,
            ground_truth_normalized=gt_norm,
            prediction_normalized=pred_norm,
            cer=character_error_rate(pred_norm, gt_norm),
            wer=word_error_rate(pred_norm, gt_norm),
            is_exact_match=exact_match(pred_norm, gt_norm),
            inference_time_s=inference_time_s,
        )
        self.sample_results.append(result)
        return result

    def aggregate(self, model_name: str, dataset_name: str) -> AggregatedResult:
        subset = [
            r for r in self.sample_results
            if r.model_name == model_name and r.dataset_name == dataset_name
        ]
        if not subset:
            raise ValueError(f"No results for {model_name} on {dataset_name}")

        cers = [r.cer for r in subset]
        wers = [r.wer for r in subset]
        times = [r.inference_time_s for r in subset]

        return AggregatedResult(
            model_name=model_name,
            dataset_name=dataset_name,
            num_samples=len(subset),
            mean_cer=statistics.mean(cers),
            median_cer=statistics.median(cers),
            std_cer=statistics.stdev(cers) if len(cers) > 1 else 0.0,
            mean_wer=statistics.mean(wers),
            median_wer=statistics.median(wers),
            std_wer=statistics.stdev(wers) if len(wers) > 1 else 0.0,
            exact_match_accuracy=sum(r.is_exact_match for r in subset) / len(subset),
            mean_inference_time_s=statistics.mean(times),
            total_inference_time_s=sum(times),
        )

    def aggregate_all(self) -> list[AggregatedResult]:
        pairs = set((r.model_name, r.dataset_name) for r in self.sample_results)
        return [self.aggregate(m, d) for m, d in sorted(pairs)]
