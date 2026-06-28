import logging

from tqdm import tqdm

from benchmark.config import BenchmarkConfig
from benchmark.datasets import load_dataset_by_name
from benchmark.datasets.preprocessing import preprocess_for_benchmark
from benchmark.models import get_model
from benchmark.evaluation.evaluator import Evaluator
from benchmark.results.exporter import ResultExporter
from benchmark.results.reporter import print_summary

logger = logging.getLogger(__name__)


def run_benchmark(config: BenchmarkConfig) -> None:
    evaluator = Evaluator()
    exporter = ResultExporter(config.output_dir)

    for dataset_name in config.datasets:
        logger.info("Loading dataset: %s", dataset_name)
        dataset = load_dataset_by_name(dataset_name, max_samples=config.max_samples)
        dataset.load()
        logger.info("  Loaded %d samples", len(dataset))

        for model_name in config.models:
            logger.info("Running model: %s on %s", model_name, dataset_name)

            model_kwargs = {}
            if model_name == "unlimited_ocr":
                visuals_dir = config.output_dir / "unlimited_ocr_visuals"
                model_kwargs = {
                    "image_size": config.unlimited_ocr_image_size,
                    "base_size": config.unlimited_ocr_base_size,
                    "crop_mode": config.unlimited_ocr_crop_mode,
                    "save_visuals_dir": visuals_dir,
                }

            model = get_model(model_name, **model_kwargs)

            try:
                with model:
                    for sample in tqdm(
                        dataset,
                        desc=f"{model_name}/{dataset_name}",
                        total=len(dataset),
                    ):
                        image = preprocess_for_benchmark(sample.image, dataset_name)
                        result = model.predict(image, sample_id=sample.sample_id)

                        evaluator.add_result(
                            sample_id=sample.sample_id,
                            dataset_name=dataset_name,
                            model_name=model_name,
                            ground_truth=sample.ground_truth,
                            prediction=result.text,
                            inference_time_s=result.inference_time_s,
                        )
            except Exception:
                logger.exception("Failed to run %s on %s", model_name, dataset_name)
                continue

            logger.info("  Completed %s on %s", model_name, dataset_name)

    aggregated = evaluator.aggregate_all()

    sample_csv = exporter.export_sample_results_csv(evaluator)
    agg_csv = exporter.export_aggregated_csv(aggregated)
    agg_json = exporter.export_aggregated_json(aggregated)

    print_summary(aggregated)

    logger.info("Sample-level results: %s", sample_csv)
    logger.info("Aggregated CSV:       %s", agg_csv)
    logger.info("Aggregated JSON:      %s", agg_json)
