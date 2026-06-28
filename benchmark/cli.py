import argparse
import logging
from pathlib import Path

from benchmark.config import BenchmarkConfig, MODEL_NAMES, DATASET_NAMES
from benchmark.pipeline import run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ocr-benchmark",
        description="Benchmark OCR models across standard datasets",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=MODEL_NAMES,
        default=MODEL_NAMES,
        help="Models to benchmark (default: all)",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=DATASET_NAMES,
        default=DATASET_NAMES,
        help="Datasets to evaluate on (default: all)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=500,
        help="Max samples per dataset (default: 500)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--unlimited-ocr-image-size",
        type=int,
        default=640,
        choices=[512, 640, 1024, 1280],
        help="Image size for Unlimited-OCR model",
    )
    parser.add_argument(
        "--unlimited-ocr-no-crop",
        action="store_true",
        help="Disable crop mode for Unlimited-OCR",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = BenchmarkConfig(
        models=args.models,
        datasets=args.datasets,
        max_samples=args.max_samples,
        seed=args.seed,
        unlimited_ocr_image_size=args.unlimited_ocr_image_size,
        unlimited_ocr_crop_mode=not args.unlimited_ocr_no_crop,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    run_benchmark(config)
