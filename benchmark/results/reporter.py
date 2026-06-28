from benchmark.evaluation.evaluator import AggregatedResult


def print_summary(results: list[AggregatedResult]) -> None:
    if not results:
        print("No results to display.")
        return

    datasets = sorted(set(r.dataset_name for r in results))
    models = sorted(set(r.model_name for r in results))

    for dataset in datasets:
        print(f"\n{'=' * 78}")
        print(f"  Dataset: {dataset}")
        print(f"{'=' * 78}")
        print(
            f"{'Model':<20} {'Samples':>8} {'CER':>8} {'WER':>8} "
            f"{'ExMatch':>8} {'Avg Time':>10}"
        )
        print(f"{'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

        for model in models:
            match = [
                r for r in results
                if r.model_name == model and r.dataset_name == dataset
            ]
            if not match:
                continue
            r = match[0]
            print(
                f"{r.model_name:<20} "
                f"{r.num_samples:>8} "
                f"{r.mean_cer:>8.4f} "
                f"{r.mean_wer:>8.4f} "
                f"{r.exact_match_accuracy:>7.1%} "
                f"{r.mean_inference_time_s:>9.4f}s"
            )

    print(f"\n{'=' * 78}")
