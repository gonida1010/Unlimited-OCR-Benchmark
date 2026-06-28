# Unlimited-OCR-Benchmark

Baidu의 3B Unlimited-OCR 모델을 다양한 공개 데이터셋으로 평가하고, 다른 OCR 모델들과 성능 수치를 비교하기 위한 벤치마크 파이프라인입니다.

## Target Models

| Model | Type | GPU Required |
| ----- | ---- | :----------: |
| [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) | 3B VLM (BF16) | Yes |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | Detection + Recognition | No |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Detection + Recognition | No |
| WinRT OCR | Windows Built-in | No |

## Datasets

| Dataset | Type | Test Samples |
| ------- | ---- | :----------: |
| [IIIT5K](https://huggingface.co/datasets/MiXaiLL76/IIIT5K_OCR) | Scene text (word) | ~3,000 |
| [ICDAR2015](https://huggingface.co/datasets/MiXaiLL76/ICDAR2015_OCR) | Camera text (word) | ~2,077 |
| [IAM](https://huggingface.co/datasets/Teklia/IAM-line) | Handwriting (line) | ~2,915 |

## Metrics

- **CER** (Character Error Rate) — Levenshtein distance at character level
- **WER** (Word Error Rate) — Levenshtein distance at word level
- **Exact Match Accuracy** — fraction of perfect predictions
- **Inference Time** — per-sample wall-clock time (seconds)

## Quick Start

```bash
# Install
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run full benchmark
python -m benchmark

# Quick test (5 samples, 2 models)
python -m benchmark -m easyocr paddleocr -d iiit5k -n 5 -v
```

## Project Structure

```text
benchmark/
├── datasets/       # Dataset loaders + preprocessing
├── models/         # OCR model runners (common interface)
├── evaluation/     # Metrics, text normalization, evaluator
├── results/        # CSV/JSON export + console reporter
├── pipeline.py     # Main orchestrator
└── cli.py          # CLI argument parsing
```

자세한 환경 설정은 [ENVIRONMENT.md](ENVIRONMENT.md)를 참고하세요.
