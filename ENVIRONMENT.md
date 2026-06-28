# Environment Setup Guide

## System Requirements

| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| **OS** | Windows 10 (WinRT OCR 필수) | Windows 11 |
| **Python** | 3.10+ | 3.12 |
| **GPU** | NVIDIA GPU 8GB VRAM (Unlimited-OCR) | NVIDIA GPU 12GB+ VRAM |
| **CUDA** | 12.0+ | 12.4+ |
| **RAM** | 16GB | 32GB |
| **Disk** | 20GB (모델 캐시 포함) | 30GB+ |

> **Note:** WinRT OCR는 Windows 전용입니다. Linux/macOS에서는 `--models paddleocr easyocr unlimited_ocr` 로 WinRT OCR를 제외하고 실행하세요.

---

## Model-Specific Requirements

### 1. Baidu Unlimited-OCR (3B BF16)

| 항목 | 값 |
|------|-----|
| HuggingFace ID | `baidu/Unlimited-OCR` |
| 모델 크기 | ~6GB (BF16) |
| GPU 필수 | Yes (CUDA) |
| 주요 의존성 | `torch`, `transformers>=4.40,<5.0`, `torchvision`, `pymupdf`, `einops` |

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.40,<5.0" pymupdf addict easydict einops
```

> `transformers>=5.0`에서는 `is_torch_fx_available` 제거로 호환 문제 발생 가능. `<5.0`으로 고정 권장.

### 2. PaddleOCR

| 항목 | 값 |
|------|-----|
| 패키지 | `paddleocr` |
| 모델 크기 | ~10-50MB (자동 다운로드) |
| GPU 필수 | No (CPU/GPU 둘 다 가능) |
| 주요 의존성 | `paddlepaddle` (자동 설치) |

```bash
pip install paddleocr
```

### 3. EasyOCR

| 항목 | 값 |
|------|-----|
| 패키지 | `easyocr` |
| 모델 크기 | ~100-200MB (첫 실행 시 다운로드) |
| GPU 필수 | No (GPU 가속 지원) |
| 주요 의존성 | `torch`, `opencv-python` |

```bash
pip install easyocr
```

### 4. WinRT OCR (Windows Built-in)

| 항목 | 값 |
|------|-----|
| 패키지 | `winocr` |
| 모델 크기 | 0 (OS 내장) |
| GPU 필수 | No (CPU only) |
| OS 제한 | **Windows 10+ 전용** |

```bash
pip install winocr
```

> Windows 설정 → 시간 및 언어 → 언어 및 지역에서 **English (United States)** 언어 팩이 설치되어 있어야 합니다.

---

## Quick Install (All-in-one)

```bash
# 1. 가상환경 생성
python -m venv .venv
.venv\Scripts\activate

# 2. PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. 전체 의존성
pip install -r requirements.txt
```

---

## Datasets

데이터셋은 HuggingFace `datasets` 라이브러리를 통해 자동 다운로드됩니다. 별도 수동 다운로드가 필요 없습니다.

| 데이터셋 | HuggingFace ID | 유형 | 테스트 규모 |
|----------|----------------|------|------------|
| IIIT5K | `MiXaiLL76/IIIT5K_OCR` | 장면 텍스트 (단어) | ~3,000 |
| ICDAR2015 | `MiXaiLL76/ICDAR2015_OCR` | 카메라 텍스트 (단어) | ~2,077 |
| IAM | `Teklia/IAM-line` | 손글씨 (라인) | ~2,915 |

> **IAM 데이터셋** 이용 시 HuggingFace에서 이용약관 동의가 필요할 수 있습니다.

---

## Usage

```bash
# 전체 벤치마크 실행
python -m benchmark

# 특정 모델/데이터셋만 실행
python -m benchmark -m easyocr paddleocr -d iiit5k -n 100

# 상세 로그
python -m benchmark -v

# Unlimited-OCR 해상도 조정
python -m benchmark -m unlimited_ocr --unlimited-ocr-image-size 1024
```

### CLI Options

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-m, --models` | 벤치마크할 모델 | 전체 (4개) |
| `-d, --datasets` | 평가 데이터셋 | 전체 (3개) |
| `-n, --max-samples` | 데이터셋당 최대 샘플 수 | 전체 |
| `-o, --output-dir` | 결과 저장 경로 | `outputs/results/` |
| `-v, --verbose` | 상세 로그 출력 | Off |
| `--unlimited-ocr-image-size` | Unlimited-OCR 이미지 크기 | 640 |
| `--unlimited-ocr-no-crop` | Unlimited-OCR crop 모드 비활성화 | False |

---

## Output

결과는 `outputs/results/` 디렉토리에 저장됩니다:

- `sample_results.csv` — 샘플별 상세 결과 (GT, 예측, CER, WER, 시간)
- `aggregated_results.csv` — 모델×데이터셋별 집계 결과
- `aggregated_results.json` — 동일 집계 결과 (JSON)
- 콘솔에 요약 테이블 출력
