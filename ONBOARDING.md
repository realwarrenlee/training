# NESTLAB Speech Onboarding - Guide

Welcome to NESTLAB! In this repository, you will learn the basics of a speech to text pipeline.

> [!IMPORTANT]
> To run this locally, please ensure that you have at least 16GB of system RAM and (ideally) a NVIDIA GPU with at least 4GB of VRAM.
>
> Otherwise, please consider running the notebooks on Google Colab instead https://colab.google.com. See the requirements section for more details.

## Slides

For slides, please refer to https://drive.google.com/drive/folders/1UR8zqxdgCa_cTjkOuuwmdGHk7GOdkf85. Please use them as reference materials throughout the onboarding.

## Outline

Learning objectives:

- Understanding what a speech-to-text (STT) pipeline is
- Performing inference with Whisper
- Fine-tuning Whisper on a dataset
- Evaluation and data analysis
- Serve on FastAPI (unified API)
- Deployment on Docker
- Integration with ClearML
- MLOps & ML Lifecycle

## Requirements

- Docker Desktop
- Python 3.11+ (for local development)
- NVIDIA GPU with 4GB+ VRAM (recommended)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install evaluate jiwer "accelerate>=0.26.0"
   ```

2. Run the API:
   ```bash
   MODELS_DIR=./models RUNS_DIR=./runs python3 api.py
   ```

3. Or with Docker:
   ```bash
   docker-compose up --build
   ```

---

## Project Structure

```
.
├── src/
│   ├── api.py              # Unified API (inference + pipeline + model management)
│   ├── train.py            # Training script implementation
│   ├── data.py             # Data path helpers
│   ├── inference.py        # Inference helper exports
│   ├── pipeline.py         # Pipeline helper exports
│   └── config.py           # Path configuration
├── api.py                  # Root API entrypoint
├── train.py                # Root training entrypoint
├── requirements.txt        # Combined dependencies
├── Dockerfile              # Container image
├── docker-compose.yml      # Docker Compose with GPU support
├── dataset/
│   ├── raw/                # Source parquet files
│   ├── processed/          # Optional processed outputs
│   └── dataset_scripts/    # Dataset utilities
├── models/                 # Model artifacts
├── checkpoints/            # Checkpoints (optional)
├── runs/                   # Pipeline manifests/reports/logs
├── scripts/                # Helper scripts
├── notebooks/              # Example files
└── test.ipynb              # API tests
```

---

## Unified API

The API provides both inference and training capabilities in a single service.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/status` | GET | Model loaded status |
| `/load` | POST | Load active/base model |
| `/unload` | POST | Unload model |
| `/transcribe` | POST | Transcribe audio |
| `/evaluate` | POST | Transcribe + WER |
| `/train` | POST | Start training job |
| `/train` | GET | List training jobs |
| `/train/{job_id}` | GET | Get job status + logs |
| `/train/{job_id}` | DELETE | Cancel job |
| `/train/dataset` | POST | Upload dataset |
| `/pipeline/run` | POST | Start full pipeline run |
| `/pipeline/run` | GET | List pipeline runs |
| `/pipeline/run/{run_id}` | GET | Get pipeline run details |
| `/pipeline/run/{run_id}/promote` | POST | Promote run model |
| `/models` | GET | List available models |
| `/models/active` | GET | Get active model |
| `/models/active` | POST | Switch active model |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `openai/whisper-base` | Whisper model |
| `MODELS_DIR` | `./models` | Model storage directory |
| `RUNS_DIR` | `./runs` | Pipeline artifacts directory |
| `CLEARML_ENABLED` | `false` | Enable ClearML logging |
| `CLEARML_CONFIG_FILE` | unset | Path to ClearML credentials config |
| `HF_HOME` | default cache path | Set to writable path if cache permission issues appear |

### Usage Examples

#### Transcription
```python
import requests

with open("audio.wav", "rb") as f:
    r = requests.post("http://localhost:8000/transcribe", files={"file": f})
print(r.json())  # {"text": "transcribed text"}
```

#### Start Training
```python
import requests

config = {
    "model_id": "openai/whisper-small",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5
}
r = requests.post("http://localhost:8000/train", json=config)
job_id = r.json()["job_id"]
```

#### Start Full Pipeline
```python
import requests

payload = {
    "dataset": {"source_type": "local_parquet", "train_path": "./dataset/raw/train.parquet", "test_path": "./dataset/raw/test.parquet"},
    "split": {"train": 0.9, "test": 0.1, "seed": 42},
    "audio": {"target_sr": 16000, "preprocess": False},
    "text": {"normalization": "atc_number_aware"},
    "model": {"base_model": "openai/whisper-small", "language": "English", "decode_language": "en"},
    "training": {"epochs": 3, "batch_size": 4, "learning_rate": 1e-5, "eval_steps": 500, "save_steps": 500},
    "reporting": {"clearml": True},
    "promote_on_success": False
}
run = requests.post("http://localhost:8000/pipeline/run", json=payload).json()
print(run["run_id"])
```

#### Check Training Status
```python
r = requests.get(f"http://localhost:8000/train/{job_id}")
print(r.json()["status"])  # "running", "completed", "failed"
```

#### Switch Model
```python
r = requests.post(
    "http://localhost:8000/models/active",
    json={"model_id": "run_20240301"}
)
```

---

## Training Pipeline

### Dataset

Uses local parquet dataset from `dataset/raw/`:
- `train.parquet` - Training data
- `test.parquet` - Test data

**Schema:**
- `audio` - Audio bytes (dict with 'bytes' key)
- `text` - Transcription text

### Running Training via API

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "openai/whisper-tiny",
    "num_epochs": 1,
    "batch_size": 1
  }'
```

### Training Script (src/train.py)

Used by the API to run training. Can also be run standalone:

```bash
python src/train.py --model-id openai/whisper-small --epochs 3 --batch-size 4
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | train or upload-dataset |
| `--model-id` | `openai/whisper-small` | Model to fine-tune |
| `--output-dir` | `whisper-finetuned` | Output directory |
| `--epochs` | 3 | Number of epochs |
| `--batch-size` | 4 | Batch size |
| `--learning-rate` | 1e-5 | Learning rate |
| `--eval-steps` | 500 | Eval interval (steps) |
| `--save-steps` | 500 | Checkpoint save interval (steps) |
| `--train-path` | None | Override training parquet path |
| `--test-path` | None | Override test parquet path |
| `--no-clearml` | False | Disable ClearML |

---

## ClearML Integration

ClearML is optional and controlled via `CLEARML_ENABLED` env var.

### Initialize and Enable ClearML
```bash
clearml-init
```

Then run with tracking:
```bash
CLEARML_ENABLED=true python api.py
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `CLEARML_PROJECT` | Project name (default: "NESTLAB Speech") |

---

## Technical Details

### Whisper Model Configuration

| Model | Parameters | VRAM (inference) | VRAM (training) |
|-------|------------|------------------|-----------------|
| tiny | 39M | ~1GB | ~2GB |
| base | 74M | ~1GB | ~3GB |
| small | 244M | ~2GB | ~5GB |
| medium | 769M | ~5GB | ~10GB |
| large | 1550M | ~10GB | ~20GB |

### Audio Processing

- Resampling: 16kHz (Whisper requirement)
- Format support: WAV, MP3, FLAC, OGG, M4A, MP4
- Mono/Stereo: Converted to mono

---

## Testing

Run the API tests in `test.ipynb`:

```bash
# Start the server first
MODELS_DIR=./models RUNS_DIR=./runs python3 api.py

# Then run cells in test.ipynb
```

---

## Troubleshooting

1. **Server won't start**
   - Check if port 8000 is available
   - Verify dependencies installed: `pip install -r requirements.txt`

2. **GPU not available**
   - Install NVIDIA Docker runtime
   - Check `nvidia-smi` works

3. **Training fails**
   - Check dataset exists in `dataset/raw/` directory
   - Verify parquet files have correct schema
   - Ensure `evaluate`, `jiwer`, and `accelerate` are installed

4. **Model switching fails**
   - Ensure fine-tuned model exists in `models/fine-tuned/`

5. **Cache/permission errors**
   - Set `HF_HOME` to a writable directory, for example: `HF_HOME=/tmp/hf_cache`

---

## References

- [Whisper Model](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ClearML](https://clear.ml/)
- [ATCO2 Dataset](https://www.atco2.org/)
