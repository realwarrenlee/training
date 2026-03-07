# NESTLAB Speech API

FastAPI speech-to-text service using Whisper, with model training and model management.

## What Changed (Inference Goals)

The API now supports:
- Lazy model loading: model is **not** loaded at startup.
- On-demand inference: model loads when `/load` is called or when first inference runs.
- Evaluation mode inference: loaded model is set to `eval()` and inference uses `torch.inference_mode()`.

## Quick Start

### Local (Windows venv example)
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install evaluate jiwer "accelerate>=0.26.0"
$env:MODELS_DIR = "./models"
.\.venv\Scripts\python.exe .\api.py
```

### Local URLs
- API root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Use `127.0.0.1` or `localhost` in browser. `0.0.0.0` is only the bind address.

### Docker
```bash
docker-compose up --build
```

## Project Structure

```text
.
|-- src/
|   |-- api.py               # Unified API (inference + pipeline + model management)
|   |-- train.py             # Whisper fine-tuning script
|   |-- data.py              # Data path helpers
|   |-- inference.py         # Inference helper exports
|   |-- pipeline.py          # Pipeline helper exports
|   `-- config.py            # Path configuration
|-- api.py                   # Root API entrypoint
|-- train.py                 # Root training entrypoint (delegates to src/train.py)
|-- requirements.txt         # Dependencies
|-- Dockerfile               # Container image
|-- docker-compose.yml       # Compose stack
|-- dataset/
|   |-- raw/                 # Source parquet files
|   |-- processed/           # Optional processed datasets
|   `-- dataset_scripts/     # Dataset helpers
|-- models/                  # Model artifacts (default)
|-- checkpoints/             # Checkpoints (optional)
|-- runs/                    # Pipeline manifests/reports/logs
|-- inference/               # Inference artifacts (optional)
|-- scripts/                 # Utility scripts
|-- notebooks/               # Examples
`-- test.ipynb               # API tests
```

## API Endpoints

### Core / Inference
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Service health |
| `/status` | GET | Model loaded status |
| `/load` | POST | Load active/base model into memory |
| `/unload` | POST | Unload model and free memory |
| `/transcribe` | POST | Transcribe audio file |
| `/evaluate` | POST | Transcribe + WER against reference text |

### Training
| Endpoint | Method | Description |
|---|---|---|
| `/train` | POST | Start training job |
| `/train` | GET | List training jobs |
| `/train/{job_id}` | GET | Job status + logs |
| `/train/{job_id}` | DELETE | Cancel running job |
| `/train/dataset` | POST | Upload dataset |
| `/pipeline/run` | POST | Start full dataset->train->eval DAG |
| `/pipeline/run` | GET | List pipeline runs |
| `/pipeline/run/{run_id}` | GET | Pipeline run status + stage outputs |
| `/pipeline/run/{run_id}/promote` | POST | Promote run model as active |

### Model Management
| Endpoint | Method | Description |
|---|---|---|
| `/models` | GET | List available models |
| `/models/active` | GET | Current active model |
| `/models/active` | POST | Switch active model |

## Recommended Validation Flow

1. Start API.
2. `GET /status` -> should show not loaded after startup.
3. `POST /transcribe` with audio -> triggers on-demand model load + inference.
4. `GET /status` -> should show loaded.
5. `POST /evaluate` with `file` + `reference` -> returns transcription and WER.
6. `POST /unload` -> release model from memory.

## Example Usage

### Check model status
```bash
curl http://127.0.0.1:8000/status
```

### Load model explicitly (optional)
```bash
curl -X POST http://127.0.0.1:8000/load
```

### Transcribe
```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@audio.wav"
```

### Evaluate WER
```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -F "file=@audio.wav" \
  -F "reference=your reference transcript"
```

### Unload model
```bash
curl -X POST http://127.0.0.1:8000/unload
```

### Start training
```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "openai/whisper-small",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 1e-5
  }'
```

### Start full pipeline run
```bash
curl -X POST http://127.0.0.1:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {
      "source_type": "local_parquet",
      "train_path": "./dataset/raw/train.parquet",
      "test_path": "./dataset/raw/test.parquet"
    },
    "split": {"train": 0.9, "test": 0.1, "seed": 42},
    "audio": {"target_sr": 16000, "preprocess": false},
    "text": {"normalization": "atc_number_aware"},
    "model": {"base_model": "openai/whisper-small", "language": "English", "decode_language": "en"},
    "training": {"epochs": 3, "batch_size": 4, "learning_rate": 1e-5, "eval_steps": 500, "save_steps": 500},
    "reporting": {"clearml": true},
    "promote_on_success": true
  }'
```

### Check pipeline status
```bash
curl http://127.0.0.1:8000/pipeline/run
curl http://127.0.0.1:8000/pipeline/run/<run_id>
```

### Promote a run model
```bash
curl -X POST http://127.0.0.1:8000/pipeline/run/<run_id>/promote \
  -H "Content-Type: application/json" \
  -d '{}'
```

## ClearML Setup

Initialize ClearML once using the CLI prompt:

```bash
clearml-init
```

If you keep a repo-local config file:

```bash
CLEARML_CONFIG_FILE=./clearml.conf python3 api.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `openai/whisper-base` | Base model ID fallback |
| `MODELS_DIR` | `./models` | Model storage root |
| `RUNS_DIR` | `./runs` | Pipeline run artifacts root |
| `CLEARML_ENABLED` | `false` | Enable ClearML logging |
| `CLEARML_CONFIG_FILE` | unset | Path to ClearML config file |
| `HF_HOME` | default cache path | Hugging Face cache root (set writable path if needed) |

## Requirements

- Python 3.11+
- NVIDIA GPU recommended for faster inference/training
- Docker + NVIDIA runtime (if containerized GPU usage is needed)
