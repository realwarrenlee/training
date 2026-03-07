# GUI Runbook (IDE + `index.html`)

## 1. Start the API from your IDE (Run button)

1. Open [api.py](/mnt/c/Users/warren/Downloads/training-main/api.py) in your IDE.
2. Make sure your interpreter is your project virtual env (`.venv`).
3. Set environment variables in your IDE Run Configuration:
   - `MODELS_DIR=.\\models`
   - `RUNS_DIR=.\\runs`
   - Optional for ClearML: `CLEARML_CONFIG_FILE=.\\clearml.conf`
4. Click the Run/Play button on `api.py`.

## 2. Open Simple GUI (`/`)

1. Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
2. Confirm Base URL is `http://127.0.0.1:8000`.
3. Use this page for all actions below.

Fallback: Swagger UI remains available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
API metadata JSON is available at [http://127.0.0.1:8000/info](http://127.0.0.1:8000/info).

## 3. Basic checks in GUI (`index.html`)

1. Click `Health` -> `Send Request`
2. Click `Status` -> `Send Request`

Expected: service is reachable and model status is shown.

## 4. Transcribe in GUI

1. Open `Transcribe`
2. Upload file: `notebooks/example.wav`
3. Click `Transcribe`

You should get JSON with `text`.

## 5. Evaluate WER in GUI

1. Open `Evaluate`
2. Upload file: `notebooks/example.wav`
3. Set `reference` to the real transcript (not placeholder text)
4. Click `Evaluate`

You should get `wer`, normalized fields, and `acceptable`.

## 6. Start training job in GUI

1. Open `Start Job`
2. Fill:
   - `Model ID`: `openai/whisper-small`
   - `Num Epochs`: `3`
   - `Batch Size`: `4`
   - `Learning Rate`: `1e-5`
3. Click `Start Training`
4. Copy `job_id` from response.

Monitor with:
- `List Jobs`
- `Job Status`

## 7. Run full pipeline in GUI

1. Open `Start Run`
2. Paste config JSON:
   ```json
   {
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
     "promote_on_success": false
   }
   ```
3. Click `Start Run`
4. Copy `run_id` from response.

Monitor with:
- `List Runs`
- `Get Run`

Check WER in pipeline output:
- `outputs.stt_whisper.summary.raw_wer`
- `outputs.stt_whisper.summary.normalized_wer`
- `outputs.stt_whisper.summary.num_samples`

## 8. Promote model in GUI

1. Open `Promote`
2. Enter `run_id`
3. Click `Promote`

Confirm with:
- `GET /models/active`

## 9. Backend Contract (must match GUI)

- `POST /transcribe`: multipart form with `file`
- `POST /evaluate`: multipart form with `file` + `reference`
- `POST /train`: JSON with `model_id`, `num_epochs`, `batch_size`, `learning_rate`
- `POST /models/active`: JSON with `model_id`
- `POST /pipeline/run/{run_id}/promote`: JSON body `{}` (or with `model_path`)
- `POST /train/dataset`: multipart form with `files` (supports multiple)

## 10. Optional ClearML UI check

If ClearML is enabled, open your task in the ClearML web app and verify:
- Scalars/logs
- Artifacts
- Task status
