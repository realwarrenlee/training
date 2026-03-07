import os
import io
import math
import uuid
import re
import sys
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import librosa
import scipy.signal as spsig
import soundfile as sf
import pandas as pd
import torch
import evaluate as hf_evaluate
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from transformers import WhisperProcessor, WhisperForConditionalGeneration
try:
    from clearml import Dataset as ClearMLDataset
except Exception:
    ClearMLDataset = None

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-small")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))
RUNS_DIR = Path(os.getenv("RUNS_DIR", str(PROJECT_ROOT / "runs")))
INDEX_HTML = PROJECT_ROOT / "index.html"
CLEARML_ENABLED = os.getenv("CLEARML_ENABLED", "false").lower() == "true"
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "English")
TASK = os.getenv("WHISPER_TASK", "transcribe")
DECODE_LANGUAGE = os.getenv("WHISPER_DECODE_LANGUAGE", "en")
ENABLE_AUDIO_PREPROCESS = os.getenv("ENABLE_AUDIO_PREPROCESS", "false").lower() == "true"

WHISPER_SAMPLE_RATE = 16000
WHISPER_CHUNK_LENGTH = 30
WHISPER_N_SAMPLES = WHISPER_SAMPLE_RATE * WHISPER_CHUNK_LENGTH
MAX_FILE_SIZE = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}
ALLOWED_CONTENT_TYPES = {"audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/mp4"}


def preprocess_audio(audio: np.ndarray, sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    audio = audio.copy()
    audio = librosa.util.normalize(audio) * 0.9
    nyq = sample_rate / 2
    sos = spsig.butter(4, 80 / nyq, btype="high", output="sos")
    audio = spsig.sosfiltfilt(sos, audio)
    noise_profile = audio[: int(sample_rate * 0.3)]
    noise_mean = np.mean(noise_profile**2)
    noise_threshold = noise_mean * 1.5
    audio[audio**2 < noise_threshold] = 0
    return audio


def _normalize_basic_text(text: str) -> list[str]:
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _consume_number_words(tokens: list[str], start: int) -> tuple[int | None, int]:
    total = 0
    current = 0
    i = start
    saw = False

    while i < len(tokens):
        t = tokens[i]
        if t in _NUM_WORDS:
            current += _NUM_WORDS[t]
            saw = True
            i += 1
            continue
        if t == "hundred" and saw:
            current *= 100
            i += 1
            continue
        if t == "thousand" and saw:
            total += current * 1000
            current = 0
            i += 1
            continue
        break

    if not saw:
        return None, start
    return total + current, i


def _number_to_digit_tokens(num: int) -> list[str]:
    return list(str(num))


def normalize_text_for_wer(text: str) -> str:
    tokens = _normalize_basic_text(text)
    out: list[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok.isdigit():
            out.extend(list(tok))
            i += 1
            continue

        if tok in {"oh", "o"}:
            out.append("0")
            i += 1
            continue

        num, j = _consume_number_words(tokens, i)
        if num is not None and j > i:
            out.extend(_number_to_digit_tokens(num))
            i = j
            continue

        out.append(tok)
        i += 1

    return " ".join(out)

training_jobs: dict = {}
pipeline_runs: dict = {}
active_model_path: Optional[Path] = None
model_lock = threading.Lock()

PIPELINE_RUNS_DIR = RUNS_DIR


model_status = "not_loaded"
processor: WhisperProcessor | None = None
model: WhisperForConditionalGeneration | None = None
wer_metric = None
device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_status

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "base").mkdir(exist_ok=True)
    (MODELS_DIR / "fine-tuned").mkdir(exist_ok=True)
    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep startup fast: load model only when explicitly requested or first inference.
    model_status = "not_loaded"
    yield


def _get_active_model_path() -> Path:
    active_link = MODELS_DIR / "active"
    if active_link.exists() or active_link.is_symlink():
        try:
            resolved = active_link.resolve()
            if resolved.exists():
                return resolved
        except OSError:
            pass

    active_marker = MODELS_DIR / "active_model.txt"
    if active_marker.exists():
        marker_value = active_marker.read_text(encoding="utf-8").strip()
        if marker_value:
            marker_path = Path(marker_value)
            if marker_path.exists():
                return marker_path

    return MODELS_DIR / "base" / MODEL_ID.replace("/", "_")


def _set_active_model_path(model_path: Path) -> None:
    active_link = MODELS_DIR / "active"
    active_marker = MODELS_DIR / "active_model.txt"

    try:
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()
        active_link.symlink_to(model_path, target_is_directory=True)
    except OSError as e:
        # Symlink creation is commonly restricted on Windows without Developer Mode/admin.
        print(f"Active symlink unavailable, using marker file fallback: {e}")

    active_marker.write_text(str(model_path), encoding="utf-8")


def _resolve_model_source() -> tuple[Path | None, str]:
    model_path = _get_active_model_path()
    if model_path.exists():
        return model_path, str(model_path)
    return None, MODEL_ID


def _load_model_if_needed() -> None:
    global model_status, processor, model, device, active_model_path

    if model is not None and processor is not None:
        return

    with model_lock:
        if model is not None and processor is not None:
            return

        model_status = "loading"
        try:
            model_path, source = _resolve_model_source()
            active_model_path = model_path

            processor = WhisperProcessor.from_pretrained(source, language=LANGUAGE, task=TASK)
            loaded_model = WhisperForConditionalGeneration.from_pretrained(source)
            loaded_model.generation_config.language = LANGUAGE
            loaded_model.generation_config.task = TASK
            loaded_model.generation_config.forced_decoder_ids = None
            loaded_model.eval()

            if torch.cuda.is_available():
                loaded_model = loaded_model.to("cuda")
                device = "cuda"
            else:
                device = "cpu"

            model = loaded_model
            model_status = "loaded"
            print(f"Model loaded from {source} on {device}")
        except Exception as e:
            model_status = "error"
            raise HTTPException(500, f"Failed to load model: {e}")


def _unload_model() -> None:
    global model_status, processor, model, active_model_path
    with model_lock:
        processor = None
        model = None
        active_model_path = None
        model_status = "not_loaded"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def chunk_audio(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) == 0:
        return []
    if len(audio) <= WHISPER_N_SAMPLES:
        return [audio]

    n_chunks = max(1, math.ceil(len(audio) / WHISPER_N_SAMPLES))
    chunks = []
    for i in range(n_chunks):
        chunk = audio[i * WHISPER_N_SAMPLES : (i + 1) * WHISPER_N_SAMPLES]
        chunks.append(chunk)
    return chunks


def _transcribe_bytes(content: bytes) -> str:
    _load_model_if_needed()

    audio, _ = librosa.load(
        io.BytesIO(content),
        sr=WHISPER_SAMPLE_RATE,
        mono=True,
        res_type="kaiser_fast",
    )

    if ENABLE_AUDIO_PREPROCESS:
        audio = preprocess_audio(audio)

    if len(audio) < 1600:
        raise HTTPException(400, "Audio too short")
    if np.max(np.abs(audio)) < 1e-4:
        raise HTTPException(400, "Audio appears to be silent")

    chunks = chunk_audio(audio)
    transcription_parts = []

    for chunk in chunks:
        inputs = processor.feature_extractor(
            chunk,
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = inputs["input_features"]
        if device == "cuda":
            input_features = input_features.to("cuda")

        model.eval()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=440,
                language=DECODE_LANGUAGE,
                task=TASK,
                temperature=0.0,
            )

        part = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        transcription_parts.append(part.strip())

    return " ".join(filter(None, transcription_parts))


class DatasetConfig(BaseModel):
    source_type: str = "local_parquet"
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    dataset_id: Optional[str] = None


class SplitConfig(BaseModel):
    train: float = 0.9
    test: float = 0.1
    seed: int = 42
    stratify_by: Optional[str] = None


class AudioPrepConfig(BaseModel):
    target_sr: int = 16000
    preprocess: bool = False


class TextPrepConfig(BaseModel):
    normalization: str = "basic"


class ModelRunConfig(BaseModel):
    base_model: str = "openai/whisper-small"
    language: str = "English"
    decode_language: str = "en"


class TrainingRunConfig(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5
    eval_steps: int = 500
    save_steps: int = 500


class ReportingConfig(BaseModel):
    clearml: bool = True


class PipelineRunConfig(BaseModel):
    dataset: DatasetConfig
    split: SplitConfig
    audio: AudioPrepConfig
    text: TextPrepConfig
    model: ModelRunConfig
    training: TrainingRunConfig
    reporting: ReportingConfig
    promote_on_success: bool = False


class PromoteRequest(BaseModel):
    model_path: Optional[str] = None


def _now_iso() -> str:
    return datetime.now().isoformat()


def _resolve_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=2)


def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _get_audio_bytes(audio_value: Any) -> bytes:
    if isinstance(audio_value, dict):
        content = audio_value.get("bytes")
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
    if isinstance(audio_value, (bytes, bytearray)):
        return bytes(audio_value)
    raise ValueError("Invalid audio column format, expected bytes or {'bytes': bytes}")


def _persist_pipeline_run(run_id: str) -> None:
    run_data = pipeline_runs.get(run_id)
    if not run_data:
        return
    status_path = Path(run_data["run_dir"]) / "run_status.json"
    _write_json(status_path, run_data)


def _record_stage(
    run_id: str,
    stage_name: str,
    status: str,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
    output: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    run = pipeline_runs[run_id]
    run["stages"].append(
        {
            "name": stage_name,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "output": output,
            "error": error,
        }
    )
    _persist_pipeline_run(run_id)


def _stage_select_dataset_source(dataset_cfg: DatasetConfig) -> dict[str, Any]:
    source_type = dataset_cfg.source_type
    resolved_files: list[str] = []

    if source_type == "local_parquet":
        train_path = _resolve_path(dataset_cfg.train_path) if dataset_cfg.train_path else None
        test_path = _resolve_path(dataset_cfg.test_path) if dataset_cfg.test_path else None
        if not train_path and not test_path:
            raise ValueError("local_parquet requires train_path or test_path")
        if train_path and not train_path.exists():
            raise ValueError(f"Missing train parquet: {train_path}")
        if test_path and not test_path.exists():
            raise ValueError(f"Missing test parquet: {test_path}")
        if train_path:
            resolved_files.append(str(train_path))
        if test_path:
            resolved_files.append(str(test_path))
        dataset_root = str((train_path or test_path).parent)
        return {
            "dataset_root": dataset_root,
            "resolved_files": resolved_files,
            "meta": {"source_type": source_type},
        }

    if source_type == "upload_bundle":
        if not dataset_cfg.train_path:
            raise ValueError("upload_bundle expects train_path to point to bundle root")
        bundle_root = _resolve_path(dataset_cfg.train_path)
        if not bundle_root.exists() or not bundle_root.is_dir():
            raise ValueError(f"Bundle directory not found: {bundle_root}")
        candidates = [
            bundle_root / "train.parquet",
            bundle_root / "test.parquet",
            bundle_root / "full.parquet",
        ]
        found = [str(p) for p in candidates if p.exists()]
        if not found:
            raise ValueError("Bundle must contain train.parquet/test.parquet or full.parquet")
        return {
            "dataset_root": str(bundle_root),
            "resolved_files": found,
            "meta": {"source_type": source_type},
        }

    if source_type == "clearml_dataset":
        if ClearMLDataset is None:
            raise ValueError("clearml package is not installed")
        if not dataset_cfg.dataset_id:
            raise ValueError("clearml_dataset source requires dataset_id")
        dataset = ClearMLDataset.get(dataset_id=dataset_cfg.dataset_id)
        local_copy = Path(dataset.get_local_copy())
        if not local_copy.exists():
            raise ValueError(f"ClearML dataset local copy missing: {local_copy}")
        found = [str(p) for p in local_copy.glob("*.parquet")]
        if not found:
            raise ValueError("ClearML dataset contains no parquet files")
        return {
            "dataset_root": str(local_copy),
            "resolved_files": sorted(found),
            "meta": {"source_type": source_type, "dataset_id": dataset_cfg.dataset_id},
        }

    raise ValueError(f"Unsupported source_type: {source_type}")


def _stage_select_dataset_validate(stage1_output: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    parquet_files = [Path(p) for p in stage1_output["resolved_files"] if str(p).endswith(".parquet")]
    if not parquet_files:
        raise ValueError("No parquet files resolved from source")

    rows_total = 0
    rows_invalid = 0
    duration_sec_estimate = 0.0
    clean_records: list[dict[str, Any]] = []

    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        if "audio" not in df.columns or "text" not in df.columns:
            raise ValueError(f"Missing required columns in {parquet_path}; required: audio, text")
        rows_total += len(df)

        for _, row in df.iterrows():
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                rows_invalid += 1
                continue

            try:
                audio_bytes = _get_audio_bytes(row["audio"])
                audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, res_type="kaiser_fast")
                if len(audio) == 0:
                    rows_invalid += 1
                    continue
                duration_sec_estimate += float(len(audio)) / float(sr or WHISPER_SAMPLE_RATE)
            except Exception:
                rows_invalid += 1
                continue

            clean_records.append({"audio": {"bytes": audio_bytes}, "text": text.strip()})

    if not clean_records:
        raise ValueError("No valid rows remain after validation")

    clean_df = pd.DataFrame(clean_records)
    clean_dataset_path = run_dir / "clean.parquet"
    clean_df.to_parquet(clean_dataset_path, index=False)

    return {
        "validated": True,
        "stats": {
            "rows_total": rows_total,
            "rows_invalid": rows_invalid,
            "duration_sec_estimate": round(duration_sec_estimate, 2),
        },
        "clean_dataset_path": str(clean_dataset_path),
    }


def _stage_train_test_split(
    dataset_path: str,
    split_cfg: SplitConfig,
    run_dir: Path,
    full_config: PipelineRunConfig,
) -> dict[str, Any]:
    if split_cfg.train <= 0 or split_cfg.test <= 0:
        raise ValueError("Split ratios must be positive")
    if abs((split_cfg.train + split_cfg.test) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    df = pd.read_parquet(dataset_path).reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to split train/test")

    if split_cfg.stratify_by:
        if split_cfg.stratify_by not in df.columns:
            raise ValueError(f"Stratify column not found: {split_cfg.stratify_by}")
        train_idx: list[int] = []
        for _, group in df.groupby(split_cfg.stratify_by):
            sampled = group.sample(frac=split_cfg.train, random_state=split_cfg.seed)
            train_idx.extend(sampled.index.tolist())
        train_df = df.loc[sorted(set(train_idx))].copy()
    else:
        train_df = df.sample(frac=split_cfg.train, random_state=split_cfg.seed).copy()

    test_df = df.drop(index=train_df.index).copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Split produced an empty train or test set")

    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_path = splits_dir / "train.parquet"
    test_path = splits_dir / "test.parquet"
    manifest_path = splits_dir / "manifest.json"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    manifest = {
        "seed": split_cfg.seed,
        "split": {"train": split_cfg.train, "test": split_cfg.test},
        "rows": {"train": len(train_df), "test": len(test_df), "total": len(df)},
        "commit_hash": _get_commit_hash(),
        "params": full_config.model_dump(),
        "created_at": _now_iso(),
    }
    _write_json(manifest_path, manifest)

    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "manifest_path": str(manifest_path),
    }


def _normalize_for_mode(text: str, mode: str) -> str:
    if mode == "basic":
        return " ".join(_normalize_basic_text(text))
    if mode == "atc_number_aware":
        return normalize_text_for_wer(text)
    raise ValueError(f"Unsupported text normalization mode: {mode}")


def _prepare_audio_bytes(audio_bytes: bytes, target_sr: int, preprocess: bool) -> bytes:
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=target_sr, mono=True, res_type="kaiser_fast")
    if preprocess:
        audio = preprocess_audio(audio, sample_rate=target_sr)
    out = io.BytesIO()
    sf.write(out, audio, target_sr, format="WAV")
    return out.getvalue()


def _build_prepared_dataset(input_path: Path, output_path: Path, audio_cfg: AudioPrepConfig, text_cfg: TextPrepConfig) -> None:
    df = pd.read_parquet(input_path)
    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        audio_bytes = _get_audio_bytes(row["audio"])
        prepared_audio = _prepare_audio_bytes(audio_bytes, audio_cfg.target_sr, audio_cfg.preprocess)
        text = row["text"] if isinstance(row["text"], str) else ""
        normalized_text = _normalize_for_mode(text, text_cfg.normalization)
        out_rows.append({"audio": {"bytes": prepared_audio}, "text": normalized_text})
    pd.DataFrame(out_rows).to_parquet(output_path, index=False)


def _stage_select_dataset(
    train_path: str,
    test_path: str,
    audio_cfg: AudioPrepConfig,
    text_cfg: TextPrepConfig,
    run_dir: Path,
) -> dict[str, Any]:
    prepared_dir = run_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_train_path = prepared_dir / "train.parquet"
    prepared_test_path = prepared_dir / "test.parquet"

    _build_prepared_dataset(Path(train_path), prepared_train_path, audio_cfg, text_cfg)
    _build_prepared_dataset(Path(test_path), prepared_test_path, audio_cfg, text_cfg)

    return {
        "prepared_train_ref": str(prepared_train_path),
        "prepared_test_ref": str(prepared_test_path),
        "prep_meta": {"target_sr": audio_cfg.target_sr},
    }


def _extract_training_metrics(output_dir: Path) -> dict[str, Any]:
    metrics = {"eval_wer_best": None, "eval_loss_best": None}
    trainer_state_path = output_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        return metrics

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    if isinstance(state.get("best_metric"), (int, float)):
        metrics["eval_wer_best"] = round(float(state["best_metric"]), 4)

    best_eval_loss = None
    for item in state.get("log_history", []):
        if "eval_loss" in item:
            loss = float(item["eval_loss"])
            if best_eval_loss is None or loss < best_eval_loss:
                best_eval_loss = loss
    if best_eval_loss is not None:
        metrics["eval_loss_best"] = round(best_eval_loss, 4)

    return metrics


def _stage_train_whisper(
    model_cfg: ModelRunConfig,
    training_cfg: TrainingRunConfig,
    reporting_cfg: ReportingConfig,
    prepared_train_path: str,
    prepared_test_path: str,
) -> dict[str, Any]:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / "fine-tuned" / f"run_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "training.log"
    train_job_id = str(uuid.uuid4())

    python_bin = os.getenv("PYTHON_BIN") or sys.executable or "python3"
    cmd = [
        python_bin,
        str(Path(__file__).with_name("train.py")),
        "--mode",
        "train",
        "--model-id",
        model_cfg.base_model,
        "--epochs",
        str(training_cfg.epochs),
        "--batch-size",
        str(training_cfg.batch_size),
        "--learning-rate",
        str(training_cfg.learning_rate),
        "--eval-steps",
        str(training_cfg.eval_steps),
        "--save-steps",
        str(training_cfg.save_steps),
        "--train-path",
        prepared_train_path,
        "--test-path",
        prepared_test_path,
        "--output-dir",
        str(output_dir),
    ]
    if not reporting_cfg.clearml:
        cmd.append("--no-clearml")

    env = os.environ.copy()
    env["WHISPER_LANGUAGE"] = model_cfg.language
    env["CLEARML_ENABLED"] = "true" if reporting_cfg.clearml else "false"

    with open(logs_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"train_whisper failed with return code {proc.returncode}")

    metrics = _extract_training_metrics(output_dir)
    return {
        "best_model_path": str(output_dir),
        "metrics": metrics,
        "logs_path": str(logs_path),
        "run_id": train_job_id,
    }


def _transcribe_bytes_with_model(
    content: bytes,
    local_processor: WhisperProcessor,
    local_model: WhisperForConditionalGeneration,
    local_device: str,
    decode_language: str,
    decode_task: str,
) -> str:
    audio, _ = librosa.load(
        io.BytesIO(content),
        sr=WHISPER_SAMPLE_RATE,
        mono=True,
        res_type="kaiser_fast",
    )
    chunks = chunk_audio(audio)
    parts: list[str] = []

    for chunk in chunks:
        inputs = local_processor.feature_extractor(chunk, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
        input_features = inputs["input_features"]
        if local_device == "cuda":
            input_features = input_features.to("cuda")
        with torch.inference_mode():
            generated_ids = local_model.generate(
                input_features=input_features,
                max_new_tokens=440,
                language=decode_language,
                task=decode_task,
                temperature=0.0,
            )
        text = local_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        parts.append(text.strip())

    return " ".join(filter(None, parts))


def _stage_stt_whisper(
    model_path: str,
    test_path: str,
    model_cfg: ModelRunConfig,
    run_dir: Path,
) -> dict[str, Any]:
    local_processor = WhisperProcessor.from_pretrained(model_path, language=model_cfg.language, task=TASK)
    local_model = WhisperForConditionalGeneration.from_pretrained(model_path)
    local_model.eval()
    local_device = "cuda" if torch.cuda.is_available() else "cpu"
    if local_device == "cuda":
        local_model = local_model.to("cuda")

    df = pd.read_parquet(test_path)
    wer_metric_local = hf_evaluate.load("wer")
    raw_refs: list[str] = []
    raw_hyps: list[str] = []
    norm_refs: list[str] = []
    norm_hyps: list[str] = []
    examples: list[dict[str, str]] = []

    for _, row in df.iterrows():
        reference = row["text"] if isinstance(row["text"], str) else ""
        if not reference.strip():
            continue
        audio_bytes = _get_audio_bytes(row["audio"])
        hypothesis = _transcribe_bytes_with_model(
            audio_bytes,
            local_processor,
            local_model,
            local_device,
            model_cfg.decode_language,
            TASK,
        )
        raw_refs.append(reference)
        raw_hyps.append(hypothesis)

        normalized_reference = normalize_text_for_wer(reference)
        normalized_hypothesis = normalize_text_for_wer(hypothesis)
        norm_refs.append(normalized_reference)
        norm_hyps.append(normalized_hypothesis)

        if len(examples) < 5:
            examples.append(
                {
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "normalized_reference": normalized_reference,
                    "normalized_hypothesis": normalized_hypothesis,
                }
            )

    if not raw_refs:
        raise ValueError("No valid rows available for stt_whisper scoring")

    raw_wer = 100 * wer_metric_local.compute(predictions=raw_hyps, references=raw_refs)
    norm_wer = 100 * wer_metric_local.compute(predictions=norm_hyps, references=norm_refs)

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_payload = {
        "summary": {
            "raw_wer": round(raw_wer, 2),
            "normalized_wer": round(norm_wer, 2),
            "num_samples": len(raw_refs),
        },
        "examples": examples,
    }
    _write_json(report_path, report_payload)

    return {
        "summary": report_payload["summary"],
        "examples": examples,
        "artifact_report": str(report_path),
    }


def _stage_promote_model(model_path: str) -> dict[str, Any]:
    target = Path(model_path)
    if not target.exists():
        raise ValueError(f"Model path does not exist: {target}")
    _set_active_model_path(target)
    _unload_model()
    return {
        "active_model_path": str(MODELS_DIR / "active_model.txt"),
        "status": "promoted",
    }


def _execute_stage(run_id: str, stage_name: str, fn, *args, **kwargs) -> dict[str, Any]:
    started_at = _now_iso()
    try:
        output = fn(*args, **kwargs)
        _record_stage(
            run_id,
            stage_name,
            "completed",
            started_at=started_at,
            ended_at=_now_iso(),
            output=output,
        )
        pipeline_runs[run_id]["outputs"][stage_name] = output
        _persist_pipeline_run(run_id)
        return output
    except Exception as e:
        _record_stage(
            run_id,
            stage_name,
            "failed",
            started_at=started_at,
            ended_at=_now_iso(),
            error=str(e),
        )
        raise


def run_pipeline_job(run_id: str, config: PipelineRunConfig) -> None:
    run = pipeline_runs[run_id]
    try:
        s1 = _execute_stage(run_id, "select_dataset_source", _stage_select_dataset_source, config.dataset)
        s2 = _execute_stage(run_id, "select_dataset_validate", _stage_select_dataset_validate, s1, Path(run["run_dir"]))
        s3 = _execute_stage(
            run_id,
            "train_test_split",
            _stage_train_test_split,
            s2["clean_dataset_path"],
            config.split,
            Path(run["run_dir"]),
            config,
        )
        s4 = _execute_stage(
            run_id,
            "select_dataset",
            _stage_select_dataset,
            s3["train_path"],
            s3["test_path"],
            config.audio,
            config.text,
            Path(run["run_dir"]),
        )
        s5 = _execute_stage(
            run_id,
            "train_whisper",
            _stage_train_whisper,
            config.model,
            config.training,
            config.reporting,
            s4["prepared_train_ref"],
            s4["prepared_test_ref"],
        )
        _execute_stage(
            run_id,
            "stt_whisper",
            _stage_stt_whisper,
            s5["best_model_path"],
            s3["test_path"],
            config.model,
            Path(run["run_dir"]),
        )
        if config.promote_on_success:
            _execute_stage(run_id, "promote_model", _stage_promote_model, s5["best_model_path"])
        run["status"] = "completed"
    except Exception as e:
        run["status"] = "failed"
        run["error"] = str(e)
    run["finished_at"] = _now_iso()
    _persist_pipeline_run(run_id)


app = FastAPI(
    title="Whisper STT API",
    description="Speech-to-Text API with Training Support",
    version="1.2.0",
    lifespan=lifespan,
)


class TrainConfig(BaseModel):
    model_id: str = "openai/whisper-small"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-5


class SwitchModelRequest(BaseModel):
    model_id: str


@app.get("/")
async def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {
        "title": app.title,
        "description": app.description,
        "version": app.version,
        "status": "operational",
        "model_status": model_status,
        "device": device,
        "endpoints": {
            "info": "/info (GET)",
            "status": "/status (GET)",
            "load": "/load (POST)",
            "unload": "/unload (POST)",
            "transcribe": "/transcribe (POST)",
            "evaluate": "/evaluate (POST)",
            "train": "/train (POST)",
            "train_status": "/train/{job_id} (GET)",
            "pipeline_run": "/pipeline/run (POST)",
            "pipeline_status": "/pipeline/run/{run_id} (GET)",
            "pipeline_promote": "/pipeline/run/{run_id}/promote (POST)",
            "models": "/models (GET)",
            "switch_model": "/models/active (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)",
        },
    }


@app.get("/info")
async def info():
    return {
        "title": app.title,
        "description": app.description,
        "version": app.version,
        "status": "operational",
        "model_status": model_status,
        "device": device,
        "endpoints": {
            "ui": "/ (GET)",
            "docs": "/docs (GET)",
            "status": "/status (GET)",
            "load": "/load (POST)",
            "unload": "/unload (POST)",
            "transcribe": "/transcribe (POST)",
            "evaluate": "/evaluate (POST)",
            "train": "/train (POST)",
            "train_status": "/train/{job_id} (GET)",
            "pipeline_run": "/pipeline/run (POST)",
            "pipeline_status": "/pipeline/run/{run_id} (GET)",
            "pipeline_promote": "/pipeline/run/{run_id}/promote (POST)",
            "models": "/models (GET)",
            "switch_model": "/models/active (POST)",
            "health": "/health (GET)",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_status == "loaded" else "initializing",
        "device": device,
        "model_status": model_status,
        "is_training": any(j["status"] == "running" for j in training_jobs.values()),
    }


@app.get("/status")
async def status():
    return {
        "loaded": model is not None and processor is not None,
        "model_status": model_status,
        "device": device,
        "active_model_path": str(active_model_path) if active_model_path else None,
    }


@app.post("/load")
async def load_model():
    _load_model_if_needed()
    return {
        "status": "loaded",
        "device": device,
        "source": str(active_model_path) if active_model_path else MODEL_ID,
    }


@app.post("/unload")
async def unload_model():
    _unload_model()
    return {"status": "unloaded"}


@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    if any(j["status"] == "running" for j in training_jobs.values()):
        raise HTTPException(503, "Training in progress, inference temporarily unavailable")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid audio format")

    content = await file.read()

    try:
        text = await run_in_threadpool(_transcribe_bytes, content)
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")


@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...), reference: str = Form(...)):
    global wer_metric

    if any(j["status"] == "running" for j in training_jobs.values()):
        raise HTTPException(503, "Training in progress, inference temporarily unavailable")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Invalid audio format")

    content = await file.read()

    try:
        hypothesis = await run_in_threadpool(_transcribe_bytes, content)

        if wer_metric is None:
            wer_metric = hf_evaluate.load("wer")

        hypothesis_normalized = normalize_text_for_wer(hypothesis)
        reference_normalized = normalize_text_for_wer(reference)
        wer = 100 * wer_metric.compute(
            predictions=[hypothesis_normalized],
            references=[reference_normalized],
        )
        return {
            "transcription": hypothesis,
            "reference": reference,
            "normalized_transcription": hypothesis_normalized,
            "normalized_reference": reference_normalized,
            "wer": round(wer, 2),
            "acceptable": wer < 30.0,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Evaluation error: {str(e)}")


def run_training_subprocess(job_id: str, config: TrainConfig):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = MODELS_DIR / "fine-tuned" / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_path = output_dir / "training.log"

    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "config": config.model_dump(),
        "output_dir": str(output_dir),
        "logs_path": str(logs_path),
        "started_at": datetime.now().isoformat(),
        "pid": None,
    }

    python_bin = os.getenv("PYTHON_BIN") or sys.executable or "python"
    cmd = [
        python_bin,
        str(Path(__file__).with_name("train.py")),
        "--mode",
        "train",
        "--model-id",
        config.model_id,
        "--epochs",
        str(config.num_epochs),
        "--batch-size",
        str(config.batch_size),
        "--learning-rate",
        str(config.learning_rate),
        "--output-dir",
        str(output_dir),
    ]

    if not CLEARML_ENABLED:
        cmd.append("--no-clearml")

    env = os.environ.copy()
    env["CLEARML_ENABLED"] = "true" if CLEARML_ENABLED else "false"

    try:
        cwd = os.getcwd()
        with open(logs_path, "w") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=cwd,
            )
            training_jobs[job_id]["pid"] = proc.pid

            proc.wait()

            if proc.returncode == 0:
                training_jobs[job_id]["status"] = "completed"
                training_jobs[job_id]["returncode"] = 0
                try:
                    _set_active_model_path(output_dir)
                except Exception as e:
                    training_jobs[job_id]["activation_warning"] = str(e)
            else:
                training_jobs[job_id]["status"] = "failed"
                training_jobs[job_id]["returncode"] = proc.returncode
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)

    training_jobs[job_id]["finished_at"] = datetime.now().isoformat()


@app.post("/train")
async def start_training(config: TrainConfig, background_tasks: BackgroundTasks):
    if any(j["status"] == "running" for j in training_jobs.values()):
        raise HTTPException(409, "Training already in progress")

    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_training_subprocess, job_id, config)

    return {
        "job_id": job_id,
        "status": "started",
        "message": "Training job started",
    }


@app.get("/train/{job_id}")
async def get_training_status(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(404, "Job not found")

    job = training_jobs[job_id].copy()

    logs_path = job.get("logs_path")
    if logs_path and Path(logs_path).exists():
        with open(logs_path, "r") as f:
            lines = f.readlines()
            job["logs"] = "".join(lines[-100:]) if len(lines) > 100 else "".join(lines)
            job["log_lines"] = len(lines)

    return job


@app.get("/train")
async def list_training_jobs():
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
            }
            for job_id, job in training_jobs.items()
        ]
    }


@app.delete("/train/{job_id}")
async def cancel_training(job_id: str):
    if job_id not in training_jobs:
        raise HTTPException(404, "Job not found")

    job = training_jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(400, f"Cannot cancel job with status: {job['status']}")

    pid = job.get("pid")
    if pid:
        try:
            import signal

            os.kill(pid, signal.SIGTERM)
            job["status"] = "cancelled"
        except Exception as e:
            raise HTTPException(500, f"Failed to cancel job: {e}")

    return {"message": "Job cancelled"}


@app.post("/pipeline/run")
async def start_pipeline_run(config: PipelineRunConfig, background_tasks: BackgroundTasks):
    if any(j["status"] == "running" for j in training_jobs.values()):
        raise HTTPException(409, "A training job is already running")

    run_id = str(uuid.uuid4())
    run_dir = PIPELINE_RUNS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pipeline_runs[run_id] = {
        "run_id": run_id,
        "status": "running",
        "started_at": _now_iso(),
        "finished_at": None,
        "run_dir": str(run_dir),
        "config": config.model_dump(),
        "stages": [],
        "outputs": {},
        "error": None,
    }
    _persist_pipeline_run(run_id)
    background_tasks.add_task(run_pipeline_job, run_id, config)

    return {"run_id": run_id, "status": "started", "run_dir": str(run_dir)}


@app.get("/pipeline/run/{run_id}")
async def get_pipeline_run(run_id: str):
    run = pipeline_runs.get(run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found")
    return run


@app.get("/pipeline/run")
async def list_pipeline_runs():
    return {
        "runs": [
            {
                "run_id": run_id,
                "status": run.get("status"),
                "started_at": run.get("started_at"),
                "finished_at": run.get("finished_at"),
            }
            for run_id, run in pipeline_runs.items()
        ]
    }


@app.post("/pipeline/run/{run_id}/promote")
async def promote_pipeline_model(run_id: str, request: PromoteRequest):
    run = pipeline_runs.get(run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found")

    model_path = request.model_path
    if not model_path:
        model_path = run.get("outputs", {}).get("train_whisper", {}).get("best_model_path")
    if not model_path:
        raise HTTPException(400, "No model_path provided and run has no train_whisper output")

    try:
        output = _stage_promote_model(model_path)
    except Exception as e:
        raise HTTPException(400, f"Promotion failed: {e}")

    return output


@app.get("/models")
async def list_models():
    base_models = list((MODELS_DIR / "base").iterdir())
    fine_tuned = list((MODELS_DIR / "fine-tuned").iterdir())

    active = None
    active_path = _get_active_model_path()
    if active_path.exists():
        active = active_path.name

    return {
        "base_models": [m.name for m in base_models],
        "fine_tuned_models": [m.name for m in fine_tuned],
        "active": active,
    }


@app.post("/models/active")
async def switch_model(req: SwitchModelRequest):
    global model, processor, active_model_path, model_status, device

    model_path = MODELS_DIR / "fine-tuned" / req.model_id
    if not model_path.exists():
        raise HTTPException(404, f"Model not found: {req.model_id}")

    with model_lock:
        try:
            processor = WhisperProcessor.from_pretrained(str(model_path), language=LANGUAGE, task=TASK)
            model = WhisperForConditionalGeneration.from_pretrained(str(model_path))
            model.generation_config.language = LANGUAGE
            model.generation_config.task = TASK
            model.generation_config.forced_decoder_ids = None
            model.eval()

            if torch.cuda.is_available():
                model = model.to("cuda")
                device = "cuda"
            else:
                device = "cpu"

            active_model_path = model_path
            model_status = "loaded"

            _set_active_model_path(model_path)

        except Exception as e:
            raise HTTPException(500, f"Failed to load model: {e}")

    return {
        "message": "Model switched successfully",
        "active_model": req.model_id,
    }


@app.get("/models/active")
async def get_active_model():
    active_path = _get_active_model_path()
    if active_path.exists():
        return {"active_model": active_path.name}
    return {"active_model": MODEL_ID}


@app.post("/train/dataset")
async def upload_dataset(files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    temp_dir = Path("/tmp/dataset_upload")
    temp_dir.mkdir(exist_ok=True)

    import pandas as pd

    records = []
    for file in files:
        content = await file.read()
        records.append({"audio": {"bytes": content}, "text": ""})

    parquet_path = temp_dir / f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path)

    return {
        "message": "Dataset uploaded",
        "path": str(parquet_path),
        "samples": len(records),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)






















