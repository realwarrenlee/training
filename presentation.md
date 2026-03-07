# NESTLAB Speech API - Project Presentation

## Overview

Built a unified Speech-to-Text API using Whisper with FastAPI that supports both inference and training, containerized with Docker.

---

## What Was Built

### 1. Unified API (main.py)
- **Single FastAPI service** handling both inference and training
- RESTful endpoints for all operations
- GPU-accelerated inference
- Async job tracking for training

### 2. Inference Pipeline
- Audio preprocessing (normalization, high-pass filter, noise gating)
- Chunking for long audio files (30-second segments)
- Multiple audio format support (WAV, MP3, FLAC, OGG, M4A)
- Configurable Whisper models

### 3. Training Pipeline
- Fine-tuning Whisper on custom datasets
- Subprocess-based training (non-blocking)
- Job status tracking with log streaming
- Model hot-swapping without restart

### 4. Model Management
- Volume-based model storage
- Symlink-based active model switching
- Support for base and fine-tuned models

### 5. Deployment
- Unified Dockerfile (GPU-enabled)
- Docker Compose for orchestration
- Health checks and auto-restart

---

## API Endpoints

```
Inference:
  POST /transcribe          Transcribe audio file
  GET  /health             Health check

Training:
  POST /train              Start training job
  GET  /train              List all jobs
  GET  /train/{job_id}     Get status + logs
  DELETE /train/{job_id}   Cancel job
  POST /train/dataset      Upload dataset

Models:
  GET  /models             List available models
  GET  /models/active      Get current model
  POST /models/active      Switch model
```

---

## Key Learnings

### 1. GPU Resource Management
- **Challenge:** Training and inference both want GPU memory
- **Solution:** Simple time-slicing - reject inference during training
- **Learn:** Need lock/semaphore for proper GPU sharing in production

### 2. Model Hot-Swapping
- **Challenge:** Switching models without downtime
- **Solution:** Thread lock + symlink + reload in memory
- **Learn:** Keep global model reference, protect swap with lock

### 3. Audio Preprocessing
- **Challenge:** Noisy audio = poor transcription
- **Solution:** Normalization + high-pass filter + noise gating
- **Learn:** Preprocessing helps but can't fix fundamental audio quality issues

### 4. Language Parameter
- **Challenge:** Auto language detection produced gibberish
- **Solution:** Explicitly set `language="en"` in generate()
- **Learn:** Always specify language when you know it; Whisper defaults to multilingual which struggles with domain-specific audio

### 5. Whisper Model Constraints
- **Challenge:** MEL spectrogram length errors
- **Solution:** Pad audio chunks to exactly 30 seconds (3000 frames)
- **Learn:** Whisper expects fixed-length input; always pad

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Unified API | Single service easier to deploy than multiple |
| Subprocess for training | Doesn't block FastAPI event loop |
| Symlink for models | Atomic switch, easy to track |
| In-memory job tracking | Simple, sufficient for single-instance |
| Time-slice GPU | Simplest approach for now |
| Language parameter | Better accuracy for known language |
| Pad to 3000 frames | Fixes Whisper MEL length errors |

---

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| GPU resource contention | Time-slice - reject inference during training |
| Model switching downtime | Thread lock + symlink hot-reload |
| Noisy audio | Preprocessing pipeline |
| Language detection gibberish | Explicit language parameter |
| MEL spectrogram length error | Pad to 3000 frames |

---

## Future Improvements

1. **ML-based Denoising** - Use Speech Enhancement models for better audio quality

2. **Better GPU Scheduling** - Redis queue for job management, separate containers

3. **Model Versioning** - Track model versions, support A/B testing

4. **Production Monitoring** - Prometheus metrics, logging, error tracking

---

## Demo

```bash
# Start server
MODELS_DIR=./models python3 main.py

# Transcribe
curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe

# Train
curl -X POST http://localhost:8000/train \
  -d '{"model_id": "openai/whisper-tiny", "num_epochs": 1}'

# Switch model
curl -X POST http://localhost:8000/models/active \
  -d '{"model_id": "run_20240301"}'
```
