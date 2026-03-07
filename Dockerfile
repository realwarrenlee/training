FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=openai/whisper-small \
    MODELS_DIR=/app/models \
    RUNS_DIR=/app/runs \
    CLEARML_ENABLED=false \
    WHISPER_LANGUAGE=English \
    WHISPER_TASK=transcribe \
    WHISPER_DECODE_LANGUAGE=en \
    ENABLE_AUDIO_PREPROCESS=false

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api.py ./
COPY config.py ./

RUN mkdir -p ${MODELS_DIR}/base ${MODELS_DIR}/fine-tuned ${RUNS_DIR}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
