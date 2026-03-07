#!/usr/bin/env bash
set -euo pipefail
curl -X POST http://127.0.0.1:8000/pipeline/run \
  -H "Content-Type: application/json" \
  -d @${1:-pipeline_payload.json}
