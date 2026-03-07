#!/usr/bin/env bash
set -euo pipefail
MODELS_DIR=${MODELS_DIR:-./models} python3 api.py
