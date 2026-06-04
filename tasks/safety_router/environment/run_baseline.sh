#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" train.py \
  --train-path data/train.npz \
  --val-path data/val.npz \
  --output model_artifacts/router_model.npz

"$PYTHON_BIN" evaluate_local.py \
  --model model_artifacts/router_model.npz \
  --split data/test_public.npz
