#!/bin/bash
set -euo pipefail

# Navigation inference example (no training)
#
# Usage:
#   # Terminal A (start env server)
#   python -m vagen.server.server server.port=5000
#
#   # Terminal B (run inference)
#   ./scripts/examples/inference/navigation/run.sh vllm
#   ./scripts/examples/inference/navigation/run.sh openai
#
# Notes:
# - Navigation env requires ai2thor installed (and system deps, see vagen/env/README.md).
# - vagen/inference/run_inference.py imports wandb at import-time; install wandb even if use_wandb=false.

MODEL_PROVIDER="${1:-vllm}" # vllm | openai | openrouter
PORT="${PORT:-5000}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EXPERIMENT_NAME="inference-navigation-${MODEL_PROVIDER}"
DATA_DIR="$REPO_ROOT/data/$EXPERIMENT_NAME"
mkdir -p "$DATA_DIR"

ENV_CONFIG="$SCRIPT_DIR/env_config.yaml"
INFER_CONFIG="$SCRIPT_DIR/inference_config.yaml"

if [[ "$MODEL_PROVIDER" == "openai" ]]; then
  MODEL_CONFIG="$SCRIPT_DIR/model_config_openai.yaml"
elif [[ "$MODEL_PROVIDER" == "openrouter" ]]; then
  MODEL_CONFIG="$SCRIPT_DIR/model_config_openrouter.yaml"
else
  MODEL_CONFIG="$SCRIPT_DIR/model_config_vllm.yaml"
fi

echo "Repo root: $REPO_ROOT"
echo "Data dir:   $DATA_DIR"
echo "Model cfg:  $MODEL_CONFIG"
echo

echo "Generating parquet with seeds/configs (for inference)..."
python -m vagen.env.create_dataset \
  --yaml_path "$ENV_CONFIG" \
  --train_path "$DATA_DIR/train.parquet" \
  --test_path "$DATA_DIR/test.parquet" \
  --force_gen

echo
echo "Make sure env server is running at: http://localhost:$PORT"
if command -v curl >/dev/null 2>&1; then
  if ! curl -fsS "http://localhost:$PORT/health" >/dev/null; then
    echo "ERROR: env server not reachable. Start it first:"
    echo "  python -m vagen.server.server server.port=$PORT"
    exit 2
  fi
else
  echo "WARN: curl not found; skipping /health check."
fi

echo
echo "Running inference..."
python -m vagen.inference.run_inference \
  --inference_config_path "$INFER_CONFIG" \
  --model_config_path "$MODEL_CONFIG" \
  --val_files_path "$DATA_DIR/test.parquet" \
  --wandb_path_name "$EXPERIMENT_NAME"


