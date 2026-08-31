#!/usr/bin/env bash
# Base Llama-3.2-3B-Instruct benchmark settings.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR=/home/test/testdata/models/Llama-3.2-3B-Instruct
TOKENIZER_PATH=$MODEL_DIR
SERVED_MODEL_NAME=llama3_2_3b_instruct
OUTPUT_DIR=results_llama3_2_3b_instruct
TAG=llama3_2_3b_instruct_base_${DATASET}
PORT=8750
TP=1
PASS_K=24
NPROC=64
JUDGE_NPROC=4
JUDGE_MODEL=deepseek-v4-flash-0731
JUDGE_BACKEND=openai
JUDGE_BASE_URL=https://yeysai.com/v1

source "$SCRIPT_DIR/benchmark_model.sh"
