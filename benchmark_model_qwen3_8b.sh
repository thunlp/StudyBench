#!/usr/bin/env bash
# Base Qwen3-8B benchmark settings.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR=/home/test/testdata/models/Qwen3-8B
TOKENIZER_PATH=$MODEL_DIR
SERVED_MODEL_NAME=qwen3_8b
DATASET=textbook
OUTPUT_DIR=base_qwen3_8b
TAG=qwen3_8b_textbook_problem_2
PORT=9000
TP=1
PASS_K=16
NPROC=64
JUDGE_NPROC=4
JUDGE_MODEL=deepseek-v4-flash-0731
JUDGE_BACKEND=openai
JUDGE_BASE_URL=https://yeysai.com/v1

source "$SCRIPT_DIR/benchmark_model.sh"
