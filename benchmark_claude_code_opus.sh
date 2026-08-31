#!/usr/bin/env bash
# Baseline Claude Opus 4.7 inside the Claude Code agent harness -- no method
# conditioning at all, so this is the reference every evolve method is compared
# against.
#
# Examples:
#   bash benchmark_claude_code_opus.sh
#   TEST_START=0 TEST_END=1 bash benchmark_claude_code_opus.sh
#   DATASET=competition NPROC=2 bash benchmark_claude_code_opus.sh
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-textbook}"
OUTPUT_DIR="${OUTPUT_DIR:-results_opus_claude_code}"

source "$SCRIPT_DIR/benchmark_claude_code.sh"
