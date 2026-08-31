#!/usr/bin/env bash
# Common evaluator for deployable, OpenAI-compatible models.
#
# Concrete benchmark entry points set their variables directly and source this
# file. The public dataset selector is DATASET; DATA_PATH is the simple escape
# hatch for a custom JSON file.
# Claude Code/Anthropic runs intentionally stay outside this helper.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[[ -f "$SCRIPT_DIR/env_local.sh" ]] && source "$SCRIPT_DIR/env_local.sh"

RUN_MODE="${RUN_MODE:-full}"
case "$RUN_MODE" in
    full) MODE_ARGS=(--resume_eval) ;;
    generate) MODE_ARGS=(--only_generate) ;;
    eval) MODE_ARGS=(--only_eval --resume_eval) ;;
    *) echo "RUN_MODE must be full, generate, or eval; got '$RUN_MODE'" >&2; exit 2 ;;
esac


DATASET="${DATASET:-competition}"
DATA_PATH="${DATA_PATH:-}"
if [[ -z "$DATA_PATH" ]]; then
    case "$DATASET" in
        textbook) DATA_PATH="data/qwen3_8b_textbook_problem.json" ;;
        competition) DATA_PATH="data/qwen3_8b_competition_problem.json" ;;
        *)
            echo "DATASET must be 'textbook' or 'competition', or set DATA_PATH to a custom JSON file; got '$DATASET'" >&2
            exit 2
            ;;
    esac
fi

MODEL_DIR="${MODEL_DIR:-${MODEL:-}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${SERVED_NAME:-}}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_DIR}"
PORT="${PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
TP="${TP:-1}"
VLLM_BIN="${VLLM_BIN:-/home/test/test1708/anaconda3/envs/vllm/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-results_model}"
TAG="${TAG:-${SERVED_MODEL_NAME:-model}}"
BACKEND="${BACKEND:-api}"
NPROC="${NPROC:-16}"
JUDGE_NPROC="${JUDGE_NPROC:-$NPROC}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-}"
PASS_K="${PASS_K:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
SERVE_MODEL="${SERVE_MODEL:-1}"
GENERATION_BASE_URL="${GENERATION_BASE_URL:-}"

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_BACKEND="${JUDGE_BACKEND:-openai}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${OPENAI_BASE_URL:-}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
JUDGE_NPROC="${JUDGE_NPROC:-$NPROC}"
JUDGE_DSV4_MODEL_DIR="${JUDGE_DSV4_MODEL_DIR:-}"
JUDGE_THINKING_MODE="${JUDGE_THINKING_MODE:-thinking}"

abspath() {
    local path="$1"
    if [[ "$path" = /* ]]; then printf '%s\n' "$path"; else printf '%s\n' "$SCRIPT_DIR/$path"; fi
}

# run_benchmark.py anchors relative paths at eval/. A path which exists at the
# repository root therefore needs ../ when passed to the Python runner.
runner_path() {
    local path="$1"
    [[ "$path" = /* ]] && { printf '%s\n' "$path"; return; }
    if [[ "$path" == eval/* || -e "$SCRIPT_DIR/$path" ]]; then
        printf '../%s\n' "$path"
    else
        printf '%s\n' "$path"
    fi
}

require_file_or_dir() {
    local path="$1" label="$2" kind="$3"
    [[ "$RUN_MODE" == eval && "$label" == model ]] && return
    if [[ "$kind" == file && ! -f "$path" ]] || [[ "$kind" == dir && ! -d "$path" ]]; then
        echo "$label not found: $path" >&2
        exit 2
    fi
}

print_command() {
    local redact=0 arg
    for arg in "$@"; do
        if ((redact)); then
            printf '%s ' '***REDACTED***'
            redact=0
        elif [[ "$arg" == --judge_api_key ]]; then
            printf '%s ' "$arg"
            redact=1
        else
            printf '%q ' "$arg"
        fi
    done
    printf '\n'
}

if [[ -z "$SERVED_MODEL_NAME" ]]; then
    echo "SERVED_MODEL_NAME is required" >&2
    exit 2
fi
if [[ "$RUN_MODE" != eval && -z "$GENERATION_BASE_URL" && "${DRY_RUN:-0}" != 1 ]]; then
    [[ -n "$MODEL_DIR" ]] || { echo "MODEL_DIR is required for generation" >&2; exit 2; }
    MODEL_DIR="$(abspath "$MODEL_DIR")"
    require_file_or_dir "$MODEL_DIR" model dir
fi
# Optional standard verl/FSDP merge. Configs can override the source/target
# directories; the operation is idempotent and lock-protected.
MERGE_FSDP="${MERGE_FSDP:-0}"
if [[ "$MERGE_FSDP" == 1 && "$RUN_MODE" != eval && "${DRY_RUN:-0}" != 1 ]]; then
    MERGE_PYTHON="${MERGE_PYTHON:-$PYTHON_BIN}"
    MERGE_BACKEND="${MERGE_BACKEND:-fsdp}"
    MERGE_LOCAL_DIR="${MERGE_LOCAL_DIR:-$MODEL_DIR}"
    MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$MERGE_LOCAL_DIR/merged_hf}"
    MERGE_LOCAL_DIR="$(abspath "$MERGE_LOCAL_DIR")"
    MERGED_MODEL_DIR="$(abspath "$MERGED_MODEL_DIR")"
    if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
        [[ -f "$MERGE_LOCAL_DIR/fsdp_config.json" ]] || {
            echo "FSDP checkpoint metadata not found: $MERGE_LOCAL_DIR/fsdp_config.json" >&2
            exit 2
        }
        mkdir -p "$MERGED_MODEL_DIR"
        exec 9>"$MERGE_LOCAL_DIR/.benchmark_model.merge.lock"
        flock 9
        if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
            "$MERGE_PYTHON" -m verl.model_merger merge \
                --backend "$MERGE_BACKEND" --local_dir "$MERGE_LOCAL_DIR" \
                --target_dir "$MERGED_MODEL_DIR"
        fi
        flock -u 9
        exec 9>&-
    fi
    MODEL_DIR="$MERGED_MODEL_DIR"
fi

if [[ -n "$TOKENIZER_PATH" && "$TOKENIZER_PATH" != /* && -e "$SCRIPT_DIR/$TOKENIZER_PATH" ]]; then
    TOKENIZER_PATH="$(abspath "$TOKENIZER_PATH")"
fi
if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_LOG_DIR="$OUTPUT_DIR"
elif [[ "$OUTPUT_DIR" == eval/* ]]; then
    OUTPUT_LOG_DIR="$(abspath "$OUTPUT_DIR")"
else
    # run_benchmark.py stores relative output_dir values below eval/.
    OUTPUT_LOG_DIR="$SCRIPT_DIR/eval/results/$OUTPUT_DIR/$TAG"
fi
mkdir -p "$OUTPUT_LOG_DIR"

VLLM_PID=""
cleanup() { [[ -z "$VLLM_PID" ]] || kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ "$RUN_MODE" != eval && "${DRY_RUN:-0}" != 1 ]]; then
    if [[ -n "$GENERATION_BASE_URL" ]]; then
        GENERATION_URL="${GENERATION_BASE_URL%/}"
        if [[ "${DRY_RUN:-0}" != 1 ]] && ! curl -fsS "${GENERATION_URL}/models" 2>/dev/null |
            grep -Fq '"id":"'"$SERVED_MODEL_NAME"'"'; then
            echo "Generation endpoint does not serve '$SERVED_MODEL_NAME': $GENERATION_URL" >&2
            exit 1
        fi
    elif [[ "$SERVE_MODEL" == 1 ]]; then
        require_file_or_dir "$VLLM_BIN" vLLM file
        VLLM_CMD=("$VLLM_BIN" serve "$MODEL_DIR" --served-model-name "$SERVED_MODEL_NAME" \
            --host "$VLLM_HOST" --port "$PORT" --tensor-parallel-size "$TP" \
            --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --trust-remote-code --enable-prefix-caching --dtype bfloat16)
        echo "[serve] ${VLLM_CMD[*]}"
        "${VLLM_CMD[@]}" >"$OUTPUT_LOG_DIR/vllm.log" 2>&1 &
        VLLM_PID=$!
        GENERATION_URL="http://${VLLM_HOST}:${PORT}/v1"
        deadline=$((SECONDS + READY_TIMEOUT))
        while ! curl -fsS "${GENERATION_URL}/models" 2>/dev/null | grep -Fq '"id":"'"$SERVED_MODEL_NAME"'"'; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "vLLM exited before becoming ready; see $OUTPUT_LOG_DIR/vllm.log" >&2
                tail -n 40 "$OUTPUT_LOG_DIR/vllm.log" >&2 || true
                exit 1
            fi
            ((SECONDS < deadline)) || { echo "Timed out waiting for $GENERATION_URL" >&2; exit 1; }
            sleep 5
        done
    else
        echo "Set GENERATION_BASE_URL or SERVE_MODEL=1 for generation" >&2
        exit 2
    fi
    echo "[generate] endpoint=$GENERATION_URL model=$SERVED_MODEL_NAME"
    export OPENAI_BASE_URL="$GENERATION_URL"
    export OPENAI_API_KEY="EMPTY"
fi

if [[ -z "$JUDGE_BASE_URL" && "$RUN_MODE" != generate ]]; then
    echo "JUDGE_BASE_URL is empty; set it explicitly (or configure OPENAI_BASE_URL)." >&2
    exit 2
fi

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/eval/run_benchmark.py" --model "$SERVED_MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" --backend "$BACKEND" --nproc "$NPROC" --tag "$TAG" \
    --max_tokens "$MAX_TOKENS" --pass_k "$PASS_K" --temperature "$TEMPERATURE")
CMD+=(--data_paths "$(runner_path "$DATA_PATH")")
[[ -n "$TOKENIZER_PATH" ]] && CMD+=(--tokenizer_path "$TOKENIZER_PATH")
[[ -n "$TOP_P" ]] && CMD+=(--top_p "$TOP_P")
[[ -n "$TOP_K" ]] && CMD+=(--top_k "$TOP_K")
[[ -n "$MAX_COMPLETION_TOKENS" ]] && CMD+=(--max_completion_tokens "$MAX_COMPLETION_TOKENS")
[[ "${STRIP_THINK_FOR_MODEL_EVAL:-1}" == 1 ]] && CMD+=(--strip_think_for_model_eval)
[[ -n "${ACE_PLAYBOOK_PATH:-}" ]] && CMD+=(--ace_playbook_path "$(runner_path "$ACE_PLAYBOOK_PATH")")
[[ -n "${EVOSKILL_SKILLS_PATH:-}" ]] && CMD+=(--evoskill_skills_path "$(runner_path "$EVOSKILL_SKILLS_PATH")")
[[ -n "${GEPA_PROMPT_PATH:-}" ]] && CMD+=(--gepa_prompt_path "$(runner_path "$GEPA_PROMPT_PATH")")
[[ -n "${GEPA_ITERATION:-}" ]] && CMD+=(--gepa_iteration "$GEPA_ITERATION")
[[ -n "${GEPA_ROLLOUTS:-}" ]] && CMD+=(--gepa_rollouts "$GEPA_ROLLOUTS")
[[ -n "${GUIDANCE_PATH:-}" ]] && CMD+=(--guidance_path "$(runner_path "$GUIDANCE_PATH")")
[[ -n "${TEST_START:-}" ]] && CMD+=(--test_start "$TEST_START")
[[ -n "${TEST_END:-}" ]] && CMD+=(--test_end "$TEST_END")
CMD+=("${MODE_ARGS[@]}")

if [[ "$RUN_MODE" != generate ]]; then
    CMD+=(--judge_model "$JUDGE_MODEL" --judge_backend "$JUDGE_BACKEND" \
        --judge_nproc "$JUDGE_NPROC" --judge_base_url "$JUDGE_BASE_URL" \
        --judge_api_key "$JUDGE_API_KEY" --judge_thinking_mode "$JUDGE_THINKING_MODE")
    [[ -n "$JUDGE_DSV4_MODEL_DIR" ]] && CMD+=(--judge_dsv4_model_dir "$JUDGE_DSV4_MODEL_DIR")
fi

printf '[benchmark] '
print_command "${CMD[@]}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "[benchmark] DRY_RUN=1; command not executed"
    exit 0
fi
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_LOG_DIR/benchmark.log"
