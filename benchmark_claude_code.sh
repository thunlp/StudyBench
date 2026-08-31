#!/usr/bin/env bash
# Common evaluator for StudyBench runs inside the Claude Code agent harness.
#
# Concrete benchmark entry points set their variables directly and source this
# file. It is deliberately separate from benchmark_model.sh: that helper serves
# deployable OpenAI-compatible models (vLLM serving, FSDP merging, tokenizer
# paths), none of which applies here. What this file owns instead is the
# claude_code backend's own surface -- agent project dir, system-prompt append,
# tool allow-list, permission mode.
#
# Conditioning note: run_benchmark.py intentionally does NOT plumb
# --ace_playbook_path / --evoskill_skills_path / --gepa_prompt_path into the
# claude_code backend (see the comment at eval/run_benchmark.py
# `_generate_parent_claude_code`). A method conditions the agent through
# AGENT_PROJECT_DIR and/or SYSTEM_PROMPT_APPEND_FILE. GUIDANCE_PATH does
# apply -- it rewrites the sub-question user turns, which every backend
# shares.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Supplies the OpenAI-compatible judge credentials used by this repository.
[[ -f "$SCRIPT_DIR/env_local.sh" ]] && source "$SCRIPT_DIR/env_local.sh"

RUN_MODE="${RUN_MODE:-full}"
case "$RUN_MODE" in
    full) MODE_ARGS=(--resume_eval) ;;
    generate) MODE_ARGS=(--only_generate) ;;
    eval) MODE_ARGS=(--only_eval --resume_eval) ;;
    *) echo "RUN_MODE must be full, generate, or eval; got '$RUN_MODE'" >&2; exit 2 ;;
esac

DATASET="${DATASET:-textbook}"
DATA_PATH="${DATA_PATH:-}"
DEFAULT_TAG=""
if [[ -z "$DATA_PATH" ]]; then
    case "$DATASET" in
        textbook)
            DATA_PATH="data/qwen3_8b_textbook_problem.json"
            DEFAULT_TAG="textbook_problems"
            ;;
        competition)
            DATA_PATH="data/qwen3_8b_competition_problem.json"
            DEFAULT_TAG="competition_problems"
            ;;
        *)
            echo "DATASET must be 'textbook' or 'competition', or set DATA_PATH to a custom JSON file; got '$DATASET'" >&2
            exit 2
            ;;
    esac
fi

MODEL="${MODEL:-claude-opus-4-7}"
AGENT_MODEL="${AGENT_MODEL:-$MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-results_claude_code}"
TAG="${TAG:-${DEFAULT_TAG:-claude_code}}"
NPROC="${NPROC:-4}"
JUDGE_NPROC="${JUDGE_NPROC:-$NPROC}"
PASS_K="${PASS_K:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"
JUDGE_BACKEND="${JUDGE_BACKEND:-openai}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://llm-center.ali.modelbest.cn/llm/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${OPENAI_API_KEY:-}}"
JUDGE_THINKING_MODE="${JUDGE_THINKING_MODE:-thinking}"

AGENT_PROJECT_DIR="${AGENT_PROJECT_DIR:-}"
SYSTEM_PROMPT_APPEND_FILE="${SYSTEM_PROMPT_APPEND_FILE:-}"
AGENT_PERMISSION_MODE="${AGENT_PERMISSION_MODE:-bypassPermissions}"
# AGENT_TOOLS is an array; leave it unset to keep run_benchmark.py's defaults
# (Bash/Read/Write/Edit/Glob/Grep/TodoWrite/BashOutput).
GUIDANCE_PATH="${GUIDANCE_PATH:-}"

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

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    if ! command -v claude >/dev/null 2>&1; then
        echo "Claude Code is not installed or not on PATH." >&2
        exit 2
    fi
    if ! "$PYTHON_BIN" -c "import claude_agent_sdk" >/dev/null 2>&1; then
        echo "Python package claude-agent-sdk is missing; install requirements.txt." >&2
        exit 2
    fi
    if [[ ! -f "${HOME}/.claude/settings.json" ]] \
        && [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]] \
        && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "Claude Code credentials are not configured in ~/.claude/settings.json or the environment." >&2
        exit 2
    fi
    if [[ "$RUN_MODE" != generate && -z "$JUDGE_API_KEY" ]]; then
        echo "Set JUDGE_API_KEY or OPENAI_API_KEY for the benchmark judge." >&2
        exit 2
    fi
    if [[ "$RUN_MODE" != generate && -z "$JUDGE_BASE_URL" ]]; then
        echo "JUDGE_BASE_URL is empty; set it explicitly (or configure OPENAI_BASE_URL)." >&2
        exit 2
    fi
fi

if [[ -n "${TEST_START:-}" || -n "${TEST_END:-}" ]]; then
    if [[ -z "${TEST_START:-}" || -z "${TEST_END:-}" ]]; then
        echo "TEST_START and TEST_END must be set together." >&2
        exit 2
    fi
fi
# run_benchmark.py rejects this combination outright; fail here with a clearer
# message than an argparse abort halfway through setup.
if [[ "${FORCE_REGENERATE:-0}" == 1 && "$RUN_MODE" == eval ]]; then
    echo "FORCE_REGENERATE=1 is meaningless with RUN_MODE=eval (which skips generation)." >&2
    exit 2
fi

# run_benchmark.py stores results under eval/results/<basename output_dir>/.
OUTPUT_LOG_DIR="$SCRIPT_DIR/eval/results/$(basename "${OUTPUT_DIR%/}")"
mkdir -p "$OUTPUT_LOG_DIR"

# Extension point: a wrapper may define prepare_run() to materialize artefacts
# that depend on OUTPUT_LOG_DIR (the ACE / GEPA wrappers write their
# system-prompt append there, so each run records the exact conditioning
# it used).
if declare -F prepare_run >/dev/null; then
    prepare_run
fi

CMD=("$PYTHON_BIN" "$SCRIPT_DIR/eval/run_benchmark.py" --model "$MODEL" \
    --agent_model "$AGENT_MODEL" --backend claude_code \
    --data_paths "$(runner_path "$DATA_PATH")" --output_dir "$OUTPUT_DIR" \
    --tag "$TAG" --strip_think_for_model_eval \
    --nproc "$NPROC" --judge_nproc "$JUDGE_NPROC" \
    --max_tokens "$MAX_TOKENS" --pass_k "$PASS_K" --temperature "$TEMPERATURE")
[[ -n "$TOP_P" ]] && CMD+=(--top_p "$TOP_P")
[[ -n "$TOP_K" ]] && CMD+=(--top_k "$TOP_K")
[[ -n "$AGENT_PROJECT_DIR" ]] && CMD+=(--agent_project_dir "$(runner_path "$AGENT_PROJECT_DIR")")
[[ -n "$SYSTEM_PROMPT_APPEND_FILE" ]] && CMD+=(--system_prompt_append_file "$(runner_path "$SYSTEM_PROMPT_APPEND_FILE")")
[[ -n "$AGENT_PERMISSION_MODE" ]] && CMD+=(--agent_permission_mode "$AGENT_PERMISSION_MODE")
[[ -n "$GUIDANCE_PATH" ]] && CMD+=(--guidance_path "$(runner_path "$GUIDANCE_PATH")")
[[ -n "${TEST_START:-}" ]] && CMD+=(--test_start "$TEST_START" --test_end "$TEST_END")
[[ "${FORCE_REGENERATE:-0}" == 1 ]] && CMD+=(--force_regenerate)
CMD+=("${MODE_ARGS[@]}")

if [[ "$RUN_MODE" != generate ]]; then
    CMD+=(--judge_model "$JUDGE_MODEL" --judge_backend "$JUDGE_BACKEND" \
        --judge_base_url "$JUDGE_BASE_URL" --judge_api_key "$JUDGE_API_KEY" \
        --judge_thinking_mode "$JUDGE_THINKING_MODE")
fi

# --agent_tools is variadic, so it has to come last or it would swallow the
# following flags.
if [[ -n "${AGENT_TOOLS+x}" && "${#AGENT_TOOLS[@]}" -gt 0 ]]; then
    CMD+=(--agent_tools "${AGENT_TOOLS[@]}")
fi

echo "[claude-code] model=$AGENT_MODEL dataset=${DATASET} data=$DATA_PATH tag=$TAG output=$OUTPUT_DIR mode=$RUN_MODE"
[[ -n "$AGENT_PROJECT_DIR" ]] && echo "[claude-code] project_dir=$AGENT_PROJECT_DIR"
[[ -n "$SYSTEM_PROMPT_APPEND_FILE" ]] && echo "[claude-code] system_append=$SYSTEM_PROMPT_APPEND_FILE"
[[ -n "$GUIDANCE_PATH" ]] && echo "[claude-code] guidance=$GUIDANCE_PATH"
printf '[benchmark] '
print_command "${CMD[@]}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "[benchmark] DRY_RUN=1; command not executed"
    exit 0
fi

claude --version
"${CMD[@]}" 2>&1 | tee -a "$OUTPUT_LOG_DIR/benchmark.log"
