#!/usr/bin/env bash
# End-to-end KP -> textbook-coverage -> guidance pipeline over ALL 699
# competition problems.
#
# Existing knowledge points, textbook matches, and guidance cache entries are
# reused where present. Every stage is independently resumable, so re-running
# after an interruption is safe and cheap.
#
# Usage:
#   bash filter/run_full_competition_pipeline.sh            # all stages
#   START_AT=3 bash filter/run_full_competition_pipeline.sh # from stage 3
#   DRY_RUN=1 bash filter/run_full_competition_pipeline.sh  # print only
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# shellcheck disable=SC1091
source env_local.sh

# The embedding model is cached locally; staying offline avoids a hard
# failure when the HF hub is unreachable.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

MODEL=${MODEL:-deepseek-v4-pro}
CONCURRENCY=${CONCURRENCY:-8}
START_AT=${START_AT:-1}
DRY_RUN=${DRY_RUN:-0}

STUDY=studybench_data
CORPUS=$STUDY/competition_problems/competition_problems_full.json
COVERAGE=$STUDY/competition_problems_full.with_coverage.json
GUIDANCE=$STUDY/level3_guidance_full.json

run() {
  echo "+ $*"
  [ "$DRY_RUN" = "1" ] || "$@"
}

stage() {
  local n=$1; shift
  if [ "$START_AT" -gt "$n" ]; then
    echo "== stage $n: $1 -- SKIPPED (START_AT=$START_AT)"
    return 1
  fi
  echo
  echo "=============================================================="
  echo "== stage $n: $1"
  echo "=============================================================="
  return 0
}

# --- 1. extract knowledge points -------------------------------------------
if stage 1 "extract knowledge points"; then
  run python3 filter/extract_knowledge_points.py \
    --input "$CORPUS" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY"
fi

# --- 2. canonicalise KPs across the whole corpus ---------------------------
# kp_id is sha1(type::normalised_name), so ids from the original run survive
# the corpus growing; only genuinely new KPs get new ids.
if stage 2 "canonicalise knowledge points"; then
  run python3 filter/canonicalize_knowledge_points.py --input "$CORPUS"
fi

# --- 2.5 retrieve candidate textbook fragments for NEW kp_ids only ---------
if stage 3 "retrieve textbook candidates (new KPs only)"; then
  run python3 filter/retrieve_textbook_candidates.py \
    --kps "$STUDY/knowledge_points.jsonl" \
    --fragments "$STUDY/textbook_fragments.jsonl" \
    --out "$STUDY/kp_candidates.jsonl"
fi

# --- 3. LLM-verify the candidates for NEW kp_ids only ----------------------
if stage 4 "verify textbook matches (new KPs only)"; then
  run python3 filter/verify_textbook_matches.py \
    --candidates "$STUDY/kp_candidates.jsonl" \
    --fragments "$STUDY/textbook_fragments.jsonl" \
    --out "$STUDY/kp_matches.jsonl" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY"
fi

# --- 4. join verified matches back onto every sub-problem ------------------
if stage 5 "join coverage"; then
  run python3 filter/join_coverage.py \
    --input "$CORPUS" \
    --matches "$STUDY/kp_matches.jsonl" \
    --out "$COVERAGE"
fi

# --- 5. teacher-model guidance, with cache resume and leakage double gate ---
if stage 6 "build guidance (resume existing guidance and cache)"; then
  run python3 filter/build_kp_guidance.py \
    --coverage "$COVERAGE" \
    --fragments "$STUDY/textbook_fragments.jsonl" \
    --out "$GUIDANCE" \
    --cache "$GUIDANCE.cache.jsonl" \
    --resume-from eval/data/level3_guidance_full.json \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY"
fi

echo
echo "done. guidance -> $GUIDANCE"
