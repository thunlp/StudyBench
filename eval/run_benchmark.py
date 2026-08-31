#!/usr/bin/env python
"""End-to-end evaluation runner for benchmark.

The benchmark dataset (under ``problems/<source>/<year>.json``) follows the
schema in ``problems/README.md``. Each parent has:

  * multi-subquestion case: ``parent.problem`` holds the shared stem (the
    text common to every sub-question, often empty for textbook problems),
    and ``sub_problems[i]`` carries the per-sub ``problem`` / ``solution`` /
    ``answer`` / ``answer_type``;
  * single-question case: ``parent.problem`` holds the entire question
    text, with the gold ``answer`` / ``answer_type`` / LLM-cleaned
    ``solution`` lifted directly onto the parent.

Both shapes are read **only** from ``parent.problem`` — the old ``stem``
field is gone and the runner no longer touches it.

This runner abstracts over the two shapes by materialising a uniform
``attempt['completions']`` slot list — one entry per sub-question for
multi-sub parents, exactly one entry built from the parent fields for
single-question parents. Each parent stores ``attempts`` for pass@k:
one attempt is a full sequential solve of all sub-questions.

This script:

  1. Loads the JSON dataset and validates that every parent has a usable
     gold answer (per-sub for multi-sub parents, parent-level for
     single-question parents).
  2. Generates model completions **one sub-question at a time**, including
     the preceding sub-questions *and the model's own earlier answers* in the
     chat history, but never leaking any solution information.
  3. Persists the full OpenAI / vLLM response dict (``raw_response``) together
     with the prompt it was generated from.
  4. Supports resume: a second run re-uses any attempt/sub-question whose
     ``raw_response`` yields a non-empty completion.
  5. Scores each sub-question with the shared ``Judger`` (via
     ``eval.eval_file``) and prints parent and sub-question pass@k
     breakdowns by answer type and source.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from judge_utils import read_json  # noqa: E402
from persistent_state import PersistentState  # noqa: E402
from judge import DEFAULT_JUDGE_MODEL  # noqa: E402
import eval

FIXED_MIN_P = 0.0
FIXED_PRESENCE_PENALTY = 1.5
FIXED_REPETITION_PENALTY = 1.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _get_sub_problems(item: dict) -> list[dict]:
    """Return the parent's sub-question list, tolerating the legacy field name.

    The current schema stores it as ``sub_problems``; pre-rename result
    files used ``problems``. We always read both names so a stale
    ``parent['problems']`` from an old generation file doesn't get
    mistaken for "no sub-questions".
    """
    subs = item.get("sub_problems")
    if subs is None:
        subs = item.get("problems")
    return subs or []


HF_COMPETITIONS = ("apho", "eupho", "ioaa", "ipho", "nbpho", "opho", "textbook_astro", "textbook_physics")


def _validate_and_index_items(items: list[dict]) -> list[dict]:
    """Validate benchmark parent records and attach stable source indices."""
    for item in items:
        subproblems = _get_sub_problems(item)
        sid = item.get("source_problem_id")
        if subproblems:
            missing = [
                s.get("problem_id") for s in subproblems if not s.get("answer")
            ]
            assert not missing, (
                f"Missing sub-answer for {sid}: {missing}"
            )
        else:
            assert item.get("answer"), (
                f"Single-question parent {sid} missing parent.answer"
            )

    for index, item in enumerate(items):
        item["_source_index"] = index
    return items


def load_local_items(data_paths: list[str]) -> list[dict]:
    """Load one or more local benchmark JSON datasets and concatenate items.

    ``data_paths`` is a list of paths; the parent items from each file are
    concatenated in order. Each parent keeps its original ``sub_problems``
    list intact (or its parent-level ``answer`` / ``answer_type`` for
    single-question parents per ``problems/README.md``). Items without any
    usable gold answer are rejected with an assertion. Each item is tagged
    with a ``_source_index`` that is unique across the merged collection.
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if not data_paths:
        raise ValueError("load_local_items requires at least one data path.")

    merged: list[dict] = []
    for data_path in data_paths:
        raw = read_json(data_path)
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
            items = raw["data"]
        else:
            raise ValueError(
                "Input dataset must be a JSON list or a dict with a `data` list; "
                f"got top-level type {type(raw).__name__}."
            )
        merged.extend(items)

    return _validate_and_index_items(merged)


def load_hf_items(hf_repo: str, competition: str) -> list[dict]:
    """Load benchmark parents from a Hugging Face dataset config."""
    if not hf_repo:
        raise ValueError("hf_repo is required when data_paths is not provided.")
    if not competition:
        raise ValueError("competition is required when data_paths is not provided.")
    competition = competition.lower()
    if competition not in HF_COMPETITIONS:
        raise ValueError(
            f"Unknown competition {competition!r}; expected one of {HF_COMPETITIONS}."
        )
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Loading from Hugging Face requires the `datasets` package. "
            "Install it with: pip install -U datasets"
        ) from exc

    dataset = load_dataset(hf_repo, competition, split="train")
    items = [dict(row) for row in dataset]
    return _validate_and_index_items(items)


def load_benchmark_items(
    data_paths: Optional[list[str]] = None,
    hf_repo: Optional[str] = None,
    competition: Optional[str] = None,
) -> list[dict]:
    """Load benchmark parents, preferring explicit local paths over HF."""
    if data_paths:
        return load_local_items(data_paths)
    return load_hf_items(hf_repo or "", competition or "")


def _guidance_key(
    source: Any,
    year: Any,
    source_problem_id: Any,
    problem_id: Any,
) -> tuple:
    """Stable join key for matching a guidance entry onto a benchmark sub."""
    return (
        str(source or ""),
        str(year or ""),
        str(source_problem_id or ""),
        str(problem_id or ""),
    )


def load_guidance_index(path: str) -> dict[tuple, str]:
    """Load per-sub-question solving guidance from a JSON file.

    Expected shape matches ``level3_guidance.json``: a list of parent
    records whose ``sub_problems[i].guidance`` holds the free-text hint
    for that sub-question. Parents / subs without a non-empty
    ``guidance`` field are skipped. Keys are
    ``(source, year, source_problem_id, problem_id)`` with all
    components stringified so int/str year mismatches still join.
    """
    raw = read_json(path)
    if isinstance(raw, list):
        parents = raw
    elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
        parents = raw["data"]
    else:
        raise ValueError(
            "Guidance file must be a JSON list or a dict with a `data` list; "
            f"got top-level type {type(raw).__name__}."
        )

    index: dict[tuple, str] = {}
    for parent in parents:
        if not isinstance(parent, dict):
            continue
        for sub in _get_sub_problems(parent):
            if not isinstance(sub, dict):
                continue
            guidance = sub.get("guidance")
            if not isinstance(guidance, str) or not guidance.strip():
                continue
            key = _guidance_key(
                parent.get("source"),
                parent.get("year"),
                parent.get("source_problem_id"),
                sub.get("problem_id"),
            )
            index[key] = guidance.strip()
    return index


def attach_guidance_to_items(
    items: list[dict],
    guidance_index: dict[tuple, str],
) -> tuple[int, int]:
    """Copy matching guidance strings onto each parent's ``sub_problems``.

    Mutates ``items`` in place. Returns ``(attached, missing)`` where
    ``attached`` is the number of subs that received a non-empty guidance
    string and ``missing`` is the number of subs that had no match (those
    keep whatever ``guidance`` they already carried, if any).
    """
    attached = 0
    missing = 0
    for parent in items:
        for sub in _get_sub_problems(parent):
            if not isinstance(sub, dict):
                continue
            key = _guidance_key(
                parent.get("source"),
                parent.get("year"),
                parent.get("source_problem_id"),
                sub.get("problem_id"),
            )
            guidance = guidance_index.get(key)
            if guidance:
                sub["guidance"] = guidance
                attached += 1
            else:
                missing += 1
    return attached, missing


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _subproblem_instruction(language: str) -> str:
    if language.lower() == "zh":
        return (
            "现在你需要解答一道物理题。"
            "请一步步进行推理，并在最后一行以 `Final Answer: \\boxed{答案}` 的形式显式给出答案。"
            "若该小题有多个答案，请在 Final Answer 行中分别用多个 \\boxed{} 列出。"
            "请不要在 \\boxed{} 中包含单位。"
        )
    return (
        "Now you need to solve a physics problem. "
        "Please reason step by step, and on the last line give your final answer "
        "in the form `Final Answer: \\boxed{answer}`. "
        "If this subquestion has multiple answers, emit one \\boxed{} per answer "
        "on that final line. Do not include units inside \\boxed{}."
    )


# Matches a ``<think>...</think>`` block (closed) and, separately, any unclosed
# trailing ``<think>...`` (which only appears when the prior turn hit
# ``max_tokens`` mid-reasoning). Both must be stripped from the conversation
# history before sending it back to a reasoning model: keeping them around
# burns 5-15k tokens per turn and reliably overflows the context window after
# a few sub-questions, exactly the failure mode seen in the Qwen3 runs.
_CLOSED_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", flags=re.IGNORECASE | re.DOTALL)


def _strip_think_for_history(text: Optional[str]) -> str:
    """Return ``text`` with any ``<think>...</think>`` reasoning removed.

    Closed blocks are dropped wholesale; an unclosed trailing ``<think>`` (the
    truncation case) is also stripped to end-of-text so we never feed a half
    chain-of-thought back into the next turn.
    """
    if not isinstance(text, str) or not text:
        return ""
    cleaned = _CLOSED_THINK_RE.sub("", text)
    cleaned = _OPEN_THINK_RE.sub("", cleaned)
    return cleaned.strip()


def _empty_assistant_placeholder(language: str) -> str:
    """Filler text for prior assistant turns that produced no usable answer.

    Most often the prior turn was length-truncated mid-``<think>`` and the
    strip helper just collapsed it to ``""``; sometimes the backend returned
    an error and ``completion`` was never written. Either way an empty
    ``assistant`` slot in the chat history confuses the next turn (the model
    sees its own previous "answer" as blank and tends to either repeat the
    earlier sub-question or refuse to attempt the new one). Replacing the
    empty content with a short, factual note keeps the chat structure
    well-formed without leaking any solution information.
    """
    if language.lower() == "zh":
        return "（上一小题未能给出最终答案，请直接处理下一小题。）"
    return (
        "(I was unable to produce a final answer for the previous "
        "sub-question. Please continue with the next sub-question.)"
    )


def _completion_slot_template(parent: dict) -> list[dict]:
    """Return the canonical per-sub-question slot list for ``parent``.

    The slot list is what ``parent['completions']`` will look like before
    any model output has been written into it: one dict per sub-question
    (or one dict total, built from the parent fields, for the
    single-question case). Each slot carries ``problem_id`` / ``problem``
    / ``solution`` / ``answer`` / ``answer_type`` so that downstream
    judging can read everything off the slot without re-consulting the
    parent.

    For multi-sub parents we deep-copy each ``sub_problems[i]`` so that
    later mutations (writing ``completion`` / ``raw_response`` /
    ``correctness`` ...) don't leak back into the static ground-truth
    list. For single-question parents we synthesise a slot from
    ``parent.{problem,solution,answer,answer_type}``; the slot's
    ``problem_id`` falls back to ``source_problem_id`` so resume keys are
    well-defined.
    """
    subs = _get_sub_problems(parent)
    if subs:
        return [copy.deepcopy(s) for s in subs]
    return [
        {
            "problem_id": parent.get("source_problem_id"),
            "problem": parent.get("problem") or "",
            "solution": parent.get("solution") or "",
            "answer": parent.get("answer") or "",
            "answer_type": parent.get("answer_type") or "",
            # ``type_sequence`` is required for TUP / ALT parents (see
            # README); the judge consumes it from the per-completion
            # field copied here.
            "type_sequence": parent.get("type_sequence") or "",
        }
    ]


def _normalize_for_generation(parent: dict) -> tuple[list[dict], str]:
    """Return ``(subproblems_for_msg, shared_stem_for_msg)`` for prompt building.

    The shared-stem string is sourced from ``parent.problem`` per the
    current schema (``problems/README.md``): multi-sub parents store the
    common preamble there; solo parents store the full question text
    there. There is no ``stem`` field anymore.

    Multi-sub parents pass their ``sub_problems`` list through unchanged
    and return ``parent.problem`` as the shared stem prepended to every
    first-turn user message.  Single-question parents
    (``sub_problems == []``) get rewritten as a one-element synthetic
    list whose only ``problem`` is the parent's full text, with the
    shared-stem-for-message blanked out — this avoids duplicating the
    question in the first-turn prompt.
    """
    subs = _get_sub_problems(parent)
    parent_stem = (parent.get("problem") or "").strip()
    if subs:
        return subs, parent_stem
    return [{"problem": parent_stem}], ""


def _parse_gepa_prompt_file(path: str, raw: str) -> Optional[list[dict]]:
    """Return GEPA history rows, or ``None`` if ``raw`` is a plain-text prompt.

    A file is treated as a GEPA history when it is ``*.jsonl`` or when its
    first non-empty line is a JSON object that carries ``system_prompt``.
    Plain-text artefacts (a single evolved instruction dumped to ``.txt``)
    return ``None`` so the caller can use the file contents as-is.
    """
    stripped = raw.strip()
    if not stripped:
        raise ValueError(f"GEPA prompt file is empty: {path}")
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    first = lines[0]
    looks_jsonl = path.endswith(".jsonl") or (
        first.startswith("{") and '"system_prompt"' in first
    )
    if looks_jsonl:
        rows: list[dict] = []
        for i, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid GEPA history JSONL at {path}:{i}: {exc}"
                ) from exc
            if not isinstance(obj, dict) or "system_prompt" not in obj:
                raise ValueError(
                    f"{path}:{i} is not a GEPA history row "
                    "(missing 'system_prompt')"
                )
            rows.append(obj)
        if not rows:
            raise ValueError(f"no GEPA history rows in {path}")
        return rows
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "system_prompt" in obj:
            return [obj]
    return None


def _select_gepa_history_row(
    rows: list[dict],
    iteration: Optional[int] = None,
    rollouts: Optional[int] = None,
) -> dict:
    """Pick a checkpoint from a chronological GEPA history.

    Default is the last row (the incumbent best prompt at the end of
    training). ``iteration=N`` / ``rollouts=N`` select the last row whose
    corresponding field is ``<= N``, so a sparse log still answers "the
    prompt as of this budget".
    """
    if iteration is not None and rollouts is not None:
        raise ValueError("iteration and rollouts are mutually exclusive")

    def _field_le(row: dict, key: str, limit: int) -> bool:
        value = row.get(key)
        if value is None:
            return False
        try:
            return int(value) <= limit
        except (TypeError, ValueError):
            return False

    if iteration is not None:
        eligible = [r for r in rows if _field_le(r, "iteration", iteration)]
        if not eligible:
            iters = [r.get("iteration") for r in rows if r.get("iteration") is not None]
            span = (
                f"history spans iteration {min(iters)}–{max(iters)}"
                if iters
                else "history has no iteration field"
            )
            raise ValueError(
                f"no GEPA checkpoint with iteration <= {iteration}; {span}"
            )
        return eligible[-1]
    if rollouts is not None:
        eligible = [r for r in rows if _field_le(r, "rollouts", rollouts)]
        if not eligible:
            vals = [r.get("rollouts") for r in rows if r.get("rollouts") is not None]
            span = (
                f"history spans rollouts {min(vals)}–{max(vals)}"
                if vals
                else "history has no rollouts field"
            )
            raise ValueError(
                f"no GEPA checkpoint with rollouts <= {rollouts}; {span}"
            )
        return eligible[-1]
    return rows[-1]


def load_gepa_system_prompt(
    path: str,
    iteration: Optional[int] = None,
    rollouts: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """Load a GEPA-evolved system prompt from a history JSONL or a text file.

    GEPA (Agrawal et al., 2025, "Reflective Prompt Evolution Can Outperform
    Reinforcement Learning") searches over a candidate *instruction* with
    reflective genetic-Pareto mutation. The artefacts under
    ``baseline/gepa/*_best_system_prompt_history.jsonl`` log the incumbent
    best prompt: each line is
    ``{iteration, rollouts, best_idx, score, system_prompt, time, time_iso}``.

    The evolved instruction is the *entire* system prompt used at training
    time, so eval-time conditioning should replace StudyBench's default
    system message rather than appending to it (unlike ACE playbooks).

    Returns ``(prompt_text, meta)``. ``meta`` always includes ``source`` and
    ``format`` (``history`` or ``text``); history selections also record
    ``iteration`` / ``rollouts`` / ``score`` / ``best_idx``.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if not raw.strip():
        raise ValueError(f"GEPA prompt file is empty: {path}")

    rows = _parse_gepa_prompt_file(path, raw)
    if rows is None:
        if iteration is not None or rollouts is not None:
            raise ValueError(
                "--gepa_iteration / --gepa_rollouts require a GEPA history "
                f"JSONL (or a JSON object with those fields), not a plain-text "
                f"prompt: {path}"
            )
        return raw.strip(), {"source": path, "format": "text"}

    selected = _select_gepa_history_row(
        rows, iteration=iteration, rollouts=rollouts
    )
    prompt = selected.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"selected GEPA checkpoint has empty system_prompt: {path}"
        )
    meta = {
        "source": path,
        "format": "history",
        "iteration": selected.get("iteration"),
        "rollouts": selected.get("rollouts"),
        "score": selected.get("score"),
        "best_idx": selected.get("best_idx"),
        "time_iso": selected.get("time_iso"),
        "n_checkpoints": len(rows),
    }
    return prompt.strip(), meta


_ACE_PLAYBOOK_INSTRUCTION_EN = (
    "You also have access to a PLAYBOOK of strategies, formulas, common "
    "mistakes, and problem-solving heuristics curated from prior physics "
    "problem-solving experience. Each entry starts with a metadata prefix "
    "of the form `[bullet_id] helpful=N harmful=M ::` followed by the actual "
    "content; only the content is informative. Read it carefully and apply "
    "any bullets that are relevant to the current problem. If no bullet "
    "applies, fall back to your own reasoning. Never copy a bullet_id, "
    "`helpful=`, or `harmful=` counter into your reasoning or final answer; "
    "the playbook is for you to consult silently."
)

_ACE_PLAYBOOK_INSTRUCTION_ZH = (
    "你还可以查阅下方的 PLAYBOOK —— 一份从历史物理解题经验中沉淀下来的策略、"
    "公式、常见错误与解题启发。每条以 `[bullet_id] helpful=N harmful=M ::` 的"
    "元信息开头，后面才是真正有用的内容；只有内容是有信息量的。请仔细阅读，"
    "并把与当前题目相关的条目用上；如果没有合适条目，就按自己的推理来做。"
    "永远不要把 bullet_id、`helpful=`、`harmful=` 这些计数原文抄进你的推理或最终答案里 —— "
    "playbook 是供你私下查阅的。"
)

_GUIDANCE_PREAMBLE_EN = (
    "The following solving guidance is provided to help you approach this "
    "sub-question. Use it as a hint for relevant concepts and steps, but still "
    "derive the solution yourself and put the final answer in \\boxed{}."
)

_GUIDANCE_PREAMBLE_ZH = (
    "下面给出了本小题的解题指导，请把它当作思路提示来使用："
    "参考其中的相关概念与步骤，但仍需自行完成推导，并把最终答案写在 \\boxed{} 中。"
)


def _format_subproblem_guidance(guidance: str, language: str) -> str:
    """Wrap a raw guidance string with a short language-appropriate preamble."""
    preamble = (
        _GUIDANCE_PREAMBLE_ZH
        if language.lower() == "zh"
        else _GUIDANCE_PREAMBLE_EN
    )
    return f"## Solving Guidance\n\n{preamble}\n\n{guidance.strip()}"


def _build_system_content(
    language: str,
    ace_playbook: Optional[str],
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
) -> str:
    """Compose the system message, optionally injecting a method-specific prompt.

    Four mutually-exclusive modes:

    * **default** (no extras): return the original one-liner so prior runs
      are byte-for-byte reproducible.
    * **ACE playbook** (``ace_playbook`` set): keep the default base and
      append the language-appropriate ``## PLAYBOOK`` block; the playbook
      lives in the system role so multi-sub parents don't pay the token
      cost on every assistant turn.
    * **EvoSkill** (``evoskill_skills`` set): the supplied blob is the
      *entire* system prompt — it already carries the task description,
      constraints, and learned-skill checklists the evolved program was
      trained under. We deliberately drop the helpful-assistant filler so
      eval-time conditioning matches training-time conditioning.
    * **GEPA** (``gepa_system_prompt`` set): the evolved instruction is
      likewise the *entire* system prompt (GEPA searches over a candidate
      instruction, not a sidecar playbook). Replacing the default keeps
      eval-time conditioning aligned with GEPA training-time conditioning.

    The CLI rejects any pair of {ace, evoskill, gepa}, so this function
    asserts mutual exclusion as a defence-in-depth check.
    """
    n_set = sum(
        1
        for blob in (ace_playbook, evoskill_skills, gepa_system_prompt)
        if blob
    )
    if n_set > 1:
        raise ValueError(
            "ace_playbook, evoskill_skills, and gepa_system_prompt are "
            "mutually exclusive; the CLI layer should have caught this."
        )
    if evoskill_skills:
        return evoskill_skills.strip()
    if gepa_system_prompt:
        return gepa_system_prompt.strip()
    base = "You are a helpful assistant in solving physics problems."
    if not ace_playbook:
        return base
    instr = (
        _ACE_PLAYBOOK_INSTRUCTION_ZH
        if language.lower() == "zh"
        else _ACE_PLAYBOOK_INSTRUCTION_EN
    )
    return (
        f"{base}\n\n{instr}\n\n"
        f"## PLAYBOOK\n\n{ace_playbook.strip()}\n\n## END PLAYBOOK"
    )


def build_subproblem_messages(
    subproblems: list[dict],
    completions_list: list[dict],
    current_index: int,
    language: str,
    stem: str,
    ace_playbook: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
) -> list[dict]:
    """Build the chat history leading up to (and including) sub-question ``current_index``.

    Callers pass in already-normalised ``subproblems`` and ``stem`` (via
    :func:`_normalize_for_generation`). The ``stem`` argument is just the
    shared-preamble string sourced from ``parent.problem`` — there is no
    longer a separate ``stem`` field in the data. For single-question
    parents both arguments collapse so we never duplicate the problem
    text: ``subproblems == [{"problem": parent.problem}]`` and
    ``stem == ""``.

    Prior assistant turns are inserted with their ``<think>...</think>`` blocks
    stripped: the chat template will re-open ``<think>`` for the *current*
    turn anyway, and keeping reasoning history is what blows up the context
    window for multi-subquestion parents.

    ``ace_playbook`` (optional) is the text of an ACE-style playbook (e.g.
    the ``best_playbook.txt`` / ``epoch_*_step_*_playbook.txt`` artefact
    produced by ``baseline/ace``). When provided, it is appended to the
    system message via :func:`_build_system_content`; all other pieces
    of the prompt (sub-question instruction, stem, per-turn user content,
    prior assistant turns) are unchanged so the rest of the StudyBench
    pipeline (extraction, judging, pass@k) stays byte-for-byte aligned
    with the no-playbook baseline.

    ``evoskill_skills`` (optional) is the assembled system-prompt blob
    exported from an evolved EvoSkill program (see
    ``baseline/EvoSkill/scripts/export_program.py``). When provided it
    *replaces* the default system message (so eval-time conditioning
    matches the program's training-time conditioning). Mutually exclusive
    with ``ace_playbook`` / ``gepa_system_prompt``; the CLI layer enforces
    this.

    ``gepa_system_prompt`` (optional) is a GEPA-evolved instruction (see
    :func:`load_gepa_system_prompt`). Same replacement semantics as
    EvoSkill: the candidate prompt *is* the system message. Mutually
    exclusive with ``ace_playbook`` / ``evoskill_skills``.

    When a sub-question dict carries a non-empty ``guidance`` field (typically
    attached via ``--guidance_path`` / :func:`attach_guidance_to_items`), that
    text is appended to the corresponding user turn under a
    ``## Solving Guidance`` heading. Prior turns in the multi-sub chat
    history rebuild with the same guidance they originally saw, so resume
    and sequential prompting stay consistent.
    """
    instruction = _subproblem_instruction(language)
    messages: list[dict] = [
        {
            "role": "system",
            "content": _build_system_content(
                language,
                ace_playbook,
                evoskill_skills=evoskill_skills,
                gepa_system_prompt=gepa_system_prompt,
            ),
        }
    ]
    stem = (stem or "").strip()

    for turn_idx in range(current_index + 1):
        sub = subproblems[turn_idx] if turn_idx < len(subproblems) else {}
        sub_text = (sub.get("problem") or "").strip() if isinstance(sub, dict) else ""
        if turn_idx == 0:
            parts = [instruction]
            if stem:
                parts.append(stem)
            parts.append(sub_text)
            user_content = "\n\n".join(p for p in parts if p)
        else:
            user_content = sub_text
        guidance = ""
        if isinstance(sub, dict):
            raw_guidance = sub.get("guidance")
            if isinstance(raw_guidance, str):
                guidance = raw_guidance.strip()
        if guidance:
            user_content = (
                f"{user_content}\n\n{_format_subproblem_guidance(guidance, language)}"
                if user_content
                else _format_subproblem_guidance(guidance, language)
            )
        messages.append({"role": "user", "content": user_content})

        if turn_idx < current_index:
            prior_entry = completions_list[turn_idx] if turn_idx < len(completions_list) else {}
            prior_text = prior_entry.get("completion") if isinstance(prior_entry, dict) else None
            assistant_content = _strip_think_for_history(prior_text)
            if not assistant_content:
                assistant_content = _empty_assistant_placeholder(language)
            messages.append({"role": "assistant", "content": assistant_content})

    return messages


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _has_cached_completion(entry: Any) -> bool:
    """True iff ``entry`` already carries a usable model output.
    """
    if not isinstance(entry, dict):
        return False
    completion = entry.get("completion")
    return isinstance(completion, str) and bool(completion)


def _parent_resume_key(item: dict) -> tuple:
    """Build a composite resume key that disambiguates same-id problems.
    """
    return (item.get("source"), item.get("year"), item.get("source_problem_id"))


def _attempt_template(parent: dict, attempt_id: int) -> dict:
    """Return a fresh pass@k attempt for ``parent``."""
    return {
        "attempt_id": attempt_id,
        "completions": _completion_slot_template(parent),
    }


def _ensure_attempts(parent: dict, pass_k: int) -> list[dict]:
    """Lazily initialise ``parent['attempts']`` in the new pass@k format."""
    attempts = parent.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    by_id = {
        a.get("attempt_id"): a
        for a in attempts
        if isinstance(a, dict) and isinstance(a.get("attempt_id"), int)
    }

    normalized: list[dict] = []
    expected_len = len(_completion_slot_template(parent))
    for attempt_id in range(pass_k):
        attempt = by_id.get(attempt_id)
        if not isinstance(attempt, dict):
            attempt = _attempt_template(parent, attempt_id)
        completions = attempt.get("completions")
        if not isinstance(completions, list) or len(completions) != expected_len:
            completions = _completion_slot_template(parent)
        attempt["attempt_id"] = attempt_id
        attempt["completions"] = completions
        normalized.append(attempt)
    parent["attempts"] = normalized
    return normalized


def merge_existing_attempts(new_items: list[dict], existing_path: str, pass_k: int) -> int:
    """Copy cached attempts from an existing new-format result file."""
    if not os.path.exists(existing_path):
        return 0
    try:
        existing = read_json(existing_path)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[resume] could not read {existing_path}: {exc}")
        return 0

    by_key: dict = {}
    for prev in existing:
        if not isinstance(prev, dict):
            continue
        sid = prev.get("source_problem_id")
        if not sid:
            continue
        key = _parent_resume_key(prev)
        if key in by_key:
            print(
                f"[resume] warning: duplicate parent in {existing_path} for "
                f"key={key}; keeping the first occurrence."
            )
            continue
        by_key[key] = prev

    matched_parents = 0
    total_reused = 0
    for parent in new_items:
        if not parent.get("source_problem_id"):
            continue
        attempts = _ensure_attempts(parent, pass_k)
        prev = by_key.get(_parent_resume_key(parent))
        if not prev:
            continue
        prev_attempts = prev.get("attempts")
        if not isinstance(prev_attempts, list):
            continue
        prev_by_attempt = {
            a.get("attempt_id"): a for a in prev_attempts if isinstance(a, dict)
        }
        reused_here = False
        for attempt in attempts:
            cached_attempt = prev_by_attempt.get(attempt["attempt_id"])
            if not isinstance(cached_attempt, dict):
                continue
            cached_completions = cached_attempt.get("completions")
            if not isinstance(cached_completions, list):
                continue
            prev_by_pid = {
                c.get("problem_id"): c
                for c in cached_completions
                if isinstance(c, dict)
            }
            merged: list[dict] = []
            for slot in attempt["completions"]:
                cached = prev_by_pid.get(slot.get("problem_id"))
                if _has_cached_completion(cached):
                    merged.append(cached)
                    reused_here = True
                    total_reused += 1
                else:
                    merged.append(slot)
            attempt["completions"] = merged
        if reused_here:
            matched_parents += 1

    if matched_parents:
        print(
            f"[resume] reused {total_reused} attempt/sub-question completion(s) across "
            f"{matched_parents} parent item(s)."
        )
    return matched_parents


# Judge-side fields that ``eval.eval_file`` writes onto every completion. We
# carry these (and only these) over when ``--resume_eval`` is on so a prior
# eval's verdicts survive into the next run; non-judge metadata stays
# whatever ``generated_path`` had.
_JUDGE_RESUME_FIELDS: tuple[str, ...] = (
    "correctness",
    "model_judge_msg",
    "extracted_answer",
    "normalized_gt",
)


def merge_existing_eval(data_list: list[dict], existing_path: str) -> int:
    """Copy judge fields from a prior ``<tag>_eval.json`` onto ``data_list``.

    Used by ``--resume_eval``: read the previously-written eval file and,
    for any completion that shares the same parent-resume-key, attempt id,
    and ``problem_id``, copy ``correctness`` / ``model_judge_msg`` /
    ``extracted_answer`` / ``normalized_gt`` onto the matching slot in
    ``data_list``. The judge loop in :func:`eval.eval_file` then
    short-circuits trusted completed results, so resumed runs only spend
    judge calls on the not-yet-judged subset. Judge infrastructure failures
    such as quota exhaustion are intentionally not reused, even when an older
    file recorded them as ``correctness: false``.

    Returns the number of completions for which judge fields were reused.
    """
    if not os.path.exists(existing_path):
        return 0
    try:
        existing = read_json(existing_path)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[resume_eval] could not read {existing_path}: {exc}")
        return 0
    if isinstance(existing, dict) and isinstance(existing.get("data"), list):
        existing = existing["data"]
    if not isinstance(existing, list):
        print(f"[resume_eval] {existing_path} is not a JSON list; skipping resume.")
        return 0

    by_key: dict = {}
    for prev in existing:
        if not isinstance(prev, dict):
            continue
        by_key.setdefault(_parent_resume_key(prev), prev)

    total_reused = 0
    total_skipped_retryable = 0
    matched_parents = 0
    for parent in data_list:
        prev = by_key.get(_parent_resume_key(parent))
        if not isinstance(prev, dict):
            continue
        prev_attempts = prev.get("attempts")
        if not isinstance(prev_attempts, list):
            continue
        prev_by_attempt = {
            a.get("attempt_id"): a
            for a in prev_attempts
            if isinstance(a, dict)
        }
        attempts = parent.get("attempts") or []
        reused_here = False
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            cached_attempt = prev_by_attempt.get(attempt.get("attempt_id"))
            if not isinstance(cached_attempt, dict):
                continue
            cached_completions = cached_attempt.get("completions")
            if not isinstance(cached_completions, list):
                continue
            prev_by_pid = {
                c.get("problem_id"): c
                for c in cached_completions
                if isinstance(c, dict)
            }
            for slot in attempt.get("completions") or []:
                if not isinstance(slot, dict):
                    continue
                cached = prev_by_pid.get(slot.get("problem_id"))
                if not isinstance(cached, dict):
                    continue
                if not eval.has_reusable_judge_result(cached):
                    if (
                        isinstance(cached.get("correctness"), bool)
                        and eval.is_retryable_judge_error(cached.get("model_judge_msg"))
                    ):
                        total_skipped_retryable += 1
                    continue
                for field in _JUDGE_RESUME_FIELDS:
                    if field in cached:
                        slot[field] = cached[field]
                total_reused += 1
                reused_here = True
        if reused_here:
            matched_parents += 1

    if matched_parents:
        print(
            f"[resume_eval] reused {total_reused} judged completion(s) across "
            f"{matched_parents} parent(s) from {existing_path}"
        )
        if total_skipped_retryable:
            print(
                f"[resume_eval] skipped {total_skipped_retryable} cached judge "
                "failure(s) that should be retried."
            )
    else:
        print(
            f"[resume_eval] {existing_path} found but no matching judged "
            f"completions to reuse."
        )
        if total_skipped_retryable:
            print(
                f"[resume_eval] skipped {total_skipped_retryable} cached judge "
                "failure(s) that should be retried."
            )
    return total_reused


# ---------------------------------------------------------------------------
# API generation (sequential per parent, threaded across parents)
# ---------------------------------------------------------------------------

def _create_openai_client(request_timeout: float = 1800.0):
    import httpx
    from openai import OpenAI

    timeout = httpx.Timeout(request_timeout, connect=10.0)
    http_client = httpx.Client(verify=False, timeout=timeout)
    return OpenAI(http_client=http_client, timeout=request_timeout, max_retries=0)


def _prompt_token_len(tokenizer, messages: list[dict]) -> Optional[int]:
    """Token length of ``messages`` under ``tokenizer``'s chat template.

    Mirrors how :func:`generate_with_vllm` measures prompts: render the chat
    with ``add_generation_prompt=True`` and count the tokens. Returns
    ``None`` when no tokenizer is available (so callers fall back to the
    unreduced ``max_tokens``).

    We deliberately render with ``tokenize=False`` and then encode the
    string, exactly like the vLLM path. The obvious ``tokenize=True``
    shortcut is *not* version-robust: on transformers >= 5.x
    ``apply_chat_template(tokenize=True)`` returns a ``BatchEncoding`` whose
    ``len()`` is the number of keys (2), not the token count -- which
    silently made this helper report a 2-token prompt for every request,
    defeating the budgeting and overflowing the server's context window.
    """
    if tokenizer is None:
        return None
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer(rendered, add_special_tokens=False).input_ids)
    except Exception:
        return None


def _process_one_parent(
    parent: dict,
    client,
    model: str,
    temperature: float,
    max_tokens: int,
    language: str,
    pass_k: int,
    state: PersistentState,
    quirks: dict,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    progress: Optional[tqdm] = None,
    ace_playbook: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
    tokenizer=None,
    max_completion_tokens: Optional[int] = None,
) -> None:
    # ``subproblems`` and ``stem`` are the *prompt-facing* views, both
    # built from ``parent.problem`` (the only schema field that still
    # carries question text): for single-question parents the parent's
    # problem becomes ``subproblems[0]['problem']`` and ``stem`` is
    # blanked, killing the duplication the old single-sub-wrapped schema
    # used to produce. Each attempt has an eval-facing ``completions``
    # list with the same length, carrying the gold answer / answer_type
    # needed by the judger.
    subproblems, stem = _normalize_for_generation(parent)
    attempts = _ensure_attempts(parent, pass_k)

    for attempt in attempts:
        completions_list = attempt["completions"]
        for i, sub in enumerate(subproblems):
            if _has_cached_completion(completions_list[i]):
                continue

            messages = build_subproblem_messages(
                subproblems=subproblems,
                completions_list=completions_list,
                current_index=i,
                language=language,
                stem=stem,
                ace_playbook=ace_playbook,
                evoskill_skills=evoskill_skills,
                gepa_system_prompt=gepa_system_prompt,
            )

            # Mirror the vLLM path: subtract this prompt's token length from
            # the given ``max_tokens`` budget so prompt+output never exceeds it
            # (which is what triggers the server's context-length 400). When no
            # tokenizer is available we fall back to the full ``max_tokens``.
            prompt_len = _prompt_token_len(tokenizer, messages)
            if prompt_len is not None:
                req_max_tokens = max(1, max_tokens - prompt_len)
            else:
                req_max_tokens = max_tokens
            if max_completion_tokens is not None:
                req_max_tokens = min(req_max_tokens, max_completion_tokens)

            # Send normal params first and, on a 400 whose message calls out
            # ``temperature`` or ``max_tokens``, flip the matching flag in the
            # shared ``quirks`` dict and retry.
            raw: dict = {}
            last_exc: Optional[BaseException] = None
            for _ in range(3):
                kwargs: dict = {
                    "model": model,
                    "messages": messages,
                    "n": 1,
                    "presence_penalty": FIXED_PRESENCE_PENALTY,
                }
                if quirks.get("use_max_completion_tokens"):
                    kwargs["max_completion_tokens"] = req_max_tokens
                else:
                    kwargs["max_tokens"] = req_max_tokens
                if not quirks.get("drop_temperature"):
                    kwargs["temperature"] = temperature
                if top_p is not None:
                    kwargs["top_p"] = top_p
                extra_body: dict[str, Any] = {
                    "min_p": FIXED_MIN_P,
                    "repetition_penalty": FIXED_REPETITION_PENALTY,
                }
                if top_k is not None:
                    extra_body["top_k"] = top_k
                kwargs["extra_body"] = extra_body

                try:
                    response = client.chat.completions.create(**kwargs)
                    raw = response.model_dump()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    status = (
                        getattr(exc, "status_code", None)
                        or getattr(exc, "http_status", None)
                    )
                    if status != 400:
                        break
                    msg_lower = str(exc).lower()
                    mutated = False
                    if "temperature" in msg_lower and not quirks.get("drop_temperature"):
                        quirks["drop_temperature"] = True
                        mutated = True
                    if (
                        ("max_tokens" in msg_lower or "max_completion_tokens" in msg_lower)
                        and not quirks.get("use_max_completion_tokens")
                    ):
                        quirks["use_max_completion_tokens"] = True
                        mutated = True
                    if not mutated:
                        break
                    print(
                        f"[api] {model} returned 400 ({exc}); adjusting quirks to "
                        f"drop_temperature={bool(quirks.get('drop_temperature'))}, "
                        f"use_max_completion_tokens="
                        f"{bool(quirks.get('use_max_completion_tokens'))} and retrying.",
                        flush=True,
                    )

            if last_exc is not None:
                print(
                    f"[api] {parent.get('source_problem_id')} "
                    f"attempt {attempt['attempt_id']} {sub.get('problem_id')} failed: "
                    f"{type(last_exc).__name__}: {last_exc}",
                    flush=True,
                )
                raw = {"error": f"{type(last_exc).__name__}: {last_exc}"}

            completion_text = ""
            if isinstance(raw, dict):
                choices = raw.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        completion_text = msg["content"]

            entry = copy.deepcopy(completions_list[i])
            entry["raw_response"] = raw
            entry["prompt_messages"] = messages
            entry["completion"] = completion_text
            completions_list[i] = entry
            state.save_async()
            if progress is not None:
                # tqdm.update is already thread-safe; no external lock needed.
                progress.update(1)


def generate_with_api(
    items: list[dict],
    model: str,
    nproc: int,
    temperature: float,
    max_tokens: int,
    language: str,
    pass_k: int,
    output_path: str,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    ace_playbook: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    max_completion_tokens: Optional[int] = None,
) -> None:
    state = PersistentState(items, output_path)
    if not items:
        state.save()
        return

    # Load a tokenizer so each request can subtract its prompt length from
    # ``max_tokens`` (same idea as the vLLM path), keeping prompt+output under
    # the server's context window. ``tokenizer_path`` defaults to the served
    # ``model`` name; if it can't be loaded (e.g. an API alias that is not a
    # local path / HF id) we log once and fall back to the full ``max_tokens``.
    tokenizer = None
    tok_src = tokenizer_path or model
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        print(f"[api] loaded tokenizer from {tok_src!r} for prompt-length budgeting.")
    except Exception as exc:
        print(
            f"[api] could not load a tokenizer from {tok_src!r} "
            f"({type(exc).__name__}: {exc}); requests will use the full "
            f"max_tokens={max_tokens} without prompt-length budgeting. "
            f"Pass --tokenizer_path to enable it.",
            flush=True,
        )

    cursor = [0]
    cursor_lock = threading.Lock()

    for parent in items:
        _ensure_attempts(parent, pass_k)

    # Progress is tracked at attempt/sub-question granularity.
    total_subs = sum(len(_completion_slot_template(p)) * pass_k for p in items)
    cached_subs = sum(
        1
        for p in items
        for attempt in (p.get("attempts") or [])
        for slot in (attempt.get("completions") or [])
        if _has_cached_completion(slot)
    )
    progress = tqdm(total=total_subs, initial=cached_subs, desc="gen-attempt-sub-questions")

    # Shared across workers: once any request discovers the server does not
    # accept ``temperature`` (or insists on ``max_completion_tokens`` over
    # ``max_tokens``) we flip these flags and every subsequent call skips
    # straight to the adjusted payload. Bool-set-to-True is GIL-atomic and
    # idempotent, so no explicit lock is needed.
    quirks: dict = {"drop_temperature": False, "use_max_completion_tokens": False}

    def worker() -> None:
        client = _create_openai_client()
        while True:
            with cursor_lock:
                if cursor[0] >= len(items):
                    return
                parent = items[cursor[0]]
                cursor[0] += 1
            try:
                _process_one_parent(
                    parent=parent,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_completion_tokens=max_completion_tokens,
                    language=language,
                    pass_k=pass_k,
                    state=state,
                    quirks=quirks,
                    top_p=top_p,
                    top_k=top_k,
                    progress=progress,
                    ace_playbook=ace_playbook,
                    evoskill_skills=evoskill_skills,
                    gepa_system_prompt=gepa_system_prompt,
                    tokenizer=tokenizer,
                )
            except Exception as exc:  # pragma: no cover - worker-level guard
                print(
                    f"[api] worker error on {parent.get('source_problem_id')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    worker_count = max(1, min(nproc, len(items)))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        time.sleep(0.05)
        t.start()
    for t in threads:
        t.join()
    progress.close()
    state.save()


# ---------------------------------------------------------------------------
# Anthropic (/v1/messages) generation
# ---------------------------------------------------------------------------
#
# Some gateways (e.g. llm-center) expose Claude models *only* through the
# native Anthropic Messages protocol and reject the OpenAI
# ``/v1/chat/completions`` schema with a 400. This backend speaks that
# protocol directly via raw httpx so we don't take a hard dependency on the
# ``anthropic`` SDK. It reuses the same prompt-building / resume / pass@k
# scaffolding as the OpenAI path; only the transport differs.

def _anthropic_endpoint() -> Tuple[str, str]:
    """Resolve the ``/v1/messages`` URL and API key from the environment.

    The base URL is taken from ``ANTHROPIC_BASE_URL`` (preferred) or
    ``OPENAI_BASE_URL`` so the existing ``env_local.sh`` setup keeps working.
    ``/messages`` is appended unless the base already ends with it.
    """
    base = (
        os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")
    if not base:
        raise ValueError(
            "The anthropic backend needs ANTHROPIC_BASE_URL or OPENAI_BASE_URL "
            "set (e.g. https://llm-center.ali.modelbest.cn/llm/v1)."
        )
    url = base if base.endswith("/messages") else f"{base}/messages"
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    return url, key


def _split_system_messages(messages: list[dict]) -> Tuple[Optional[str], list[dict]]:
    """Split an OpenAI-style message list into Anthropic ``(system, messages)``.

    Anthropic carries the system prompt as a top-level ``system`` field
    rather than a ``role: system`` entry, and only accepts ``user`` /
    ``assistant`` turns in the ``messages`` array.
    """
    system_parts: list[str] = []
    conv: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        conv.append({"role": role, "content": content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, conv


def _create_anthropic_http_client(request_timeout: float = 300.0):
    import httpx

    timeout = httpx.Timeout(request_timeout, connect=10.0)
    return httpx.Client(verify=False, timeout=timeout)


def _anthropic_completion_text(raw: Any) -> str:
    """Concatenate the ``text`` blocks from an Anthropic Messages response."""
    if not isinstance(raw, dict):
        return ""
    content = raw.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _process_one_parent_anthropic(
    parent: dict,
    http_client,
    url: str,
    headers: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    language: str,
    pass_k: int,
    state: PersistentState,
    quirks: dict,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    progress: Optional[tqdm] = None,
    ace_playbook: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
) -> None:
    subproblems, stem = _normalize_for_generation(parent)
    attempts = _ensure_attempts(parent, pass_k)

    for attempt in attempts:
        completions_list = attempt["completions"]
        for i, sub in enumerate(subproblems):
            if _has_cached_completion(completions_list[i]):
                continue

            messages = build_subproblem_messages(
                subproblems=subproblems,
                completions_list=completions_list,
                current_index=i,
                language=language,
                stem=stem,
                ace_playbook=ace_playbook,
                evoskill_skills=evoskill_skills,
                gepa_system_prompt=gepa_system_prompt,
            )
            system, conv = _split_system_messages(messages)

            raw: dict = {}
            last_exc: Optional[BaseException] = None
            for _ in range(3):
                body: dict = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": conv,
                }
                if system:
                    body["system"] = system
                if not quirks.get("drop_temperature"):
                    body["temperature"] = temperature
                if top_p is not None and not quirks.get("drop_top_p"):
                    body["top_p"] = top_p
                if top_k is not None:
                    body["top_k"] = top_k

                try:
                    resp = http_client.post(url, headers=headers, json=body)
                    if resp.status_code >= 400:
                        try:
                            err_payload = resp.json()
                        except Exception:
                            err_payload = {"raw_text": resp.text}
                        raise RuntimeError(
                            f"HTTP {resp.status_code}: {err_payload}"
                        )
                    raw = resp.json()
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    msg_lower = str(exc).lower()
                    mutated = False
                    # Anthropic recommends not setting temperature and top_p
                    # together; on a 400 calling out either, drop the offending
                    # one and retry. (vLLM-only knobs are never sent here.)
                    if (
                        "temperature" in msg_lower
                        and not quirks.get("drop_temperature")
                    ):
                        quirks["drop_temperature"] = True
                        mutated = True
                    if "top_p" in msg_lower and not quirks.get("drop_top_p"):
                        quirks["drop_top_p"] = True
                        mutated = True
                    if not mutated:
                        break
                    print(
                        f"[anthropic] {model} returned 400 ({exc}); adjusting "
                        f"drop_temperature={bool(quirks.get('drop_temperature'))}, "
                        f"drop_top_p={bool(quirks.get('drop_top_p'))} and retrying.",
                        flush=True,
                    )

            if last_exc is not None:
                print(
                    f"[anthropic] {parent.get('source_problem_id')} "
                    f"attempt {attempt['attempt_id']} {sub.get('problem_id')} failed: "
                    f"{type(last_exc).__name__}: {last_exc}",
                    flush=True,
                )
                raw = {"error": f"{type(last_exc).__name__}: {last_exc}"}

            completion_text = _anthropic_completion_text(raw)

            entry = copy.deepcopy(completions_list[i])
            entry["raw_response"] = raw
            entry["prompt_messages"] = messages
            entry["completion"] = completion_text
            completions_list[i] = entry
            state.save_async()
            if progress is not None:
                progress.update(1)


def generate_with_anthropic(
    items: list[dict],
    model: str,
    nproc: int,
    temperature: float,
    max_tokens: int,
    language: str,
    pass_k: int,
    output_path: str,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    ace_playbook: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
) -> None:
    state = PersistentState(items, output_path)
    if not items:
        state.save()
        return

    url, api_key = _anthropic_endpoint()
    headers = {
        # Different gateways accept different auth headers; send both the
        # native Anthropic ``x-api-key`` and an OpenAI-style bearer token so
        # the same key works whether the upstream is real Anthropic or a
        # translating proxy.
        "x-api-key": api_key,
        "authorization": f"Bearer {api_key}",
        "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        "content-type": "application/json",
    }
    print(f"[anthropic] POST {url}")

    cursor = [0]
    cursor_lock = threading.Lock()

    for parent in items:
        _ensure_attempts(parent, pass_k)

    total_subs = sum(len(_completion_slot_template(p)) * pass_k for p in items)
    cached_subs = sum(
        1
        for p in items
        for attempt in (p.get("attempts") or [])
        for slot in (attempt.get("completions") or [])
        if _has_cached_completion(slot)
    )
    progress = tqdm(total=total_subs, initial=cached_subs, desc="gen-attempt-sub-questions")

    quirks: dict = {"drop_temperature": False, "drop_top_p": False}

    def worker() -> None:
        client = _create_anthropic_http_client()
        while True:
            with cursor_lock:
                if cursor[0] >= len(items):
                    return
                parent = items[cursor[0]]
                cursor[0] += 1
            try:
                _process_one_parent_anthropic(
                    parent=parent,
                    http_client=client,
                    url=url,
                    headers=headers,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    language=language,
                    pass_k=pass_k,
                    state=state,
                    quirks=quirks,
                    top_p=top_p,
                    top_k=top_k,
                    progress=progress,
                    ace_playbook=ace_playbook,
                    evoskill_skills=evoskill_skills,
                    gepa_system_prompt=gepa_system_prompt,
                )
            except Exception as exc:  # pragma: no cover - worker-level guard
                print(
                    f"[anthropic] worker error on {parent.get('source_problem_id')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    worker_count = max(1, min(nproc, len(items)))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        time.sleep(0.05)
        t.start()
    for t in threads:
        t.join()
    progress.close()
    state.save()


# ---------------------------------------------------------------------------
# Claude Code (agentic harness) generation
# ---------------------------------------------------------------------------
#
# Unlike the api/anthropic/vllm backends -- which do a single stateless
# completion per sub-question -- this backend runs each sub-question through a
# full Claude Code agent (via ``claude-agent-sdk``), so the model gets real
# tools (Bash/Read/Write/...). That is what lets "opus + claude code"-style
# evolved methods (EvoSkill and friends) run under the same harness they were
# evolved in, instead of in a tool-less chat where their tool assumptions
# silently break.
#
# The method's conditioning is supplied *generically* -- the backend never
# mentions any specific method:
#   * ``--agent_project_dir``: a Claude Code project dir whose ``.claude/``
#     (skills, settings, MCP) and ``CLAUDE.md`` are loaded natively per call;
#   * ``--system_prompt_append_file``: text appended to the ``claude_code``
#     preset's system prompt.
# Any "opus + claude code" evolve method plugs in by pointing these at its own
# artefacts.
#
# Conversation handling follows "scheme B": each sub-question is an
# independent, stateless agent run. The StudyBench chat history (built by
# ``build_subproblem_messages``, with ``<think>`` stripped and placeholders
# inserted) is flattened into a single query string, preserving the exact
# StudyBench prompting protocol so results stay comparable to the other
# backends.

_AGENT_DEFAULT_TOOLS: Tuple[str, ...] = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite", "BashOutput",
)


def _flatten_conversation_for_query(messages: list[dict]) -> str:
    """Fold an OpenAI-style message list into a single Claude Code query.

    The system turn is dropped (the ``claude_code`` preset + optional append
    own the system prompt); user turns are emitted verbatim and prior
    assistant turns are labelled so the agent can tell its own earlier
    answers apart from the new sub-question. The final user turn is the
    current sub-question.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role == "system":
            continue
        if role == "assistant":
            parts.append(f"[Your previous answer]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def _prepare_agent_cwd(agent_project_dir: Optional[str]) -> str:
    """Make a fresh scratch cwd, seeding it with the method's project config.

    Scheme B keeps sub-questions stateless, so every call gets a throwaway
    working directory. When a method project dir is supplied we copy its
    Claude Code config (``.claude/``, ``CLAUDE.md``, ``.mcp.json``) into the
    scratch dir so the agent loads the method's skills / memory / MCP natively
    while any files it writes stay isolated and are discarded afterwards.
    """
    workdir = tempfile.mkdtemp(prefix="studybench_cc_")
    if agent_project_dir:
        for name in (".claude", "CLAUDE.md", ".mcp.json"):
            src = os.path.join(agent_project_dir, name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(workdir, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    return workdir


def _build_agent_options(
    *,
    system_append: Optional[str],
    cwd: str,
    tools: list[str],
    permission_mode: str,
    model: Optional[str],
):
    """Assemble ``ClaudeAgentOptions`` for one agent run."""
    from claude_agent_sdk import ClaudeAgentOptions

    system_prompt: dict[str, Any] = {"type": "preset", "preset": "claude_code"}
    if system_append:
        system_prompt["append"] = system_append
    opts = ClaudeAgentOptions(
        system_prompt=system_prompt,
        allowed_tools=list(tools),
        permission_mode=permission_mode,
        # "user" pulls ~/.claude/settings.json (the llm-center routing +
        # CLAUDE_CODE_MODEL); "project" pulls the method's seeded .claude.
        setting_sources=["user", "project"],
        cwd=cwd,
        max_buffer_size=10 * 1024 * 1024,
    )
    if model:
        opts.model = model
    return opts


async def _run_one_agent_query(
    *,
    query: str,
    system_append: Optional[str],
    tools: list[str],
    permission_mode: str,
    model: Optional[str],
    agent_project_dir: Optional[str],
) -> dict:
    """Run a single sub-question through a Claude Code agent; return a raw dict.

    Captures the agent's final text (``ResultMessage.result``) -- the boxed
    final answer lives there, so the downstream judge / ``\\boxed{}``
    extraction is unchanged. The scratch cwd is always cleaned up.
    """
    from claude_agent_sdk import ClaudeSDKClient

    cwd = _prepare_agent_cwd(agent_project_dir)
    try:
        opts = _build_agent_options(
            system_append=system_append,
            cwd=cwd,
            tools=tools,
            permission_mode=permission_mode,
            model=model,
        )
        async with ClaudeSDKClient(opts) as client:
            await client.query(query)
            messages = [m async for m in client.receive_response()]
        last = messages[-1] if messages else None
        text = getattr(last, "result", None) or ""
        usage = getattr(last, "usage", None)
        return {
            "backend": "claude_code",
            "model": model,
            "result_text": text,
            "is_error": bool(getattr(last, "is_error", False)),
            "num_turns": getattr(last, "num_turns", None),
            "total_cost_usd": getattr(last, "total_cost_usd", None),
            "usage": usage if isinstance(usage, dict) else (
                None if usage is None else str(usage)
            ),
            "session_id": getattr(last, "session_id", None),
        }
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def _process_one_parent_claude_code(
    parent: dict,
    *,
    language: str,
    pass_k: int,
    state: PersistentState,
    system_append: Optional[str],
    tools: list[str],
    permission_mode: str,
    model: Optional[str],
    agent_project_dir: Optional[str],
    progress: Optional[tqdm] = None,
) -> None:
    subproblems, stem = _normalize_for_generation(parent)
    attempts = _ensure_attempts(parent, pass_k)

    for attempt in attempts:
        completions_list = attempt["completions"]
        for i, sub in enumerate(subproblems):
            if _has_cached_completion(completions_list[i]):
                continue

            # ace_playbook / evoskill_skills / gepa_system_prompt are
            # intentionally NOT plumbed through here: under the Claude Code
            # harness the method's conditioning arrives via
            # agent_project_dir / system_append, so this backend stays
            # method-agnostic.
            messages = build_subproblem_messages(
                subproblems=subproblems,
                completions_list=completions_list,
                current_index=i,
                language=language,
                stem=stem,
                ace_playbook=None,
                evoskill_skills=None,
                gepa_system_prompt=None,
            )
            query = _flatten_conversation_for_query(messages)

            raw: dict = {}
            last_exc: Optional[BaseException] = None
            for _ in range(2):
                try:
                    raw = asyncio.run(_run_one_agent_query(
                        query=query,
                        system_append=system_append,
                        tools=tools,
                        permission_mode=permission_mode,
                        model=model,
                        agent_project_dir=agent_project_dir,
                    ))
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc

            if last_exc is not None:
                print(
                    f"[claude_code] {parent.get('source_problem_id')} "
                    f"attempt {attempt['attempt_id']} {sub.get('problem_id')} failed: "
                    f"{type(last_exc).__name__}: {last_exc}",
                    flush=True,
                )
                raw = {
                    "backend": "claude_code",
                    "error": f"{type(last_exc).__name__}: {last_exc}",
                }

            completion_text = raw.get("result_text", "") if isinstance(raw, dict) else ""

            entry = copy.deepcopy(completions_list[i])
            entry["raw_response"] = raw
            entry["prompt_messages"] = messages
            entry["completion"] = completion_text
            completions_list[i] = entry
            state.save_async()
            if progress is not None:
                progress.update(1)


def generate_with_claude_code(
    items: list[dict],
    *,
    nproc: int,
    language: str,
    pass_k: int,
    output_path: str,
    system_append: Optional[str] = None,
    agent_project_dir: Optional[str] = None,
    tools: Optional[list[str]] = None,
    permission_mode: str = "bypassPermissions",
    model: Optional[str] = None,
) -> None:
    state = PersistentState(items, output_path)
    if not items:
        state.save()
        return

    # The claude CLI refuses to launch nested inside another Claude Code
    # session (it checks CLAUDECODE). The eval process itself is not a session
    # that needs it, so drop it here; child agents then inherit an env without
    # it. A no-op when the benchmark is run from a plain shell.
    os.environ.pop("CLAUDECODE", None)

    tools = list(tools) if tools else list(_AGENT_DEFAULT_TOOLS)
    for parent in items:
        _ensure_attempts(parent, pass_k)

    total_subs = sum(len(_completion_slot_template(p)) * pass_k for p in items)
    cached_subs = sum(
        1
        for p in items
        for attempt in (p.get("attempts") or [])
        for slot in (attempt.get("completions") or [])
        if _has_cached_completion(slot)
    )
    progress = tqdm(total=total_subs, initial=cached_subs, desc="gen-attempt-sub-questions")
    print(
        f"[claude_code] tools={tools} permission={permission_mode} "
        f"project_dir={agent_project_dir or '<none>'} "
        f"append={'yes' if system_append else 'no'} "
        f"model={model or '<settings>'}",
        flush=True,
    )

    cursor = [0]
    cursor_lock = threading.Lock()

    def worker() -> None:
        while True:
            with cursor_lock:
                if cursor[0] >= len(items):
                    return
                parent = items[cursor[0]]
                cursor[0] += 1
            try:
                _process_one_parent_claude_code(
                    parent,
                    language=language,
                    pass_k=pass_k,
                    state=state,
                    system_append=system_append,
                    tools=tools,
                    permission_mode=permission_mode,
                    model=model,
                    agent_project_dir=agent_project_dir,
                    progress=progress,
                )
            except Exception as exc:  # pragma: no cover - worker-level guard
                print(
                    f"[claude_code] worker error on {parent.get('source_problem_id')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    worker_count = max(1, min(nproc, len(items)))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        time.sleep(0.05)
        t.start()
    for t in threads:
        t.join()
    progress.close()
    state.save()


# ---------------------------------------------------------------------------
# Local (vLLM) generation - stepwise batching across parents
# ---------------------------------------------------------------------------

def _record_vllm_error(
    attempt: dict,
    ridx: int,
    messages: list[dict],
    err: BaseException,
) -> None:
    """Mirror the API path's error slot: keep generation moving but record why."""
    entry = copy.deepcopy(attempt["completions"][ridx])
    entry["raw_response"] = {
        "backend": "vllm",
        "error": f"{type(err).__name__}: {err}",
    }
    entry["prompt_messages"] = messages
    entry["completion"] = ""
    attempt["completions"][ridx] = entry


def generate_with_vllm(
    items: list[dict],
    model_path: str,
    tensor_parallel_size: int,
    temperature: float,
    max_tokens: int,
    language: str,
    pass_k: int,
    output_path: str,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    ace_playbook: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    state = PersistentState(items, output_path)
    if not items:
        state.save()
        return

    for parent in items:
        _ensure_attempts(parent, pass_k)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        swap_space=32,
        gpu_memory_utilization=0.8,
        max_model_len=max_tokens,
    )
    # ``max_model_len`` is the total length budget shared by prompt+output.
    # The per-batch sampling ``max_tokens`` is derived from it below by
    # subtracting the longest prompt in that batch (see the generation loop),
    # so we keep the base sampling kwargs here without a fixed ``max_tokens``.
    max_model_len = max_tokens
    base_sampling_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p if top_p is not None else 1.0,
        "min_p": FIXED_MIN_P,
        "presence_penalty": FIXED_PRESENCE_PENALTY,
        "repetition_penalty": FIXED_REPETITION_PENALTY,
        "n": 1,
    }
    if top_k is not None:
        base_sampling_kwargs["top_k"] = top_k

    # Attempt/sub-question level progress, identical to the API path.
    parent_slot_lens = [len(_completion_slot_template(p)) for p in items]
    total_subs = sum(parent_slot_lens) * pass_k
    cached_subs = sum(
        1
        for p in items
        for attempt in (p.get("attempts") or [])
        for slot in (attempt.get("completions") or [])
        if _has_cached_completion(slot)
    )
    progress = tqdm(total=total_subs, initial=cached_subs, desc="gen-attempt-sub-questions")

    # Pre-compute the prompt-facing view (subproblems, shared_stem) per
    # parent so we don't recompute it every round; ``items`` and
    # ``parent_slot_lens`` stay aligned. The shared-stem string is
    # ``parent.problem`` for multi-sub parents and ``""`` for solo.
    parent_views = [_normalize_for_generation(p) for p in items]

    max_rounds = max(parent_slot_lens, default=0)
    try:
        for attempt_idx in range(pass_k):
            for round_idx in range(max_rounds):
                # Step 1: collect this attempt/round's batch.
                batch: list[Tuple[dict, dict, int, int, list[dict], str]] = []
                for parent, (subproblems, stem) in zip(items, parent_views):
                    if round_idx >= len(subproblems):
                        continue
                    attempt = parent["attempts"][attempt_idx]
                    completions_list = attempt["completions"]
                    if _has_cached_completion(completions_list[round_idx]):
                        continue
                    messages = build_subproblem_messages(
                        subproblems=subproblems,
                        completions_list=completions_list,
                        current_index=round_idx,
                        language=language,
                        stem=stem,
                        ace_playbook=ace_playbook,
                        evoskill_skills=evoskill_skills,
                        gepa_system_prompt=gepa_system_prompt,
                    )
                    try:
                        rendered = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                    except Exception as exc:
                        print(
                            f"[local] {parent.get('source_problem_id')} "
                            f"attempt {attempt_idx} round {round_idx} template render failed: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        _record_vllm_error(attempt, round_idx, messages, exc)
                        progress.update(1)
                        continue
                    batch.append((parent, attempt, attempt_idx, round_idx, messages, rendered))

                if not batch:
                    state.save_async()
                    continue

                print(
                    f"[local] attempt {attempt_idx} round {round_idx}: "
                    f"generating {len(batch)} sub-question(s)",
                    flush=True,
                )

                # vLLM rejects a request whose prompt_len + max_tokens exceeds
                # ``max_model_len``. Because the whole batch shares one
                # ``SamplingParams.max_tokens``, size it against the *longest*
                # prompt in this batch: subtract that prompt's token count from
                # the model-length budget so even the longest prompt leaves room
                # to generate, leaving at least 1 token.
                prompt_token_lens = [
                    len(tokenizer(entry[5], add_special_tokens=False).input_ids)
                    for entry in batch
                ]
                max_prompt_len = max(prompt_token_lens) if prompt_token_lens else 0
                batch_max_tokens = max(1, max_model_len - max_prompt_len)
                if batch_max_tokens < max_model_len:
                    print(
                        f"[local] attempt {attempt_idx} round {round_idx}: "
                        f"longest prompt {max_prompt_len} tok -> capping "
                        f"sampling max_tokens at {batch_max_tokens} "
                        f"(max_model_len={max_model_len})",
                        flush=True,
                    )
                sampling = SamplingParams(
                    **base_sampling_kwargs, max_tokens=batch_max_tokens
                )

                try:
                    outputs = llm.generate([entry[5] for entry in batch], sampling)
                except Exception as exc:
                    print(
                        f"[local] llm.generate failed at attempt {attempt_idx} "
                        f"round {round_idx}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    for _parent, attempt, _aidx, ridx, messages, _rendered in batch:
                        _record_vllm_error(attempt, ridx, messages, exc)
                        progress.update(1)
                    state.save_async()
                    continue

                # Step 3: stitch outputs back into completion slots.
                for (_parent, attempt, _aidx, ridx, messages, _rendered), output in zip(batch, outputs):
                    if not output.outputs:
                        text = ""
                        finish_reason = None
                    else:
                        text = output.outputs[0].text
                        finish_reason = getattr(output.outputs[0], "finish_reason", None)
                    entry = copy.deepcopy(attempt["completions"][ridx])
                    entry["raw_response"] = {
                        "backend": "vllm",
                        "model": model_path,
                        "finish_reason": finish_reason,
                    }
                    entry["prompt_messages"] = messages
                    entry["completion"] = text
                    attempt["completions"][ridx] = entry
                    progress.update(1)
                state.save_async()
    finally:
        progress.close()
        state.save()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _optional_env_float(names: tuple[str, ...], default: Optional[float] = None) -> Optional[float]:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def _optional_env_int(names: tuple[str, ...], default: Optional[int] = None) -> Optional[int]:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def summarise_eval(data_list: list[dict], report_path: str, pass_k: int) -> dict:
    parent_rows: list[dict] = []
    attempt_rows: list[dict] = []
    sub_rows: list[dict] = []
    missing_correctness = 0

    for parent_index, parent in enumerate(data_list):
        attempts = parent.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError(
                f"Parent {parent.get('source_problem_id')} missing new-format attempts."
            )

        source = parent.get("source") or parent.get("subject")
        year = parent.get("year")
        sid = parent.get("source_problem_id")
        year_label = year if year is not None else "?"
        parent_label = f"{year_label}/{sid}"

        parent_passed = False
        parent_correct_attempts = 0
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            completions = attempt.get("completions")
            if not isinstance(completions, list):
                completions = []

            sub_correct_flags: list[bool] = []
            for sub_index, c in enumerate(completions):
                if not isinstance(c, dict):
                    missing_correctness += 1
                    sub_correct_flags.append(False)
                    sub_rows.append({
                        "subquestion_key": (parent_index, sub_index),
                        "source": source,
                        "year": year,
                        "source_problem_id": sid,
                        "parent_label": parent_label,
                        "attempt_id": attempt.get("attempt_id"),
                        "problem_id": None,
                        "answer_type": "",
                        "correctness": False,
                        "model_judge_used": False,
                    })
                    continue
                raw_correct = c.get("correctness")
                if raw_correct is None:
                    missing_correctness += 1
                    correctness = False
                else:
                    try:
                        correctness = bool(raw_correct)
                    except Exception:
                        missing_correctness += 1
                        correctness = False
                c["correctness"] = correctness
                c.setdefault("model_judge_msg", None)
                sub_correct_flags.append(correctness)
                sub_rows.append({
                    # The completion slot is the stable identity of a
                    # sub-question across attempts.  It is more reliable
                    # than ``problem_id`` here because some datasets reuse
                    # parent/problem labels.
                    "subquestion_key": (parent_index, sub_index),
                    "source": source,
                    "year": year,
                    "source_problem_id": sid,
                    "parent_label": parent_label,
                    "attempt_id": attempt.get("attempt_id"),
                    "problem_id": c.get("problem_id"),
                    "answer_type": c.get("answer_type") or "",
                    "correctness": correctness,
                    "model_judge_used": c["model_judge_msg"] is not None,
                })

            attempt_correct = bool(completions) and all(sub_correct_flags)
            attempt["correctness"] = attempt_correct
            parent_passed = parent_passed or attempt_correct
            if attempt_correct:
                parent_correct_attempts += 1
            attempt_rows.append({
                "source": source,
                "year": year,
                "source_problem_id": sid,
                "parent_label": parent_label,
                "attempt_id": attempt.get("attempt_id"),
                "correctness": attempt_correct,
            })

        parent["pass_at_k"] = parent_passed
        parent["correct_attempts"] = parent_correct_attempts
        parent_rows.append({
            "source": source,
            "year": year,
            "source_problem_id": sid,
            "problem_type": parent.get("problem_type") or "",
            "parent_label": parent_label,
            "pass_at_k": parent_passed,
            "attempts": len(attempts),
            "correct_attempts": parent_correct_attempts,
        })

    if not parent_rows:
        report_text = "No parents to summarise."
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(report_text)
        return {
            "parents": 0,
            "parents_with_any_correct_attempt": 0,
            f"parent_pass@{pass_k}": 0.0,
            "missing_correctness": missing_correctness,
        }

    parent_df = pd.DataFrame(parent_rows)
    attempt_df = pd.DataFrame(attempt_rows)
    sub_df = pd.DataFrame(sub_rows)

    parents = len(parent_df)
    parents_any_correct = int(parent_df["pass_at_k"].sum())
    parent_pass_at_k = _safe_ratio(parents_any_correct, parents)
    attempt_total = len(attempt_df)
    attempt_correct = int(attempt_df["correctness"].sum()) if attempt_total else 0
    attempt_rate = _safe_ratio(attempt_correct, attempt_total)
    attempt_sub_total = len(sub_df)
    if attempt_sub_total:
        # A sub-question passes at k when at least one of its k attempts is
        # correct.  Aggregate by the canonical completion slot before
        # calculating the rate; averaging all attempt-level judgements would
        # produce ordinary per-attempt accuracy instead.
        sub_pass_df = (
            sub_df.groupby("subquestion_key", as_index=False)
                  .agg(
                      source=("source", "first"),
                      year=("year", "first"),
                      source_problem_id=("source_problem_id", "first"),
                      parent_label=("parent_label", "first"),
                      problem_id=("problem_id", "first"),
                      answer_type=("answer_type", "first"),
                      pass_at_k=("correctness", "any"),
                      attempts_evaluated=("correctness", "count"),
                  )
        )
    else:
        sub_pass_df = pd.DataFrame(
            columns=[
                "subquestion_key", "source", "year", "source_problem_id",
                "parent_label", "problem_id", "answer_type", "pass_at_k",
                "attempts_evaluated",
            ]
        )
    sub_total = len(sub_pass_df)
    sub_passed = int(sub_pass_df["pass_at_k"].sum()) if sub_total else 0
    sub_pass_at_k = _safe_ratio(sub_passed, sub_total)
    judge_ratio = (
        _safe_ratio(int(sub_df["model_judge_used"].sum()), attempt_sub_total)
        if attempt_sub_total else 0.0
    )

    by_source = (
        parent_df.groupby("source", dropna=False)
          .agg(parents=("pass_at_k", "count"),
               passed=("pass_at_k", "sum"))
          .assign(pass_at_k=lambda d: d["passed"] / d["parents"])
          .reset_index()
          .sort_values("pass_at_k", ascending=False)
    )
    by_problem_type = (
        parent_df.groupby("problem_type", dropna=False)
          .agg(parents=("pass_at_k", "count"),
               passed=("pass_at_k", "sum"))
          .assign(pass_at_k=lambda d: d["passed"] / d["parents"])
          .reset_index()
          .sort_values("pass_at_k", ascending=False)
    )
    by_parent = parent_df[
        ["parent_label", "attempts", "correct_attempts", "pass_at_k"]
    ].sort_values("parent_label")

    label_w = 32
    lines = [
        f"{'Parents':<{label_w}}: {parents}",
        f"{'Parents with >=1 correct attempt':<{label_w}}: {parents_any_correct}",
        f"{f'Parent pass@{pass_k}':<{label_w}}: {parent_pass_at_k:.4f}",
        f"{'Attempts':<{label_w}}: {attempt_total}",
        f"{'All-correct attempts':<{label_w}}: {attempt_correct}",
        f"{'Attempt success rate':<{label_w}}: {attempt_rate:.4f}",
        f"{'Attempt sub-questions':<{label_w}}: {attempt_sub_total}",
        f"{'Unique sub-questions':<{label_w}}: {sub_total}",
        f"{'Sub-questions with >=1 correct attempt':<{label_w}}: {sub_passed}",
        f"{f'Sub-question pass@{pass_k}':<{label_w}}: {sub_pass_at_k:.4f}",
        f"{'Model judge ratio':<{label_w}}: {judge_ratio:.4f}",
    ]
    if missing_correctness:
        lines.append(
            f"Missing/invalid correctness (counted as wrong): "
            f"{missing_correctness}"
        )
    if sub_total:
        by_type = (
            sub_pass_df.groupby("answer_type", dropna=False)
              .agg(total=("pass_at_k", "count"),
                   passed=("pass_at_k", "sum"))
              .reset_index()
              .assign(
                  pass_at_k=lambda d: d["passed"] / d["total"],
              )
              .sort_values("pass_at_k", ascending=False)
              [["answer_type", "total", "passed", "pass_at_k"]]
        )
        lines.extend([
            "",
            f"Parent pass@{pass_k} by source:",
            by_source.to_string(index=False),
            "",
            f"Parent pass@{pass_k} by problem_type:",
            by_problem_type.to_string(index=False),
            "",
            f"Sub-question pass@{pass_k} by answer_type:",
            by_type.to_string(index=False),
            "",
            f"Parent pass@{pass_k} by (year/source_problem_id):",
            by_parent.to_string(index=False),
        ])

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(report_text)
    result = {
        "parents": parents,
        "parents_with_any_correct_attempt": parents_any_correct,
        f"parent_pass@{pass_k}": parent_pass_at_k,
        "attempts": attempt_total,
        "all_correct_attempts": attempt_correct,
        "attempt_success_rate": attempt_rate,
        # ``sub_questions`` historically meant attempt-level rows. Keep that
        # field stable and expose the unique-question denominator separately.
        "sub_questions": attempt_sub_total,
        "unique_sub_questions": sub_total,
        "sub_questions_with_any_correct_attempt": sub_passed,
        f"sub_question_pass@{pass_k}": sub_pass_at_k,
        # Keep the old result key for callers that consume the summary dict;
        # its value now follows the report's sub-question pass@k definition.
        "sub_question_accuracy": sub_pass_at_k,
        "judge_fallback_ratio": judge_ratio,
        "missing_correctness": missing_correctness,
        "by_source": by_source.to_dict(orient="records"),
        "by_problem_type": by_problem_type.to_dict(orient="records"),
        "by_parent_label": by_parent.to_dict(orient="records"),
    }
    if sub_total:
        result["by_answer_type"] = by_type.to_dict(orient="records")
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _infer_backend(model_path: str, backend: str) -> str:
    if backend != "auto":
        return backend
    if os.path.exists(model_path) or "/" in model_path:
        return "local"
    return "api"


def run_benchmark(
    model_path: str,
    data_paths: Optional[list[str]],
    output_dir: str,
    hf_repo: Optional[str] = None,
    competition: Optional[str] = None,
    backend: str = "auto",
    tensor_parallel_size: int = 1,
    nproc: int = 16,
    judge_nproc: Optional[int] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: int = 8192,
    max_completion_tokens: Optional[int] = None,
    precision: float = 1e-2,
    language: str = "EN",
    tag: str = "ioaa",
    pass_k: int = 1,
    only_generate: bool = False,
    only_eval: bool = False,
    strip_think_for_model_eval: bool = False,
    test_part: Optional[Tuple[int, int]] = None,
    judger: Optional[eval.Judger] = None,
    force_regenerate: bool = False,
    ace_playbook: Optional[str] = None,
    ace_playbook_path: Optional[str] = None,
    evoskill_skills: Optional[str] = None,
    evoskill_skills_path: Optional[str] = None,
    gepa_system_prompt: Optional[str] = None,
    gepa_prompt_path: Optional[str] = None,
    gepa_prompt_meta: Optional[dict[str, Any]] = None,
    guidance_index: Optional[dict[tuple, str]] = None,
    guidance_path: Optional[str] = None,
    resume_eval: bool = False,
    agent_project_dir: Optional[str] = None,
    agent_system_append: Optional[str] = None,
    agent_tools: Optional[list[str]] = None,
    agent_permission_mode: str = "bypassPermissions",
    agent_model: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> None:
    """End-to-end: load → sequential generation → eval → report.

    Artefacts are stored at
    ``eval/results/<method_model>/<tag>/{tag}[.json|_eval.json|_report.txt]``.
    ``output_dir`` names the first-level ``method_model`` directory. For
    compatibility with older wrappers, a trailing component equal to ``tag``
    is ignored (for example ``results_base_qwen3_8b/my_exp`` is treated as
    ``results_base_qwen3_8b`` when ``tag == "my_exp"``).

    When ``test_part`` is provided the artefact filenames get a
    ``_{start}:{end}`` suffix so partial-slice runs do not overwrite any
    previously-produced full-run artefacts in the same directory.

    ``pass_k`` controls how many full parent attempts are generated.

    ``force_regenerate=True`` skips the resume-from-disk step entirely:
    no attempt completion is reused even if ``<tag>.json`` already exists on disk.
    The generation backend then writes every attempt slot from scratch and
    ``state.save()`` atomically replaces the old file. Has no effect when
    ``only_eval=True`` (in fact the two are rejected as a combination by
    the CLI).
    """
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    data_paths = data_paths or []
    if pass_k < 1:
        raise ValueError(f"pass_k must be >= 1, got {pass_k}.")

    if test_part is not None:
        path_tag = f"{tag}_{test_part[0]}-{test_part[1]}"
    else:
        path_tag = tag

    model_name = model_path.rstrip("/").split("/")[-1]
    normalized_output_dir = os.path.normpath(output_dir)
    if os.path.basename(normalized_output_dir) == tag:
        normalized_output_dir = os.path.dirname(normalized_output_dir)
    method_model = os.path.basename(normalized_output_dir)
    if method_model in ("", ".", os.path.sep):
        raise ValueError(
            "--output_dir must identify a method_model directory, for example "
            "'results_base_qwen3_8b'."
        )
    results_root = os.path.join(THIS_DIR, "results")
    run_dir = os.path.join(results_root, method_model, path_tag)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run] results will be written under {run_dir}")

    generated_path = os.path.join(run_dir, f"{path_tag}.json")
    evaluated_path = os.path.join(run_dir, f"{path_tag}_eval.json")
    report_path = os.path.join(run_dir, f"{path_tag}_report.txt")

    # Drop a sidecar copy of the playbook actually used, so the run is
    # self-describing. We do this even when only_eval=True since the user
    # may still want to look up the playbook later.
    if ace_playbook is not None:
        playbook_sidecar = os.path.join(run_dir, f"{path_tag}_ace_playbook.txt")
        try:
            with open(playbook_sidecar, "w", encoding="utf-8") as f:
                if ace_playbook_path:
                    f.write(f"# source: {ace_playbook_path}\n\n")
                f.write(ace_playbook)
            print(
                f"[ace] using playbook ({len(ace_playbook)} chars) "
                f"from {ace_playbook_path or '<inline>'}; "
                f"copy saved to {playbook_sidecar}"
            )
        except OSError as exc:  # pragma: no cover - best-effort sidecar
            print(f"[ace] could not write playbook sidecar: {exc}", flush=True)

    # Same self-describing-run treatment for EvoSkill. The blob is whatever
    # ``baseline/EvoSkill/scripts/export_program.py`` produced for the chosen
    # frontier program; dropping a copy next to the results makes it
    # straightforward to attribute scores back to a specific program version
    # months later.
    if evoskill_skills is not None:
        evoskill_sidecar = os.path.join(run_dir, f"{path_tag}_evoskill_skills.txt")
        try:
            with open(evoskill_sidecar, "w", encoding="utf-8") as f:
                if evoskill_skills_path:
                    f.write(f"# source: {evoskill_skills_path}\n\n")
                f.write(evoskill_skills)
            print(
                f"[evoskill] using skills blob ({len(evoskill_skills)} chars) "
                f"from {evoskill_skills_path or '<inline>'}; "
                f"copy saved to {evoskill_sidecar}"
            )
        except OSError as exc:  # pragma: no cover - best-effort sidecar
            print(f"[evoskill] could not write skills sidecar: {exc}", flush=True)

    # Same self-describing-run treatment for GEPA. The selected checkpoint
    # (last row of the history JSONL by default, or a specific iteration /
    # rollout budget) is the exact system prompt that will replace the
    # default StudyBench system message.
    if gepa_system_prompt is not None:
        gepa_sidecar = os.path.join(run_dir, f"{path_tag}_gepa_system_prompt.txt")
        try:
            with open(gepa_sidecar, "w", encoding="utf-8") as f:
                if gepa_prompt_path:
                    f.write(f"# source: {gepa_prompt_path}\n")
                if gepa_prompt_meta:
                    for key in (
                        "format",
                        "iteration",
                        "rollouts",
                        "score",
                        "best_idx",
                        "time_iso",
                        "n_checkpoints",
                    ):
                        if key in gepa_prompt_meta and gepa_prompt_meta[key] is not None:
                            f.write(f"# {key}: {gepa_prompt_meta[key]}\n")
                if gepa_prompt_path or gepa_prompt_meta:
                    f.write("\n")
                f.write(gepa_system_prompt)
            extra = ""
            if gepa_prompt_meta:
                bits = []
                if gepa_prompt_meta.get("iteration") is not None:
                    bits.append(f"iteration={gepa_prompt_meta['iteration']}")
                if gepa_prompt_meta.get("rollouts") is not None:
                    bits.append(f"rollouts={gepa_prompt_meta['rollouts']}")
                if gepa_prompt_meta.get("score") is not None:
                    bits.append(f"score={gepa_prompt_meta['score']}")
                if bits:
                    extra = " (" + ", ".join(bits) + ")"
            print(
                f"[gepa] using system prompt ({len(gepa_system_prompt)} chars)"
                f"{extra} from {gepa_prompt_path or '<inline>'}; "
                f"copy saved to {gepa_sidecar}"
            )
        except OSError as exc:  # pragma: no cover - best-effort sidecar
            print(f"[gepa] could not write system-prompt sidecar: {exc}", flush=True)

    if guidance_index is not None:
        guidance_meta = os.path.join(run_dir, f"{path_tag}_guidance_meta.txt")
        try:
            with open(guidance_meta, "w", encoding="utf-8") as f:
                if guidance_path:
                    f.write(f"# source: {guidance_path}\n")
                f.write(f"# indexed_entries: {len(guidance_index)}\n")
            print(
                f"[guidance] loaded {len(guidance_index)} sub-question hint(s) "
                f"from {guidance_path or '<inline>'}; "
                f"meta saved to {guidance_meta}"
            )
        except OSError as exc:  # pragma: no cover - best-effort sidecar
            print(f"[guidance] could not write guidance meta: {exc}", flush=True)

    if not only_eval:
        items = load_benchmark_items(
            data_paths=data_paths,
            hf_repo=hf_repo,
            competition=competition,
        )
        if test_part is not None:
            items = items[test_part[0]:test_part[1]]
        if data_paths:
            print(
                f"[load] {len(items)} parent item(s) from "
                f"{len(data_paths)} local file(s): {data_paths}"
            )
        else:
            print(
                f"[load] {len(items)} parent item(s) from "
                f"Hugging Face dataset {hf_repo} config={competition}"
            )

        if guidance_index is not None:
            attached, missing = attach_guidance_to_items(items, guidance_index)
            print(
                f"[guidance] attached to {attached} sub-question(s); "
                f"{missing} sub-question(s) had no matching hint "
                f"(those runs keep the baseline prompt)."
            )

        if force_regenerate:
            # Don't touch ``generated_path`` on disk yet -- the backend's
            # first ``state.save()`` will atomically overwrite it. Keeping
            # the old file around buys us a recovery option if the new
            # run dies before producing a single completion.
            print(
                f"[generate] --force_regenerate set: ignoring any cached "
                f"attempts in {generated_path}; every attempt slot will be "
                f"regenerated from scratch."
            )
        else:
            merge_existing_attempts(items, generated_path, pass_k=pass_k)

        resolved_backend = _infer_backend(model_path=model_path, backend=backend)
        print(f"[generate] model={model_name} backend={resolved_backend} output={generated_path}")
        if resolved_backend == "local":
            generate_with_vllm(
                items=items,
                model_path=model_path,
                tensor_parallel_size=tensor_parallel_size,
                temperature=temperature,
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                language=language,
                pass_k=pass_k,
                output_path=generated_path,
                top_p=top_p,
                top_k=top_k,
                ace_playbook=ace_playbook,
                evoskill_skills=evoskill_skills,
                gepa_system_prompt=gepa_system_prompt,
            )
        elif resolved_backend == "anthropic":
            generate_with_anthropic(
                items=items,
                model=model_path,
                nproc=nproc,
                temperature=temperature,
                max_tokens=max_tokens,
                language=language,
                pass_k=pass_k,
                output_path=generated_path,
                top_p=top_p,
                top_k=top_k,
                ace_playbook=ace_playbook,
                evoskill_skills=evoskill_skills,
                gepa_system_prompt=gepa_system_prompt,
            )
        elif resolved_backend == "claude_code":
            generate_with_claude_code(
                items=items,
                nproc=nproc,
                language=language,
                pass_k=pass_k,
                output_path=generated_path,
                system_append=agent_system_append,
                agent_project_dir=agent_project_dir,
                tools=agent_tools,
                permission_mode=agent_permission_mode,
                model=agent_model,
            )
        else:
            generate_with_api(
                items=items,
                model=model_path,
                nproc=nproc,
                temperature=temperature,
                max_tokens=max_tokens,
                language=language,
                pass_k=pass_k,
                output_path=generated_path,
                top_p=top_p,
                top_k=top_k,
                ace_playbook=ace_playbook,
                evoskill_skills=evoskill_skills,
                gepa_system_prompt=gepa_system_prompt,
                tokenizer_path=tokenizer_path,
            )
        print(f"[generate] wrote {generated_path}")

    if only_generate:
        return

    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated file not found: {generated_path}. Run generation first or drop --only_eval.")

    print(f"[eval] scoring {generated_path}")
    data_list = read_json(generated_path)
    if isinstance(data_list, dict) and isinstance(data_list.get("data"), list):
        data_list = data_list["data"]
    if not isinstance(data_list, list):
        raise ValueError(f"Input data must be a JSON list or a dict with a `data` list; got {type(data_list).__name__}.")
    if resume_eval:
        merge_existing_eval(data_list, evaluated_path)
    attempt_counts = [
        len(p.get("attempts") or [])
        for p in data_list
        if isinstance(p, dict)
    ]
    report_pass_k = max(attempt_counts, default=pass_k)
    resolved_judge_nproc = nproc if judge_nproc is None else judge_nproc
    eval.eval_file(
        data_list=data_list,
        save_path=evaluated_path,
        precision=precision,
        nproc=resolved_judge_nproc,
        strip_think_for_model_eval=strip_think_for_model_eval,
        judger=judger,
    )

    print(f"[report] summarising {evaluated_path}")
    summarise_eval(data_list, report_path, pass_k=report_pass_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IOAA benchmark with sequential per-subquestion prompting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to local model or OpenAI API model name.")
    parser.add_argument("--data_paths", nargs="*", default=None,
                        help="Optional local benchmark JSON paths. When provided, "
                             "these take precedence over Hugging Face loading. "
                             "Multiple files are concatenated into a single run.")
    parser.add_argument("--hf_repo", default=None,
                        help="Hugging Face dataset repo id used when --data_paths "
                             "is omitted, e.g. yourname/physics-olympiad-bench.")
    parser.add_argument("--competition", choices=HF_COMPETITIONS, default=None,
                        help="Dataset config / competition to evaluate from "
                             "Hugging Face when --data_paths is omitted.")
    parser.add_argument(
        "--output_dir",
        default="results_base",
        help=(
            "First-level method_model directory under eval/results. Results "
            "are written to eval/results/<method_model>/<tag>/. A legacy "
            "trailing /<tag> component is accepted and removed."
        ),
    )
    parser.add_argument("--tag", default="ioaa",
                        help="Short name used for artefact filenames.")
    parser.add_argument("--backend", choices=["auto", "local", "api", "anthropic", "claude_code"], default="auto")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--nproc", type=int, default=16,
                        help="Parallel workers for generation (API / Anthropic / "
                             "claude_code backends).")
    parser.add_argument("--judge_nproc", type=int, default=None,
                        help="Parallel workers for LLM-judge evaluation. "
                             "Defaults to --nproc when omitted.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus-sampling top_p. When omitted, falls back "
                             "to the TOP_P / TOPP env var; if neither is set, "
                             "the API backend omits the field entirely and the "
                             "vLLM backend uses 1.0 (no truncation).")
    parser.add_argument("--top_k", type=int, default=None,
                        help="top_k sampling cutoff. When omitted, falls back "
                             "to the TOP_K / TOPK env var; if neither is set, "
                             "both backends omit the field (backend default).")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument(
        "--max_completion_tokens",
        type=int,
        default=None,
        help=(
            "API backend only: optional hard cap for generated tokens per "
            "request, applied after subtracting prompt length from --max_tokens. "
            "Useful for preventing runaway completions while retaining a large "
            "total context budget."
        ),
    )
    parser.add_argument("--tokenizer_path", default=None,
                        help="API backend only: path/HF-id of the tokenizer used "
                             "to measure each prompt's token length so the "
                             "request's max_tokens is reduced by it (keeping "
                             "prompt+output within the server's context window, "
                             "mirroring the vLLM path). Defaults to --model; set "
                             "this when the served model name is an API alias "
                             "that AutoTokenizer cannot load.")
    parser.add_argument("--precision", type=float, default=1e-2,
                        help="Relative precision for numerical judge.")
    parser.add_argument("--language", default="EN", choices=["EN", "ZH"])
    parser.add_argument("--pass_k", type=int, default=1,
                        help="Number of full parent attempts to sample. "
                             "A parent passes if any attempt gets all "
                             "sub-questions correct.")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL,
                        help="LLM used as the fallback judge when rule-based match fails.")
    parser.add_argument("--judge_backend", choices=["auto", "openai", "dsv4"],
                        default=None,
                        help="LLM-judge transport. 'openai' = the standard "
                             "/v1/chat/completions SDK call (default for "
                             "non-DSv4 models). 'dsv4' = raw /v1/completions "
                             "with the official DeepSeek-V4 encoder, for the "
                             "local serve_deepseek_v4_flash.sh server. "
                             "'auto' (default) inspects --judge_model and "
                             "picks 'dsv4' iff it starts with 'DeepSeek-V4'. "
                             "Falls back to the JUDGE_BACKEND env var.")
    parser.add_argument("--judge_base_url", default=None,
                        help="LLM-judge endpoint (e.g. 'http://127.0.0.1:8000/v1' "
                             "for the local DSv4 server). Falls back to the "
                             "JUDGE_BASE_URL env var, then to OPENAI_BASE_URL.")
    parser.add_argument("--judge_api_key", default=None,
                        help="LLM-judge API key. Falls back to JUDGE_API_KEY "
                             "env var, then to OPENAI_API_KEY. The local DSv4 "
                             "server does not require a real key; pass any "
                             "non-empty string or leave unset.")
    parser.add_argument("--judge_dsv4_model_dir", default=None,
                        help="Path to the DeepSeek-V4-Flash model directory "
                             "(only used by --judge_backend dsv4 to import "
                             "the official encoder). Falls back to "
                             "JUDGE_DSV4_MODEL_DIR or "
                             "/home/tsinghua/cyh/models/DeepSeek-V4-Flash.")
    parser.add_argument("--judge_thinking_mode",
                        choices=["thinking", "chat"], default=None,
                        help="DSv4-only: 'thinking' lets the judge reason "
                             "before emitting TRUE/FALSE (default; recommended); "
                             "'chat' is faster but skips the thinking trace. "
                             "Falls back to JUDGE_THINKING_MODE env var.")
    parser.add_argument("--strict_extract", action=argparse.BooleanOptionalAction, default=True,
                        help="Require explicit \\boxed{} answers when extracting. "
                             "Pass --no-strict_extract to allow the legacy speculative "
                             "fallback (last latex / last number).")
    parser.add_argument("--strip_think_for_model_eval", action="store_true",
                        help="Strip <think>...</think> from completions before BOTH "
                             "rule-based auto_judge and LLM aux_judge, so both paths see "
                             "the same post-think text.")
    parser.add_argument("--only_generate", action="store_true",
                        help="Generate completions and skip evaluation.")
    parser.add_argument("--only_eval", action="store_true",
                        help="Skip generation and only re-score existing completions.")
    parser.add_argument("--force_regenerate", action="store_true",
                        help="Ignore any cached completions in <tag>.json and "
                             "regenerate every sub-question from scratch. The "
                             "first state save will atomically overwrite the "
                             "old file. Cannot be combined with --only_eval.")
    parser.add_argument("--test_start", type=int, default=None,
                        help="Inclusive start index for a dataset slice.")
    parser.add_argument("--test_end", type=int, default=None,
                        help="Exclusive end index for a dataset slice.")
    parser.add_argument("--resume_eval", action="store_true",
                        help="Reuse judge verdicts from a previously written "
                             "<tag>_eval.json: load it, copy each completion's "
                             "correctness / model_judge_msg / extracted_answer "
                             "/ normalized_gt onto the matching slot in the "
                             "fresh data, and let the judge loop short-circuit "
                             "any completion that already has a boolean "
                             "``correctness``. Useful when a prior eval was "
                             "interrupted, when only some completions need to "
                             "be (re-)judged, or when several --only_eval runs "
                             "want to share work. No-op when no <tag>_eval.json "
                             "exists yet. Off by default.")
    parser.add_argument("--ace_playbook_path", default=None,
                        help="Optional path to an ACE-style playbook (e.g. "
                             "baseline/ace/results/.../intermediate_playbooks/"
                             "epoch_*_step_*_playbook.txt). When provided, the "
                             "file's contents are appended to the system "
                             "message used by every sub-question turn. All "
                             "other StudyBench behaviour (multi-sub chat "
                             "history, \\boxed{} extraction, auto_judge / "
                             "aux_judge, pass@k, sampling params) is "
                             "unchanged so results stay directly comparable "
                             "to a no-playbook baseline. Mutually exclusive "
                             "with --evoskill_skills_path / --gepa_prompt_path.")
    parser.add_argument("--evoskill_skills_path", default=None,
                        help="Optional path to an assembled EvoSkill system "
                             "prompt (produced by "
                             "baseline/EvoSkill/scripts/export_program.py). "
                             "The file already carries the task description, "
                             "constraints, and learned-skill checklists the "
                             "evolved program was trained under, so it "
                             "*replaces* the default system message instead "
                             "of being appended. Mutually exclusive with "
                             "--ace_playbook_path / --gepa_prompt_path. All "
                             "other StudyBench behaviour (multi-sub chat "
                             "history, \\boxed{} extraction, judging, "
                             "pass@k) is unchanged.")
    parser.add_argument("--gepa_prompt_path", default=None,
                        help="Optional path to a GEPA-evolved system prompt. "
                             "Accepts a chronological history JSONL (e.g. "
                             "baseline/gepa/qwen3_8b_best_system_prompt_"
                             "history.jsonl, each line "
                             "{iteration, rollouts, score, system_prompt}) "
                             "or a plain-text prompt file. The selected "
                             "instruction *replaces* the default system "
                             "message so eval-time conditioning matches "
                             "GEPA training-time conditioning. Default "
                             "selection is the last JSONL row (incumbent "
                             "best at the end of training); use "
                             "--gepa_iteration / --gepa_rollouts to pin a "
                             "checkpoint. Mutually exclusive with "
                             "--ace_playbook_path / --evoskill_skills_path.")
    parser.add_argument("--gepa_iteration", type=int, default=None,
                        help="Select the last GEPA history row whose "
                             "iteration is <= this value. Requires "
                             "--gepa_prompt_path pointing at a history "
                             "JSONL. Mutually exclusive with "
                             "--gepa_rollouts.")
    parser.add_argument("--gepa_rollouts", type=int, default=None,
                        help="Select the last GEPA history row whose "
                             "rollouts is <= this value. Requires "
                             "--gepa_prompt_path pointing at a history "
                             "JSONL. Mutually exclusive with "
                             "--gepa_iteration.")
    parser.add_argument("--guidance_path", default=None,
                        help="Optional path to a per-sub-question solving-"
                             "guidance JSON (e.g. level3_guidance.json). "
                             "Each matching ``sub_problems[i].guidance`` is "
                             "appended to that sub-question's user turn under "
                             "a ``## Solving Guidance`` heading. Subs without "
                             "a match keep the baseline prompt. Compatible "
                             "with --ace_playbook_path / --evoskill_skills_path "
                             "/ --gepa_prompt_path.")
    # --- Claude Code harness backend (--backend claude_code) ---------------
    parser.add_argument("--agent_project_dir", default=None,
                        help="Claude Code backend only: path to a project dir "
                             "whose .claude/ (skills, settings, MCP) and "
                             "CLAUDE.md are copied into each agent run's "
                             "scratch cwd and loaded natively. This is the "
                             "method-agnostic way to condition the agent: any "
                             "'opus + claude code' evolve method plugs in by "
                             "pointing here at its own artefacts. The backend "
                             "never special-cases any method.")
    parser.add_argument("--system_prompt_append_file", default=None,
                        help="Claude Code backend only: path to a text file "
                             "whose contents are appended to the claude_code "
                             "preset's system prompt. Composes with "
                             "--agent_project_dir.")
    parser.add_argument("--agent_tools", nargs="*", default=None,
                        help="Claude Code backend only: tool allow-list for the "
                             "agent. Defaults to Bash/Read/Write/Edit/Glob/"
                             "Grep/TodoWrite/BashOutput.")
    parser.add_argument("--agent_permission_mode", default="bypassPermissions",
                        help="Claude Code backend only: permission mode for the "
                             "agent. Eval is unattended, so the default is "
                             "'bypassPermissions' (the agent really executes "
                             "Bash/etc. in its scratch cwd).")
    parser.add_argument("--agent_model", default=None,
                        help="Claude Code backend only: model id sent to the "
                             "agent SDK. When omitted, the spawned Claude Code "
                             "process uses CLAUDE_CODE_MODEL from the loaded "
                             "user settings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.test_start is None) ^ (args.test_end is None):
        raise SystemExit("Please provide both --test_start and --test_end, or neither.")
    test_part = (
        (args.test_start, args.test_end)
        if args.test_start is not None and args.test_end is not None
        else None
    )

    if args.force_regenerate and args.only_eval:
        raise SystemExit(
            "--force_regenerate is meaningless with --only_eval (the latter "
            "skips generation entirely). Drop one of them."
        )
    if args.pass_k < 1:
        raise SystemExit("--pass_k must be >= 1.")
    if args.max_completion_tokens is not None and args.max_completion_tokens < 1:
        raise SystemExit("--max_completion_tokens must be >= 1 when provided.")
    data_paths = args.data_paths or []
    if not data_paths and not args.only_eval:
        if not args.hf_repo or not args.competition:
            raise SystemExit(
                "Provide --data_paths for local loading, or provide both "
                "--hf_repo and --competition for Hugging Face loading."
            )

    judger = (
        None
        if args.only_generate
        else eval.make_default_judger(
            strict_extract=args.strict_extract,
            judge_model=args.judge_model,
            judge_backend=args.judge_backend,
            judge_base_url=args.judge_base_url,
            judge_api_key=args.judge_api_key,
            judge_dsv4_model_dir=args.judge_dsv4_model_dir,
            judge_thinking_mode=args.judge_thinking_mode,
        )
    )

    # Anchor relative paths at this script's directory rather than the
    # invocation CWD. This preserves the historical behaviour the
    # ``benchmark.sh`` callers rely on (e.g. ``--data_paths
    # ../problems/ioaa/2023.json`` resolving to
    # ``study_bench/problems/ioaa/2023.json``) without us having to
    # ``os.chdir`` and clobber the caller's working directory.
    def _anchor(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(THIS_DIR, p))

    top_p = (
        args.top_p
        if args.top_p is not None
        else _optional_env_float(("TOP_P", "TOPP"), default=None)
    )
    top_k = (
        args.top_k
        if args.top_k is not None
        else _optional_env_int(("TOP_K", "TOPK"), default=None)
    )

    method_flags = [
        flag
        for flag, value in (
            ("--ace_playbook_path", args.ace_playbook_path),
            ("--evoskill_skills_path", args.evoskill_skills_path),
            ("--gepa_prompt_path", args.gepa_prompt_path),
        )
        if value
    ]
    if len(method_flags) > 1:
        raise SystemExit(
            f"{' and '.join(method_flags)} are mutually exclusive: ACE "
            "appends a playbook to the default system message, while "
            "EvoSkill and GEPA each replace the system message with their "
            "evolved instruction. Pick one."
        )
    if args.gepa_iteration is not None and args.gepa_rollouts is not None:
        raise SystemExit(
            "--gepa_iteration and --gepa_rollouts are mutually exclusive."
        )
    if (
        args.gepa_iteration is not None or args.gepa_rollouts is not None
    ) and not args.gepa_prompt_path:
        raise SystemExit(
            "--gepa_iteration / --gepa_rollouts require --gepa_prompt_path."
        )

    ace_playbook: Optional[str] = None
    ace_playbook_path: Optional[str] = None
    if args.ace_playbook_path:
        ace_playbook_path = _anchor(args.ace_playbook_path)
        if not os.path.exists(ace_playbook_path):
            raise SystemExit(
                f"--ace_playbook_path does not exist: {ace_playbook_path}"
            )
        with open(ace_playbook_path, "r", encoding="utf-8") as f:
            ace_playbook = f.read()
        if not ace_playbook.strip():
            raise SystemExit(
                f"--ace_playbook_path is empty: {ace_playbook_path}"
            )

    evoskill_skills: Optional[str] = None
    evoskill_skills_path: Optional[str] = None
    if args.evoskill_skills_path:
        evoskill_skills_path = _anchor(args.evoskill_skills_path)
        if not os.path.exists(evoskill_skills_path):
            raise SystemExit(
                f"--evoskill_skills_path does not exist: {evoskill_skills_path}"
            )
        with open(evoskill_skills_path, "r", encoding="utf-8") as f:
            evoskill_skills = f.read()
        if not evoskill_skills.strip():
            raise SystemExit(
                f"--evoskill_skills_path is empty: {evoskill_skills_path}"
            )

    gepa_system_prompt: Optional[str] = None
    gepa_prompt_path: Optional[str] = None
    gepa_prompt_meta: Optional[dict[str, Any]] = None
    if args.gepa_prompt_path:
        gepa_prompt_path = _anchor(args.gepa_prompt_path)
        if not os.path.exists(gepa_prompt_path):
            raise SystemExit(
                f"--gepa_prompt_path does not exist: {gepa_prompt_path}"
            )
        try:
            gepa_system_prompt, gepa_prompt_meta = load_gepa_system_prompt(
                gepa_prompt_path,
                iteration=args.gepa_iteration,
                rollouts=args.gepa_rollouts,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    guidance_index: Optional[dict[tuple, str]] = None
    guidance_path: Optional[str] = None
    if args.guidance_path:
        guidance_path = _anchor(args.guidance_path)
        if not os.path.exists(guidance_path):
            raise SystemExit(
                f"--guidance_path does not exist: {guidance_path}"
            )
        guidance_index = load_guidance_index(guidance_path)
        if not guidance_index:
            raise SystemExit(
                f"--guidance_path contains no usable guidance entries: "
                f"{guidance_path}"
            )

    # --- Claude Code harness backend inputs --------------------------------
    agent_project_dir: Optional[str] = None
    if args.agent_project_dir:
        agent_project_dir = _anchor(args.agent_project_dir)
        if not os.path.isdir(agent_project_dir):
            raise SystemExit(
                f"--agent_project_dir is not a directory: {agent_project_dir}"
            )
    agent_system_append: Optional[str] = None
    if args.system_prompt_append_file:
        append_path = _anchor(args.system_prompt_append_file)
        if not os.path.exists(append_path):
            raise SystemExit(
                f"--system_prompt_append_file does not exist: {append_path}"
            )
        with open(append_path, "r", encoding="utf-8") as f:
            agent_system_append = f.read()
        if not agent_system_append.strip():
            raise SystemExit(
                f"--system_prompt_append_file is empty: {append_path}"
            )
    if args.backend != "claude_code" and (
        args.agent_project_dir
        or args.system_prompt_append_file
        or args.agent_tools is not None
        or args.agent_model
    ):
        raise SystemExit(
            "--agent_project_dir / --system_prompt_append_file / --agent_tools "
            "/ --agent_model only apply to --backend claude_code."
        )

    run_benchmark(
        model_path=args.model,
        data_paths=[_anchor(p) for p in data_paths],
        output_dir=_anchor(args.output_dir),
        hf_repo=args.hf_repo,
        competition=args.competition,
        backend=args.backend,
        tensor_parallel_size=args.tensor_parallel_size,
        nproc=args.nproc,
        judge_nproc=args.judge_nproc,
        temperature=args.temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=args.max_tokens,
        max_completion_tokens=args.max_completion_tokens,
        precision=args.precision,
        language=args.language,
        tag=args.tag,
        pass_k=args.pass_k,
        only_generate=args.only_generate,
        only_eval=args.only_eval,
        strip_think_for_model_eval=args.strip_think_for_model_eval,
        test_part=test_part,
        judger=judger,
        force_regenerate=args.force_regenerate,
        ace_playbook=ace_playbook,
        ace_playbook_path=ace_playbook_path,
        evoskill_skills=evoskill_skills,
        evoskill_skills_path=evoskill_skills_path,
        gepa_system_prompt=gepa_system_prompt,
        gepa_prompt_path=gepa_prompt_path,
        gepa_prompt_meta=gepa_prompt_meta,
        guidance_index=guidance_index,
        guidance_path=guidance_path,
        resume_eval=args.resume_eval,
        agent_project_dir=agent_project_dir,
        agent_system_append=agent_system_append,
        agent_tools=args.agent_tools,
        agent_permission_mode=args.agent_permission_mode,
        agent_model=args.agent_model,
        tokenizer_path=args.tokenizer_path,
    )


if __name__ == "__main__":
    main()
