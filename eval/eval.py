import time
from typing import Optional
import re
from tqdm import tqdm
import threading
from judge_utils import *
import argparse
import numpy as np
from judge import Judger, DEFAULT_JUDGE_MODEL
import pandas as pd
from persistent_state import PersistentState


_RETRYABLE_JUDGE_ERROR_MARKERS = (
    "[aux_judge error]",
    "[auto_judge error]",
    "[eval crash]",
    "token quota is not enough",
    "quota is not enough",
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    "permissiondeniederror",
    "authenticationerror",
    "ratelimiterror",
    "rate limit",
    "badrequesterror",
    "bad_response_status_code",
    "openai_error",
)


def is_retryable_judge_error(msg: object) -> bool:
    """Return whether a judge message is an infrastructure failure to retry."""
    if not isinstance(msg, str):
        return False
    low = msg.lower()
    return any(marker in low for marker in _RETRYABLE_JUDGE_ERROR_MARKERS)


def has_reusable_judge_result(completion: dict) -> bool:
    """True only for completed judge results that resume should trust."""
    return (
        isinstance(completion.get("correctness"), bool)
        and not is_retryable_judge_error(completion.get("model_judge_msg"))
    )


def make_default_judger(
    *,
    strict_extract: bool = True,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_backend: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_dsv4_model_dir: str | None = None,
    judge_thinking_mode: str | None = None,
) -> Judger:
    """Single construction site for the rule-based + LLM-fallback judger.

    Kept as a thin factory so external callers can either build their own
    ``Judger`` (for tests / custom configs) or rely on this default. There is
    intentionally **no** module-level singleton: importing :mod:`eval` should
    not pay the LaTeX-parser / OpenAI-client cost when only generation is
    needed, and the singleton pattern was forcing callers to monkey-patch the
    module to swap judge models.

    The default ``judge_model`` comes from :data:`judge.DEFAULT_JUDGE_MODEL`
    so the CLI, this factory, and ``Judger.__init__`` all share a single
    source of truth.

    The ``judge_backend`` / ``judge_base_url`` / ... knobs route the LLM
    call through a different backend than the default OpenAI-compatible
    chat endpoint. Most useful values:
      * ``judge_backend="dsv4"`` -- talk to a local DeepSeek-V4-Flash vLLM
        server via raw ``/v1/completions`` (the default
        ``serve_deepseek_v4_flash.sh`` ships no chat template, so the
        chat endpoint is not available without extra setup).
      * ``judge_base_url="http://127.0.0.1:8000/v1"`` -- override endpoint.
    Any of these may be left as ``None``: the corresponding env var
    (``JUDGE_BACKEND`` / ``JUDGE_BASE_URL`` / ...) is then consulted, and
    finally the documented default is used. See ``judge.make_judge_backend``
    for the full precedence ladder.
    """
    return Judger(
        strict_extract=strict_extract,
        judge_model=judge_model,
        judge_backend=judge_backend,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_dsv4_model_dir=judge_dsv4_model_dir,
        judge_thinking_mode=judge_thinking_mode,
    )

def strip_think_blocks(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _eval_one_parent(parent_item,
                     judger: Judger,
                     quirks: dict,
                     progress: Optional[tqdm] = None,
                     precision: float = 1e-8,
                     strip_think_for_model_eval: bool = False
    ):
    # Each sub-question is fully isolated: any uncaught exception from
    # extraction / ``auto_judge`` / ``aux_judge`` is swallowed here, and the
    # loop moves on. Transient judge infrastructure failures are recorded
    # without ``correctness`` so ``--resume_eval`` can retry them later.
    # This guarantees:
    #   - the progress bar always advances exactly once per completion;
    #   - completed judge results get ``correctness`` while retryable judge
    #     infrastructure failures stay incomplete for the next resume.
    #
    # The shared-preamble string passed to ``aux_judge`` (as the ``stem``
    # kwarg) is sourced from ``parent.problem`` per the current schema --
    # there is no ``stem`` field anymore. For single-question parents the
    # completion's own ``problem`` field already holds the entire problem
    # text, so we pass ``""`` here to avoid duplicating the question in
    # the LLM judge prompt; multi-sub parents pass the shared preamble
    # through so the judge sees it before the per-sub question.
    has_subs = bool(parent_item.get("sub_problems") or parent_item.get("problems"))
    parent_stem = (parent_item.get("problem") or "") if has_subs else ""
    attempts = parent_item.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError(
            f"Parent {parent_item.get('source_problem_id')} missing new-format attempts."
        )
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        completions = attempt.get("completions")
        if not isinstance(completions, list):
            completions = []
        for completion in completions:
            # ``--resume_eval`` short-circuit: when ``run_benchmark`` has
            # merged a prior <tag>_eval.json onto this slot the completion
            # already carries a boolean ``correctness`` (and its
            # accompanying judge fields). Skipping here means we never
            # spend an ``auto_judge`` / ``aux_judge`` call on something
            # that was already decided. Without ``--resume_eval`` no
            # completion ever reaches this point with ``correctness``
            # populated, so this guard is a no-op for the default flow.
            if isinstance(completion, dict) and has_reusable_judge_result(completion):
                if progress is not None:
                    progress.update(1)
                continue
            try:
                answer_type = str(completion.get('answer_type', '')).upper()
                # ``type_sequence`` is a comma-separated string per the
                # README schema; only TUP / ALT use it. We pass the raw
                # field through to ``auto_judge`` which parses it.
                type_sequence = completion.get('type_sequence') or ''
                raw_pred = completion.get('completion') or ''
                gold_text = completion.get('answer') or ''
                if strip_think_for_model_eval:
                    pred_text = strip_think_blocks(raw_pred)
                else:
                    pred_text = raw_pred

                # Record the same normalized lists ``auto_judge`` actually
                # compares, so eval artefacts let humans see *what* was
                # extracted and *what* gold was matched against without
                # re-running extraction.
                try:
                    extracted_pred, normalized_gt = judger.extract_normalized_lists(
                        pred_text, gold_text, answer_type,
                    )
                except Exception as exc:  # pragma: no cover - extraction must not block eval
                    extracted_pred, normalized_gt = [], []
                    print(
                        f"[eval] extract failed for "
                        f"{parent_item.get('source_problem_id')} / "
                        f"attempt {attempt.get('attempt_id')} / {completion.get('problem_id')}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                completion["extracted_answer"] = extracted_pred
                completion["normalized_gt"] = normalized_gt

                try:
                    correctness = judger.auto_judge(
                        pred_text,
                        gold_text,
                        answer_type=answer_type,
                        precision=precision,
                        type_sequence=type_sequence,
                    )
                    msg: Optional[str] = None
                except Exception as exc:
                    err = (
                        f"[auto_judge error] {type(exc).__name__}: {exc}"
                    )
                    print(
                        f"[eval] auto_judge failed for "
                        f"{parent_item.get('source_problem_id')} / "
                        f"attempt {attempt.get('attempt_id')} / {completion.get('problem_id')}: "
                        f"{err}",
                        flush=True,
                    )
                    correctness = False
                    msg = err

                if not correctness:
                    try:
                        correctness, aux_msg = judger.aux_judge(
                            pred_text,
                            gold_text,
                            completion.get('problem') or '',
                            completion.get('solution') or '',
                            stem=parent_stem,
                        )
                    except Exception as exc:
                        err = (
                            f"[aux_judge error] {type(exc).__name__}: {exc}"
                        )
                        print(
                            f"[eval] aux_judge failed for "
                            f"{parent_item.get('source_problem_id')} / "
                            f"attempt {attempt.get('attempt_id')} / {completion.get('problem_id')}: "
                            f"{err}",
                            flush=True,
                        )
                        correctness = False
                        aux_msg = err
                    # Stitch the auto-judge error (if any) onto the aux-judge
                    # message so a single ``model_judge_msg`` field carries
                    # the full failure trail when both paths bailed out.
                    if msg and aux_msg:
                        msg = f"{msg}\n{aux_msg}"
                    else:
                        msg = aux_msg or msg

                completion["model_judge_msg"] = msg
                if is_retryable_judge_error(msg):
                    completion.pop("correctness", None)
                else:
                    completion["correctness"] = bool(correctness)
            except Exception as exc:  # pragma: no cover - last-ditch guard
                print(
                    f"[eval] sub-question crashed for "
                    f"{parent_item.get('source_problem_id')} / "
                    f"attempt {attempt.get('attempt_id')} / "
                    f"{completion.get('problem_id') if isinstance(completion, dict) else '<?>'}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if isinstance(completion, dict):
                    completion.setdefault("extracted_answer", [])
                    completion.setdefault("normalized_gt", [])
                    completion["model_judge_msg"] = (
                        f"[eval crash] {type(exc).__name__}: {exc}"
                    )
                    completion.pop("correctness", None)
            finally:
                if progress is not None:
                    progress.update(1)
        valid_completions = [c for c in completions if isinstance(c, dict)]
        attempt["correctness"] = (
            bool(completions)
            and len(valid_completions) == len(completions)
            and all(bool(c.get("correctness")) for c in valid_completions)
        )
    parent_item["pass_at_k"] = any(
        bool(a.get("correctness")) for a in attempts if isinstance(a, dict)
    )

def eval_file(data_list: list[dict],
              save_path: str,
              precision: float = 1e-8,
              nproc: int = 16,
              strip_think_for_model_eval: bool = False,
              judger: Optional[Judger] = None):
    """
    Evaluate the generated response.

    Parameters
    ----------
    judger
        Optional pre-built :class:`Judger`. When ``None`` a default instance is
        created via :func:`make_default_judger`. Pass an explicit one to
        override the judge model, share configuration with another component,
        or inject a stub from tests.
    """
    if judger is None:
        judger = make_default_judger()

    state = PersistentState(data_list, save_path, save_interval_sec=0.0)
    if not data_list:
        state.save()
        return

    cursor = [0]
    cursor_lock = threading.Lock()
    # ``_eval_one_parent`` iterates attempts[*].completions (one
    # ``progress.update`` per slot), so size the bar off the new format.
    def _slot_count(p: dict) -> int:
        attempts = p.get("attempts")
        if not isinstance(attempts, list):
            return 0
        return sum(
            len(a.get("completions") or [])
            for a in attempts
            if isinstance(a, dict)
        )
    total_subs = sum(_slot_count(p) for p in data_list)
    progress = tqdm(total=total_subs, desc="eval-attempt-sub-questions")
    quirks: dict = {"drop_temperature": False, "use_max_completion_tokens": False}

    def worker() -> None:
        while True:
            with cursor_lock:
                if cursor[0] >= len(data_list):
                    return
                parent = data_list[cursor[0]]
                cursor[0] += 1
            try:
                _eval_one_parent(
                    parent_item=parent,
                    judger=judger,
                    quirks=quirks,
                    progress=progress,
                    precision=precision,
                    strip_think_for_model_eval=strip_think_for_model_eval,
                )
                state.save()
            except Exception as exc:  # pragma: no cover - worker-level guard
                print(
                    f"[api] worker error on {parent.get('source_problem_id')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    worker_count = min(nproc, len(data_list))
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(worker_count)]
    for t in threads:
        time.sleep(0.05)
        t.start()
    for t in threads:
        t.join()
    progress.close()

    state.save()