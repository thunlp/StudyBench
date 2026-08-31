"""Extract required knowledge points for every sub-problem of the competition
competition corpus (``studybench_data/competition_problems/competition_problems_full.json``).

This is the FIRST stage of the coverage-argument pipeline:

    (1) [this script]  sub-problem -> knowledge points
    (2) knowledge point -> matching textbook spans in 11 PhysicsBooks
    (3) matched spans -> LLM writes a study guide -> Qwen3-8B solves with it

The goal of Stage 1 is to decompose each sub-problem into a small, named
list of "physics knowledge units" a student would need to solve it. The
output is used by Stage 2 to retrieve textbook spans and by Stage 3 to
generate study guides; it also directly supports the coverage / gap
analysis (textbook x knowledge-point matrix).

Design decisions worth remembering:

  * We label ONE sub-problem at a time. The parent statement and the
    statements (only) of previous sub-problems are shown as CONTEXT.
  * We show the model both the sub-problem STATEMENT and its REFERENCE
    SOLUTION -- solutions are far more informative than statements. To
    let downstream code filter out solution-leaked points, every point
    carries a ``source`` field marking whether it comes from the problem
    statement, the solution, or both.
  * Each point is one of four typed classes:
        concept          -- named physics concept
        law_or_equation  -- named law or key formula
        technique        -- problem-solving technique / approximation trick
        assumption       -- physical assumption or idealisation used
    Typing prevents both under-generation ("mechanics") and
    over-generation (a step of algebra).
  * We do NOT canonicalise names across sub-problems; that is a separate
    stage (embed + cluster + LLM naming), following the same
    single-responsibility convention as the sibling label scripts.

The competition JSON's top level is a plain list, unlike the textbook file
which is ``{"data": [...]}``. We auto-detect and preserve the original
shape when writing back.

``BASE_URL`` / ``API_KEY`` are intentionally blank -- fill them in or pass
``--base-url`` / ``--api-key`` / set ``LABEL_BASE_URL`` / ``LABEL_API_KEY``.
The default endpoint matches ``env_local.sh``'s LLMCenter proxy and the
default model is ``deepseek-v4-pro`` (same as ``repair_failed_answers.py``).

Usage::

    # single-run (uses LABEL_BASE_URL / LABEL_API_KEY env vars if set)
    python filter/extract_knowledge_points.py \\
        --base-url https://llm-center.ali.modelbest.cn/llm/v1 \\
        --api-key   <key>                                         \\
        --model     deepseek-v4-pro                               \\
        --concurrency 4

    # smoke test on 3 parent problems first
    python filter/extract_knowledge_points.py --limit 3

    # re-label everything from scratch
    python filter/extract_knowledge_points.py --force

Output: each sub-problem gets a new ``knowledge_points`` field, written
back in place (atomic temp-file + rename). Runs are resumable; re-running
skips sub-problems that already have a valid label.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    sys.exit("The 'openai' package is required: pip install openai")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kwargs):  # type: ignore
        return it


# ---------------------------------------------------------------------------
# Configuration -- fill BASE_URL / API_KEY in, or override on the CLI / env.
# ---------------------------------------------------------------------------

BASE_URL = ""   # e.g. "https://llm-center.ali.modelbest.cn/llm/v1"
API_KEY = ""    # e.g. "sk-..."
MODEL = "deepseek-v4-pro"

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "studybench_data" / "competition_problems" / "competition_problems_full.json"

ALLOWED_TYPES = {"concept", "law_or_equation", "technique", "assumption"}
ALLOWED_SOURCES = {"problem", "solution", "both"}

# Per-sub soft limits. The prompt asks for 3..8 points; we accept a bit of
# slack to avoid dropping otherwise-well-formed extractions.
MIN_POINTS = 1
MAX_POINTS = 15


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a physics-olympiad content analyst. Your job is to read ONE \
sub-problem from an olympiad competition (with its parent context and \
reference solution) and enumerate the MINIMAL SUFFICIENT set of named \
physics-knowledge units required to solve it.

Think of this as building a "look-up list" a strong student would need \
BEFORE attempting the sub-problem, plus everything the reference solution \
visibly relies on. Each unit is a self-contained, retrievable piece of \
physics knowledge -- something a textbook can teach in a paragraph, a \
subsection, or a worked example. Do NOT enumerate algebraic steps or \
arithmetic; those are the student's job, not the textbook's.

Every knowledge point MUST have a "type" field drawn from this fixed set:

  - "concept":         a named physics concept whose definition or \
qualitative meaning the student must know
                       (examples: "moment of inertia", "adiabatic \
invariant", "Roche limit", "tidal locking", "phase velocity",
                       "reduced mass").

  - "law_or_equation": a named law, principle, or key formula used \
directly in the derivation
                       (examples: "Kepler's third law", "conservation of \
angular momentum", "Bernoulli's equation",
                       "Lorentz force law", "Planck distribution").

  - "technique":       a problem-solving technique / mathematical device \
that is applied
                       (examples: "small-angle approximation", \
"perturbation expansion", "series expansion of 1/sqrt(1+x)",
                       "iterative fixed-point solve", "separation of \
variables", "dimensional analysis").

  - "assumption":      a physical assumption / idealisation that the \
problem states or the solution silently uses
                       (examples: "circular orbit", "isolated \
Earth-Moon system", "quasi-static process", "point particle",
                       "rigid body", "non-relativistic limit").

Naming rules:

  - "name" is a SHORT canonical noun phrase (2 to 6 words). Prefer the \
term as it appears in a standard physics textbook (Griffiths, Morin, \
Purcell, Blundell, Carroll & Ostlie, Taylor & Wheeler, ...). Do not \
include the problem's specific numbers or variables in the name.
  - "description" is 1 to 2 sentences explaining SPECIFICALLY why this \
unit is needed for THIS sub-problem (not a generic definition).
  - "source" records where the evidence for this unit came from:
      "problem"  -- the sub-problem statement itself asks for/uses it,
      "solution" -- only the reference solution reveals it,
      "both"     -- both places make it plain.
  - "keywords" is a list of 2 to 6 SHORT retrieval keywords (each 1-5 \
words) that would help find this unit in a physics textbook via keyword \
or embedding search. Include synonyms and closely related terms.

Coverage rules:

  - Prefer 3 to 8 points per sub-problem. Fewer is fine for a very \
small step; more only if the sub-problem is genuinely multi-topic.
  - Aim for MINIMAL and SUFFICIENT: every point must be actually needed; \
together, they should be enough that a student who knew all of them \
could reproduce the reference solution.
  - Do NOT include "algebra", "arithmetic", "calculus in general", \
"physics in general", or any other unit so broad that any olympiad \
problem trivially depends on it. Do NOT include the final numeric answer \
as a point. Do NOT invent physics that the problem/solution does not \
support.
  - Sub-problems often reuse setups from earlier parts. If the current \
sub-problem's derivation needs a QUANTITY OR RESULT DERIVED in an \
earlier sub-part, model that as an "assumption" or (rarely) a \
"technique" -- but only if the reference solution really imports that \
prior result. Do not re-list the underlying physics of the earlier part.

Output format:

After any reasoning, end your reply with a single JSON object wrapped in \
a ```json code block, in EXACTLY this shape:

```json
{
  "knowledge_points": [
    {
      "type": "concept | law_or_equation | technique | assumption",
      "name": "<2-6 word canonical noun phrase>",
      "description": "<1-2 sentences on why this is needed for this sub-problem>",
      "source": "problem | solution | both",
      "keywords": ["kw1", "kw2", ...]
    }
  ]
}
```

Return only the JSON object at the end. No trailing prose after the \
closing fence.\
"""


USER_TEMPLATE = """\
{header}

[MAIN PROBLEM STATEMENT (context)]
<<<
{parent_problem}
>>>

{prev_subs_block}\
[CURRENT SUB-PROBLEM {cur_id}]
<<<
{cur_problem}
>>>

[REFERENCE SOLUTION for the current sub-problem]
<<<
{cur_solution}
>>>

[REFERENCE ANSWER for the current sub-problem]
{cur_answer}

Enumerate the minimal sufficient set of named knowledge points required \
to solve the CURRENT sub-problem. End with the JSON object in a ```json \
code block."""


# ---------------------------------------------------------------------------
# I/O helpers  (supports both list-at-top and {"data": [...]} layouts)
# ---------------------------------------------------------------------------

def load_blob(path: Path) -> tuple[Any, list[dict[str, Any]], bool]:
    """Return (raw_blob, items_list, wrapped_in_data).

    ``wrapped_in_data`` tells us whether the original file used the
    ``{"data": [...]}`` wrapper so we can preserve it on write-back.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw, raw["data"], True
    if isinstance(raw, list):
        return raw, raw, False
    sys.exit(f"{path}: top level must be a list or {{'data': [...] }}.")


def dump_blob(path: Path, raw: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _header(item: dict[str, Any]) -> str:
    bits = [
        str(item.get("source", "")).strip(),
        str(item.get("year", "")).strip(),
        str(item.get("source_problem_id", "")).strip(),
        str(item.get("title", "")).strip(),
    ]
    line = " | ".join(b for b in bits if b)
    return f"[COMPETITION HEADER] {line}" if line else ""


def render_prompt_inputs(
    parent: dict[str, Any], sub_idx: int
) -> dict[str, str]:
    """Assemble the fields consumed by ``USER_TEMPLATE`` for one sub-problem.

    Previous sub-problems (STATEMENTS only, not their solutions) are shown
    as context; the current sub-problem gets its full solution + answer.
    """
    subs = parent.get("sub_problems") or []
    cur = subs[sub_idx]

    prev_bits: list[str] = []
    for k in range(sub_idx):
        prev = subs[k]
        if not isinstance(prev, dict):
            continue
        pid = str(prev.get("problem_id", "")).strip() or f"sub[{k}]"
        text = str(prev.get("problem", "")).strip()
        if text:
            prev_bits.append(f"[PREVIOUS SUB-PROBLEM {pid} (context only)]\n<<<\n{text}\n>>>")

    prev_block = ("\n\n".join(prev_bits) + "\n\n") if prev_bits else ""

    return {
        "header": _header(parent),
        "parent_problem": str(parent.get("problem", "")).strip() or "(none)",
        "prev_subs_block": prev_block,
        "cur_id": str(cur.get("problem_id", "")).strip() or f"sub[{sub_idx}]",
        "cur_problem": str(cur.get("problem", "")).strip() or "(none)",
        "cur_solution": str(cur.get("solution", "")).strip() or "(none)",
        "cur_answer": str(cur.get("answer", "")).strip() or "(none)",
    }


# ---------------------------------------------------------------------------
# Response parsing (tolerant of thinking output; take the last valid JSON)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _iter_balanced_objects(text: str):
    """Yield every balanced ``{...}`` substring, string/escape aware."""
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
            j += 1
        i = j + 1


def _looks_like_kp_wrapper(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("knowledge_points"), list)
    )


def _normalise_point(raw: Any) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    t = str(raw.get("type", "") or "").strip().lower()
    if t not in ALLOWED_TYPES:
        return None
    name = str(raw.get("name", "") or "").strip()
    if not name or len(name) > 200:
        return None
    desc = str(raw.get("description", "") or "").strip()
    if not desc:
        return None
    src = str(raw.get("source", "") or "").strip().lower()
    if src not in ALLOWED_SOURCES:
        src = "solution"  # conservative default: assume solution-derived
    kws_raw = raw.get("keywords")
    kws: list[str] = []
    if isinstance(kws_raw, list):
        for k in kws_raw:
            if isinstance(k, str):
                s = k.strip()
                if s and len(s) <= 100:
                    kws.append(s)
    return {
        "type": t,
        "name": name,
        "description": desc,
        "source": src,
        "keywords": kws,
    }


def parse_knowledge_points(text: str) -> Optional[list[dict[str, Any]]]:
    """Extract the ``knowledge_points`` list from the model reply.

    Prefer the JSON in the last ```json``` fence; fall back to scanning the
    whole reply. Reject wrapper objects that lack a list.
    """
    if not text:
        return None

    fences = _FENCE_RE.findall(text)
    search_spaces = ([fences[-1]] if fences else []) + [text]

    wrapper: Optional[dict[str, Any]] = None
    for space in search_spaces:
        last = None
        for cand in _iter_balanced_objects(space):
            try:
                parsed = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if _looks_like_kp_wrapper(parsed):
                last = parsed
        if last is not None:
            wrapper = last
            break
    if wrapper is None:
        return None

    points: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in wrapper["knowledge_points"]:
        p = _normalise_point(raw)
        if p is None:
            continue
        key = (p["type"], p["name"].lower())
        if key in seen:
            continue
        seen.add(key)
        points.append(p)
        if len(points) >= MAX_POINTS:
            break

    if len(points) < MIN_POINTS:
        return None
    return points


def kp_label_is_valid(label: Any) -> bool:
    """Whether an existing ``knowledge_points`` field is complete enough
    to skip on resume."""
    if not isinstance(label, list) or not label:
        return False
    for p in label:
        if not isinstance(p, dict):
            return False
        if p.get("type") not in ALLOWED_TYPES:
            return False
        if not str(p.get("name", "")).strip():
            return False
        if not str(p.get("description", "")).strip():
            return False
        if p.get("source") not in ALLOWED_SOURCES:
            return False
    return True


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def extract_one(
    client: OpenAI,
    model: str,
    fields: dict[str, str],
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
) -> dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(**fields)
    last_err = ""
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            points = parse_knowledge_points(text)
            if points is not None:
                return {"points": points}
            last_err = f"unparseable reply: {text[:200]!r}"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return {"points": None, "error": last_err or "failed after retries"}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_targets(
    items: list[dict[str, Any]],
    parent_indices: list[int],
    force: bool,
):
    """Yield (parent_idx, sub_idx) pairs that still need labelling."""
    for i in parent_indices:
        parent = items[i]
        subs = parent.get("sub_problems") or []
        for k, sub in enumerate(subs):
            if not isinstance(sub, dict):
                continue
            if not str(sub.get("problem", "")).strip():
                continue
            if not force and kp_label_is_valid(sub.get("knowledge_points")):
                continue
            yield i, k


def process_file(
    client: OpenAI,
    model: str,
    in_path: Path,
    concurrency: int,
    limit: Optional[int],
    indices: Optional[list[int]],
    max_tokens: int,
    temperature: float,
    force: bool = False,
    save_every: int = 50,
) -> dict[str, Any]:
    raw, items, wrapped = load_blob(in_path)

    if indices is not None:
        parent_indices = [i for i in indices if 0 <= i < len(items)]
    else:
        cap = len(items) if limit is None else min(limit, len(items))
        parent_indices = list(range(cap))
    scope = set(parent_indices)

    todo = list(iter_targets(items, parent_indices, force))

    already = sum(
        1
        for i in parent_indices
        for sub in (items[i].get("sub_problems") or [])
        if isinstance(sub, dict) and kp_label_is_valid(sub.get("knowledge_points"))
    )
    total_units = sum(
        1
        for i in parent_indices
        for sub in (items[i].get("sub_problems") or [])
        if isinstance(sub, dict) and str(sub.get("problem", "")).strip()
    )
    scope_desc = (
        f"indices={sorted(scope)[:8]}{'...' if len(scope) > 8 else ''} "
        f"({len(parent_indices)} parents)"
        if indices is not None
        else f"cap={len(parent_indices)}"
    )
    print(
        f"[{in_path.name}] {len(items)} parents ({scope_desc}), "
        f"{total_units} sub-problems in scope, "
        f"{already} already labelled, {len(todo)} to do.",
        flush=True,
    )

    lock = threading.Lock()
    done_since_save = [0]

    def work(task: tuple[int, int]) -> tuple[int, int, dict[str, Any]]:
        pi, si = task
        fields = render_prompt_inputs(items[pi], si)
        result = extract_one(client, model, fields, max_tokens, temperature)
        return pi, si, result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(work, t) for t in todo]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=in_path.name
        ):
            pi, si, result = fut.result()
            sub = items[pi]["sub_problems"][si]
            with lock:
                if result.get("points") is not None:
                    sub["knowledge_points"] = result["points"]
                    sub.pop("knowledge_points_error", None)
                else:
                    # Preserve any earlier valid label; only record the error.
                    sub["knowledge_points_error"] = result.get("error", "")
                done_since_save[0] += 1
                if done_since_save[0] >= save_every:
                    dump_blob(in_path, raw if wrapped else items)
                    done_since_save[0] = 0

    dump_blob(in_path, raw if wrapped else items)

    # ---- summary --------------------------------------------------------
    n_subs = 0
    n_labelled = 0
    n_err = 0
    type_hist: dict[str, int] = {t: 0 for t in ALLOWED_TYPES}
    source_hist: dict[str, int] = {s: 0 for s in ALLOWED_SOURCES}
    len_hist: dict[str, int] = {}
    for i in parent_indices:
        for sub in (items[i].get("sub_problems") or []):
            if not isinstance(sub, dict):
                continue
            if not str(sub.get("problem", "")).strip():
                continue
            n_subs += 1
            pts = sub.get("knowledge_points")
            if kp_label_is_valid(pts):
                n_labelled += 1
                for p in pts:
                    type_hist[p["type"]] = type_hist.get(p["type"], 0) + 1
                    source_hist[p["source"]] = source_hist.get(p["source"], 0) + 1
                bucket = str(len(pts))
                len_hist[bucket] = len_hist.get(bucket, 0) + 1
            if sub.get("knowledge_points_error"):
                n_err += 1
    return {
        "total_subproblems": n_subs,
        "labelled": n_labelled,
        "errors": n_err,
        "points_per_subproblem_hist": len_hist,
        "type_hist": type_hist,
        "source_hist": source_hist,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract per-sub-problem knowledge points for the competition "
            "test set (first stage of the coverage-argument pipeline)."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LABEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or BASE_URL,
        help="OpenAI-compatible endpoint. Falls back to LABEL_BASE_URL / OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LABEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or API_KEY,
        help="API key. Falls back to LABEL_API_KEY / OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("LABEL_MODEL", MODEL),
        help="Model name (default: deepseek-v4-pro).",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Label sub-problems for only the first N parent problems "
             "(ignored if --indices is given).",
    )
    parser.add_argument(
        "--indices", type=str, default=None,
        help="Comma-separated list of specific parent indices to process "
             "(e.g. '0,5,17,42'). Overrides --limit. Useful for stratified "
             "samples across sources / problem_types.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-label every sub-problem, overwriting existing knowledge_points.",
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    indices: Optional[list[int]] = None
    if args.indices is not None:
        try:
            indices = sorted({int(x) for x in args.indices.split(",") if x.strip()})
        except ValueError:
            sys.exit(f"--indices must be comma-separated integers, got: {args.indices!r}")

    if not args.base_url or not args.api_key:
        sys.exit(
            "BASE_URL / API_KEY are empty. Fill them in at the top of this "
            "file, pass --base-url/--api-key, or set LABEL_BASE_URL/"
            "LABEL_API_KEY (env_local.sh's OPENAI_BASE_URL/OPENAI_API_KEY "
            "are also picked up as a fallback)."
        )
    if not args.input.exists():
        sys.exit(f"missing input: {args.input}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    stats = process_file(
        client=client,
        model=args.model,
        in_path=args.input,
        concurrency=args.concurrency,
        limit=args.limit,
        indices=indices,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        force=args.force,
        save_every=args.save_every,
    )
    print(f"\n   -> {args.input.name} (updated in place)", flush=True)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
