#!/usr/bin/env python3
"""LLM verification of retrieved KP -> textbook fragment matches.

Stage 4 of the coverage-argument pipeline.  ``retrieve_textbook_candidates.py``
gives us ~30 candidate fragments per canonical knowledge point.  Many of
those are retrieval false positives -- fragments that share vocabulary with
the KP but don't actually teach or exemplify it.  This stage asks an LLM
to look at every candidate and decide whether the fragment genuinely
covers the KP, then extract a short verbatim quote as evidence.

Design decisions -- mirrored on ``extract_knowledge_points.py`` so
operators only need to learn one pattern:

  * One LLM call per KP.  All candidates for a KP go into a single
    prompt as a numbered list; the model returns a JSON array with one
    entry per candidate.  This exploits shared context (definition of
    the KP, scoring rubric) and cuts total calls by ``top_verify`` x.
  * ``score`` is on a 0-3 rubric:
        0  irrelevant / off-topic
        1  mentions or touches the KP but doesn't teach it
        2  worked example that clearly USES the KP, or an exposition
           passage that partially defines it
        3  direct exposition / statement of the KP -- the fragment
           essentially is the textbook's teaching of that KP
  * ``quote`` is a short (<= 2 sentence) verbatim span COPIED from the
    fragment.  We verify server-side that the quote actually appears
    in the fragment's ``text_md``; if not the match is demoted.
    (LLM hallucination guard.)  The check is LaTeX-aware -- see
    ``_normalise_quote``; an earlier strict version demoted 2621
    legitimate matches, 87% of them purely for containing math markup
    (``knowledge_points/COVERAGE_REPORT.md`` "缺陷 2").
  * Same run infrastructure as Stage 1: OpenAI-compatible client,
    ``LABEL_BASE_URL`` / ``LABEL_API_KEY`` env vars (falling back to
    ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``), ThreadPoolExecutor +
    atomic write every ``--save-every`` calls, resumable.

Input:  ``studybench_data/kp_candidates.jsonl``  +  ``studybench_data/textbook_fragments.jsonl``
Output: ``studybench_data/kp_matches.jsonl`` -- one record per KP::

    {
      "kp_id":            "abc123deadbeef",
      "canonical_name":   "Conservation of angular momentum",
      "type":             "law_or_equation",
      "n_candidates":     20,
      "verifier_model":   "deepseek-v4-pro",
      "matches": [ {fragment_id, book_display, ..., score, quote,
                    quote_verified, reason, candidate_rank}, ... ],
      "n_strong": 2, "n_partial": 3, "n_weak": 1, "n_zero": 14,
      "verifier_error": null
    }

Usage::

    # smoke: verify the first 20 KPs, top-15 candidates each
    LABEL_BASE_URL=... LABEL_API_KEY=... \\
      python filter/verify_textbook_matches.py --limit 20 --top-verify 15

    # full pass (uses env_local.sh vars automatically); resumes by default
    python filter/verify_textbook_matches.py

    # re-verify a specific KP
    python filter/verify_textbook_matches.py --kp-ids abc123deadbeef --force
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("The 'openai' package is required: pip install openai")

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **_):
        return it

ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = ROOT / "studybench_data"
FRAGMENTS_PATH = STUDY_DIR / "textbook_fragments.jsonl"
CANDIDATES_PATH = STUDY_DIR / "kp_candidates.jsonl"
OUT_PATH = STUDY_DIR / "kp_matches.jsonl"
DEFAULT_MODEL = "deepseek-v4-pro"


SYSTEM_PROMPT = """You are a physics-olympiad content analyst deciding whether each textbook fragment covers a specific knowledge point (KP).

You will be given ONE KP (its name, type, description, and the physics sub-problems it is required for), and a numbered list of candidate textbook FRAGMENTS.

For every candidate fragment, decide a coverage score on this rubric:

  3  DIRECT EXPOSITION -- the fragment is a textbook's teaching of the KP: it defines, derives, or states the KP as its primary subject. Section headings like "Kepler's third law" that then explain the law are the typical case.
  2  APPLIED / EXAMPLE -- the fragment is a worked example, or an exposition passage that uses the KP as a tool or partial ingredient (so a student would learn the KP by studying it), but the KP is not the primary subject.
  1  MENTIONS / TOUCHES -- the fragment mentions the KP or a closely related idea in passing (a single sentence, a formula in a list, a footnote), but does not teach it.
  0  IRRELEVANT -- keyword overlap only, or the fragment is off-topic.

Rules for the "quote" field:

  * If score >= 2, "quote" MUST be a SHORT VERBATIM span (<= 2 sentences, <= 300 characters) COPIED EXACTLY from the fragment. It should be the single most convincing sentence proving coverage.
  * If score == 1, "quote" may be a shorter span (a phrase is fine).
  * If score == 0, "quote" MUST be the empty string "".
  * The quote must appear byte-for-byte in the fragment. Do not paraphrase, translate, or shorten with "...".
  * If the fragment has ONLY equations for the relevant part, quote a short surrounding sentence + the equation (e.g. "L = I omega. This angular momentum is conserved."). It is fine for the quote to contain LaTeX math; keep it verbatim.

Rules for the "reason" field:

  * ONE clause, at most 20 words, explaining WHY this score was chosen.
  * When score >= 2, mention which physical concept in the fragment matches the KP (e.g. "Directly states omega ^2 r^3 = GM.").
  * When score == 0, give the confounder (e.g. "About viscosity, not Archimedes.").

Output format:

Return exactly ONE JSON object wrapped in a ```json fence, with a single key "matches" whose value is a list of length equal to the number of candidates. Each entry has {"index", "score", "quote", "reason"}. Indexes MUST match the numbered candidates in the prompt.

```json
{
  "matches": [
    {"index": 1, "score": 3, "quote": "...", "reason": "..."},
    {"index": 2, "score": 0, "quote": "",    "reason": "..."},
    ...
  ]
}
```

Return only the JSON object; do not add prose after the closing fence."""


USER_TEMPLATE = """[KNOWLEDGE POINT under evaluation]
type: {kp_type}
name: {kp_name}
aliases: {kp_aliases}
description(s):
{kp_descriptions}
keywords: {kp_keywords}
required by {n_subs} sub-problem(s) in problem_type(s): {kp_domains}

[CANDIDATE FRAGMENTS ({n_cand} total)]
{candidates_block}

Score every candidate. End with the JSON object described in the system prompt (single fenced ```json block, "matches" list of length {n_cand})."""


class SlidingWindowRateLimiter:
    """Thread-safe request limiter for the endpoint's per-key RPM quota.

    Mirrors the implementation in ``review_modern_astrophysics_self_containment.py``
    so operators only have to learn one pattern.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                delay = 60 - (now - self._timestamps[0]) + 0.05
            time.sleep(max(delay, 0.05))


class RateLimitedClient:
    """Expose the subset of the OpenAI client interface ``verify_one_kp`` uses."""

    def __init__(self, client: OpenAI, limiter: SlidingWindowRateLimiter) -> None:
        create = client.chat.completions.create

        def rate_limited_create(*args, **kwargs):
            limiter.acquire()
            return create(*args, **kwargs)

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=rate_limited_create)
        )


def render_candidate(i: int, frag_snippet: str, cand: dict[str, Any]) -> str:
    """One numbered candidate block for the prompt."""
    section = " > ".join(cand.get("section_path") or [])
    kind = "EXAMPLE" if cand.get("kind") == "example" else "EXPOSITION"
    manual = " [SOLUTION MANUAL]" if cand.get("is_solution_manual") else ""
    return (f"[{i}]  [{kind}{manual}]  "
            f"{cand.get('book_display', '')}  ::  {section}"
            f"\n<<<\n{frag_snippet}\n>>>")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _iter_balanced_objects(text: str):
    """Yield every balanced {...} span, so we can recover the payload even
    when the model forgets the fence or trails prose after it."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:i + 1]
                start = -1


def _extract_matches_object(text: str) -> Optional[dict]:
    """Pull the {"matches": [...]} object out of a model reply."""
    for m in _FENCE_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and isinstance(obj.get("matches"), list):
                return obj
        except json.JSONDecodeError:
            pass
    for blob in _iter_balanced_objects(text or ""):
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("matches"), list):
            return obj
    return None


# ---------------------------------------------------------------------------
# Verbatim-quote guard (LaTeX aware)
# ---------------------------------------------------------------------------

_ALNUM_ONLY = re.compile(r"[^0-9a-z]+")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_ELLIPSIS = re.compile(r"\.\.\.+|…")


def _normalise_quote(s: str) -> str:
    """Aggressive normalisation for verbatim-in-fragment checks.

    LLMs frequently reformat LaTeX ("PV=NkT" vs "P V = N k T"), swap
    curly quotes for straight ones, or add/remove spaces around ``$``.
    They also drop or reformat LaTeX command names ("\\frac", "\\mathbf",
    "\\theta") and Unicode math glyphs (subscripts, Greek letters) that
    appear in the source fragment.  We (1) NFKD-normalise to fold
    accents/compatibility glyphs, (2) strip LaTeX command names, then
    (3) collapse to alphanumerics only for the substring check, so those
    non-semantic differences don't produce false 'not verbatim' verdicts.
    """
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = _LATEX_CMD.sub(" ", s)
    return _ALNUM_ONLY.sub("", s)


def _quote_in_fragment(quote: str, fragment_text: str) -> bool:
    """True if ``quote`` appears in ``fragment_text`` modulo formatting.

    Ellipsis-stitched quotes ("A ... B") are checked segment by segment,
    since the model is told not to elide but sometimes does anyway.  A
    segment shorter than 20 normalised characters fails the check: a
    handful of alphanumerics (typically a bare formula like ``k=n\\pi/L``
    stripped of its LaTeX commands) will substring-match almost any
    fragment, so accepting it would wave through hallucinated quotes.

    The 20-char floor was tuned against the 15452 already-adjudicated
    quotes in the previous run's ``kp_matches.jsonl`` (92.3% agreement,
    the best of every threshold tried).

    Also imported by ``recover_demoted_matches.py``.
    """
    if not quote:
        return False
    haystack = _normalise_quote(fragment_text)
    if not haystack:
        return False
    for seg in _ELLIPSIS.split(quote):
        needle = _normalise_quote(seg)
        if len(needle) < 20:
            return False
        if needle not in haystack:
            return False
    return True


def parse_verifier_response(text: str, candidates: list[dict[str, Any]],
                            fragments_by_id: dict[str, dict]) -> Optional[list[dict]]:
    """Merge the model's scores back onto the candidate metadata.

    Returns None if the reply is unusable, so the caller can retry.
    """
    obj = _extract_matches_object(text)
    if obj is None:
        return None

    by_index: dict[int, dict] = {}
    for entry in obj["matches"]:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(candidates):
            by_index[idx] = entry

    out: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates, 1):
        entry = by_index.get(i)
        if entry is None:
            continue
        try:
            score = int(entry.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(3, score))
        quote = str(entry.get("quote", "") or "")
        reason = str(entry.get("reason", "") or "")

        frag = fragments_by_id.get(cand["fragment_id"]) or {}
        verified = _quote_in_fragment(quote, frag.get("text_md", ""))

        score_original = score
        note = ""
        # A >=2 claim has to be backed by a quote we can actually find.
        if score >= 2 and not verified:
            score = 1
            note = ("quote not found verbatim in fragment" if quote
                    else "score>=2 but no quote returned")

        out.append({
            "fragment_id": cand["fragment_id"],
            "book": cand.get("book", ""),
            "book_display": cand.get("book_display", ""),
            "domain": cand.get("domain", ""),
            "chapter": cand.get("chapter", ""),
            "chapter_title": cand.get("chapter_title", ""),
            "section_path": cand.get("section_path") or [],
            "kind": cand.get("kind", ""),
            "is_solution_manual": bool(cand.get("is_solution_manual")),
            "candidate_rank": i,
            "bm25_rank": cand.get("bm25_rank"),
            "emb_rank": cand.get("emb_rank"),
            "final_score": cand.get("final_score"),
            "score": score,
            "score_original": score_original,
            "quote": quote,
            "quote_verified": verified,
            "reason": reason,
            "note": note,
        })
    return out or None


def _summarise_scores(matches: list[dict]) -> dict[str, int]:
    return {
        "n_strong": sum(1 for m in matches if m["score"] == 3),
        "n_partial": sum(1 for m in matches if m["score"] == 2),
        "n_weak": sum(1 for m in matches if m["score"] == 1),
        "n_zero": sum(1 for m in matches if m["score"] == 0),
    }


def verify_one_kp(client: Any, model: str, kp_row: dict[str, Any],
                  candidates: list[dict[str, Any]],
                  fragments_by_id: dict[str, dict],
                  fragment_snippet_chars: int, max_tokens: int,
                  temperature: float, max_retries: int) -> dict[str, Any]:
    descs: list[str] = []
    for d in (kp_row.get("descriptions") or [])[:4]:
        sub_id = d.get("sub_id", "")
        text = (d.get("description", "") or "")[:280]
        descs.append(f"  - ({sub_id}) {text}")
    desc_block = "\n".join(descs) or "  - (none)"

    keywords = ", ".join((kp_row.get("keywords_union") or [])[:20]) or "(none)"
    domains = ", ".join(kp_row.get("expected_domains") or []) or "(unspecified)"

    blocks = []
    for i, cand in enumerate(candidates, 1):
        frag = fragments_by_id.get(cand["fragment_id"])
        snippet = (frag or {}).get("text_md", "(fragment missing)")
        blocks.append(render_candidate(i, snippet[:fragment_snippet_chars], cand))

    user = USER_TEMPLATE.format(
        kp_type=kp_row.get("type", ""),
        kp_name=kp_row.get("canonical_name", ""),
        kp_aliases=", ".join(kp_row.get("aliases") or []) or "(none)",
        kp_descriptions=desc_block,
        kp_keywords=keywords,
        n_subs=kp_row.get("n_sub_problems", 0),
        kp_domains=domains,
        n_cand=len(candidates),
        candidates_block="\n\n".join(blocks),
    )

    last_err = "failed after retries"
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            matches = parse_verifier_response(text, candidates, fragments_by_id)
            if matches is not None:
                return {"matches": matches, "error": None}
            last_err = "unparseable reply: " + text[:200]
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return {"matches": [], "error": last_err}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dump_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    ap.add_argument("--fragments", type=Path, default=FRAGMENTS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=os.environ.get("LABEL_MODEL", DEFAULT_MODEL))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--top-verify", type=int, default=30,
                    help="verify at most N candidates per KP")
    ap.add_argument("--fragment-snippet-chars", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kp-ids", default="")
    ap.add_argument("--force", action="store_true",
                    help="re-verify KPs already present in --out")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--skip-singleton-assumptions", action="store_true",
                    help="skip 'assumption' KPs used by only one sub-problem. "
                         "These are problem-specific modelling assumptions "
                         "('neglect air density', 'uniform wind speed') that "
                         "textbooks do not state, so they verify as uncovered "
                         "at high cost -- the assumption type already scores "
                         "lowest for coverage (see COVERAGE_REPORT.md)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-request timeout in seconds (default 180). The "
                         "SDK default is 600s with 2 retries, so one stuck "
                         "request can occupy a worker for 30 minutes")
    ap.add_argument("--requests-per-minute", type=int, default=50,
                    help="client-side RPM cap matching the endpoint quota "
                         "(default 50). Exceeding it returns 429s that burn "
                         "retries and land as verifier_error placeholders")
    args = ap.parse_args()

    base_url = (args.base_url or os.environ.get("LABEL_BASE_URL")
                or os.environ.get("OPENAI_BASE_URL"))
    api_key = (args.api_key or os.environ.get("LABEL_API_KEY")
               or os.environ.get("OPENAI_API_KEY"))
    if not api_key:
        sys.exit("[verify] no API key (set LABEL_API_KEY or OPENAI_API_KEY)")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)
    if args.requests_per_minute > 0:
        client = RateLimitedClient(
            client, SlidingWindowRateLimiter(args.requests_per_minute))
        print(f"[verify] rate limited to {args.requests_per_minute} req/min")

    fragments_by_id = {f["fragment_id"]: f for f in load_jsonl(args.fragments)}
    candidate_rows = load_jsonl(args.candidates)
    print(f"[verify] {len(candidate_rows)} KP candidate rows, "
          f"{len(fragments_by_id)} fragments")

    # --- resume: keep already-verified KPs untouched ------------------------
    # Only SUCCESSFUL records count as done. A record carrying a
    # verifier_error is a placeholder from a failed run (API outage, quota
    # exhaustion, ...); treating it as done would silently bake the failure
    # in, since it looks identical to a genuine "no matches found" result
    # downstream.
    done: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    if args.out.exists() and not args.force:
        for r in load_jsonl(args.out):
            if r.get("verifier_error"):
                failed[r["kp_id"]] = r
            else:
                done[r["kp_id"]] = r
        print(f"[verify] {len(done)} KPs already verified in {args.out.name}")
        if failed:
            print(f"[verify] {len(failed)} KPs carry a previous error "
                  f"and will be RETRIED")

    wanted = {s.strip() for s in args.kp_ids.split(",") if s.strip()}
    todo = [r for r in candidate_rows
            if (not wanted or r["kp_id"] in wanted)
            and (args.force or r["kp_id"] not in done)]
    if args.skip_singleton_assumptions:
        before = len(todo)
        todo = [r for r in todo
                if not (r.get("type") == "assumption"
                        and (r.get("n_sub_problems") or 0) <= 1)]
        # Say what was dropped -- a silent filter reads as full coverage later.
        print(f"[verify] skipped {before - len(todo)} singleton-assumption KPs "
              f"(--skip-singleton-assumptions); they stay unverified and will "
              f"count as uncovered downstream")
    if args.limit:
        todo = todo[:args.limit]
    print(f"[verify] {len(todo)} KPs to verify with {args.model}, "
          f"{len(candidate_rows) - len(todo)} skipped")
    if not todo:
        print("[verify] nothing to do")
        return

    results: dict[str, dict] = dict(done)
    lock = threading.Lock()
    n_done = 0

    def flush() -> None:
        # Carry forward not-yet-retried error placeholders so an interrupted
        # retry run doesn't drop them from the output file entirely.
        merged = {**failed, **results}
        ordered = [merged[r["kp_id"]] for r in candidate_rows
                   if r["kp_id"] in merged]
        dump_jsonl(args.out, ordered)

    def work(row: dict[str, Any]) -> dict[str, Any]:
        cands = (row.get("candidates") or [])[:args.top_verify]
        res = verify_one_kp(client, args.model, row, cands, fragments_by_id,
                            args.fragment_snippet_chars, args.max_tokens,
                            args.temperature, args.max_retries)
        rec = {
            "kp_id": row["kp_id"],
            "canonical_name": row.get("canonical_name", ""),
            "type": row.get("type", ""),
            "n_candidates": len(cands),
            "verifier_model": args.model,
            "verifier_error": res["error"],
            "matches": res["matches"],
        }
        rec.update(_summarise_scores(res["matches"]))
        return rec

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(work, row): row for row in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="verify"):
            rec = fut.result()
            with lock:
                results[rec["kp_id"]] = rec
                n_done += 1
                if n_done % args.save_every == 0:
                    flush()

    flush()

    fresh = [results[r["kp_id"]] for r in todo if r["kp_id"] in results]
    n_err = sum(1 for r in fresh if r["verifier_error"])
    n_ok = len(fresh) - n_err
    covered = sum(1 for r in fresh
                  if not r["verifier_error"] and (r["n_strong"] or r["n_partial"]))
    print(f"[verify] wrote {len(results)} records -> {args.out}")
    print(f"[verify] new: {len(fresh)}, succeeded: {n_ok}, errors: {n_err}")
    # Denominator is successful calls -- dividing by all attempts would
    # understate coverage whenever a run hits API failures.
    print(f"[verify] covered (>=1 match scoring >=2): {covered}/{n_ok}")
    if n_err:
        print(f"[verify] {n_err} KPs failed; re-run this command to retry "
              f"them (errored records are not treated as done)")


if __name__ == "__main__":
    main()
