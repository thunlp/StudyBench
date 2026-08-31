#!/usr/bin/env python3
"""Knowledge-point-driven guidance generation with a strong teacher (DeepSeek-V4-Pro).

This is the successor to ``filter/pipeline.py``.  The old pipeline quoted whole
textbooks by ``problem_type`` (noisy) and fed the full gold solution to a weaker
teacher (V4-Flash) with only a soft "don't reveal the answer" instruction.

The new pipeline exploits the coverage-argument artefacts:

  * ``studybench_data/competition_problems_full.with_coverage.json``
    -- every sub-problem's knowledge points, each carrying ``textbook_matches``
       with the top verified exposition / example fragments (fragment_id + quote
       + section + verifier score).
  * ``studybench_data/textbook_fragments.jsonl``
    -- the full ``text_md`` of every fragment, looked up by ``fragment_id``.

For each sub-problem we therefore hand the teacher *exactly the textbook passages
that were independently verified to teach this sub-problem's knowledge points*,
plus the problem and (for the teacher's own understanding) the gold solution.

The teacher writes a **methodological** guidance: which textbook concepts /
formulae / worked examples to use and in what order -- WITHOUT revealing the
final answer or the specific key calculation.  We enforce that principle with a
**dual leakage gate**:

  1. Rule gate  -- reuse ``build_targeted_guidance``'s ``sanitize_reference`` +
     ``leakage_reasons`` (strips \\boxed, final-answer lines, the literal gold
     answer, and paragraphs that state a conclusion with a gold number).
  2. LLM gate   -- V4-Pro reviews whether the guidance reveals the final answer
     OR performs a key calculation step that would hand the teacher's ability to
     the student.

A guidance that fails either gate is regenerated (up to ``--max-retry`` times);
the last attempt is rule-sanitised as a hard fallback so no leak survives.

Output matches the benchmark-shaped competition corpus (list of parents; each
``sub_problems[i].guidance`` holds the hint), so ``eval/run_benchmark.py
--guidance_path`` consumes it unchanged.

Usage::

    source env_local.sh          # OPENAI_API_KEY / OPENAI_BASE_URL for llm-center
    python filter/build_kp_guidance.py \\
        --coverage studybench_data/competition_problems_full.with_coverage.json \\
        --skeleton studybench_data/competition_problems/competition_problems_full.json \\
        --fragments studybench_data/textbook_fragments.jsonl \\
        --out studybench_data/level3_guidance_full.json \\
        --model deepseek-v4-pro --concurrency 8

    # smoke on the first 5 parents:
    python filter/build_kp_guidance.py --limit 5 --out /tmp/kp_guidance_smoke.json
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
    def tqdm(it, **_kw):
        return it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "filter"))
# Reuse the battle-tested redaction / leakage helpers.
from build_targeted_guidance import (  # noqa: E402
    answer_bodies,
    leakage_reasons,
    sanitize_reference,
)

DEFAULT_COVERAGE = ROOT / "studybench_data/competition_problems_full.with_coverage.json"
DEFAULT_SKELETON = ROOT / "studybench_data/competition_problems/competition_problems_full.json"
DEFAULT_FRAGMENTS = ROOT / "studybench_data/textbook_fragments.jsonl"
DEFAULT_OUT = ROOT / "studybench_data/level3_guidance_full.json"
# Optional legacy output used to carry completed guidance into the new cache.
# Missing files are treated as an empty resume source.
DEFAULT_RESUME_FROM = ROOT / "eval/data/level3_guidance_full.json"
DEFAULT_MODEL = os.environ.get("LABEL_MODEL", "deepseek-v4-pro")

Key = tuple[str, str, str, str]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parents_of(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"]
    raise ValueError("Expected a JSON list or {data: [...]}")


def guidance_results(path: Path) -> dict[Key, dict]:
    """Read non-empty guidance from a previous benchmark-shaped JSON file."""
    if not path.exists():
        print(f"[kp-guidance] resume source not found: {path}")
        return {}

    results: dict[Key, dict] = {}
    for parent in parents_of(read_json(path)):
        for sub in parent.get("sub_problems") or []:
            guidance = sub.get("guidance")
            if not isinstance(guidance, str) or not guidance.strip():
                continue
            results[sub_key(parent, sub)] = {
                "guidance": guidance,
                "guidance_check": sub.get("guidance_check") or {},
            }
    return results


def sub_key(parent: dict, sub: dict) -> Key:
    return (
        str(parent.get("source") or ""),
        str(parent.get("year") or ""),
        str(parent.get("source_problem_id") or ""),
        str(sub.get("problem_id") or ""),
    )


def load_fragments(path: Path) -> dict[str, dict]:
    frags: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frags[d["fragment_id"]] = d
    return frags


# ---------------------------------------------------------------------------
# Material assembly (KP -> verified textbook fragments)
# ---------------------------------------------------------------------------

def _is_covered(tm: dict) -> bool:
    return (tm.get("n_strong", 0) + tm.get("n_partial", 0)) >= 1


def build_material(
    sub: dict,
    fragments: dict[str, dict],
    max_frag_chars: int,
    max_kps: int,
) -> tuple[str, list[str]]:
    """Return (material_text, kp_names) for one sub-problem.

    We walk the sub-problem's knowledge points, and for each *covered* KP pull
    its best verified exposition fragment (falling back to example).  Fragment
    full text is deduplicated across KPs so a shared passage is shown once.
    """
    kp_names: list[str] = []
    blocks: list[str] = []
    seen_frag: set[str] = set()

    kps = sub.get("knowledge_points") or []
    # Prefer KPs with strong matches first, then partial; keep ordering stable.
    def kp_rank(kp: dict) -> tuple[int, int]:
        tm = kp.get("textbook_matches") or {}
        return (-tm.get("n_strong", 0), -tm.get("n_partial", 0))

    ordered = sorted(
        [kp for kp in kps if _is_covered(kp.get("textbook_matches") or {})],
        key=kp_rank,
    )

    for kp in ordered[:max_kps]:
        tm = kp.get("textbook_matches") or {}
        name = kp.get("name") or tm.get("canonical_name") or "(unnamed)"
        # top_exposition preferred; each entry is a scored match with fragment_id
        picks = (tm.get("top_exposition") or []) + (tm.get("top_example") or [])
        picks = [p for p in picks if p.get("score", 0) >= 2]
        if not picks:
            continue
        kp_names.append(name)
        frag_texts: list[str] = []
        for p in picks[:2]:
            fid = p.get("fragment_id")
            if not fid or fid in seen_frag:
                continue
            seen_frag.add(fid)
            frag = fragments.get(fid)
            if not frag:
                continue
            book = frag.get("book_display", "")
            sect = " > ".join(frag.get("section_path") or [])
            body = (frag.get("text_md") or "")[:max_frag_chars].strip()
            kind = "worked example" if frag.get("kind") == "example" else "exposition"
            frag_texts.append(
                f"[{book} — {sect} ({kind})]\n{body}"
            )
        if frag_texts:
            blocks.append(
                f"### Knowledge point: {name}\n" + "\n\n".join(frag_texts)
            )

    return ("\n\n".join(blocks), kp_names)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GEN_SYSTEM = (
    "You are an expert physics teacher writing study guidance grounded strictly "
    "in provided textbook passages. You understand the full solution, but your "
    "job is to teach the student HOW to think, not to hand them the answer."
)

GEN_TEMPLATE = """You are given a physics problem, its reference solution (for YOUR understanding only), the knowledge points it requires, and the exact textbook passages that teach those knowledge points.

Write a solving **guidance** for a student who has studied these textbook passages.

STRICT RULES — the guidance MUST NOT transfer the teacher's ability to the student:
1. Do NOT state the final answer, any intermediate numerical result, or any boxed expression.
2. Do NOT perform the key calculation. You may name the formula/law/example to use and explain WHY and in what ORDER, but leave every substitution and every arithmetic/algebraic step for the student to carry out.
3. Ground every formula, concept, and worked-example reference in the textbook passages below. Do not introduce facts absent from them.
4. Phrase it as method: "First use <concept/formula from the textbook> to ..., because ...; then relate it to ... using <formula>; finally combine them to isolate the requested quantity." Point to the relevant textbook example by its idea, not by copying its numbers.

### Problem
{problem}

### Reference solution (for your understanding — DO NOT reveal its results)
{solution}

### Required knowledge points
{kp_list}

### Textbook passages (the only sources you may rely on)
{material}

Output only the guidance text, no preamble, under the heading:
### Guidance
<your guidance>
"""

REVIEW_SYSTEM = (
    "You are a strict examiner checking whether study guidance leaks the answer "
    "or the teacher's key calculations to the student."
)

REVIEW_TEMPLATE = """A guidance is acceptable ONLY if it teaches method without transferring the teacher's ability. It FAILS if it does any of:
- states or strongly implies the final answer or any intermediate numeric result;
- carries out a key calculation (substitutes numbers into a formula and computes, or does the decisive algebraic isolation) that the student should do themselves;
- reveals a boxed expression or an explicit "answer is ..." statement.

It PASSES if it only names which textbook concepts/formulae/examples to use and in what order, leaving all substitution and computation to the student.

### Problem
{problem}

### Gold answer (what must NOT be revealed)
{answer}

### Guidance under review
{guidance}

Return exactly:
### Verdict
PASS or FAIL
### Reason
<one short sentence>
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call(client: OpenAI, model: str, system: str, user: str,
         max_tokens: int, temperature: float, max_retries: int = 3) -> str:
    last = ""
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
    return f"__CALL_ERROR__ {last}"


def extract_guidance(text: str) -> str:
    if "### Guidance" in text:
        text = text.split("### Guidance", 1)[-1]
    return text.strip()


def review_pass(text: str) -> tuple[bool, str]:
    # Parse "### Verdict\nPASS/FAIL"
    verdict = ""
    reason = ""
    if "### Verdict" in text:
        after = text.split("### Verdict", 1)[-1]
        vpart = after.split("### Reason", 1)
        verdict = vpart[0].strip().upper()
        if len(vpart) > 1:
            reason = vpart[1].strip()
    else:
        verdict = text.strip().upper()
    is_pass = "PASS" in verdict and "FAIL" not in verdict
    return is_pass, reason


# ---------------------------------------------------------------------------
# Per-sub-problem generation with dual leakage gate
# ---------------------------------------------------------------------------

def generate_one(
    client: OpenAI,
    model: str,
    problem: str,
    solution: str,
    answer: str,
    kp_names: list[str],
    material: str,
    max_tokens: int,
    max_retry: int,
) -> dict[str, Any]:
    kp_list = "\n".join(f"- {n}" for n in kp_names) or "(none identified)"
    gen_user = GEN_TEMPLATE.format(
        problem=problem, solution=solution or "(not provided)",
        kp_list=kp_list, material=material or "(no verified passages)",
    )
    attempts: list[dict[str, Any]] = []
    best_guidance = ""
    for attempt in range(max(1, max_retry)):
        raw = call(client, model, GEN_SYSTEM, gen_user,
                   max_tokens=max_tokens, temperature=0.7)
        if raw.startswith("__CALL_ERROR__"):
            attempts.append({"stage": "gen", "error": raw})
            continue
        guidance = extract_guidance(raw)
        best_guidance = guidance

        # Gate 1: rule-based leakage detection
        rule_hits = leakage_reasons(guidance, answer)

        # Gate 2: LLM review
        rev = call(client, model, REVIEW_SYSTEM,
                   REVIEW_TEMPLATE.format(problem=problem, answer=answer or "(n/a)",
                                          guidance=guidance),
                   max_tokens=512, temperature=0.0)
        llm_pass, llm_reason = review_pass(rev) if not rev.startswith("__CALL_ERROR__") else (False, rev)

        attempts.append({
            "stage": "check", "rule_hits": rule_hits,
            "llm_pass": llm_pass, "llm_reason": llm_reason,
        })
        if not rule_hits and llm_pass:
            return {
                "guidance": guidance,
                "guidance_check": {"pass": True, "attempts": attempt + 1,
                                   "kp_used": kp_names},
            }

    # Hard fallback: rule-sanitise the last guidance so no literal leak survives.
    sanitized = sanitize_reference(best_guidance, answer, max_chars=4000)
    residual = leakage_reasons(sanitized, answer)
    return {
        "guidance": sanitized,
        "guidance_check": {
            "pass": not residual,
            "fallback_sanitized": True,
            "residual_leak": residual,
            "attempts": len(attempts),
            "kp_used": kp_names,
            "history": attempts[-2:],
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    ap.add_argument("--skeleton", type=Path, default=DEFAULT_SKELETON,
                    help="Output skeleton (benchmark-shaped); guidance is written onto its subs.")
    ap.add_argument("--fragments", type=Path, default=DEFAULT_FRAGMENTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="Optional JSON manifest of per-sub check results.")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retry", type=int, default=3)
    ap.add_argument("--max-frag-chars", type=int, default=2000)
    ap.add_argument("--max-kps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N parents (smoke).")
    ap.add_argument("--cache", type=Path, default=None,
                    help="JSONL result cache for resumability. Defaults to "
                         "<out>.cache.jsonl. Completed subs are skipped on rerun.")
    ap.add_argument("--resume-from", type=Path, default=DEFAULT_RESUME_FROM,
                    help="Optional previous guidance JSON to import into the "
                         "cache before generation. Missing files are ignored.")
    args = ap.parse_args()

    if not args.api_key or not args.base_url:
        sys.exit("Set OPENAI_API_KEY / OPENAI_BASE_URL (source env_local.sh).")

    fragments = load_fragments(args.fragments)
    print(f"[kp-guidance] loaded {len(fragments)} fragments")

    # coverage: build per-sub material index keyed by sub_key
    cov_parents = parents_of(read_json(args.coverage))
    material_by_key: dict[Key, dict] = {}
    for parent in cov_parents:
        for sub in parent.get("sub_problems") or []:
            k = sub_key(parent, sub)
            material, kp_names = build_material(
                sub, fragments, args.max_frag_chars, args.max_kps
            )
            material_by_key[k] = {
                "material": material,
                "kp_names": kp_names,
                "problem": sub.get("problem") or "",
                "solution": sub.get("solution") or "",
                "answer": str(sub.get("answer") or ""),
            }

    # skeleton: the file we write guidance onto (benchmark consumes this shape)
    skel_raw = read_json(args.skeleton)
    skel_parents = parents_of(skel_raw)
    if args.limit is not None:
        skel_parents = skel_parents[: args.limit]

    # Flatten the work list: one task per sub-problem that has material.
    tasks: list[tuple[dict, dict, Key]] = []
    for parent in skel_parents:
        for sub in parent.get("sub_problems") or []:
            k = sub_key(parent, sub)
            tasks.append((parent, sub, k))

    print(f"[kp-guidance] {len(tasks)} sub-problems to process "
          f"({len(skel_parents)} parents)")

    cache_path = args.cache or args.out.with_suffix(args.out.suffix + ".cache.jsonl")

    # Resume: load any previously completed sub results from the cache.
    results: dict[Key, dict] = {}
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                results[tuple(rec["key"])] = rec["res"]
        print(f"[kp-guidance] resumed {len(results)} cached results from {cache_path.name}")

    # Import completed guidance from an older benchmark-shaped output in the
    # same keyed format as the JSONL cache. Existing cache entries win, so a
    # manually resumed or newly generated result is never overwritten.
    imported = guidance_results(args.resume_from)
    imported_count = 0
    if imported:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as f:
            for key, res in imported.items():
                if key in results:
                    continue
                results[key] = res
                f.write(json.dumps({"key": list(key), "res": res},
                                   ensure_ascii=False) + "\n")
                imported_count += 1
        print(f"[kp-guidance] imported {imported_count} results from "
              f"{args.resume_from}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    lock = threading.Lock()
    manifest: list[dict] = []
    stats = {"pass": 0, "fallback": 0, "no_material": 0}
    cache_fh = cache_path.open("a", encoding="utf-8")

    def work(task: tuple[dict, dict, Key]) -> tuple[Key, dict]:
        _parent, sub, k = task
        mat = material_by_key.get(k)
        problem = (mat or {}).get("problem") or sub.get("problem") or ""
        solution = (mat or {}).get("solution") or sub.get("solution") or ""
        answer = (mat or {}).get("answer") or str(sub.get("answer") or "")
        material = (mat or {}).get("material") or ""
        kp_names = (mat or {}).get("kp_names") or []
        if not material:
            # No verified textbook passages for this sub — skip generation,
            # leave existing guidance untouched (honest: we can't ground it).
            return k, {"skipped": "no_material"}
        res = generate_one(
            client, args.model, problem, solution, answer, kp_names, material,
            max_tokens=args.max_tokens, max_retry=args.max_retry,
        )
        return k, res

    todo = [t for t in tasks if t[2] not in results]
    print(f"[kp-guidance] {len(todo)} to run, {len(tasks) - len(todo)} already cached")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(work, t) for t in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="guidance"):
            k, res = fut.result()
            with lock:
                results[k] = res
                cache_fh.write(json.dumps({"key": list(k), "res": res},
                                          ensure_ascii=False) + "\n")
                cache_fh.flush()
    cache_fh.close()

    # Write guidance onto the skeleton parents.
    for parent in skel_parents:
        for sub in parent.get("sub_problems") or []:
            k = sub_key(parent, sub)
            res = results.get(k) or {}
            if res.get("skipped"):
                stats["no_material"] += 1
                manifest.append({"key": list(k), "status": "no_material"})
                continue
            sub["guidance"] = res.get("guidance", "")
            sub["guidance_check"] = res.get("guidance_check", {})
            chk = res.get("guidance_check", {})
            if chk.get("fallback_sanitized"):
                stats["fallback"] += 1
            elif chk.get("pass"):
                stats["pass"] += 1
            manifest.append({
                "key": list(k),
                "status": "fallback" if chk.get("fallback_sanitized") else
                          ("pass" if chk.get("pass") else "fail"),
                "attempts": chk.get("attempts"),
                "kp_used": chk.get("kp_used"),
                "residual_leak": chk.get("residual_leak"),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(skel_parents if isinstance(skel_raw, list) else {"data": skel_parents},
                  f, ensure_ascii=False, indent=2)
    print(f"[kp-guidance] wrote {args.out}")
    print(f"[kp-guidance] clean pass: {stats['pass']}  "
          f"fallback-sanitized: {stats['fallback']}  no-material: {stats['no_material']}")

    if args.manifest:
        with args.manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[kp-guidance] wrote manifest {args.manifest}")


if __name__ == "__main__":
    main()
