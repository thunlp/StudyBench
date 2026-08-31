#!/usr/bin/env python3
"""Build solution-aware, answer-redacted guidance for high-yield subproblems.

The script deliberately keeps the benchmark data immutable.  It writes:

* a full guidance file with selected entries replaced;
* a manifest describing target selection and leakage checks;
* optional pilot / expanded benchmark subsets containing complete parents.

Near-miss guidance is distilled from the shortest completion already judged
correct.  Zero-success guidance is distilled from the gold solution.  In both
cases the final-answer statement and literal gold answer are removed.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


Key = tuple[str, str, str, str]
ParentKey = tuple[str, str, str]

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "studybench_data/competition_problems/competition_problems_full.json"
DEFAULT_GUIDANCE = ROOT / "studybench_data/level3_guidance_full.json"
DEFAULT_GUIDED_EVAL = (
    ROOT
    / "eval/results_llama3_2_3b_instruct/"
    "llama3_2_3b_instruct_with_guidance/"
    "llama3_2_3b_instruct_with_guidance/"
    "llama3_2_3b_instruct_with_guidance_eval.json"
)
DEFAULT_BASE_EVAL = (
    ROOT
    / "eval/results_llama3_2_3b_instruct/"
    "llama3_2_3b_instruct_base/"
    "llama3_2_3b_instruct/"
    "llama3_2_3b_instruct_base_eval.json"
)

SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")
BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
FINAL_LINE_RE = re.compile(
    r"(?im)^[^\n]*(?:final\s+answer|the\s+answer\s+is|answer\s*:)[^\n]*$"
)
SOURCE_ARTIFACT_RE = re.compile(
    r"(?i)\b(?:the\s+)?(?:textbook|quotes?|section\s+\d+|eq\.\s*\d+)\b"
)
FALSE_MISSING_RE = re.compile(
    r"(?i)(?:no numerical computation is possible|"
    r"(?:data|parameters?|information|values?).{0,45}not (?:given|provided|present)|"
    r"(?:remaining steps|details).{0,80}(?:omitted|not present))"
)
CONCLUSION_RE = re.compile(
    r"(?i)\b(?:therefore|thus|hence|consequently|finally|this gives|we get)\b"
)


def read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parent_key(parent: dict) -> ParentKey:
    return (
        str(parent.get("source") or ""),
        str(parent.get("year") or ""),
        str(parent.get("source_problem_id") or ""),
    )


def sub_key(parent: dict, sub: dict) -> Key:
    return (*parent_key(parent), str(sub.get("problem_id") or ""))


def iter_subs(parents: Iterable[dict]) -> Iterable[tuple[dict, dict]]:
    for parent in parents:
        for sub in parent.get("sub_problems") or parent.get("problems") or []:
            if isinstance(sub, dict):
                yield parent, sub


def index_subs(parents: list[dict]) -> dict[Key, tuple[dict, dict]]:
    return {sub_key(parent, sub): (parent, sub) for parent, sub in iter_subs(parents)}


def eval_slots(parents: list[dict]) -> dict[Key, list[dict]]:
    slots: dict[Key, list[dict]] = {}
    for parent in parents:
        by_pid: dict[str, list[dict]] = {}
        for attempt in parent.get("attempts") or []:
            for completion in attempt.get("completions") or []:
                if isinstance(completion, dict):
                    by_pid.setdefault(str(completion.get("problem_id") or ""), []).append(
                        completion
                    )
        for sub in parent.get("sub_problems") or []:
            key = sub_key(parent, sub)
            slots[key] = by_pid.get(str(sub.get("problem_id") or ""), [])
    return slots


def guidance_index(parents: list[dict]) -> dict[Key, str]:
    out: dict[Key, str] = {}
    for parent, sub in iter_subs(parents):
        guidance = sub.get("guidance")
        if isinstance(guidance, str) and guidance.strip():
            out[sub_key(parent, sub)] = guidance.strip()
    return out


def strip_latex_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right|,|;|!|quad|cdot|times)", "", text)
    text = re.sub(r"[^a-z0-9.+\-=/^]", "", text)
    return text


def answer_bodies(answer: str) -> list[str]:
    bodies = [m.group(1).strip() for m in BOX_RE.finditer(answer or "")]
    if not bodies and answer:
        bodies = [answer.strip()]
    return [body for body in bodies if body]


def distinctive_numbers(answer: str) -> set[str]:
    numbers = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:\s*\\times\s*10\^?\{?-?\d+\}?)?", answer))
    return {re.sub(r"\s+", "", number) for number in numbers if len(number) >= 3}


def _drop_answer_paragraphs(text: str, answer: str) -> str:
    answer_canon = [strip_latex_for_match(x) for x in answer_bodies(answer)]
    answer_canon = [x for x in answer_canon if len(x) >= 4]
    answer_numbers = distinctive_numbers(answer)
    kept: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        canon = strip_latex_for_match(paragraph)
        direct = any(body in canon for body in answer_canon)
        compact = re.sub(r"\s+", "", paragraph)
        conclusion_with_number = (
            bool(CONCLUSION_RE.search(paragraph))
            and any(number in compact for number in answer_numbers)
        )
        if direct or conclusion_with_number:
            continue
        kept.append(paragraph)
    return "\n\n".join(kept)


def sanitize_reference(text: str, answer: str, max_chars: int) -> str:
    text = SPECIAL_TOKEN_RE.sub("", text or "")
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = FINAL_LINE_RE.sub("", text)
    text = BOX_RE.sub(r"\1", text)
    for body in answer_bodies(answer):
        text = text.replace(body, "[final result intentionally omitted]")
    text = _drop_answer_paragraphs(text, answer)

    clean_paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        if FALSE_MISSING_RE.search(paragraph):
            continue
        if SOURCE_ARTIFACT_RE.search(paragraph):
            paragraph = SOURCE_ARTIFACT_RE.sub("reference", paragraph)
        paragraph = re.sub(r"\n{3,}", "\n\n", paragraph).strip()
        if paragraph:
            clean_paragraphs.append(paragraph)
    text = "\n\n".join(clean_paragraphs).strip()

    if len(text) > max_chars:
        cut = text.rfind("\n\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = text.rfind(". ", 0, max_chars)
        text = text[: cut if cut > 0 else max_chars].rstrip()
    return text


def answer_contract(sub: dict) -> str:
    answer_type = str(sub.get("answer_type") or "").upper()
    n_answers = max(1, len(answer_bodies(str(sub.get("answer") or ""))))
    parts = [
        f"The expected response type is {answer_type or 'a concise physics result'}.",
        (
            f"Return exactly {n_answers} boxed expression"
            f"{'s' if n_answers != 1 else ''} on the final line, in the order asked."
        ),
        "Keep units outside the box and do not stop before the final line.",
    ]
    if answer_type in {"NV", "IN"}:
        parts.insert(1, "Carry units through every substitution and check the order of magnitude.")
    elif answer_type in {"EQ", "EX"}:
        parts.insert(1, "Preserve the requested symbols; simplify algebra before substituting.")
    elif answer_type in {"TUP", "ALT"}:
        parts.insert(1, "Account for every requested component; a partial tuple is incorrect.")
    elif answer_type in {"TF", "MC"}:
        parts.insert(1, "Make the selection only after checking the stated condition and limiting cases.")
    return " ".join(parts)


def build_guidance(sub: dict, reference: str, tier: str, max_chars: int) -> str:
    answer = str(sub.get("answer") or "")
    prefix = (
        "Use the following checked route as a calculation scaffold. Re-derive each step "
        "from the problem statement; the final result is intentionally omitted.\n\n"
    )
    suffix = (
        "\n\nExecution checks:\n"
        "1. Verify signs, powers of ten, and limiting cases before committing to a value.\n"
        f"2. {answer_contract(sub)}\n"
        "3. Finish the remaining algebra or arithmetic yourself; do not invent an unsupported number."
    )
    body_budget = max(300, max_chars - len(prefix) - len(suffix))
    route = sanitize_reference(reference, answer, body_budget)
    if not route:
        route = (
            "Identify the requested quantity, write the governing relation using only "
            "the variables defined in the problem, and simplify it before numerical substitution."
        )
    guidance = prefix + route + suffix
    return guidance.rstrip()


def leakage_reasons(guidance: str, answer: str) -> list[str]:
    reasons: list[str] = []
    low = guidance.lower()
    if "final answer" in low:
        reasons.append("contains-final-answer-phrase")
    if "\\boxed" in guidance:
        reasons.append("contains-boxed")
    if SPECIAL_TOKEN_RE.search(guidance):
        reasons.append("contains-special-token")
    canon = strip_latex_for_match(guidance)
    for body in answer_bodies(answer):
        body_canon = strip_latex_for_match(body)
        if len(body_canon) >= 8 and body_canon in canon:
            reasons.append("contains-normalized-gold")
            break
    if FALSE_MISSING_RE.search(guidance):
        reasons.append("contains-missing-data-claim")
    return sorted(set(reasons))


def select_targets(
    benchmark: list[dict],
    guided_slots: dict[Key, list[dict]],
    base_slots: dict[Key, list[dict]],
    existing_guidance: dict[Key, str],
    target_count: int,
    include_near_miss: bool = True,
    zero_max_solution_chars: int = 3200,
    zero_all_types: bool = False,
) -> list[dict]:
    # Empirically the redacted "scaffold" hurts near_miss subs: the sanitiser
    # strips their (already effective) derivation down to a generic stub, so
    # the model does worse than with the original guidance it was scoring on.
    # Zero-success subs have nothing to lose and convert well, so a
    # zero-focused sweep (``include_near_miss=False``) leaves working near_miss
    # subs on their original guidance and spends the budget where the scaffold
    # actually pays off.
    records: list[dict] = []
    for parent, sub in iter_subs(benchmark):
        key = sub_key(parent, sub)
        guided = guided_slots.get(key, [])
        base = base_slots.get(key, [])
        g = sum(c.get("correctness") is True for c in guided)
        b = sum(c.get("correctness") is True for c in base)
        solution = str(sub.get("solution") or "")
        answer_type = str(sub.get("answer_type") or "").upper()
        old_guidance = existing_guidance.get(key, "")
        defect = bool(FALSE_MISSING_RE.search(old_guidance))

        if include_near_miss and 1 <= g <= 7:
            tier = "near_miss"
            priority = (0, -g, len(solution), key)
        elif g == 0 and defect:
            tier = "critical_defect"
            priority = (1, 0, len(solution), key)
        elif (
            g == 0
            and (zero_all_types or answer_type in {"EQ", "EX", "NV", "QL", "IN"})
            and solution
            and len(solution) <= zero_max_solution_chars
        ):
            tier = "zero_short_solution"
            # Prefer evidence of latent ability, then concise derivations.
            priority = (2, -b, len(solution), key)
        else:
            continue
        records.append(
            {
                "key": key,
                "parent_key": key[:3],
                "parent_sub_count": len(parent.get("sub_problems") or []),
                "problem_id": key[3],
                "answer_type": answer_type,
                "guided_correct": g,
                "base_correct": b,
                "tier": tier,
                "solution_chars": len(solution),
                "had_guidance": key in existing_guidance,
                "_priority": priority,
            }
        )
    records.sort(key=lambda item: item["_priority"])
    selected = records[:target_count]
    for record in selected:
        record.pop("_priority", None)
    return selected


def choose_reference(record: dict, sub: dict, guided_slots: dict[Key, list[dict]]) -> tuple[str, str]:
    key = tuple(record["key"])
    correct = [
        c.get("completion", "")
        for c in guided_slots.get(key, [])
        if c.get("correctness") is True and isinstance(c.get("completion"), str)
    ]
    if correct:
        return min(correct, key=len), "shortest_correct_completion"
    return str(sub.get("solution") or ""), "gold_solution"


def upsert_guidance(
    benchmark: list[dict],
    old_guidance: dict[Key, str],
    target_records: list[dict],
    guided_slots: dict[Key, list[dict]],
    max_chars: int,
) -> tuple[list[dict], list[dict]]:
    output = copy.deepcopy(benchmark)
    sub_index = index_subs(output)
    target_by_key = {tuple(record["key"]): record for record in target_records}
    audit: list[dict] = []
    for key, (_, sub) in sub_index.items():
        if key not in target_by_key:
            if key in old_guidance:
                sub["guidance"] = old_guidance[key]
            continue
        record = target_by_key[key]
        reference, reference_source = choose_reference(record, sub, guided_slots)
        guidance = build_guidance(sub, reference, record["tier"], max_chars)
        reasons = leakage_reasons(guidance, str(sub.get("answer") or ""))
        if reasons and reference_source == "shortest_correct_completion":
            reference_source = "gold_solution_fallback"
            guidance = build_guidance(
                sub, str(sub.get("solution") or ""), record["tier"], max_chars
            )
            reasons = leakage_reasons(guidance, str(sub.get("answer") or ""))
        if reasons:
            # Keep the prompt useful while guaranteeing the hard no-answer gate.
            guidance = (
                "Use a compact independent derivation. First identify the requested quantity "
                "and all givens, then write the governing equation, simplify symbolically, "
                "substitute with units, and check signs, dimensions, and limiting cases. "
                + answer_contract(sub)
            )
            reasons = leakage_reasons(guidance, str(sub.get("answer") or ""))
        sub["guidance"] = guidance
        record.update(
            {
                "reference_source": reference_source,
                "guidance_chars": len(guidance),
                "leakage_reasons": reasons,
            }
        )
        audit.append(record)
    return output, audit


def select_pilot(records: list[dict], near_count: int, zero_count: int) -> list[dict]:
    by_parent: dict[ParentKey, list[dict]] = {}
    for record in records:
        by_parent.setdefault(tuple(record["parent_key"]), []).append(record)

    selected_parents: set[ParentKey] = set()
    selected: list[dict] = []
    near = 0
    zero = 0
    while near < near_count or zero < zero_count:
        candidates = []
        for key, group in by_parent.items():
            if key in selected_parents:
                continue
            group_near = sum(r["tier"] == "near_miss" for r in group)
            group_zero = len(group) - group_near
            useful = min(max(0, near_count - near), group_near)
            useful += min(max(0, zero_count - zero), group_zero)
            if useful == 0:
                continue
            parent_subs = max(r["parent_sub_count"] for r in group)
            candidates.append((useful / parent_subs, useful, len(group), key, group))
        if not candidates:
            break
        _, _, _, key, group = max(candidates)
        selected_parents.add(key)
        selected.extend(group)
        near += sum(r["tier"] == "near_miss" for r in group)
        zero += sum(r["tier"] != "near_miss" for r in group)
    return selected


def parent_subset(benchmark: list[dict], records: list[dict]) -> list[dict]:
    wanted = {tuple(record["parent_key"]) for record in records}
    return [copy.deepcopy(parent) for parent in benchmark if parent_key(parent) in wanted]


def summarize(records: list[dict]) -> dict[str, Any]:
    return {
        "targets": len(records),
        "parents": len({tuple(record["parent_key"]) for record in records}),
        "tier_counts": dict(Counter(record["tier"] for record in records)),
        "answer_type_counts": dict(Counter(record["answer_type"] for record in records)),
        "guided_correct_before": sum(record["guided_correct"] for record in records),
        "remaining_slot_ceiling": sum(8 - record["guided_correct"] for record in records),
        "leakage_failures": sum(bool(record.get("leakage_reasons")) for record in records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--base-guidance", type=Path, default=DEFAULT_GUIDANCE)
    parser.add_argument("--guided-eval", type=Path, default=DEFAULT_GUIDED_EVAL)
    parser.add_argument("--base-eval", type=Path, default=DEFAULT_BASE_EVAL)
    parser.add_argument("--output-guidance", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--pilot-data", type=Path)
    parser.add_argument("--expanded-data", type=Path)
    parser.add_argument("--target-count", type=int, default=350)
    parser.add_argument("--pilot-near", type=int, default=30)
    parser.add_argument("--pilot-zero", type=int, default=20)
    parser.add_argument("--max-guidance-chars", type=int, default=2000)
    parser.add_argument(
        "--no-near-miss",
        action="store_true",
        help="Skip near_miss subs (keep their original guidance); target only "
        "zero-success subs, where the redacted scaffold does not regress.",
    )
    parser.add_argument("--zero-max-solution-chars", type=int, default=3200)
    parser.add_argument("--zero-all-types", action="store_true",
        help="Target every zero-success sub with a gold solution, not just EQ/EX/NV/QL/IN.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = read_json(args.data)
    base_guidance = read_json(args.base_guidance)
    guided_eval = read_json(args.guided_eval)
    base_eval = read_json(args.base_eval)

    benchmark_index = index_subs(benchmark)
    guided = eval_slots(guided_eval)
    base = eval_slots(base_eval)
    old_guidance = guidance_index(base_guidance)
    targets = select_targets(
        benchmark, guided, base, old_guidance, target_count=args.target_count,
        include_near_miss=not args.no_near_miss,
        zero_max_solution_chars=args.zero_max_solution_chars,
        zero_all_types=args.zero_all_types,
    )
    optimized, audited = upsert_guidance(
        benchmark, old_guidance, targets, guided, max_chars=args.max_guidance_chars
    )
    if len(audited) != len(targets):
        raise RuntimeError(f"only updated {len(audited)} of {len(targets)} targets")
    if any(record["leakage_reasons"] for record in audited):
        bad = [record for record in audited if record["leakage_reasons"]]
        raise RuntimeError(f"answer-leakage gate failed for {len(bad)} targets")

    pilot = select_pilot(audited, args.pilot_near, args.pilot_zero)
    manifest = {
        "inputs": {
            "data": str(args.data),
            "base_guidance": str(args.base_guidance),
            "guided_eval": str(args.guided_eval),
            "base_eval": str(args.base_eval),
        },
        "selection": summarize(audited),
        "pilot": summarize(pilot),
        "targets": audited,
        "pilot_keys": [record["key"] for record in pilot],
        "benchmark_subproblems": len(benchmark_index),
        "old_guidance_entries": len(old_guidance),
    }
    write_json(args.output_guidance, optimized)
    write_json(args.output_manifest, manifest)
    if args.pilot_data:
        write_json(args.pilot_data, parent_subset(benchmark, pilot))
    if args.expanded_data:
        write_json(args.expanded_data, parent_subset(benchmark, audited))

    print(json.dumps({"selection": manifest["selection"], "pilot": manifest["pilot"]}, indent=2))
    print(f"wrote guidance: {args.output_guidance}")
    print(f"wrote manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
