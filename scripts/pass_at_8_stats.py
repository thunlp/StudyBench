#!/usr/bin/env python3
"""Compute repeated pass@8 accuracy and error-bar stddev from 24-attempt eval data.

The benchmark result contains 24 attempts for each parent question.  This
script treats consecutive groups of eight attempts as three independent
pass@8 runs.  A parent passes a group when at least one attempt is completely
correct (all of its sub-questions are correct).  A sub-question passes when
at least one completion for that slot is correct.

Hard-split groups are a high-variance estimator: an item that succeeds on
1 of 24 attempts lands in only ~29% of eight-attempt groups, so a handful of
marginal items can swing the group accuracy by ten points or more.  The
script therefore also reports the unbiased pass@k estimator of Chen et al.
(2021), "Evaluating Large Language Models Trained on Code", which pools all
``n`` attempts per item::

    pass@k = 1 - C(n - c, k) / C(n, k)

where ``c`` is the number of correct attempts.  This is the expectation of
the hard-split statistic over every way of drawing ``k`` of the ``n``
attempts, so it uses the same data with far less sampling noise.

By default the script reads the Qwen3-8B competition result in this checkout::

    python3 scripts/pass_at_8_stats.py

Use ``--output`` to write the complete per-group and per-parent result as
JSON.  The printed error bar is the sample standard deviation of the three
group accuracies (divide by 2).  The population standard deviation (divide
by 3) is printed as an additional reference value.

``--exclude-groups`` drops whole groups from the unbiased estimator's attempt
pool (the hard-split report is left untouched).  Use it when some attempts
are not exchangeable with the rest -- for example when a group was the
selection run that decided which items enter the dataset, in which case its
attempts are conditioned on success and would bias the pooled estimate::

    python3 scripts/pass_at_8_stats.py --exclude-groups 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT
    / "eval"
    / "results"
    / "base_qwen3_8b"
    / "qwen3_8b_competition_problem"
    / "qwen3_8b_competition_problem_eval.json"
)


def _as_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError(f"expected a JSON list of objects in {path}")
    return raw


def _attempts_in_order(parent: dict[str, Any], expected: int, label: str) -> list[dict[str, Any]]:
    attempts = parent.get("attempts")
    if not isinstance(attempts, list) or not all(isinstance(a, dict) for a in attempts):
        raise ValueError(f"{label}: missing or invalid attempts list")
    if len(attempts) != expected:
        raise ValueError(f"{label}: expected {expected} attempts, found {len(attempts)}")

    ids = [a.get("attempt_id") for a in attempts]
    if all(isinstance(attempt_id, int) for attempt_id in ids) and len(set(ids)) == len(ids):
        attempts = sorted(attempts, key=lambda a: a["attempt_id"])
    return attempts


def _label(parent: dict[str, Any], index: int) -> str:
    year = str(parent.get("year") or "")
    problem_id = str(parent.get("source_problem_id") or index)
    source = str(parent.get("source") or "")
    prefix = f"{year}/{problem_id}" if year else problem_id
    return f"{source}/{prefix}" if source else prefix


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _stddev(values: list[float], ddof: int) -> float:
    """Standard deviation of ``values`` (``ddof=1`` is the sample / error-bar std)."""
    if len(values) <= ddof:
        return math.nan
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - ddof))


def _stderr(values: list[float]) -> float:
    """Standard error of the mean across items (item-level sampling noise)."""
    if len(values) <= 1:
        return math.nan
    return _stddev(values, 1) / math.sqrt(len(values))


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Return the unbiased pass@k estimate for one item (Chen et al., 2021).

    ``n`` attempts were sampled and ``c`` of them were correct.  The estimate
    is ``1 - C(n - k, k) / C(n, k)``, the probability that a uniformly drawn
    subset of ``k`` attempts contains at least one correct attempt.  It is
    computed as a running product to stay exact in floating point for the
    item counts seen here.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= c <= n:
        raise ValueError(f"c must lie in [0, {n}], got {c}")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError(f"k={k} exceeds the {n} available attempts")
    # Fewer than k failures means every k-subset contains a correct attempt.
    if n - c < k:
        return 1.0
    # prod_{i=0}^{k-1} (n - c - i) / (n - i) == C(n - c, k) / C(n, k)
    fail = 1.0
    for i in range(k):
        fail *= (n - c - i) / (n - i)
    return 1.0 - fail


def compute(
    records: list[dict[str, Any]],
    *,
    groups: int = 3,
    group_size: int = 8,
    exclude_groups: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Return repeated pass@k statistics and auditable group-level counts."""
    if groups <= 0 or group_size <= 0:
        raise ValueError("groups and group_size must be positive")
    expected_attempts = groups * group_size
    if not records:
        raise ValueError("input contains no parent questions")
    for group_index in sorted(exclude_groups):
        if not 0 <= group_index < groups:
            raise ValueError(f"--exclude-groups index {group_index} is outside [0, {groups - 1}]")
    kept_slots = [
        slot for slot in range(expected_attempts) if slot // group_size not in exclude_groups
    ]
    if len(kept_slots) < group_size:
        raise ValueError(
            f"excluding {sorted(exclude_groups)} leaves {len(kept_slots)} attempts, "
            f"fewer than k={group_size}; the unbiased estimator needs at least k"
        )

    parent_group_passed = [[False] * len(records) for _ in range(groups)]
    sub_group_passed: list[list[bool]] = [[] for _ in range(groups)]
    sub_total = 0
    parent_details: list[dict[str, Any]] = []
    warnings = 0
    parent_unbiased: list[float] = []
    sub_unbiased: list[float] = []

    for parent_index, parent in enumerate(records):
        label = _label(parent, parent_index)
        attempts = _attempts_in_order(parent, expected_attempts, label)
        first_completions = attempts[0].get("completions")
        if not isinstance(first_completions, list):
            raise ValueError(f"{label}: first attempt has no completions list")
        slot_count = len(first_completions)
        if slot_count == 0:
            raise ValueError(f"{label}: attempts contain no completion slots")

        slot_ids = [
            str(completion.get("problem_id") or slot_index)
            if isinstance(completion, dict)
            else str(slot_index)
            for slot_index, completion in enumerate(first_completions)
        ]
        for attempt in attempts:
            completions = attempt.get("completions")
            if not isinstance(completions, list) or len(completions) != slot_count:
                raise ValueError(f"{label}: completion slot count changes between attempts")

        sub_total += slot_count
        for group_index in range(groups):
            start = group_index * group_size
            group_attempts = attempts[start : start + group_size]
            parent_pass = any(attempt.get("correctness") is True for attempt in group_attempts)
            parent_group_passed[group_index][parent_index] = parent_pass
            if any("correctness" not in attempt for attempt in group_attempts):
                warnings += 1

            sub_passes: list[bool] = []
            for slot_index in range(slot_count):
                passed = any(
                    isinstance(attempt.get("completions"), list)
                    and attempt["completions"][slot_index].get("correctness") is True
                    for attempt in group_attempts
                    if isinstance(attempt.get("completions"), list)
                    and isinstance(attempt["completions"][slot_index], dict)
                )
                sub_passes.append(passed)
            sub_group_passed[group_index].extend(sub_passes)

        # Unbiased pass@k over the retained attempt pool, pooled across groups.
        pool = [attempts[slot] for slot in kept_slots]
        pool_size = len(pool)
        parent_correct = sum(1 for attempt in pool if attempt.get("correctness") is True)
        parent_estimate = unbiased_pass_at_k(pool_size, parent_correct, group_size)
        parent_unbiased.append(parent_estimate)

        sub_correct: list[int] = []
        sub_estimates: list[float] = []
        for slot_index in range(slot_count):
            correct = sum(
                1
                for attempt in pool
                if isinstance(attempt.get("completions"), list)
                and isinstance(attempt["completions"][slot_index], dict)
                and attempt["completions"][slot_index].get("correctness") is True
            )
            estimate = unbiased_pass_at_k(pool_size, correct, group_size)
            sub_correct.append(correct)
            sub_estimates.append(estimate)
            sub_unbiased.append(estimate)

        parent_details.append(
            {
                "parent_index": parent_index,
                "label": label,
                "source": parent.get("source", ""),
                "year": parent.get("year", ""),
                "source_problem_id": parent.get("source_problem_id", ""),
                "sub_questions": slot_count,
                "sub_question_ids": slot_ids,
                "parent_passed": [row[parent_index] for row in parent_group_passed],
                "parent_correct_attempts": parent_correct,
                "parent_pool_attempts": pool_size,
                "parent_unbiased_pass_k": parent_estimate,
                "sub_correct_attempts": sub_correct,
                "sub_unbiased_pass_k": sub_estimates,
            }
        )

    parent_total = len(records)
    parent_accuracy = [sum(row) / parent_total for row in parent_group_passed]
    sub_accuracy = [sum(row) / sub_total for row in sub_group_passed]

    return {
        "input_parents": parent_total,
        "attempts_per_parent": expected_attempts,
        "groups": groups,
        "group_size": group_size,
        "pass_k": group_size,
        "excluded_groups": sorted(exclude_groups),
        "unbiased_pool_attempts": len(kept_slots),
        "parent": {
            "denominator": parent_total,
            "passed_by_group": [sum(row) for row in parent_group_passed],
            "accuracy_by_group": parent_accuracy,
            "mean_accuracy": _mean(parent_accuracy),
            "stddev_population": _stddev(parent_accuracy, 0),
            "stddev_sample": _stddev(parent_accuracy, 1),
            "unbiased_pass_k": _mean(parent_unbiased),
            "unbiased_stderr": _stderr(parent_unbiased),
        },
        "sub_question": {
            "denominator": sub_total,
            "passed_by_group": [sum(row) for row in sub_group_passed],
            "accuracy_by_group": sub_accuracy,
            "mean_accuracy": _mean(sub_accuracy),
            "stddev_population": _stddev(sub_accuracy, 0),
            "stddev_sample": _stddev(sub_accuracy, 1),
            "unbiased_pass_k": _mean(sub_unbiased),
            "unbiased_stderr": _stderr(sub_unbiased),
        },
        "parent_details": parent_details,
        "warnings": warnings,
    }


def _print_metric(name: str, metric: dict[str, Any], pass_k: int, pool: int) -> None:
    accuracies = ", ".join(f"{value:.6f}" for value in metric["accuracy_by_group"])
    print(f"{name}:")
    print(f"  denominator: {metric['denominator']}")
    print(f"  passed by group: {metric['passed_by_group']}")
    print(f"  pass@{pass_k} accuracy by group: [{accuracies}]")
    print(f"  mean accuracy: {metric['mean_accuracy']:.6f}")
    print(f"  stddev / error bar (sample, /2): {metric['stddev_sample']:.8f}")
    print(f"  stddev (population, /3): {metric['stddev_population']:.8f}")
    stderr = metric["unbiased_stderr"]
    stderr_text = "n/a" if math.isnan(stderr) else f"{stderr:.6f}"
    print(f"  unbiased pass@{pass_k} (pooled over {pool} attempts): {metric['unbiased_pass_k']:.6f}")
    print(f"  unbiased pass@{pass_k} stderr: {stderr_text}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"eval JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--groups", type=int, default=3, help="number of pass@8 groups (default: 3)")
    parser.add_argument("--group-size", type=int, default=8, help="attempts per group (default: 8)")
    parser.add_argument(
        "--exclude-groups",
        type=int,
        nargs="+",
        default=(),
        metavar="INDEX",
        help=(
            "group indices to drop from the unbiased estimator's attempt pool "
            "(the per-group report still shows them); use for non-exchangeable "
            "attempts such as a selection run"
        ),
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    try:
        result = compute(
            _as_records(args.input),
            groups=args.groups,
            group_size=args.group_size,
            exclude_groups=frozenset(args.exclude_groups),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Input: {args.input}")
    print(f"Parents: {result['input_parents']}; {result['groups']} groups x {result['group_size']} attempts")
    if result["excluded_groups"]:
        print(
            f"Unbiased estimator excludes group(s) {result['excluded_groups']}; "
            f"pooling {result['unbiased_pool_attempts']} attempts per item"
        )
    pool = result["unbiased_pool_attempts"]
    _print_metric("Parent (大题, all sub-questions correct)", result["parent"], result["pass_k"], pool)
    _print_metric("Sub-question (小题)", result["sub_question"], result["pass_k"], pool)
    if result["warnings"]:
        print(f"Warnings: {result['warnings']} group-parent rows had missing attempt correctness", file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Details: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
