"""Join verified KP matches back into per-sub-problem coverage records.

Stage 5 of the coverage-argument pipeline.  Consumes:

  * ``studybench_data/kp_matches.jsonl``       (Stage 4 output)
  * ``studybench_data/knowledge_points.jsonl`` (Stage 2 canonical KPs -> subs)
  * the competition JSON at ``studybench_data/competition_problems/competition_problems_full.json``
    (or the smoke backup)

and produces:

  * ``studybench_data/competition_problems_full.with_coverage.json``
    -- a *copy* of the input with a ``textbook_matches`` field added to
    every KP entry inside every ``sub_problems[i].knowledge_points[j]``.
    The source competition JSON is NEVER modified in place; this is
    deliberately a separate file so the Stage-1 extractor and the full
    scan can keep running against the original file.
  * ``studybench_data/coverage_summary.md`` -- human-readable overview
  * ``studybench_data/coverage_matrix.csv`` -- KP x book coverage matrix
  * ``studybench_data/subproblem_coverage.csv`` -- per-sub-problem coverage row

The ``textbook_matches`` block attached to each KP is::

    "textbook_matches": {
      "kp_id":           "...",
      "canonical_name":  "...",
      "n_strong":        2,
      "n_partial":       1,
      "top_exposition":  [{fragment_id, book_display, section_path,
                          score, quote, reason}, ...],   # up to 3
      "top_example":     [ ... ],                        # up to 2
      "verifier_error":  null | "..."
    }

We intentionally keep this compact -- if the full fragment text is
needed downstream, look it up via ``fragment_id`` in
``studybench_data/textbook_fragments.jsonl``.

Coverage scoring rules (per sub-problem):

  * A KP is COVERED  if at least one match has score >= 2 in the
    verifier output.
  * A KP is STRONGLY COVERED if at least one match has score == 3.
  * A sub-problem is FULLY COVERED if every one of its KPs is
    covered; STRONGLY FULLY COVERED if every KP is strongly covered.

Usage::

    # smoke: use the backup that already has 139 labelled subs
    python filter/join_coverage.py \\
        --input studybench_data/competition_problems/competition_problems_full.json \\
        --matches studybench_data/kp_matches.smoke.jsonl \\
        --out studybench_data/competition_problems_full.with_coverage.json

    # full pass (defaults)
    python filter/join_coverage.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "studybench_data" / "competition_problems" / "competition_problems_full.json"
DEFAULT_OUT   = ROOT / "studybench_data" / "competition_problems_full.with_coverage.json"
STUDY_DIR = ROOT / "studybench_data"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_json(path: Path) -> tuple[Any, list[dict[str, Any]], bool]:
    with path.open() as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw, raw["data"], True
    if isinstance(raw, list):
        return raw, raw, False
    sys.exit(f"{path}: top level must be a list or {{'data': [...]}}.")


def dump_json(path: Path, raw: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Build helper indices
# ---------------------------------------------------------------------------

def _kp_lookup_from_canonical(canonical_kps: list[dict]) -> dict[tuple[str, str], str]:
    """Given the canonicalise-KP output, build a lookup keyed by
    ``(type, normalised_name)`` returning ``kp_id``.  Any alias present
    in the cluster is also added, so a sub-problem KP whose original
    name matched an alias still finds its canonical row.
    """
    import re
    _WS = re.compile(r"\s+")
    _PUNCT = re.compile(r"[.,;:!?()\[\]{}\"']")

    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = _PUNCT.sub(" ", s)
        s = _WS.sub(" ", s).strip()
        return s

    lut: dict[tuple[str, str], str] = {}
    for k in canonical_kps:
        t = k["type"]
        kp_id = k["kp_id"]
        lut[(t, norm(k["canonical_name"]))] = kp_id
        for a in k.get("aliases") or []:
            lut[(t, norm(a))] = kp_id
    return lut


# ---------------------------------------------------------------------------
# Building the compact textbook_matches block per KP
# ---------------------------------------------------------------------------

def _compact_match(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "fragment_id":       m["fragment_id"],
        "book":              m["book"],
        "book_display":      m["book_display"],
        "domain":            m.get("domain"),
        "chapter_title":     m.get("chapter_title"),
        "section_path":      m.get("section_path"),
        "kind":              m["kind"],
        "is_solution_manual": m.get("is_solution_manual", False),
        "score":             m["score"],
        "quote":             m["quote"],
        "reason":            m.get("reason", ""),
    }


def _build_kp_matches_block(match_row: Optional[dict[str, Any]],
                            top_exposition: int,
                            top_example: int) -> dict[str, Any]:
    if match_row is None:
        return {
            "kp_id":          None,
            "canonical_name": None,
            "n_strong":       0,
            "n_partial":      0,
            "top_exposition": [],
            "top_example":    [],
            "verifier_error": "kp not verified yet",
        }

    matches = match_row.get("matches") or []
    # only consider score >= 1; drop noisy 0s
    exp = [m for m in matches if m["kind"] == "exposition" and m["score"] >= 1]
    exa = [m for m in matches if m["kind"] == "example"    and m["score"] >= 1]
    # sort: higher score first, verified quote wins ties, then rank ascending
    def key(m: dict[str, Any]) -> tuple:
        return (-int(m["score"]),
                0 if m.get("quote_verified") else 1,
                m.get("candidate_rank", 999))
    exp.sort(key=key)
    exa.sort(key=key)

    return {
        "kp_id":          match_row.get("kp_id"),
        "canonical_name": match_row.get("canonical_name"),
        "n_strong":       int(match_row.get("n_strong", 0) or 0),
        "n_partial":      int(match_row.get("n_partial", 0) or 0),
        "top_exposition": [_compact_match(m) for m in exp[:top_exposition]],
        "top_example":    [_compact_match(m) for m in exa[:top_example]],
        "verifier_error": match_row.get("verifier_error"),
    }


# ---------------------------------------------------------------------------
# Sub-problem coverage classification
# ---------------------------------------------------------------------------

def _sub_coverage_status(kp_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not kp_blocks:
        return {"n_kps": 0, "n_covered": 0, "n_strong": 0,
                "coverage_status": "no_kps"}
    n_cov = 0
    n_strong = 0
    unverified = 0
    for b in kp_blocks:
        if b.get("verifier_error") == "kp not verified yet":
            unverified += 1
            continue
        if b["n_strong"] >= 1:
            n_strong += 1
            n_cov += 1
        elif b["n_partial"] >= 1:
            n_cov += 1
    n = len(kp_blocks)
    if unverified > 0:
        status = "unverified"
    elif n_strong == n:
        status = "strongly_covered"
    elif n_cov == n:
        status = "covered"
    elif n_cov > 0:
        status = "partial"
    else:
        status = "uncovered"
    return {
        "n_kps":       n,
        "n_covered":   n_cov,
        "n_strong":    n_strong,
        "n_unverified": unverified,
        "coverage_status": status,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def join(input_path: Path,
         out_path: Path,
         canonical_kps: list[dict[str, Any]],
         match_rows: list[dict[str, Any]],
         top_exposition: int,
         top_example: int,
         summary_md: Path,
         coverage_csv: Path,
         subproblem_csv: Path) -> dict[str, Any]:

    raw, items, wrapped = load_json(input_path)
    kp_id_from_kp = _kp_lookup_from_canonical(canonical_kps)
    match_by_kp = {r["kp_id"]: r for r in match_rows}

    # per-KP aggregate for coverage matrix
    per_kp_book_scores: dict[str, dict[str, int]] = defaultdict(dict)
    per_kp_meta: dict[str, dict[str, Any]] = {}
    for k in canonical_kps:
        per_kp_meta[k["kp_id"]] = {
            "canonical_name": k["canonical_name"],
            "type":           k["type"],
            "n_sub_problems": k.get("n_sub_problems", 0),
        }

    subproblem_rows: list[dict[str, Any]] = []
    global_stats = Counter()

    import re
    _WS = re.compile(r"\s+")
    _PUNCT = re.compile(r"[.,;:!?()\[\]{}\"']")
    def norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = _PUNCT.sub(" ", s)
        s = _WS.sub(" ", s).strip()
        return s

    for pi, parent in enumerate(items):
        if not isinstance(parent, dict):
            continue
        for si, sub in enumerate(parent.get("sub_problems") or []):
            if not isinstance(sub, dict):
                continue
            kps = sub.get("knowledge_points")
            if not (isinstance(kps, list) and kps):
                continue
            for kp in kps:
                if not isinstance(kp, dict):
                    continue
                t = str(kp.get("type", "") or "").strip().lower()
                name = kp.get("name", "")
                kp_id = kp_id_from_kp.get((t, norm(name)))
                match_row = match_by_kp.get(kp_id) if kp_id else None
                block = _build_kp_matches_block(match_row, top_exposition, top_example)
                # rewrite: attach in place (input JSON is mutated locally
                # but we output to a NEW file only)
                kp["textbook_matches"] = block

                # accumulate per-KP x book scores
                if kp_id is not None and match_row is not None:
                    per_kp_meta.setdefault(kp_id, {
                        "canonical_name": block["canonical_name"],
                        "type": t,
                        "n_sub_problems": 0,
                    })
                    for m in match_row.get("matches") or []:
                        if m["score"] < 2:
                            continue
                        b = m["book"]
                        prev = per_kp_book_scores[kp_id].get(b, 0)
                        per_kp_book_scores[kp_id][b] = max(prev, m["score"])

            status = _sub_coverage_status([kp["textbook_matches"] for kp in kps
                                            if isinstance(kp, dict)])
            sub["coverage_status"] = status
            global_stats[status["coverage_status"]] += 1

            subproblem_rows.append({
                "parent_index":     pi,
                "sub_index":        si,
                "source":           parent.get("source", ""),
                "year":             parent.get("year", ""),
                "source_problem_id": parent.get("source_problem_id", ""),
                "sub_problem_id":   sub.get("problem_id", ""),
                "problem_type":     parent.get("problem_type", ""),
                "n_kps":            status["n_kps"],
                "n_covered":        status["n_covered"],
                "n_strong":         status["n_strong"],
                "n_unverified":     status["n_unverified"],
                "coverage_status":  status["coverage_status"],
            })

    dump_json(out_path, raw if wrapped else items)

    # coverage matrix CSV: rows = KPs, cols = books
    all_books = sorted({b for scores in per_kp_book_scores.values() for b in scores})
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    with coverage_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kp_id", "type", "canonical_name", "n_sub_problems",
                    "n_books_with_ge2", "n_books_with_strong"] + all_books)
        for kp_id, meta in sorted(per_kp_meta.items(),
                                  key=lambda kv: -(kv[1].get("n_sub_problems") or 0)):
            row_scores = per_kp_book_scores.get(kp_id, {})
            n_ge2 = sum(1 for s in row_scores.values() if s >= 2)
            n_strong = sum(1 for s in row_scores.values() if s >= 3)
            w.writerow(
                [kp_id, meta["type"], meta["canonical_name"],
                 meta.get("n_sub_problems", 0), n_ge2, n_strong]
                + [row_scores.get(b, "") for b in all_books]
            )

    # per-sub-problem CSV
    with subproblem_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(subproblem_rows[0].keys()) if subproblem_rows else [
            "parent_index","sub_index","source","year","source_problem_id",
            "sub_problem_id","problem_type","n_kps","n_covered","n_strong",
            "n_unverified","coverage_status"])
        w.writeheader()
        for r in subproblem_rows:
            w.writerow(r)

    # summary markdown
    _write_summary_md(summary_md, subproblem_rows, per_kp_meta,
                      per_kp_book_scores, all_books, global_stats)

    return {
        "n_subproblems":            len(subproblem_rows),
        "coverage_status_counts":   dict(global_stats),
        "n_kps_with_ge2_match":     sum(1 for kp_id, sc in per_kp_book_scores.items()
                                        if any(s >= 2 for s in sc.values())),
        "n_kps_with_strong_match":  sum(1 for kp_id, sc in per_kp_book_scores.items()
                                        if any(s >= 3 for s in sc.values())),
        "n_kps_uncovered":          len(per_kp_meta) - sum(
                                        1 for kp_id in per_kp_meta
                                        if per_kp_book_scores.get(kp_id)),
        "output_file":              str(out_path),
        "matrix_csv":               str(coverage_csv),
        "subproblem_csv":           str(subproblem_csv),
        "summary_md":               str(summary_md),
    }


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def _write_summary_md(path: Path,
                      subproblem_rows: list[dict[str, Any]],
                      per_kp_meta: dict[str, dict[str, Any]],
                      per_kp_book_scores: dict[str, dict[str, int]],
                      all_books: list[str],
                      global_stats: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    total_sub = len(subproblem_rows)
    n_kp = len(per_kp_meta)
    n_kp_ge2 = sum(1 for kp in per_kp_meta
                   if any(s >= 2 for s in per_kp_book_scores.get(kp, {}).values()))
    n_kp_strong = sum(1 for kp in per_kp_meta
                      if any(s >= 3 for s in per_kp_book_scores.get(kp, {}).values()))
    n_kp_uncov = n_kp - sum(1 for kp in per_kp_meta if per_kp_book_scores.get(kp))

    with tmp.open("w", encoding="utf-8") as f:
        f.write("# Textbook coverage summary\n\n")
        f.write(f"- Sub-problems with knowledge points: **{total_sub}**\n")
        f.write("- Sub-problem coverage status counts:\n")
        for status, n in sorted(global_stats.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / max(1, total_sub)
            f.write(f"    - `{status}` : **{n}** ({pct:.1f}%)\n")
        f.write("\n")
        f.write(f"- Canonical KPs total: **{n_kp}**\n")
        f.write(f"    - with at least one strong (score=3) match: **{n_kp_strong}** "
                f"({100.0 * n_kp_strong / max(1, n_kp):.1f}%)\n")
        f.write(f"    - with at least one partial-or-strong (score>=2) match: **{n_kp_ge2}** "
                f"({100.0 * n_kp_ge2 / max(1, n_kp):.1f}%)\n")
        f.write(f"    - completely uncovered (no match >= 2 in any book): **{n_kp_uncov}** "
                f"({100.0 * n_kp_uncov / max(1, n_kp):.1f}%)\n\n")

        # per-book contribution
        book_kp_ge2 = Counter()
        book_kp_strong = Counter()
        for kp, scores in per_kp_book_scores.items():
            for b, s in scores.items():
                if s >= 2:
                    book_kp_ge2[b] += 1
                if s >= 3:
                    book_kp_strong[b] += 1
        f.write("## Per-book contribution\n\n")
        f.write("| Book | KPs with score>=2 | KPs with score=3 |\n")
        f.write("|------|-------------------|------------------|\n")
        for b in sorted(all_books):
            f.write(f"| {b} | {book_kp_ge2.get(b, 0)} | {book_kp_strong.get(b, 0)} |\n")
        f.write("\n")

        # per-problem-type coverage
        f.write("## Coverage by parent problem_type\n\n")
        by_pt: dict[str, Counter] = defaultdict(Counter)
        for row in subproblem_rows:
            by_pt[row["problem_type"]][row["coverage_status"]] += 1
        f.write("| problem_type | strong | covered | partial | uncovered | unverified | no_kps |\n")
        f.write("|--------------|--------|---------|---------|-----------|------------|--------|\n")
        for pt, c in sorted(by_pt.items()):
            f.write(f"| {pt} | {c.get('strongly_covered',0)} | {c.get('covered',0)} | "
                    f"{c.get('partial',0)} | {c.get('uncovered',0)} | "
                    f"{c.get('unverified',0)} | {c.get('no_kps',0)} |\n")
        f.write("\n")

        # top-covered KPs
        f.write("## Top 20 best-covered KPs\n\n")
        by_ge2: list[tuple[str, int, dict[str, int]]] = []
        for kp, scores in per_kp_book_scores.items():
            n_ge2 = sum(1 for s in scores.values() if s >= 2)
            by_ge2.append((kp, n_ge2, scores))
        by_ge2.sort(key=lambda t: -t[1])
        for kp_id, n, scores in by_ge2[:20]:
            m = per_kp_meta[kp_id]
            best_books = sorted(
                [b for b, s in scores.items() if s >= 2],
                key=lambda b: -scores[b],
            )[:5]
            f.write(f"- **{m['canonical_name']}** ({m['type']}, n_sub={m['n_sub_problems']}) "
                    f"-- in {n} book(s): {best_books}\n")
        f.write("\n")

        # uncovered KPs
        f.write("## Uncovered KPs (score < 2 in every book)\n\n")
        uncov = [(kp, per_kp_meta[kp]) for kp in per_kp_meta
                 if not any(s >= 2 for s in per_kp_book_scores.get(kp, {}).values())]
        # sort by n_sub_problems desc (higher-priority uncovered first)
        uncov.sort(key=lambda t: -(t[1].get("n_sub_problems") or 0))
        for kp_id, meta in uncov[:60]:
            f.write(f"- `{meta['type']}` **{meta['canonical_name']}** "
                    f"(n_sub={meta['n_sub_problems']})\n")
        if len(uncov) > 60:
            f.write(f"- _(+ {len(uncov) - 60} more)_\n")

    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input",     type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--kps",       type=Path,
                    default=STUDY_DIR / "knowledge_points.jsonl")
    ap.add_argument("--matches",   type=Path,
                    default=STUDY_DIR / "kp_matches.jsonl")
    ap.add_argument("--out",       type=Path, default=DEFAULT_OUT)
    ap.add_argument("--summary-md", type=Path,
                    default=STUDY_DIR / "coverage_summary.md")
    ap.add_argument("--coverage-csv", type=Path,
                    default=STUDY_DIR / "coverage_matrix.csv")
    ap.add_argument("--subproblem-csv", type=Path,
                    default=STUDY_DIR / "subproblem_coverage.csv")
    ap.add_argument("--top-exposition", type=int, default=3,
                    help="How many exposition matches to include per KP.")
    ap.add_argument("--top-example", type=int, default=2,
                    help="How many example matches to include per KP.")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing input: {args.input}")
    if not args.kps.exists():
        sys.exit(f"missing kps: {args.kps}")
    if not args.matches.exists():
        sys.exit(f"missing matches: {args.matches}\n"
                 f"Run filter/verify_textbook_matches.py first.")

    canonical_kps = load_jsonl(args.kps)
    match_rows    = load_jsonl(args.matches)

    stats = join(
        input_path=args.input, out_path=args.out,
        canonical_kps=canonical_kps, match_rows=match_rows,
        top_exposition=args.top_exposition, top_example=args.top_example,
        summary_md=args.summary_md, coverage_csv=args.coverage_csv,
        subproblem_csv=args.subproblem_csv,
    )
    print(f"[join_coverage] wrote {args.out}")
    print(f"[join_coverage] wrote {args.summary_md}")
    print(f"[join_coverage] wrote {args.coverage_csv}")
    print(f"[join_coverage] wrote {args.subproblem_csv}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
