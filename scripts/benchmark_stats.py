"""Summarise Study-Bench contents.

The script prints two top-level sections and a final sanity check.

(1) **Test sets**  --  the held-out evaluation problems served by the
    benchmark runner.  Both test sets live under ``eval/data/``:

    (a) ``qwen3_8b_competition_problem.json``  --  olympiad / competition
        problems that Qwen3-8B failed all 8 attempts; ``source`` is the
        competition code (APhO, EuPhO, IOAA, IPhO, NBPhO, OPhO).

    (b) ``qwen3_8b_textbook_problem.json``     --  textbook problems that
        Qwen3-8B failed all 8 attempts; ``source`` is a textbook code
        (Purcell_EM, Morin_ClassicalMechanics, ...).

    Solo parents (no ``sub_problems``) count as one flattened sub-problem,
    matching the evaluation runner.  For each test set we report total parents
    and sub-problems; the competition set additionally gets a per-``source`` and
    per-sub-discipline breakdown (from the ``problem_type`` field; the textbook
    set's per-source counts already appear in section (2)).

(2) **Training/test coverage**  --  first, a table with one row per ``source``
    code (book or competition).  The second column is the book's sub-discipline
    (physics area).  For each book it reports the raw-markdown corpus
    size (Qwen3 tokens, book + solution manual), the number of instructions
    attributed to it (split across the four ``train_data`` instr files) and how
    many carry a final answer, and how many textbook test-set problems come
    from it.  The instruction ``source`` is read directly from ``metadata`` in
    the four instruction files; competition sources show ``-`` for corpus, and
    the ``hard`` instr column overlaps the test column by construction (hard
    instructions are recycled from the textbook test set).

    A second table aggregates the books by sub-discipline and reports each
    discipline's total and percentage of all training instructions and textbook
    test problems.  Test parents and flattened sub-problems are both shown.

The closing **sanity check** verifies that every test-set unit is
"all-answered": for every parent the answer of every sub-problem (or the
parent itself, if solo) is non-empty AND its ``answer_type`` lies in the
judgeable set ``{NV, EX, EQ, IN, MC, TF, QL, TUP, ALT}``.  Any failure is
printed in detail and the script exits with a non-zero code.

The script is read-only and prints plain text to stdout.

Usage::

    python scripts/benchmark_stats.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EVAL_DATA_DIR  = ROOT / "eval" / "data"
TRAIN_DATA_DIR = ROOT / "train_data"
RAW_BOOKS_DIR  = TRAIN_DATA_DIR / "PhysicsBooks"

COMPETITION_TESTSET = EVAL_DATA_DIR / "qwen3_8b_competition_problem.json"
TEXTBOOK_TESTSET    = EVAL_DATA_DIR / "qwen3_8b_textbook_problem.json"

INSTR_VERIFIABLE       = TRAIN_DATA_DIR / "verifiable_instr.json"
INSTR_VERIFIABLE_MULTI = TRAIN_DATA_DIR / "verifiable_instr_multi_subs.json"
INSTR_UNVERIFIABLE     = TRAIN_DATA_DIR / "unverifiable_instr.json"
INSTR_VERIFIABLE_HARD  = TRAIN_DATA_DIR / "verifiable_instr_hard.json"

# Per-book rollup (section 2) joins the instruction files to their source book
# using ``metadata.source`` stored directly in each record.
INSTR_FILES = [
    ("verif",   INSTR_VERIFIABLE),
    ("multi",   INSTR_VERIFIABLE_MULTI),
    ("unverif", INSTR_UNVERIFIABLE),
    ("hard",    INSTR_VERIFIABLE_HARD),
]

# PhysicsBooks directory name -> source code used in metadata / test sets.
# Mirrors ``BOOK_META`` in ``train_data/recover_metadata.py`` (the source of
# truth); keep the two in sync.  Solution-manual directories share the parent
# book's code (see ``_dir_to_source``), so a book's corpus includes its manual.
BOOK_SOURCE_MAP = {
    "concepts_in_thermal_physics":         "Blundell_ThermalPhysics",
    "electricity_and_magnetism":           "Purcell_EM",
    "introduction_to_classical_mechanics": "Morin_ClassicalMechanics",
    "introduction_to_electrodynamics":     "Griffiths_Electrodynamics",
    "introduction_to_mechanics":           "Kleppner_Mechanics",
    "modern_astrophysics":                 "CarrollOstlie_ModernAstrophysics",
    "physics_of_waves":                    "Georgi_Waves",
    "quantum_physics":                     "EisbergResnick_QuantumPhysics",
    "schaums_astronomy":                   "Palen_SchaumAstronomy",
    "spacetime_physics":                   "TaylorWheeler_SpacetimePhysics",
    "special_relativity":                  "French_SpecialRelativity",
}

# Source code -> sub-discipline (physics area the textbook covers).  Competition
# sources (IOAA, ...) are not textbooks and have no entry -> shown as ``-``.
SOURCE_SUBDISCIPLINE = {
    "Morin_ClassicalMechanics":         "Mechanics",
    "Kleppner_Mechanics":               "Mechanics",
    "Purcell_EM":                       "Electromagnetism",
    "Griffiths_Electrodynamics":        "Electromagnetism",
    "Blundell_ThermalPhysics":          "Thermal Physics",
    "Georgi_Waves":                     "Waves",
    "EisbergResnick_QuantumPhysics":    "Quantum Physics",
    "French_SpecialRelativity":         "Relativity",
    "TaylorWheeler_SpacetimePhysics":   "Relativity",
    "CarrollOstlie_ModernAstrophysics": "Astrophysics",
    "Palen_SchaumAstronomy":            "Astrophysics",
}

# Display order for sub-disciplines in the section (2) rollup, matching the
# order they appear in section (1)'s competition ``By sub-discipline`` breakdown
# (which is sorted by problem count).  The competition ``problem_type`` labels
# differ slightly from the textbook labels, so the correspondence is:
#   Mechanics -> Mechanics, Electromagnetic Fields -> Electromagnetism,
#   Astronomy and Astrophysics -> Astrophysics, Quantum Physics -> Quantum
#   Physics, Thermodynamics and Statistical Physics -> Thermal Physics,
#   Relativity -> Relativity, Oscillations and Waves -> Waves.
SUBDISCIPLINE_ORDER = [
    "Mechanics",
    "Electromagnetism",
    "Astrophysics",
    "Quantum Physics",
    "Thermal Physics",
    "Relativity",
    "Waves",
]


def _dir_to_source(dir_name: str) -> str | None:
    """Map a PhysicsBooks dir to its book source code, or ``None`` if unknown.

    Solution-manual directories (``<book>_solution_manual`` /
    ``<book>_solution_manual_by_chapter``) fold into the parent book's code so
    the rollup counts the manual as part of that book's corpus.
    """
    base = dir_name
    for suffix in ("_solution_manual_by_chapter", "_solution_manual"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return BOOK_SOURCE_MAP.get(base)


JUDGEABLE_ANSWER_TYPES = {
    "NV", "EX", "EQ", "IN", "MC", "TF", "QL", "TUP", "ALT",
}

# Tokenizer used to report token counts for the raw markdown corpus.  We use
# the actual Qwen3-8B tokenizer (HF tokenizers JSON, no torch required) so the
# numbers line up with what the served model would see.  Override via the
# ``QWEN3_TOKENIZER_PATH`` env var; if the file is missing or the tokenizers
# package is unavailable, we warn and report ``-`` in the tokens column.
QWEN3_TOKENIZER_PATH = Path(
    os.environ.get("QWEN3_TOKENIZER_PATH", "/home/test/testdata/models/Qwen3-8B/tokenizer.json")
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_parents(path: Path):
    """Yield parent records from a ``.json`` (``{"data": [...]}``) or ``.jsonl``."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    yield from raw


def _load_instr(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    return list(raw)


# ---------------------------------------------------------------------------
# Tiny table formatter
# ---------------------------------------------------------------------------

def _print_table(headers, rows, *, indent: str = "  ") -> None:
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(str(headers[i])), *(len(r[i]) for r in cells)) if cells else len(str(headers[i]))
        for i in range(len(headers))
    ]
    sep = "  "
    line = sep.join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    print(indent + line)
    print(indent + sep.join("-" * widths[i] for i in range(len(headers))))
    for r in cells:
        print(indent + sep.join(r[i].ljust(widths[i]) for i in range(len(headers))))


# ---------------------------------------------------------------------------
# Tokenizer (Qwen3-8B) -- used by the markdown corpus stats
# ---------------------------------------------------------------------------

def _load_qwen_tokenizer():
    """Return a callable ``encode(text) -> list[int]`` for Qwen3-8B, or ``None``.

    Loading failures (missing ``tokenizers`` package or missing tokenizer
    file) are reported to stderr exactly once; downstream code substitutes
    ``-`` in the token column instead of crashing.
    """
    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except Exception as e:
        print(f"[WARN] cannot import 'tokenizers' ({e}); token counts will be skipped.",
              file=sys.stderr)
        return None
    if not QWEN3_TOKENIZER_PATH.is_file():
        print(f"[WARN] tokenizer file not found: {QWEN3_TOKENIZER_PATH} "
              f"(override with QWEN3_TOKENIZER_PATH); token counts will be skipped.",
              file=sys.stderr)
        return None
    tok = Tokenizer.from_file(str(QWEN3_TOKENIZER_PATH))
    return lambda text: tok.encode(text, add_special_tokens=False).ids


# ---------------------------------------------------------------------------
# Section 1 -- test sets
# ---------------------------------------------------------------------------

def _summarise_testset(label: str, path: Path, *, breakdowns: bool = True) -> None:
    print(f"\n--- {label}")
    print(f"    source: {path.relative_to(ROOT)}")
    if not path.exists():
        print(f"    [WARN] file does not exist")
        return

    parents = list(_read_parents(path))
    n_parents = len(parents)
    solo = sum(1 for p in parents if not (p.get("sub_problems") or []))
    multi = n_parents - solo
    n_subs = sum(len(p.get("sub_problems") or []) or 1 for p in parents)

    print(f"\n    Overall: {n_parents} parents, {n_subs} sub-problems  "
          f"(multi-sub: {multi},  solo: {solo})")

    if not breakdowns:
        return

    by_source: Counter[str] = Counter()
    sub_by_source: Counter[str] = Counter()
    by_ptype: Counter[str] = Counter()
    sub_by_ptype: Counter[str] = Counter()
    for p in parents:
        src = (p.get("source") or "(unset)")
        pt = (p.get("problem_type") or "").strip() or "(unset)"
        n = len(p.get("sub_problems") or []) or 1
        by_source[src] += 1
        sub_by_source[src] += n
        by_ptype[pt] += 1
        sub_by_ptype[pt] += n

    print("\n    By source:\n")
    _print_table(
        ["source", "parents", "sub_problems"],
        [(s, by_source[s], sub_by_source[s])
         for s in sorted(by_source, key=lambda x: (-by_source[x], x))]
        + [("TOTAL", n_parents, n_subs)],
        indent="      ",
    )

    print("\n    By sub-discipline:\n")
    _print_table(
        ["sub_discipline", "parents", "sub_problems"],
        [(t, by_ptype[t], sub_by_ptype[t])
         for t in sorted(by_ptype, key=lambda x: (-by_ptype[x], x))]
        + [("TOTAL", n_parents, n_subs)],
        indent="      ",
    )


def section_testsets() -> None:
    print("=" * 78)
    print("(1) Test sets  --  held-out evaluation problems (qwen3-8b 8/8 failures)")
    print(f"    source root: {EVAL_DATA_DIR.relative_to(ROOT)}/")
    print("    note: solo parents count as 1 flattened sub-problem")
    print("=" * 78)

    _summarise_testset("(a) Competition test set", COMPETITION_TESTSET)
    _summarise_testset("(b) Textbook test set",    TEXTBOOK_TESTSET, breakdowns=False)


# ---------------------------------------------------------------------------
# Section 2 -- training data
# ---------------------------------------------------------------------------

def _markdown_corpus_stats(
    encode=None,
) -> tuple[list[tuple[str, int, int, int, int | None]], tuple[int, int, int, int | None]]:
    """Return per-book ``(book, n_chapters, bytes, chars, tokens)`` + grand-total.

    ``tokens`` is ``None`` when ``encode`` is ``None`` (tokenizer unavailable).
    """
    rows: list[tuple[str, int, int, int, int | None]] = []
    total_files = total_bytes = total_chars = 0
    total_tokens: int | None = 0 if encode is not None else None
    for book_dir in sorted(p for p in RAW_BOOKS_DIR.iterdir() if p.is_dir()):
        mds = sorted(book_dir.rglob("full.md"))
        b = sum(p.stat().st_size for p in mds)
        c = 0
        t: int | None = 0 if encode is not None else None
        for p in mds:
            text = p.read_text(encoding="utf-8", errors="replace")
            c += len(text)
            if encode is not None:
                t += len(encode(text))  # type: ignore[operator]
        rows.append((book_dir.name, len(mds), b, c, t))
        total_files += len(mds); total_bytes += b; total_chars += c
        if encode is not None:
            total_tokens += t  # type: ignore[operator]
    return rows, (total_files, total_bytes, total_chars, total_tokens)


# ---------------------------------------------------------------------------
# Section 2 -- per-textbook rollup (corpus + instructions + test coverage)
# ---------------------------------------------------------------------------

def _corpus_tokens_by_source(md_rows) -> dict[str, tuple[int, int]]:
    """Aggregate ``md_rows`` into ``source -> (tokens_or_chars, is_tokens)``.

    Solution-manual dirs fold into their parent book.  Falls back to character
    counts when the tokenizer was unavailable (``tokens`` column is ``None``).
    """
    acc: dict[str, list[int]] = {}
    for name, _n_ch, _b, chars, tokens in md_rows:
        src = _dir_to_source(name)
        if src is None:
            continue
        val = tokens if tokens is not None else chars
        acc.setdefault(src, [0, 1 if tokens is not None else 0])
        acc[src][0] += val
        if tokens is None:
            acc[src][1] = 0
    return {s: (v[0], bool(v[1])) for s, v in acc.items()}


def _instr_by_source() -> tuple[dict, dict, list[str]]:
    """Read the four instruction files and group their records by source.

    Returns ``(per_file_total, per_file_answered, labels_order)`` where the
    first two are ``{source: {label: count}}`` and label ``"all"`` holds the
    row total across files.
    """
    per_total: dict[str, Counter] = {}
    per_ans: dict[str, Counter] = {}
    labels: list[str] = []
    for label, path in INSTR_FILES:
        labels.append(label)
        if not path.exists():
            continue
        items = _load_instr(path)
        for it in items:
            src = (it.get("metadata") or {}).get("source") or "(no-source)"
            per_total.setdefault(src, Counter())[label] += 1
            per_total[src]["all"] += 1
            if (it.get("answer") or "").strip():
                per_ans.setdefault(src, Counter())[label] += 1
                per_ans[src]["all"] += 1
    return per_total, per_ans, labels


def _test_by_source() -> tuple[Counter, Counter]:
    """Return ``(parents_by_source, subs_by_source)`` for the textbook test set."""
    par: Counter[str] = Counter()
    sub: Counter[str] = Counter()
    if not TEXTBOOK_TESTSET.exists():
        return par, sub
    for p in _read_parents(TEXTBOOK_TESTSET):
        src = p.get("source") or "(unset)"
        par[src] += 1
        sub[src] += len(p.get("sub_problems") or []) or 1
    return par, sub


def _print_subdiscipline_rollup(per_total, test_par, test_sub) -> None:
    """Print training-instruction and textbook-test shares by sub-discipline."""
    instr_by_sd: Counter[str] = Counter()
    test_par_by_sd: Counter[str] = Counter()
    test_sub_by_sd: Counter[str] = Counter()

    all_sources = set(per_total) | set(test_par) | set(test_sub)
    for source in all_sources:
        sub_discipline = SOURCE_SUBDISCIPLINE.get(source, "(unmapped)")
        instr_by_sd[sub_discipline] += per_total.get(source, Counter())["all"]
        test_par_by_sd[sub_discipline] += test_par.get(source, 0)
        test_sub_by_sd[sub_discipline] += test_sub.get(source, 0)

    total_instr = sum(instr_by_sd.values())
    total_test_par = sum(test_par_by_sd.values())
    total_test_sub = sum(test_sub_by_sd.values())
    sub_rank = {sd: i for i, sd in enumerate(SUBDISCIPLINE_ORDER)}

    def _pct(value: int, total: int) -> str:
        return f"{value / total:.2%}" if total else "-"

    rows = []
    for sd in sorted(instr_by_sd.keys() | test_par_by_sd.keys() | test_sub_by_sd.keys(),
                     key=lambda x: (sub_rank.get(x, len(sub_rank)), x)):
        n_instr = instr_by_sd[sd]
        n_test_par = test_par_by_sd[sd]
        n_test_sub = test_sub_by_sd[sd]
        rows.append([
            sd,
            f"{n_instr:,}",
            _pct(n_instr, total_instr),
            f"{n_test_par:,}",
            _pct(n_test_par, total_test_par),
            f"{n_test_sub:,}",
            _pct(n_test_sub, total_test_sub),
        ])

    rows.append([
        "TOTAL",
        f"{total_instr:,}",
        _pct(total_instr, total_instr),
        f"{total_test_par:,}",
        _pct(total_test_par, total_test_par),
        f"{total_test_sub:,}",
        _pct(total_test_sub, total_test_sub),
    ])

    print("\n    By sub-discipline -- training instructions + textbook test problems:\n")
    _print_table(
        ["sub_discipline", "instr", "instr_pct", "test_prob", "test_prob_pct",
         "test_sub", "test_sub_pct"],
        rows,
        indent="      ",
    )
    print("\n      Percentages use the corresponding column total as denominator.")
    print("      test_prob counts parent problems; test_sub counts flattened evaluation units.")
    if "(unmapped)" in instr_by_sd or "(unmapped)" in test_par_by_sd:
        print("      (unmapped) contains sources missing from SOURCE_SUBDISCIPLINE.")


def section_per_book_rollup(md_rows=None, encode=None) -> None:
    print("\n" + "=" * 78)
    print("(2) Per-textbook rollup  --  corpus + instructions + test coverage")
    print("    joined on the `source` code (book / competition), one row per source")
    print("=" * 78)

    if md_rows is None:
        md_rows, _ = _markdown_corpus_stats(encode)

    corpus = _corpus_tokens_by_source(md_rows)      # source -> (amount, is_tokens)
    per_total, per_ans, labels = _instr_by_source()
    test_par, test_sub = _test_by_source()

    # ``encode`` is global (all rows tokens, or all rows chars) -- never mixed.
    corpus_is_tokens = all(v[1] for v in corpus.values()) if corpus else False
    corpus_col = "corpus_tok" if corpus_is_tokens else "corpus_chr"

    all_sources = set(corpus) | set(per_total) | set(test_par)

    # Order by sub-discipline (following section (1)'s competition breakdown
    # order), then within a sub-discipline by instruction volume desc, then name.
    # Sources without a sub-discipline (competition, e.g. IOAA) sort last.
    sub_rank = {sd: i for i, sd in enumerate(SUBDISCIPLINE_ORDER)}

    def _sort_key(s: str):
        sd = SOURCE_SUBDISCIPLINE.get(s)
        return (sub_rank.get(sd, len(sub_rank)),
                -per_total.get(s, Counter())["all"], s)

    def _num(n):
        return f"{n:,}" if n else "-"

    rows = []
    tot = {"corpus": 0, "all": 0, "ans": 0, "par": 0, "sub": 0,
           **{lb: 0 for lb in labels}}
    for s in sorted(all_sources, key=_sort_key):
        amount, _is_tok = corpus.get(s, (None, False))
        if amount is None:
            corpus_cell = "-"
        else:
            corpus_cell = f"{amount:,}"
            tot["corpus"] += amount
        pt = per_total.get(s, Counter())
        pa = per_ans.get(s, Counter())
        row = [s, SOURCE_SUBDISCIPLINE.get(s, "-"), corpus_cell, _num(pt["all"]), _num(pa["all"])]
        row += [_num(pt[lb]) for lb in labels]
        row += [_num(test_par.get(s, 0)), _num(test_sub.get(s, 0))]
        rows.append(row)
        tot["all"] += pt["all"]; tot["ans"] += pa["all"]
        tot["par"] += test_par.get(s, 0); tot["sub"] += test_sub.get(s, 0)
        for lb in labels:
            tot[lb] += pt[lb]

    total_row = [
        "TOTAL",
        "",
        f"{tot['corpus']:,}",
        _num(tot["all"]), _num(tot["ans"]),
        *[_num(tot[lb]) for lb in labels],
        _num(tot["par"]), _num(tot["sub"]),
    ]

    headers = ["textbook", "sub_discipline", corpus_col, "instr", "w/ans",
               *labels, "test_prob", "test_sub"]
    print()
    _print_table(headers, rows + [total_row])

    print("\n    Columns:")
    print("      sub_discipline  physics area the textbook covers (`-` = competition source)")
    print(f"      {corpus_col:10s} raw markdown "
          f"{'Qwen3 tokens' if corpus_is_tokens else 'characters (Qwen3 tokenizer unavailable)'}, "
          "textbook + its solution manual; `-` = competition source (no textbook)")
    print("      instr      instructions attributed to this source (all 4 files)")
    print("      w/ans      subset of instr whose `answer` field is non-empty")
    print(f"      {'/'.join(labels):10s} instr split by file "
          "(verifiable / multi_subs / unverifiable / hard)")
    print("      test_prob  parents in eval/data/qwen3_8b_textbook_problem.json (test_sub: flattened)")
    print("\n    Notes:")
    print("      - instr `source` is read directly from each training record's metadata.")
    print("      - `hard` overlaps `test_prob`: verifiable_instr_hard is recycled")
    print("        from the textbook test set into training.")

    _print_subdiscipline_rollup(per_total, test_par, test_sub)


# ---------------------------------------------------------------------------
# Sanity check -- every test set is fully answered
# ---------------------------------------------------------------------------

def _judgeable(answer: str, answer_type: str) -> bool:
    return bool((answer or "").strip()) and ((answer_type or "").strip() in JUDGEABLE_ANSWER_TYPES)


def _check_parent_units_answered(parent: dict) -> list[str]:
    """Return human-readable failure reasons for ``parent``; empty list means OK.

    For multi-sub parents every ``sub_problems[i]`` must have a non-empty
    ``answer`` AND a judgeable ``answer_type``.  Solo parents are checked on
    their own ``answer`` / ``answer_type`` fields.
    """
    fails: list[str] = []
    subs = parent.get("sub_problems") or []
    if subs:
        for i, s in enumerate(subs):
            ans = (s.get("answer") or "")
            atype = (s.get("answer_type") or "")
            if not _judgeable(ans, atype):
                fails.append(
                    f"sub_problems[{i}] id={s.get('problem_id')!r}: "
                    f"answer={'<empty>' if not ans.strip() else '<set>'}, "
                    f"answer_type={atype!r}"
                )
    else:
        ans = (parent.get("answer") or "")
        atype = (parent.get("answer_type") or "")
        if not _judgeable(ans, atype):
            fails.append(
                f"solo parent: answer={'<empty>' if not ans.strip() else '<set>'}, "
                f"answer_type={atype!r}"
            )
    return fails


def _sanity_check_records(label: str, records: list[dict]) -> int:
    """Return the number of failing parents (also prints details)."""
    bad = 0
    for p in records:
        fails = _check_parent_units_answered(p)
        if fails:
            bad += 1
            sid = f"{p.get('source','?')}/{p.get('year','')}/{p.get('source_problem_id','?')}"
            print(f"  [FAIL] {label}  {sid}")
            for f in fails:
                print(f"           {f}")
    return bad


def sanity_check_all_test_sets() -> int:
    """Walk every test set and verify all units are answered.

    Returns the total number of failing parents (0 means OK).
    """
    print("\n" + "=" * 78)
    print("Sanity check: every test-set unit has a judgeable answer")
    print(f"    answer_type ∈ {sorted(JUDGEABLE_ANSWER_TYPES)}")
    print("=" * 78)

    total_checked = 0
    total_fail = 0

    for label, path in [
        ("competition", COMPETITION_TESTSET),
        ("textbook",    TEXTBOOK_TESTSET),
    ]:
        if not path.exists():
            print(f"  [SKIP] {label}: {path.relative_to(ROOT)} does not exist")
            continue
        records = list(_read_parents(path))
        total_checked += len(records)
        fails = _sanity_check_records(label, records)
        total_fail += fails
        marker = "OK" if fails == 0 else f"FAIL ({fails})"
        print(f"  [{marker:>9s}] {label:12s}  {len(records)} parent(s)  "
              f"({path.relative_to(ROOT)})")

    print(f"\n  Summary: {total_checked} parent(s) checked, {total_fail} failing.")
    return total_fail


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    section_testsets()
    # Compute the markdown corpus once (loading + tokenizing every full.md is the
    # expensive step); the rollup needs md_rows for per-book corpus tokens.
    encode = _load_qwen_tokenizer()
    md_rows, _ = _markdown_corpus_stats(encode)
    section_per_book_rollup(md_rows, encode)
    n_fail = sanity_check_all_test_sets()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
