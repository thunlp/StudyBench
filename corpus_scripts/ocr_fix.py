"""Clean MinerU Markdown: drop leaked VLM prompts and fix the easiest OCR artifacts.

Does not try to fix every OCR issue (nu/v mix-ups, Tl/Ti, l/1 in answer keys,
and anything else that needs a human looking at the original page).

Behavior:
  1. Recursively find .md files under <target_dir> (skip this script's ocrfix.md)
  2. Skip a directory that already has ocrfix.md unless --force
  3. Write the cleaned text to ocrfix.md next to the source file

What gets fixed:
  - Rule 2 / Ground Truth paragraphs that MinerU's VLM dumps when it sees a
    decorative horizontal rule
  - Split digits in math: `2. 9 9 8 \\times 1 0 ^ {8}` -> `2.998 \\times 10^{8}`
  - Spaces between `_` / `^` / macros and `{`, e.g. `_ {T}`, `\\mathrm {m}`
  - Leftover page numbers in chapter outlines: `PROBLEMS 169` -> `PROBLEMS`
  - Unwrap of leftover `<div class="mineru-algorithm">` wrappers
  - A few unambiguous typos (Coloumb, eletromagnetic)

Examples:
  python3 corpus_scripts/ocr_fix.py quantum_physics
  python3 corpus_scripts/ocr_fix.py quantum_physics --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

OUTPUT_NAME = "ocrfix.md"

# Phrases the MinerU VLM injects when it treats a decorative rule as an OCR
# eval item. They do not occur in the textbook.
_LEAK_MARKERS = (
    "ground truth image",
    "underscore & line rules",
    "ocr result should be empty",
    "stylistic or background line",
    "stylistic horizontal line",
    "placeholder underscore",
    "the ocr has hallucinated",
    "ocr result is inconsistent with the ground truth",
)

_LEAK_LINE_RE = re.compile(
    r"^\s*(\\_){2,}\s*$"  # leftover \_\_\_\_ from a leak paragraph
)

_MINERU_ALGO_RE = re.compile(
    r'<div\s+class="mineru-algorithm"[^>]*>\s*(.*?)\s*</div>',
    re.DOTALL | re.IGNORECASE,
)

_DISPLAY_MATH_RE = re.compile(r"\$\$[\s\S]*?\$\$")
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+)\$(?!\$)")

_QUESTIONS_PROBLEMS_PAGE_RE = re.compile(
    r"^(QUESTIONS|PROBLEMS)[ \t]+\d+[ \t]*$",
    re.MULTILINE,
)
_QUESTIONS_PROBLEMS_LONE_PAGE_RE = re.compile(
    r"^(QUESTIONS|PROBLEMS)[ \t]*\n+[ \t]*\d{2,3}[ \t]*$",
    re.MULTILINE,
)

# Unambiguous whole-book typos only. Do not put nu->v style rewrites here.
_WORD_FIXES = (
    ("Coloumb", "Coulomb"),
    ("eletromagnetic", "electromagnetic"),
    ("reflec-tion", "reflection"),
    ("QuantityConserved", "Quantity Conserved"),
)

_DIGIT_SPACE_RE = re.compile(r"(\d)\s+(?=\d)")
_DECIMAL_SPACE_RE = re.compile(r"(\d)\s*\.\s*(?=\d)")
_EXP_BRACE_RE = re.compile(r"\{\s*(-\s*)?(\d+)\s*\}")
_SCRIPT_AFTER_RE = re.compile(r"([_^])\s+")
_SCRIPT_BEFORE_RE = re.compile(r"\s+([_^])")
_MACRO_BRACE_RE = re.compile(r"(\\[A-Za-z]+)\s+\{")


# ---------------- discovery ----------------


def find_mds(root: Path) -> list[Path]:
    """Find markdown files to clean; skip this script's ocrfix.md output.

    A MinerU output directory usually contains only full.md. If a directory
    has several .md files, prefer full.md so they do not all write ocrfix.md.
    """
    by_dir: dict[Path, list[Path]] = {}
    for p in root.rglob("*.md"):
        if not p.is_file():
            continue
        if p.name == OUTPUT_NAME:
            continue
        by_dir.setdefault(p.parent, []).append(p)

    chosen: list[Path] = []
    for parent, files in by_dir.items():
        files = sorted(files)
        full = parent / "full.md"
        if full in files:
            chosen.append(full)
            continue
        if len(files) == 1:
            chosen.append(files[0])
            continue
        # Several non-full.md sources: keep the first and ignore the rest.
        chosen.append(files[0])
    return sorted(chosen)


def output_for(md: Path) -> Path:
    return md.parent / OUTPUT_NAME


def already_done(md: Path) -> bool:
    return output_for(md).is_file()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------- leak stripping ----------------


def _is_leak_paragraph(para: str) -> bool:
    compact = " ".join(para.split())
    if _LEAK_LINE_RE.match(compact):
        return True
    low = compact.lower()
    return any(marker in low for marker in _LEAK_MARKERS)


def strip_vlm_leaks(text: str) -> tuple[str, int]:
    """Drop VLM eval-prompt paragraphs. Returns (new text, paragraphs removed)."""
    parts = re.split(r"(\n{2,})", text)
    kept: list[str] = []
    n_drop = 0
    for part in parts:
        if part.startswith("\n") or part == "":
            kept.append(part)
            continue
        if _is_leak_paragraph(part):
            n_drop += 1
            continue
        kept.append(part)
    out = "".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + ("\n" if out.strip() else ""), n_drop


# ---------------- math digit / brace spacing ----------------


def _fix_exp_brace(match: re.Match[str]) -> str:
    sign = "-" if match.group(1) else ""
    return "{" + sign + match.group(2) + "}"


def collapse_math_ocr(tex: str) -> str:
    """Inside a math span: rejoin split digits and tighten ^/_/macro spacing."""
    prev = None
    while tex != prev:
        prev = tex
        tex = _DECIMAL_SPACE_RE.sub(r"\1.", tex)
        tex = _DIGIT_SPACE_RE.sub(r"\1", tex)
    tex = _EXP_BRACE_RE.sub(_fix_exp_brace, tex)
    tex = _SCRIPT_AFTER_RE.sub(r"\1", tex)
    tex = _SCRIPT_BEFORE_RE.sub(r"\1", tex)
    tex = _MACRO_BRACE_RE.sub(r"\1{", tex)
    return tex


def fix_math_spans(text: str) -> tuple[str, int]:
    """Rewrite $$...$$ and $...$ interiors. Returns (new text, spans changed)."""
    n_changed = 0

    def _sub_display(m: re.Match[str]) -> str:
        nonlocal n_changed
        old = m.group(0)
        new = collapse_math_ocr(old)
        if new != old:
            n_changed += 1
        return new

    def _sub_inline(m: re.Match[str]) -> str:
        nonlocal n_changed
        old = m.group(0)
        new = collapse_math_ocr(old)
        if new != old:
            n_changed += 1
        return new

    text = _DISPLAY_MATH_RE.sub(_sub_display, text)
    text = _INLINE_MATH_RE.sub(_sub_inline, text)
    return text, n_changed


# ---------------- other easy fixes ----------------


def unwrap_mineru_algorithm(text: str) -> tuple[str, int]:
    n = 0

    def _repl(m: re.Match[str]) -> str:
        nonlocal n
        n += 1
        inner = m.group(1).strip()
        return inner + "\n"

    return _MINERU_ALGO_RE.sub(_repl, text), n


def strip_outline_page_numbers(text: str) -> tuple[str, int]:
    n = 0

    def _count_sub(pattern: re.Pattern[str], repl: str, src: str) -> str:
        nonlocal n
        src, k = pattern.subn(repl, src)
        n += k
        return src

    text = _count_sub(_QUESTIONS_PROBLEMS_PAGE_RE, r"\1", text)
    text = _count_sub(_QUESTIONS_PROBLEMS_LONE_PAGE_RE, r"\1", text)
    return text, n


def apply_word_fixes(text: str) -> tuple[str, int]:
    n = 0
    for old, new in _WORD_FIXES:
        text, k = re.subn(re.escape(old), new, text)
        n += k
    return text, n


# ---------------- per-file ----------------


def clean_markdown(text: str) -> tuple[str, Counter]:
    stats: Counter = Counter()
    text, stats["vlm_leaks"] = strip_vlm_leaks(text)
    text, stats["algo_unwrap"] = unwrap_mineru_algorithm(text)
    text, stats["math_blocks"] = fix_math_spans(text)
    text, stats["page_nums"] = strip_outline_page_numbers(text)
    text, stats["word_fixes"] = apply_word_fixes(text)
    if not text.endswith("\n"):
        text += "\n"
    return text, stats


def process_file(src: Path, dst: Path) -> Counter:
    raw = src.read_text(encoding="utf-8")
    cleaned, stats = clean_markdown(raw)
    dst.write_text(cleaned, encoding="utf-8")
    stats["bytes_in"] = len(raw.encode("utf-8"))
    stats["bytes_out"] = len(cleaned.encode("utf-8"))
    return stats


# ---------------- CLI ----------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clean MinerU Markdown under a directory: drop VLM leaks and easy "
            "OCR artifacts, write ocrfix.md."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", help="Target directory, e.g. quantum_physics")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N pending md files (debug)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing ocrfix.md and rerun",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pending files only; do not write",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).expanduser().resolve()
    if not target.is_dir():
        print(f"ERROR: not a directory: {target}", file=sys.stderr)
        return 2

    all_mds = find_mds(target)
    if not all_mds:
        print(f"No .md files found under {target}")
        return 0

    if args.force:
        pending = list(all_mds)
        skipped: list[Path] = []
    else:
        pending, skipped = [], []
        for md in all_mds:
            (skipped if already_done(md) else pending).append(md)

    print(
        f"Scan {target}: {len(all_mds)} md files, "
        f"{len(skipped)} already done, {len(pending)} pending."
    )
    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit: processing {len(pending)} file(s).")

    if not pending:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        print("--dry-run, pending files:")
        for md in pending:
            print(f"  {_rel(md, target)} -> {_rel(output_for(md), target)}")
        return 0

    overall: Counter = Counter()
    n_written = 0
    for md in pending:
        dst = output_for(md)
        stats = process_file(md, dst)
        overall.update(stats)
        n_written += 1
        bits = []
        if stats["vlm_leaks"]:
            bits.append(f"leak={stats['vlm_leaks']}")
        if stats["math_blocks"]:
            bits.append(f"math={stats['math_blocks']}")
        if stats["page_nums"]:
            bits.append(f"page={stats['page_nums']}")
        if stats["word_fixes"]:
            bits.append(f"word={stats['word_fixes']}")
        if stats["algo_unwrap"]:
            bits.append(f"unwrap={stats['algo_unwrap']}")
        extra = f" ({', '.join(bits)})" if bits else " (no-op)"
        print(f"[done] {_rel(md, target)}{extra}")

    print("\n========== summary ==========")
    print(f"target            : {target}")
    print(f"md scanned        : {len(all_mds)}")
    print(f"already done      : {len(skipped)}")
    print(f"written           : {n_written}")
    print(f"  VLM leak paras  : {overall['vlm_leaks']}")
    print(f"  math spans      : {overall['math_blocks']}")
    print(f"  outline pages   : {overall['page_nums']}")
    print(f"  word fixes      : {overall['word_fixes']}")
    print(f"  algo unwrap     : {overall['algo_unwrap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
