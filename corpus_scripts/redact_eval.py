"""Redact textbook passages in ocrfix.md that also appear in the eval JSON.

Each eval item has a `source` field (e.g. TaylorWheeler_SpacetimePhysics).
That name is the last path component of the book directory:

    PhysicsBooks/<source>/**/ocrfix.md

N-grams from problem / solution / answer (and sub_problems) locate
candidate passages. Shared textbook phrasing produces extra short hits, so
each eval field keeps only its longest run, and only if that run covers
most of the field (one source per problem). The original ocrfix.md is left
intact; the result is written next to it as ocrfix.redacted.md.

Examples:
  python3 corpus_scripts/redact_eval.py eval/data/qwen3_8b_textbook_problem.json
  python3 corpus_scripts/redact_eval.py eval/data/qwen3_8b_textbook_problem.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_BOOKS_DIR = "PhysicsBooks"
DEFAULT_OCRFIX_NAME = "ocrfix.md"
DEFAULT_OUTPUT_NAME = "ocrfix.redacted.md"
# 8-grams fire on shared textbook phrasing (e.g. an in-chapter example
# that restates the later problem). A problem should have one source, so
# we keep only the longest run per eval field and require it to cover
# most of that field.
DEFAULT_NGRAM = 13
DEFAULT_MIN_COVER = 0.6
DEFAULT_GAP = 2
DEFAULT_PLACEHOLDER = "[REDACTED]"

ITEM_TEXT_FIELDS = ("title", "problem", "solution", "answer")
SUB_TEXT_FIELDS = ("problem", "solution", "answer")

_TOKEN_RE = re.compile(r"\S+")
_WRAP_PUNCT_RE = re.compile(r"^[^\w$\\]+|[^\w$\\]+$", re.UNICODE)


def norm_token(tok: str) -> str:
    t = (
        tok.lower()
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    t = _WRAP_PUNCT_RE.sub("", t)
    return t


def tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Whitespace tokens with original character spans, plus a match key."""
    out: list[tuple[str, int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        key = norm_token(m.group())
        if not key:
            continue
        out.append((key, m.start(), m.end()))
    return out


def iter_ngrams(keys: list[str], n: int) -> Iterable[tuple[str, ...]]:
    if n <= 0 or len(keys) < n:
        return
    for i in range(len(keys) - n + 1):
        yield tuple(keys[i : i + n])


@dataclass(frozen=True)
class Hit:
    path: Path
    start: int
    end: int
    n_tokens: int


def eval_keys(text: str) -> list[str]:
    return [k for k, _, _ in tokens_with_spans(text)]


def find_runs(
    book_text: str,
    grams: set[tuple[str, ...]],
    n: int,
    max_gap: int,
) -> list[tuple[int, int, int]]:
    """Return (char_start, char_end, token_count) runs of n-gram hits.

    Consecutive hits separated by at most max_gap unmarked tokens are merged
    so a stray OCR word does not split a real problem statement.
    """
    toks = tokens_with_spans(book_text)
    if len(toks) < n or not grams:
        return []
    marked = [False] * len(toks)
    keys = [t[0] for t in toks]
    for i, gram in enumerate(iter_ngrams(keys, n)):
        if gram in grams:
            for j in range(i, i + n):
                marked[j] = True

    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < len(marked):
        if not marked[i]:
            i += 1
            continue
        start_i = i
        end_i = i
        i += 1
        gap = 0
        while i < len(marked):
            if marked[i]:
                end_i = i
                gap = 0
                i += 1
            elif gap < max_gap:
                gap += 1
                i += 1
            else:
                break
        runs.append((toks[start_i][1], toks[end_i][2], end_i - start_i + 1))
    return runs


def best_hit(
    eval_text: str,
    book_texts: dict[Path, str],
    n: int,
    min_cover: float,
    max_gap: int,
) -> Hit | None:
    """Single best book span for one eval field, or None if too weak."""
    keys = eval_keys(eval_text)
    if len(keys) < n:
        return None
    grams = set(iter_ngrams(keys, n))
    need = max(n, int(len(keys) * min_cover + 0.999))
    best: Hit | None = None
    for path, book_text in book_texts.items():
        for start, end, ntok in find_runs(book_text, grams, n, max_gap):
            if ntok < need:
                continue
            if best is None or ntok > best.n_tokens:
                best = Hit(path, start, end, ntok)
    return best


def apply_hits(text: str, hits: list[Hit], placeholder: str) -> tuple[str, int]:
    """Replace hits that land in this file. Overlapping hits are merged."""
    if not hits:
        return text, 0
    spans = sorted((h.start, h.end) for h in hits)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out = text
    for start, end in reversed(merged):
        out = out[:start] + placeholder + out[end:]
    return out, len(merged)


def iter_item_texts(item: dict[str, Any]) -> Iterable[str]:
    for field in ITEM_TEXT_FIELDS:
        val = item.get(field)
        if isinstance(val, str) and val.strip():
            yield val
    for sub in item.get("sub_problems") or []:
        if not isinstance(sub, dict):
            continue
        for field in SUB_TEXT_FIELDS:
            val = sub.get(field)
            if isinstance(val, str) and val.strip():
                yield val


def collect_eval_texts(items: list[Any]) -> dict[str, list[str]]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        if not isinstance(source, str) or not source:
            continue
        by_source[source].extend(iter_item_texts(item))
    return by_source


def discover_books(books_dir: Path, ocrfix_name: str) -> dict[str, Path]:
    """Map last folder name -> book directory, for dirs that contain ocrfix.md."""
    found: dict[str, Path] = {}
    if not books_dir.is_dir():
        return found
    for child in sorted(books_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.rglob(ocrfix_name)):
            found[child.name] = child
    return found


def find_ocrfix(book_dir: Path, ocrfix_name: str) -> list[Path]:
    return sorted(p for p in book_dir.rglob(ocrfix_name) if p.is_file())


def output_for(src: Path, output_name: str) -> Path:
    return src.parent / output_name


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Redact the single best ocrfix.md passage for each eval field. "
            "Writes ocrfix.redacted.md next to each source."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "eval_json",
        help="Eval JSON with a top-level 'data' list (items have a 'source' field)",
    )
    parser.add_argument(
        "--books-dir",
        default=DEFAULT_BOOKS_DIR,
        help="Directory of books; last folder name must equal item['source']",
    )
    parser.add_argument(
        "--ocrfix-name",
        default=DEFAULT_OCRFIX_NAME,
        help="Cleaned markdown filename under each book",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Redacted markdown filename written next to each ocrfix.md",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=DEFAULT_NGRAM,
        help="Seed n-gram length in whitespace tokens",
    )
    parser.add_argument(
        "--min-cover",
        type=float,
        default=DEFAULT_MIN_COVER,
        help="Keep a hit only if it covers at least this fraction of the eval field",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=DEFAULT_GAP,
        help="Merge hit runs separated by at most this many unmatched tokens",
    )
    parser.add_argument(
        "--placeholder",
        default=DEFAULT_PLACEHOLDER,
        help="Replacement for overlapping spans",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N ocrfix.md files (debug)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing ocrfix.redacted.md")
    parser.add_argument("--dry-run", action="store_true", help="Report stats only; do not write")
    args = parser.parse_args(argv)

    if args.ngram < 1:
        print("ERROR: --ngram must be >= 1", file=sys.stderr)
        return 2
    if not 0 < args.min_cover <= 1:
        print("ERROR: --min-cover must be in (0, 1]", file=sys.stderr)
        return 2
    if args.gap < 0:
        print("ERROR: --gap must be >= 0", file=sys.stderr)
        return 2
    if args.output_name == args.ocrfix_name:
        print("ERROR: --output-name must differ from --ocrfix-name", file=sys.stderr)
        return 2

    eval_path = Path(args.eval_json).expanduser().resolve()
    if not eval_path.is_file():
        print(f"ERROR: not a file: {eval_path}", file=sys.stderr)
        return 2

    books_dir = Path(args.books_dir).expanduser().resolve()
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    items = payload.get("data")
    if not isinstance(items, list):
        print("ERROR: JSON has no top-level 'data' list", file=sys.stderr)
        return 2

    eval_texts = collect_eval_texts(items)
    books = discover_books(books_dir, args.ocrfix_name)
    print(
        f"Scan {books_dir}: {len(books)} book(s) with {args.ocrfix_name}; "
        f"{len(items)} eval item(s) across {len(eval_texts)} source(s)."
    )

    pending: list[tuple[str, Path]] = []
    skipped_exist = 0
    for source, book_dir in books.items():
        for src in find_ocrfix(book_dir, args.ocrfix_name):
            dst = output_for(src, args.output_name)
            if dst.exists() and not args.force and not args.dry_run:
                skipped_exist += 1
                continue
            pending.append((source, src))

    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit: processing {len(pending)} file(s).")

    print(f"Pending {args.ocrfix_name}: {len(pending)}; already done: {skipped_exist}.")
    if args.dry_run:
        for source, src in pending:
            print(f"  {source}  {_rel(src, books_dir)} -> {args.output_name}")

    overall: Counter = Counter()
    sources_without_eval: set[str] = set()
    pending_by_source: dict[str, list[Path]] = defaultdict(list)
    for source, src in pending:
        pending_by_source[source].append(src)

    for source, paths in pending_by_source.items():
        texts = eval_texts.get(source) or []
        if not texts:
            sources_without_eval.add(source)
            overall["no_eval"] += len(paths)
            continue
        book_texts = {p: p.read_text(encoding="utf-8") for p in paths}
        hits_by_path: dict[Path, list[Hit]] = defaultdict(list)
        print(f"[index] {source}: {len(texts)} eval field(s), {len(paths)} {args.ocrfix_name}")
        for field in texts:
            hit = best_hit(field, book_texts, args.ngram, args.min_cover, args.gap)
            if hit is None:
                overall["fields_unmatched"] += 1
                continue
            hits_by_path[hit.path].append(hit)
            overall["fields_matched"] += 1

        for src, raw in book_texts.items():
            hits = hits_by_path.get(src, [])
            cleaned, n_spans = apply_hits(raw, hits, args.placeholder)
            overall["files"] += 1
            if n_spans:
                overall["files_redacted"] += 1
                overall["spans"] += n_spans
            print(f"[done] {_rel(src, books_dir)}  spans={n_spans}")
            if args.dry_run:
                continue
            dst = output_for(src, args.output_name)
            dst.write_text(cleaned if cleaned.endswith("\n") else cleaned + "\n", encoding="utf-8")

    unused_eval = sorted(set(eval_texts) - set(books))
    print("\n========== summary ==========")
    print(f"eval JSON         : {eval_path}")
    print(f"books dir         : {books_dir}")
    print(f"ngram             : {args.ngram}")
    print(f"min cover         : {args.min_cover}")
    print(f"ocrfix processed  : {overall['files']}")
    print(f"  files redacted  : {overall['files_redacted']}")
    print(f"  spans           : {overall['spans']}")
    print(f"  fields matched  : {overall['fields_matched']}")
    print(f"  fields unmatched: {overall['fields_unmatched']}")
    print(f"  already done    : {skipped_exist}")
    print(f"  no eval items   : {overall['no_eval']}")
    if sources_without_eval:
        print("books with ocrfix but no eval items:")
        for name in sorted(sources_without_eval):
            print(f"  {name}")
    if unused_eval:
        print("eval sources with no matching book:")
        for name in unused_eval:
            print(f"  {name}: {len(eval_texts[name])} field(s)")

    if args.dry_run:
        print("--dry-run, not writing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
