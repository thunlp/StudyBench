"""Cluster per-sub-problem knowledge points into a global canonical KP list.

Stage-1 (``extract_knowledge_points.py``) deliberately does NOT
canonicalise names across sub-problems. But Stage 2 wants to run
retrieval once per named concept, not once per (sub, KP) pair -- else
"conservation of angular momentum" is queried once for every sub that
uses it, with 12 slightly different phrasings and 12 inconsistent
top-K lists downstream.

This script consolidates the KP records inside the competition JSON into
a global canonical KP list. Two records collapse into one canonical KP
when they share ``(type, normalise(name))`` -- lowercased, whitespace-
squashed, punctuation-lite. That is deliberately conservative: it never
merges concepts across types (e.g. an ``assumption`` "circular orbit"
stays separate from a ``concept`` "circular orbit"), and it does not
attempt embedding-based fuzzy merging (we'd rather see two nearly-
identical KPs in the review dump and merge them by hand than
accidentally fuse "adiabatic process" with "adiabatic invariant").

For each canonical KP we record:

  * a stable ``kp_id`` = sha1(type + "::" + canonical_name)[:12]
  * ``canonical_name`` -- most common original name in the cluster
    (ties broken by shortest length, then alphabetic)
  * ``aliases`` -- other original spellings, deduped
  * ``keywords_union`` -- union of retrieval keywords across the cluster
  * ``descriptions`` -- per-instance context (sub_id + description)
    so downstream retrieval can query with the SPECIFIC why-needed
    rationale rather than a bland definition
  * ``domain_hint`` -- histogram of parent ``problem_type`` values
    (competitions tag every parent as Mechanics / EM / … / Astronomy);
    used later to prefer within-domain textbook fragments
  * ``sub_problems`` -- list of (sub_id, parent_index, sub_index)
    triples for the join phase

The input JSON is read but NOT modified. Two artifacts are written:

  * ``studybench_data/knowledge_points.jsonl`` -- one record per canonical KP
  * ``studybench_data/knowledge_points.review.md`` -- a human-readable
    audit dump grouped by (type, domain, count), for spot-checking
    cluster names before firing off retrieval

Usage::

    # smoke: use the pre-full-KP backup that already has 139 labels
    python filter/canonicalize_knowledge_points.py \\
        --input studybench_data/competition_problems/competition_problems_full.json

    # once the full-scan is done, re-run against the live file
    python filter/canonicalize_knowledge_points.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "studybench_data" / "competition_problems" / "competition_problems_full.json"
OUT_DIR = ROOT / "studybench_data"
OUT_JSONL = OUT_DIR / "knowledge_points.jsonl"
OUT_REVIEW = OUT_DIR / "knowledge_points.review.md"


ALLOWED_TYPES = {"concept", "law_or_equation", "technique", "assumption"}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[.,;:!?()\[\]{}\"']")
_WS = re.compile(r"\s+")
# tokens we deliberately swallow when comparing names so trivial phrasing
# differences don't produce two canonical KPs.  Keep this list very short
# and unambiguous.
_STOP_SUFFIX = re.compile(
    r"\s+(?:formula|equation|law|principle|theorem|approximation|expansion|method|technique|effect|relation)$"
)


def normalise_name(s: str) -> str:
    """Lower-cased, punctuation-stripped, whitespace-squashed."""
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def normalise_name_loose(s: str) -> str:
    """Same as :func:`normalise_name`, but with generic domain suffixes
    stripped ("Newton's second law" ~= "Newton's second"). Only used as
    a secondary bucketing key printed in the review file, never for
    hard clustering."""
    s = normalise_name(s)
    prev = None
    while s != prev:
        prev = s
        s = _STOP_SUFFIX.sub("", s)
    return s


def kp_hash(kp_type: str, canonical_name: str) -> str:
    key = f"{kp_type}::{normalise_name(canonical_name)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Sub-problem identity
# ---------------------------------------------------------------------------

def sub_id_for(parent: dict[str, Any], sub: dict[str, Any]) -> str:
    """Globally-unique short id for a sub-problem, stable across runs."""
    bits = [
        str(parent.get("source", "")).strip(),
        str(parent.get("year", "")).strip(),
        str(parent.get("source_problem_id", "")).strip(),
        str(sub.get("problem_id", "")).strip(),
    ]
    return " | ".join(b for b in bits if b) or "(unknown-sub)"


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

def iter_kp_records(items: list[dict[str, Any]]):
    """Yield ``(kp_dict, sub_id, parent_idx, sub_idx, parent)`` for every
    valid KP entry across the corpus."""
    for pi, parent in enumerate(items):
        if not isinstance(parent, dict):
            continue
        for si, sub in enumerate(parent.get("sub_problems") or []):
            if not isinstance(sub, dict):
                continue
            kps = sub.get("knowledge_points")
            if not isinstance(kps, list) or not kps:
                continue
            sid = sub_id_for(parent, sub)
            for kp in kps:
                if not isinstance(kp, dict):
                    continue
                t = str(kp.get("type", "") or "").strip().lower()
                if t not in ALLOWED_TYPES:
                    continue
                name = str(kp.get("name", "") or "").strip()
                if not name:
                    continue
                yield kp, sid, pi, si, parent


# ---------------------------------------------------------------------------
# Cluster & pick canonical name
# ---------------------------------------------------------------------------

def _pick_canonical(name_counts: Counter) -> str:
    """Return the "best" original spelling for a cluster.

    Preference order:
      1. highest frequency
      2. shortest length (canonical names tend to be terse)
      3. alphabetically first (deterministic tiebreak)
    """
    best_count = max(name_counts.values())
    top = [n for n, c in name_counts.items() if c == best_count]
    top.sort(key=lambda n: (len(n), n))
    return top[0]


def build_canonical_kps(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # cluster key = (type, normalise_name(name))
    clusters: dict[tuple[str, str], dict[str, Any]] = {}

    for kp, sid, pi, si, parent in iter_kp_records(items):
        t = kp["type"].strip().lower()
        name = kp["name"].strip()
        key = (t, normalise_name(name))
        c = clusters.setdefault(
            key,
            {
                "type":            t,
                "name_counts":     Counter(),
                "keywords":        Counter(),
                "descriptions":    [],
                "sub_problems":    [],
                "problem_types":   Counter(),
                "problem_domains": Counter(),
                "sources":         Counter(),
            },
        )
        c["name_counts"][name] += 1
        for kw in kp.get("keywords") or []:
            if isinstance(kw, str) and kw.strip():
                c["keywords"][kw.strip()] += 1
        desc = str(kp.get("description", "") or "").strip()
        source = str(kp.get("source", "") or "").strip().lower()
        c["descriptions"].append({
            "sub_id":       sid,
            "parent_idx":   pi,
            "sub_idx":      si,
            "description":  desc,
            "kp_source":    source,
        })
        c["sub_problems"].append({
            "sub_id":     sid,
            "parent_idx": pi,
            "sub_idx":    si,
        })
        pt = str(parent.get("problem_type", "") or "").strip()
        pd = str(parent.get("domain", "") or "").strip()
        if pt: c["problem_types"][pt] += 1
        if pd: c["problem_domains"][pd] += 1
        c["sources"][source] += 1

    out: list[dict[str, Any]] = []
    for (t, norm), c in clusters.items():
        canonical = _pick_canonical(c["name_counts"])
        aliases = sorted({n for n in c["name_counts"] if n != canonical})
        # keep sub_problems in a stable order
        subs_sorted = sorted(c["sub_problems"],
                             key=lambda r: (r["parent_idx"], r["sub_idx"]))
        # dedupe (sub_id) in case a KP appears multiple times inside one sub
        # (shouldn't -- extract_knowledge_points dedupes within a sub -- but
        # be defensive).
        seen: set[str] = set()
        subs_deduped: list[dict[str, Any]] = []
        for r in subs_sorted:
            if r["sub_id"] in seen:
                continue
            seen.add(r["sub_id"])
            subs_deduped.append(r)

        rec = {
            "kp_id":            kp_hash(t, canonical),
            "type":             t,
            "canonical_name":   canonical,
            "aliases":          aliases,
            "n_records":        sum(c["name_counts"].values()),
            "n_sub_problems":   len(subs_deduped),
            "keywords_union":   [k for k, _ in c["keywords"].most_common()],
            "descriptions":     c["descriptions"],
            "sub_problems":     subs_deduped,
            "domain_hint":      dict(c["problem_types"]),
            "coarse_domain":    dict(c["problem_domains"]),
            "source_hist":      dict(c["sources"]),
        }
        out.append(rec)

    # Sort: high-frequency first, then by type, then by canonical name for
    # deterministic diffs.
    out.sort(key=lambda r: (-r["n_records"], r["type"], r["canonical_name"]))
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        return raw["data"]
    if isinstance(raw, list):
        return raw
    sys.exit(f"{path}: top level must be a list or {{'data': [...]}}.")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    tmp.replace(path)


def write_review(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bucket by (type, loose_name) so obvious near-duplicates surface.
    loose_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        loose_buckets[(r["type"], normalise_name_loose(r["canonical_name"]))].append(r)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        n = len(records)
        n_multi = sum(1 for r in records if r["n_sub_problems"] > 1)
        f.write(f"# Knowledge point canonicalisation review\n\n")
        f.write(f"Total canonical KPs: **{n}**  |  cross-sub-problem "
                f"(count > 1): **{n_multi}**\n\n")
        by_type = Counter(r["type"] for r in records)
        f.write("Per type: " + ", ".join(f"{t}={c}" for t, c in by_type.items()) + "\n\n")

        # ---- flag near-duplicates ---------------------------------------
        f.write("## Suggested merge candidates (same type, similar name)\n\n")
        f.write("_Two KPs whose loose-normalised names collide -- worth "
                "eyeballing to see whether they should be merged manually._\n\n")
        any_flagged = False
        for (t, loose), group in sorted(loose_buckets.items(),
                                        key=lambda kv: (-len(kv[1]), kv[0])):
            if len(group) < 2:
                continue
            any_flagged = True
            f.write(f"- `[{t}]` loose=`{loose}`\n")
            for r in sorted(group, key=lambda r: -r["n_sub_problems"]):
                f.write(f"    - `{r['kp_id']}`  n={r['n_sub_problems']}  "
                        f"**{r['canonical_name']}**"
                        + (f"  aliases={r['aliases']}" if r['aliases'] else "")
                        + "\n")
        if not any_flagged:
            f.write("_(none found)_\n")
        f.write("\n")

        # ---- full listing grouped by type -------------------------------
        for t in ("law_or_equation", "concept", "technique", "assumption"):
            group = [r for r in records if r["type"] == t]
            if not group:
                continue
            f.write(f"## {t}  ({len(group)} KPs)\n\n")
            for r in group:
                f.write(f"### {r['canonical_name']}  "
                        f"`{r['kp_id']}`  (n_sub={r['n_sub_problems']}, "
                        f"n_records={r['n_records']})\n\n")
                if r["aliases"]:
                    f.write(f"- aliases: {r['aliases']}\n")
                if r["keywords_union"]:
                    f.write(f"- keywords: {r['keywords_union']}\n")
                if r["domain_hint"]:
                    top_dom = sorted(r["domain_hint"].items(),
                                     key=lambda kv: -kv[1])
                    f.write(f"- domain_hint: {top_dom}\n")
                if r["descriptions"]:
                    # show at most 3 descriptions per KP
                    for d in r["descriptions"][:3]:
                        f.write(f"    - _{d['sub_id']}_: {d['description']}\n")
                    extra = len(r["descriptions"]) - 3
                    if extra > 0:
                        f.write(f"    - _(+ {extra} more)_\n")
                f.write("\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    ap.add_argument("--out-review", type=Path, default=OUT_REVIEW)
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"missing input: {args.input}")

    items = load_items(args.input)
    canonical = build_canonical_kps(items)

    write_jsonl(args.out_jsonl,  canonical)
    write_review(args.out_review, canonical)

    # summary
    n = len(canonical)
    n_multi = sum(1 for r in canonical if r["n_sub_problems"] > 1)
    n_singleton = n - n_multi
    n_records = sum(r["n_records"] for r in canonical)
    by_type = Counter(r["type"] for r in canonical)
    print(f"[canonicalize_knowledge_points] {n_records} KP records "
          f"across {sum(1 for it in items for s in (it.get('sub_problems') or []) if isinstance(s, dict) and s.get('knowledge_points'))} "
          f"labelled sub-problems -> {n} canonical KPs "
          f"({n_multi} multi-use, {n_singleton} singleton).")
    print(f"[canonicalize_knowledge_points] per type: {dict(by_type)}")
    print(f"[canonicalize_knowledge_points] wrote {args.out_jsonl}")
    print(f"[canonicalize_knowledge_points] wrote {args.out_review}")


if __name__ == "__main__":
    main()
