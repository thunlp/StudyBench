#!/usr/bin/env python3
"""Retrieve candidate textbook fragments for each canonical knowledge point.

Stage 3.5 of the coverage-argument pipeline.  Sits between
``canonicalize_knowledge_points.py`` (which produces the canonical KP list)
and ``verify_textbook_matches.py`` (which asks an LLM to score each
candidate).  The job here is purely recall: surface ~30 plausible fragments
per KP so the verifier only pays LLM cost on a short list.

Retrieval is a two-channel union fused with Reciprocal Rank Fusion:

  * **BM25** over ``text_norm`` -- lexical, catches exact formula and term
    matches.  Top ``--pool`` (200) kept.
  * **Embeddings** (``all-mpnet-base-v2``, 768-dim, cosine) -- semantic,
    catches the case where the textbook says "buoyant force" and the KP
    says "Archimedes' principle".  Top ``--pool`` (200) kept.

    An earlier revision of this pipeline had the embedding channel silently
    disabled (every record had ``n_emb_scored=0``), which cost ~17 points of
    KP coverage -- see ``knowledge_points/COVERAGE_REPORT.md`` "缺陷 1".
    ``--require-emb`` (default on) makes that failure loud instead of silent.

  * ``rrf_score = sum over channels of 1/(60 + rank)``, then
    ``final_score = rrf_score * (1.0 if in_domain else 0.6)`` -- an
    out-of-domain fragment has to be clearly better to outrank an
    in-domain one.  Both are rounded to 5 decimals.

``in_domain`` compares the fragment's ``domain`` against the domains implied
by the parent ``problem_type``s the KP appears under (``domain_hint``).

Input:  ``studybench_data/knowledge_points.jsonl`` + ``studybench_data/textbook_fragments.jsonl``
Output: ``studybench_data/kp_candidates.jsonl`` -- one record per KP::

    {
      "kp_id": "b1ceb19d2e2e",
      "type": "law_or_equation",
      "canonical_name": "ideal gas law",
      "aliases": ["Ideal gas law"],
      "n_sub_problems": 23,
      "expected_domains": ["thermal_physics", ...],
      "n_bm25_scored": 8156,
      "n_emb_scored": 200,
      "query_text": "...",
      "candidates": [ {fragment_id, book, ..., bm25_rank, emb_rank,
                       rrf_score, final_score, text_snippet}, ... ]
    }

Usage::

    # incremental: only KPs not already present in the output (the default)
    python filter/retrieve_textbook_candidates.py

    # smoke test on 20 KPs, writing elsewhere
    python filter/retrieve_textbook_candidates.py --limit 20 --out /tmp/cand.jsonl

    # recompute specific KPs
    python filter/retrieve_textbook_candidates.py --kp-ids b1ceb19d2e2e --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STUDY_DIR = ROOT / "studybench_data"
KP_PATH = STUDY_DIR / "knowledge_points.jsonl"
FRAGMENTS_PATH = STUDY_DIR / "textbook_fragments.jsonl"
OUT_PATH = STUDY_DIR / "kp_candidates.jsonl"
EMB_CACHE_PATH = STUDY_DIR / "fragment_embeddings.npy"

EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
RRF_K = 60
OUT_OF_DOMAIN_PENALTY = 0.6
SNIPPET_CHARS = 240

# Parent ``problem_type`` -> textbook domains that plausibly teach it.
# Astrophysics and wave problems lean on mechanics often enough that the
# mechanics books count as in-domain for them.
PROBLEM_TYPE_DOMAINS: dict[str, list[str]] = {
    "mechanics": ["classical_mechanics"],
    "astronomy and astrophysics": ["astrophysics", "classical_mechanics"],
    "electromagnetic fields": ["electromagnetism"],
    "thermodynamics and statistical physics": ["thermal_physics"],
    "quantum physics": ["quantum_physics"],
    "oscillations and waves": ["waves_and_oscillations", "classical_mechanics"],
    "relativity": ["special_relativity"],
}


def norm_problem_type(s: str) -> str:
    """Fold the '&' / 'and' spelling split seen in the corpus."""
    return " ".join((s or "").strip().lower().replace("&", "and").split())


def expected_domains_for(kp: dict[str, Any]) -> list[str]:
    out: set[str] = set()
    for ptype in (kp.get("domain_hint") or {}):
        out.update(PROBLEM_TYPE_DOMAINS.get(norm_problem_type(ptype), []))
    return sorted(out)


def build_query_text(kp: dict[str, Any]) -> str:
    """Build the retrieval query shared by both channels.

    Layout (recovered from, and byte-identical to, the existing
    ``kp_candidates.jsonl``)::

        <canonical_name>          x3   -- upweights the KP name for BM25
        <keywords joined by " ">  x2   -- upweights retrieval keywords
        <first 2 descriptions>[:400]   -- prose context for the embedder

    The repetition is a crude but effective BM25 term-frequency boost; the
    descriptions carry the semantic signal the embedding channel needs.
    """
    name = kp.get("canonical_name", "") or ""
    keywords = " ".join(kp.get("keywords_union") or [])
    descs = [d.get("description", "") for d in (kp.get("descriptions") or [])]
    tail = " ".join(d for d in descs[:2] if d)[:400]

    parts = [name] * 3
    if keywords:
        parts.extend([keywords] * 2)
    if tail:
        parts.append(tail)
    return "\n".join(parts)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fragment_embeddings(frags: list[dict[str, Any]], model, cache: Path,
                        batch_size: int):
    """Encode every fragment once and cache to .npy.

    The cache is keyed by fragment count alone; ``build_textbook_fragments.py``
    emits content-hashed stable ids, so a changed corpus changes the count in
    practice.  ``--refresh-emb-cache`` forces a rebuild if that ever bites.
    """
    import numpy as np

    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(frags):
            print(f"[retrieve] loaded fragment embeddings from {cache.name} "
                  f"{arr.shape}")
            return arr
        print(f"[retrieve] embedding cache stale "
              f"({arr.shape[0]} rows vs {len(frags)} fragments), rebuilding")

    print(f"[retrieve] encoding {len(frags)} fragments with {EMB_MODEL} ...")
    texts = [f.get("text_md") or f.get("text_norm") or "" for f in frags]
    arr = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    print(f"[retrieve] cached fragment embeddings -> {cache.name} {arr.shape}")
    return arr


def candidate_record(frag: dict[str, Any], bm25_rank: int | None,
                     bm25_score: float | None, emb_rank: int | None,
                     emb_score: float | None, in_domain: bool) -> dict[str, Any]:
    rrf = 0.0
    if bm25_rank:
        rrf += 1.0 / (RRF_K + bm25_rank)
    if emb_rank:
        rrf += 1.0 / (RRF_K + emb_rank)
    final = rrf * (1.0 if in_domain else OUT_OF_DOMAIN_PENALTY)
    text = frag.get("text_md") or ""
    return {
        "fragment_id": frag["fragment_id"],
        "book": frag.get("book", ""),
        "book_display": frag.get("book_display", ""),
        "domain": frag.get("domain", ""),
        "in_domain": in_domain,
        "kind": frag.get("kind", ""),
        "chapter": frag.get("chapter", ""),
        "chapter_title": frag.get("chapter_title", ""),
        "section_path": frag.get("section_path") or [],
        "is_solution_manual": bool(frag.get("is_solution_manual")),
        "bm25_rank": bm25_rank,
        "bm25_score": round(bm25_score, 4) if bm25_score is not None else None,
        "emb_rank": emb_rank,
        "emb_score": round(float(emb_score), 4) if emb_score is not None else None,
        "rrf_score": round(rrf, 5),
        "final_score": round(final, 5),
        "text_snippet": text[:SNIPPET_CHARS],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kps", type=Path, default=KP_PATH)
    ap.add_argument("--fragments", type=Path, default=FRAGMENTS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--emb-cache", type=Path, default=EMB_CACHE_PATH)
    ap.add_argument("--refresh-emb-cache", action="store_true")
    ap.add_argument("--pool", type=int, default=200,
                    help="per-channel candidate pool size (default 200)")
    ap.add_argument("--top-k", type=int, default=30,
                    help="candidates kept per KP after fusion (default 30)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="process first N KPs only")
    ap.add_argument("--kp-ids", default="", help="comma-separated kp_ids to process")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the kp_id is already in --out")
    ap.add_argument("--no-emb", action="store_true",
                    help="BM25 only (debug; degrades recall badly)")
    args = ap.parse_args()

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("[retrieve] pip install rank-bm25")
    import numpy as np

    frags = load_jsonl(args.fragments)
    if not frags:
        sys.exit(f"[retrieve] no fragments in {args.fragments}")
    kps = load_jsonl(args.kps)
    print(f"[retrieve] {len(kps)} canonical KPs, {len(frags)} fragments")

    # --- incremental: skip kp_ids already present in the output -------------
    existing: dict[str, str] = {}
    if args.out.exists() and not args.force:
        with args.out.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing[json.loads(line)["kp_id"]] = line
        print(f"[retrieve] {len(existing)} KPs already in {args.out.name}")

    wanted = {s.strip() for s in args.kp_ids.split(",") if s.strip()}
    todo = [k for k in kps if (not wanted or k["kp_id"] in wanted)
            and (args.force or k["kp_id"] not in existing)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[retrieve] {len(todo)} KPs to retrieve, "
          f"{len(kps) - len(todo)} skipped")
    if not todo:
        print("[retrieve] nothing to do")
        return

    bm25 = BM25Okapi([(f.get("text_norm") or "").split() for f in frags])

    frag_emb = None
    model = None
    if not args.no_emb:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit("[retrieve] pip install sentence-transformers "
                     "(or pass --no-emb, which badly degrades recall)")
        model = SentenceTransformer(EMB_MODEL)
        if args.refresh_emb_cache and args.emb_cache.exists():
            args.emb_cache.unlink()
        frag_emb = fragment_embeddings(frags, model, args.emb_cache, args.batch_size)

    results: list[dict[str, Any]] = []
    from tqdm import tqdm
    for kp in tqdm(todo, desc="retrieve"):
        query = build_query_text(kp)
        exp_domains = set(expected_domains_for(kp))

        # BM25 channel
        scores = bm25.get_scores(query.split())
        n_bm25_scored = int((scores > 0).sum())
        order = np.argsort(-scores)[:args.pool]
        bm25_hits = {int(i): (rank, float(scores[i]))
                     for rank, i in enumerate(order, 1) if scores[i] > 0}

        # Embedding channel
        emb_hits: dict[int, tuple[int, float]] = {}
        if frag_emb is not None:
            qv = model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True)[0]
            sims = frag_emb @ qv
            eorder = np.argsort(-sims)[:args.pool]
            emb_hits = {int(i): (rank, float(sims[i]))
                        for rank, i in enumerate(eorder, 1)}

        cands = []
        for idx in set(bm25_hits) | set(emb_hits):
            frag = frags[idx]
            b = bm25_hits.get(idx)
            e = emb_hits.get(idx)
            in_domain = (not exp_domains) or (frag.get("domain") in exp_domains)
            cands.append(candidate_record(
                frag,
                b[0] if b else None, b[1] if b else None,
                e[0] if e else None, e[1] if e else None,
                in_domain,
            ))
        cands.sort(key=lambda c: (-c["final_score"], c["fragment_id"]))
        cands = cands[:args.top_k]

        results.append({
            "kp_id": kp["kp_id"],
            "type": kp.get("type", ""),
            "canonical_name": kp.get("canonical_name", ""),
            "aliases": kp.get("aliases") or [],
            "n_sub_problems": kp.get("n_sub_problems", 0),
            "expected_domains": sorted(exp_domains),
            "n_bm25_scored": n_bm25_scored,
            "n_emb_scored": len(emb_hits),
            "query_text": query,
            "candidates": cands,
        })

    # Merge: refreshed records win, untouched ones are preserved verbatim.
    fresh = {r["kp_id"] for r in results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    n_written = 0
    with tmp.open("w", encoding="utf-8") as f:
        for kp_id, line in existing.items():
            if kp_id not in fresh:
                f.write(line + "\n")
                n_written += 1
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_written += 1
    tmp.replace(args.out)

    n_emb0 = sum(1 for r in results if r["n_emb_scored"] == 0)
    print(f"[retrieve] wrote {n_written} records -> {args.out}")
    if not args.no_emb and n_emb0:
        print(f"[retrieve] WARNING: {n_emb0}/{len(results)} new KPs got zero "
              f"embedding hits -- semantic channel may be broken")


if __name__ == "__main__":
    main()
