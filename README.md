# StudyBench

StudyBench measures how efficiently a self-evolution method turns a small, fixed set of physics textbooks into problem-solving capability. The training side of the benchmark is a **Corpus** of raw textbook passages (plus optional instruction layers built from the same books). The test side is an Application Set of hard end-of-chapter problems and a Transfer Set of olympiad problems.

This README covers **how to rebuild the Corpus**. Other parts of the release will be documented here later.

## Why this repo does not ship the books

The eleven textbooks are in-copyright commercial works. We cannot redistribute the PDFs, scanned pages, or the parsed Markdown derived from them.

What this repository ships is the **processing pipeline**. To obtain the Corpus used for training:

1. Collect your own legal copies of the listed textbooks (and, where relevant, their official solution manuals).
2. Lay them out as described below.
3. Run the pipeline. It converts PDFs to Markdown, applies light OCR cleanup, and redacts every Application Set item so the training Corpus does not contain the test problems.

Do not commit `PhysicsBooks/`, the PDFs, or the parsed Markdown to git. Put secrets such as `MINERU_TOKEN` in `env_local.sh` (gitignored), not in the scripts.

## Books to collect

Use these **directory names** exactly. `redact_eval.py` matches the last folder under `PhysicsBooks/` to the `"source"` field in the eval JSON.

| Directory (`source`) | Textbook | Area |
| --- | --- | --- |
| `Morin_ClassicalMechanics` | *Introduction to Classical Mechanics* (Morin) | Mechanics |
| `Kleppner_Mechanics` | *Introduction to Mechanics* (Kleppner & Kolenkow), plus official solution manual | Mechanics |
| `Purcell_EM` | *Electricity and Magnetism* (Purcell & Morin), plus official solution manual | Electromagnetism |
| `Griffiths_Electrodynamics` | *Introduction to Electrodynamics* (Griffiths), plus official solution manual | Electromagnetism |
| `Blundell_ThermalPhysics` | *Concepts in Thermal Physics* (Blundell & Blundell), plus official solution manual | Thermal Physics |
| `Georgi_Waves` | *The Physics of Waves* (Georgi), plus official solution manual | Waves |
| `EisbergResnick_QuantumPhysics` | *Quantum Physics* (Eisberg & Resnick) | Quantum Physics |
| `French_SpecialRelativity` | *Special Relativity* (French) | Relativity |
| `TaylorWheeler_SpacetimePhysics` | *Spacetime Physics* (Taylor & Wheeler) | Relativity |
| `CarrollOstlie_ModernAstrophysics` | *An Introduction to Modern Astrophysics* (Carroll & Ostlie) | Astrophysics |
| `Palen_SchaumAstronomy` | *Schaum's Outline of Astronomy* (Palen) | Astrophysics |

Books marked “plus official solution manual” in the paper are bundled with that manual when extracting answers. Put the manual PDFs **under the same `source` directory** so redaction can see answer keys and worked solutions that also appear in the Application Set.

## Layout

Split each book into one PDF per chapter (or similar unit). MinerU writes a sibling folder next to each PDF:

```
PhysicsBooks/
  TaylorWheeler_SpacetimePhysics/
    00_cover_toc.pdf
    01_chapter1.pdf
    ...
    01_chapter1/          # created by the pipeline
      full.md             # MinerU output
      ocrfix.md           # light cleanup
      ocrfix.redacted.md  # Corpus used for training
```

`corpus_scripts/mineru_parse.py` walks `PhysicsBooks/` recursively and skips a PDF whose output folder already contains `full.md`.

## Prerequisites

- Python 3.10+
- A [MinerU](https://mineru.net/apiManage/docs) API token
- `requests` and `tqdm` (`pip install requests tqdm`)

Create `env_local.sh` in the repo root:

```bash
export MINERU_TOKEN="your-token"
```

`corpus_process.sh` sources this file before calling MinerU.

## Run the pipeline

From the repo root:

```bash
bash corpus_process.sh
```

That is three steps. Existing outputs are skipped; add `--force` on a step to redo it.

### 1. Parse PDFs (`corpus_scripts/mineru_parse.py`)

Uploads each pending PDF to the MinerU VLM+OCR API and unpacks the zip into a same-named folder (`full.md`, layout JSON, page images).

```bash
python3 corpus_scripts/mineru_parse.py PhysicsBooks \
    --model vlm --language en \
    --batch-size 20 --poll-interval 5 --is-ocr
```

`--dry-run` lists pending PDFs without calling the API. `--limit N` is useful for a smoke test.

### 2. Light OCR cleanup (`corpus_scripts/ocr_fix.py`)

Reads `full.md` and writes `ocrfix.md` next to it. This step only applies cheap, high-precision fixes:

- drop VLM prompt leaks (`Rule 2` / `Ground Truth` paragraphs on decorative rules)
- rejoin split digits and `^` / `_` braces in math
- strip leftover outline page numbers (`PROBLEMS 169` → `PROBLEMS`)
- unwrap leftover `mineru-algorithm` divs
- a few unambiguous typos

It does **not** try to resolve ν/v, l/1, or other cases that need the original page.

```bash
python3 corpus_scripts/ocr_fix.py PhysicsBooks
```

### 3. Redact Application Set items (`corpus_scripts/redact_eval.py`)

For each eval field (`problem` / `solution` / `answer`, including `sub_problems`) whose `"source"` matches a book folder, the script finds the **single best** passage in that book's `ocrfix.md` files and replaces it with `[REDACTED]`. The eval JSON is never modified.

N-grams are only seeds. Shared textbook phrasing produces extra short hits, so each field keeps only its longest run, and only if that run covers most of the field (default: 13-gram seeds, 60% coverage). A problem is expected to have one source locus.

```bash
python3 corpus_scripts/redact_eval.py eval/data/qwen3_8b_textbook_problem.json \
    --books-dir PhysicsBooks
```

Use `--dry-run` to inspect match counts, `--force` to overwrite existing `ocrfix.redacted.md`.

## What to train on

After a successful run, the Corpus is the set of `ocrfix.redacted.md` files under `PhysicsBooks/<source>/`. Use those locally. Do not redistribute them: they are still derived from copyrighted books.

`ocrfix.md` is the unredacted cleanup and is useful for debugging the matcher's one-locus rule. Training and any shared artifacts should use the redacted files only.
