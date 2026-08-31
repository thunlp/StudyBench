"""Chunk the PhysicsBooks markdown corpus into retrievable fragments.

Stage 2 of the coverage-argument pipeline needs, for every knowledge
point, "the textbook fragment that teaches it or the worked example that
uses it". This is the offline builder that produces that fragment
index.

Layout on disk (see ``train_data/PhysicsBooks/``): every book has one
folder per chapter, each containing a single ``full.md``. Solution
manuals live in sibling ``*_solution_manual*`` folders with the same
per-chapter structure. All headings were promoted to H1 by the OCR, so
the "outline" of a chapter is just the sequence of ``# ...`` lines with
regular prose in between.

We build fragments by:

  * walking every H1 heading, classifying it as one of
      - noise             (page-running titles like ``# CHAPTER 1`` or
                          ``# TABLE 1-3``; merged into the current body)
      - section anchor    (``# 1.1 Balancing forces`` / ``# 1-1 …`` /
                          all-caps single-int like French);
                          resets ``section_path``
      - subsection        (unnumbered heading between anchors; pushed
                          onto ``section_path``)
      - example intro     (``# Example (…):``, ``# SAMPLE PROBLEM…``,
                          ``# Solved Problem…``); opens a new
                          ``kind=example`` fragment
      - solution intro    (``# Solution:``); attaches to the current
                          example fragment
      - problem-zone gate (``# Problems`` / ``# Exercises`` / ``# 
                          Questions`` / ``# Supplementary Problems``);
                          stops emitting fragments for the rest of the
                          file (those are bare problem statements
                          without teaching value)
  * every heading that is not noise flushes the current fragment;
  * fragments longer than ``MAX_CHARS`` are subdivided on blank-line
    paragraph boundaries with a small overlap.

Solution-manual chapters (folders whose ancestor name ends in
``_solution_manual*``) go through the same splitter but every
non-noise fragment is force-tagged ``kind=example`` and the
``section_path`` is prefixed with ``["Solution Manual"]``. That matches
the design decision that solution manuals are worked-example material.

Output: one JSONL record per fragment, streamed to
``<repo>/studybench_data/textbook_fragments.jsonl``. Each record::

    {
      "fragment_id":   "morin::ch02::1.1::0",
      "book":          "introduction_to_classical_mechanics",
      "book_display":  "Morin, Introduction to Classical Mechanics",
      "domain":        "classical_mechanics",
      "is_solution_manual": false,
      "chapter":       "02_chapter1",
      "chapter_title": "Statics",
      "section_path":  ["1.1 Balancing forces", "Tension"],
      "heading":       "Tension",
      "anchor":        "1.1",              # nearest numbered section id
      "kind":          "exposition",       # or "example"
      "example_title": null,
      "text_md":       "...raw markdown span...",
      "text_norm":     "...lower-cased plain text for BM25...",
      "n_chars":       1234,
      "n_tokens_est":  308                 # chars/4
    }

The fragment_id is stable across reruns (bookslug + chapterslug + anchor
+ within-chapter running index), so downstream candidate/verify files
can safely reference it.

Usage::

    python filter/build_textbook_fragments.py
    python filter/build_textbook_fragments.py --books introduction_to_classical_mechanics
    python filter/build_textbook_fragments.py --exclude-solution-manuals
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "train_data" / "PhysicsBooks"
OUT_DIR = ROOT / "studybench_data"
OUT_PATH = OUT_DIR / "textbook_fragments.jsonl"


# ---------------------------------------------------------------------------
# Book catalogue -- keep display names + coarse domain tags for retrieval
# ---------------------------------------------------------------------------

# We tag every primary book with a coarse domain and give it a nice display
# name (used later in review dumps and study-guide citations). Solution
# manuals inherit the same tags via their base slug (``*_solution_manual*``).

BOOKS: dict[str, dict[str, str]] = {
    "introduction_to_classical_mechanics": {
        "display": "Morin, Introduction to Classical Mechanics",
        "domain":  "classical_mechanics",
    },
    "introduction_to_mechanics": {
        "display": "Kleppner & Kolenkow, An Introduction to Mechanics",
        "domain":  "classical_mechanics",
    },
    "electricity_and_magnetism": {
        "display": "Purcell & Morin, Electricity and Magnetism",
        "domain":  "electromagnetism",
    },
    "introduction_to_electrodynamics": {
        "display": "Griffiths, Introduction to Electrodynamics",
        "domain":  "electromagnetism",
    },
    "concepts_in_thermal_physics": {
        "display": "Blundell & Blundell, Concepts in Thermal Physics",
        "domain":  "thermal_physics",
    },
    "physics_of_waves": {
        "display": "Georgi, The Physics of Waves",
        "domain":  "waves_and_oscillations",
    },
    "quantum_physics": {
        "display": "Eisberg & Resnick, Quantum Physics",
        "domain":  "quantum_physics",
    },
    "modern_astrophysics": {
        "display": "Carroll & Ostlie, An Introduction to Modern Astrophysics",
        "domain":  "astrophysics",
    },
    "schaums_astronomy": {
        "display": "Palen, Schaum's Outline of Astronomy",
        "domain":  "astrophysics",
    },
    "spacetime_physics": {
        "display": "Taylor & Wheeler, Spacetime Physics",
        "domain":  "special_relativity",
    },
    "special_relativity": {
        "display": "A. P. French, Special Relativity",
        "domain":  "special_relativity",
    },
}

# short slug used in fragment_id
BOOK_SLUG: dict[str, str] = {
    "introduction_to_classical_mechanics": "morin",
    "introduction_to_mechanics":           "kleppner",
    "electricity_and_magnetism":           "purcell",
    "introduction_to_electrodynamics":     "griffiths",
    "concepts_in_thermal_physics":         "blundell",
    "physics_of_waves":                    "georgi",
    "quantum_physics":                     "eisberg",
    "modern_astrophysics":                 "carroll_ostlie",
    "schaums_astronomy":                   "schaums_astro",
    "spacetime_physics":                   "taylor_wheeler",
    "special_relativity":                  "french",
}


def resolve_book(folder_name: str) -> Optional[tuple[str, dict[str, str], bool]]:
    """Map a top-level folder under ``PhysicsBooks/`` to (base_key, meta,
    is_solution_manual). Returns None for folders we should skip."""
    is_sol = False
    key = folder_name
    for suffix in ("_solution_manual_by_chapter", "_solution_manual"):
        if key.endswith(suffix):
            is_sol = True
            key = key[: -len(suffix)]
            break
    if key not in BOOKS:
        return None
    return key, BOOKS[key], is_sol


# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------

H1_RE = re.compile(r"^#\s+(.+?)\s*$")

# "# 1.1 Balancing forces"  /  "# 1.1.2 Sub-sub"  /  "# 1.1 ■ TITLE"
SECTION_DOT_RE = re.compile(
    r"^(\d+)\.(\d+)(?:\.(\d+))?\s*[■\.]?\s*(.+?)\s*$"
)
# "# 2-1 THE PHOTOELECTRIC EFFECT" (Eisberg)
SECTION_DASH_RE = re.compile(r"^(\d+)\-(\d+)\s+(.+?)\s*$")

# "# 1 Departures from Newtonian dynamics" (French).  Very restrictive so
# we don't confuse it with paragraph-leading numbers: title must start
# with a Latin letter and be reasonably long, and the integer must be
# <= 3 digits.
SECTION_ALT_RE = re.compile(r"^(\d{1,3})\s+([A-Za-z][A-Za-z0-9 ,;:\-'\u2014\u2013]{6,120})\s*$")

# Worked-example intros in various OCR flavours.
EXAMPLE_INTRO_RE = re.compile(
    r"^(?:"
    r"Example\s*(?:\d+(?:\.\d+)*)?\s*(?:\(.*?\))?\s*[:\.\-]?\s*(.*)|"
    r"Worked\s*Example\s*(?:\d+(?:\.\d+)*)?\s*[:\.\-]?\s*(.*)|"
    r"SAMPLE\s*PROBLEM\s*\d+[\-\.]?\d*\s*(.*)|"
    r"Solved\s*Problems?\s*(.*)|"                          # Schaum's zone label
    r"EXAMPLE\s*\d+(?:\.\d+)*\s*[:\.\-]?\s*(.*)"
    r")$",
    re.IGNORECASE,
)

# Bare "# Solution:" / "# SOLUTION" (attach to the current example).
SOLUTION_INTRO_RE = re.compile(r"^Solutions?\s*:?\s*$", re.IGNORECASE)

# Inline (non-H1) example/solution intros -- Morin and a few others leave
# these as regular paragraph starters instead of promoting them to H1.
# We match only at paragraph boundaries (previous line blank) to avoid
# stealing "example" verbs mid-sentence.  Match is anchored to the
# start of the line; the intro may run into a description that continues
# on the same line ("Example (Block on a plane): A block of mass M...").
_INLINE_EXAMPLE_PATTERNS = [
    # "Example (Block on a plane):"  (Morin)
    (re.compile(r"^Example\s*\((?P<title>[^)]+)\)\s*:"), "title"),
    # "Example 1.1 The Law of Cosines"  (Kleppner)
    (re.compile(r"^Example\s+(?P<num>\d+(?:\.\d+)*)\s+(?P<title>[A-Z][^\n]{2,100})?"), "title"),
    # "Worked Example 3.4 - ..."       (Blundell)
    (re.compile(r"^Worked\s+Example\s+(?P<num>\d+(?:\.\d+)*)\s*[:\-\.]\s*(?P<title>.*)"), "title"),
    # "SAMPLE PROBLEM 1-1"             (Taylor & Wheeler; title on next line)
    (re.compile(r"^SAMPLE\s+PROBLEM\s+(?P<num>\d+[\-\.]?\d*)\s*(?P<title>[A-Z][^\n]{0,120})?"), "title"),
    # "EXAMPLE 1.1 Title" (uppercase variants)
    (re.compile(r"^EXAMPLE\s+(?P<num>\d+(?:\.\d+)*)\s*[:\-\.]?\s*(?P<title>.*)"), "title"),
]

INLINE_SOLUTION_RE = re.compile(r"^Solutions?\s*:\s*$")


def match_inline_example(line: str) -> Optional[str]:
    """Return the example title (may be empty) if `line` starts with an
    inline example intro, else None."""
    for regex, title_key in _INLINE_EXAMPLE_PATTERNS:
        m = regex.match(line)
        if m is not None:
            gd = m.groupdict()
            return (gd.get(title_key) or "").strip(" :.-") or ""
    return None

# Chapter-end problem zones -- once we cross one of these, stop emitting
# fragments from this file (they are bare statements without teaching
# value).  Deliberately tight so we don't gate real content.
PROBLEM_ZONE_RE = re.compile(
    r"^(?:"
    r"Problems?|Exercises?|Questions?|"
    r"Supplementary\s+Problems?|Additional\s+Problems?|"
    r"PROBLEMS?|EXERCISES?|QUESTIONS?|"
    r"PRACTICE|"
    r"CHAPTER\s+\d+\s+EXERCISES?|"
    r"Answers?\s+to\s+(?:Odd(?:-Numbered)?\s+)?Problems?|"
    r"Multiple\s+Choice\s+Questions?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

# Noise headings: page-running titles, figure/table labels, front/back
# matter markers.  These don't reset the section stack and are folded
# into the current fragment body as regular text.
NOISE_HEADING_RE = re.compile(
    r"^(?:"
    r"PART(?:\s+[IVXLC0-9]+)?|"
    r"CHAPTER(?:\s+\d+)?|"
    r"CHAPTER\s+\d+\s*\.?\s*[A-Z ]{0,60}|"     # "CHAPTER 1. STATICS"
    r"TABLE\s*[\d\.\-]*.*|"
    r"FIG(?:URE)?\.?\s*[\d\.\-]*.*|"
    r"REFERENCES?|"
    r"ACKNOWLEDG(?:E?)MENTS?|"
    r"SUGGESTED\s+READING|"
    r"BIBLIOGRAPHY|"
    r"INDEX|"
    r"APPENDIX(?:\s+[A-Z0-9]+)?.*|"
    r"CONTENTS|"
    r"PREFACE|"
    r"NOTATION|"
    r"BOX\s*[\d\.\-]*.*|"
    r"CHAPTER\s+SUMMARY|"
    r"CHAPTER\s+CHECKLIST|"
    r"Chapter\s+Checklist|"
    r"SUMMARY|"
    r"INTRODUCTION\s+TO\s+THE\s+EXERCISES|"
    r"General|Technical|"                       # subheads under SUGGESTED READING
    r"REMARKS?:?|Remark:?|"                     # inline annotations promoted to H1
    r"HISTORICAL\s+NOTES?"
    r")\s*:?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Fragment size policy
# ---------------------------------------------------------------------------

TARGET_CHARS = 2000    # aim for fragments ~500 tokens
MAX_CHARS    = 6000    # anything larger gets split on paragraph boundary
MIN_CHARS    =  200    # fragments smaller than this get merged with previous
                       #   sibling within the same section_path
OVERLAP_CHARS = 400    # overlap when force-splitting a huge fragment


# ---------------------------------------------------------------------------
# Text normalisation for BM25
# ---------------------------------------------------------------------------

_MATH_BLOCK = re.compile(r"\$\$([\s\S]+?)\$\$")
_MATH_INLINE = re.compile(r"\$([^$\n]+)\$")
_IMG_TAG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
# Strip common LaTeX macros so tokens survive: \\frac{a}{b} -> a b,
# \\vec{v} -> v, \\hat{r} -> r, \\dot{x} -> x, \\alpha -> alpha etc.
_LATEX_MACRO = re.compile(r"\\(?:[a-zA-Z]+|.)")
_MD_FMT = re.compile(r"[*_`>]|^\s*[\-•]\s*", re.MULTILINE)
_WS = re.compile(r"\s+")


def _unwrap_math(m: re.Match) -> str:
    """Replace ``$...$`` and ``$$...$$`` with the raw LaTeX inside so
    equation tokens (``PV = nRT``, ``F = ma``) survive normalisation and
    remain searchable by BM25."""
    inner = m.group(1)
    # peel simple macros to raw letters (\alpha -> alpha)
    inner = _LATEX_MACRO.sub(lambda mm: " " + mm.group(0)[1:] + " ", inner)
    # curly braces -> whitespace
    inner = inner.replace("{", " ").replace("}", " ")
    return " " + inner + " "


def normalise_text(md: str) -> str:
    """Cheap plain-text projection used for keyword / BM25 matching.

    We deliberately KEEP the content of math spans (just peel the $ /
    $$ markers and LaTeX macros) so equation tokens like ``PV = nRT``
    or ``F = ma`` remain searchable.  Images and HTML tags are dropped.
    """
    s = md
    s = _MATH_BLOCK.sub(_unwrap_math, s)
    s = _MATH_INLINE.sub(_unwrap_math, s)
    s = _IMG_TAG.sub(" ", s)
    s = _HTML_TAG.sub(" ", s)
    s = s.replace("\\(", " ").replace("\\)", " ")
    s = s.replace("\\[", " ").replace("\\]", " ")
    s = _MD_FMT.sub(" ", s)
    s = s.lower()
    s = _WS.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    book: str
    book_display: str
    domain: str
    is_solution_manual: bool
    chapter: str
    chapter_title: str
    section_path: list[str]
    heading: str
    anchor: str
    kind: str                 # "exposition" | "example"
    example_title: Optional[str]
    text_md: str = ""

    def finalize(self, running_idx: int) -> dict:
        text = self.text_md.strip()
        chapter_slug = re.sub(r"[^0-9a-z]+", "", self.chapter.lower())
        anchor_slug = self.anchor.replace(".", "p").replace("-", "d") or "top"
        content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        frag_id = (
            f"{BOOK_SLUG.get(self.book, self.book)}"
            f"::{chapter_slug}::{anchor_slug}::{running_idx:03d}::{content_hash}"
        )
        n_chars = len(text)
        return {
            "fragment_id":       frag_id,
            "book":              self.book,
            "book_display":      self.book_display,
            "domain":            self.domain,
            "is_solution_manual": self.is_solution_manual,
            "chapter":           self.chapter,
            "chapter_title":     self.chapter_title,
            "section_path":      list(self.section_path),
            "heading":           self.heading,
            "anchor":            self.anchor,
            "kind":              self.kind,
            "example_title":     self.example_title,
            "text_md":           text,
            "text_norm":         normalise_text(text),
            "n_chars":           n_chars,
            "n_tokens_est":      max(1, n_chars // 4),
        }


# ---------------------------------------------------------------------------
# Chapter title extraction (best-effort)
# ---------------------------------------------------------------------------

def _guess_chapter_title(md: str, fallback: str) -> str:
    """Pull a chapter title out of the leading H1 block.

    Books differ: Morin uses ``# Chapter N`` + ``# <Title>``; Kleppner
    splits the title over multiple H1s (``# VECTORS`` ``# AND`` ``#
    KINEMATICS``); modern_astrophysics tags with ``# CHAPTER`` ``# N`` +
    title; Eisberg puts the title on the very first line.
    """
    lines = md.splitlines()
    # skip blank lines
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    # collect leading H1 headings until we hit the first section anchor or
    # >6 headings
    titles: list[str] = []
    while i < len(lines) and len(titles) < 6:
        m = H1_RE.match(lines[i])
        if not m:
            if lines[i].strip():
                break
            i += 1
            continue
        heading = m.group(1).strip()
        # stop at first real section anchor
        if (SECTION_DOT_RE.match(heading)
                or SECTION_DASH_RE.match(heading)
                or SECTION_ALT_RE.match(heading)
                or EXAMPLE_INTRO_RE.match(heading)
                or PROBLEM_ZONE_RE.match(heading)):
            break
        # drop pure "Chapter N", "CHAPTER 1", single-int labels
        low = heading.lower()
        if re.match(r"^(?:part|chapter)\s*[ivxlc0-9]*\s*$", low):
            i += 1
            continue
        if re.match(r"^\d+$", heading):
            i += 1
            continue
        titles.append(heading)
        i += 1
    if not titles:
        return fallback
    # dedupe consecutive shouty duplicates; join short ALL-CAPS runs
    joined = " ".join(titles)
    joined = re.sub(r"\s+", " ", joined).strip()
    # collapse "VECTORS AND KINEMATICS" if the pieces were split
    return joined[:200]


# ---------------------------------------------------------------------------
# Heading classification
# ---------------------------------------------------------------------------

@dataclass
class ClassifiedHeading:
    kind: str                        # "section" | "subsection" | "example"
                                     # | "solution" | "problem_zone" | "noise"
    anchor: str                      # "1.1" or "" for unnumbered
    title: str
    example_title: Optional[str] = None


def _match_section(heading: str) -> Optional[ClassifiedHeading]:
    m = SECTION_DOT_RE.match(heading)
    if m:
        parts = [m.group(1), m.group(2)]
        if m.group(3):
            parts.append(m.group(3))
        anchor = ".".join(parts)
        title = m.group(4).strip()
        # rule out "1.1." style trailing-period running titles by checking
        # that the title doesn't end with ALL CAPS + no lowercase letters:
        # noisy running titles are things like "1.5. SOLUTIONS"; those we
        # let through as sections anyway (the "SOLUTIONS" zone gets caught
        # by PROBLEM_ZONE_RE if it's bare).
        # Also skip pure-numeric "titles" like "1.1 2" (should not happen).
        if title and not re.fullmatch(r"\d+", title):
            return ClassifiedHeading("section", anchor, f"{anchor} {title}")
    m = SECTION_DASH_RE.match(heading)
    if m:
        anchor = f"{m.group(1)}-{m.group(2)}"
        title = m.group(3).strip()
        if title and not re.fullmatch(r"\d+", title):
            return ClassifiedHeading("section", anchor, f"{anchor} {title}")
    m = SECTION_ALT_RE.match(heading)
    if m:
        anchor = m.group(1)
        title = m.group(2).strip()
        return ClassifiedHeading("section", anchor, f"{anchor} {title}")
    return None


def _match_example(heading: str) -> Optional[ClassifiedHeading]:
    m = EXAMPLE_INTRO_RE.match(heading)
    if not m:
        return None
    # extract the "title" (the non-None captured group after the intro)
    title_bits = [g for g in m.groups() if g]
    example_title = title_bits[0].strip(" :.-") if title_bits else ""
    return ClassifiedHeading("example", "", heading, example_title or None)


def classify_heading(heading: str) -> ClassifiedHeading:
    if PROBLEM_ZONE_RE.match(heading):
        return ClassifiedHeading("problem_zone", "", heading)
    if NOISE_HEADING_RE.match(heading):
        return ClassifiedHeading("noise", "", heading)
    if SOLUTION_INTRO_RE.match(heading):
        return ClassifiedHeading("solution", "", heading)
    sec = _match_section(heading)
    if sec is not None:
        return sec
    ex = _match_example(heading)
    if ex is not None:
        return ex
    # unnumbered heading between anchors -> subsection
    return ClassifiedHeading("subsection", "", heading)


# ---------------------------------------------------------------------------
# Chapter parser
# ---------------------------------------------------------------------------

def parse_chapter(
    md_path: Path,
    book_key: str,
    meta: dict[str, str],
    is_solution_manual: bool,
) -> Iterator[dict]:
    md = md_path.read_text(encoding="utf-8", errors="replace")
    chapter_folder = md_path.parent.name
    chapter_title = _guess_chapter_title(md, chapter_folder)

    # section_stack: (level, title) with level 1 = numbered section anchor,
    # 2 = unnumbered subsection.  We only keep 2 levels of context.
    section_stack: list[tuple[int, str]] = []
    current_anchor = ""

    def new_frag(kind: str, heading: str,
                 example_title: Optional[str] = None) -> Fragment:
        # Solution manuals are entirely worked-solutions; every non-noise
        # fragment carries pedagogical value only as an example, so we
        # force kind=example there.  This matches the design decision to
        # index solution-manual fragments as examples.
        effective_kind = "example" if is_solution_manual else kind
        return Fragment(
            book=book_key,
            book_display=meta["display"],
            domain=meta["domain"],
            is_solution_manual=is_solution_manual,
            chapter=chapter_folder,
            chapter_title=chapter_title,
            section_path=(["Solution Manual"] if is_solution_manual else [])
                           + [t for _, t in section_stack],
            heading=heading,
            anchor=current_anchor,
            kind=effective_kind,
            example_title=example_title,
            text_md="",
        )

    frags: list[Fragment] = []

    def push(frag: Fragment) -> None:
        """Accept a finished fragment, merging tiny ones into the previous
        sibling within the same section/anchor/kind."""
        text = frag.text_md.strip()
        if not text:
            return
        frag.text_md = text
        if (
            frags
            and len(text) < MIN_CHARS
            and frags[-1].anchor == frag.anchor
            and frags[-1].section_path == frag.section_path
            and frags[-1].kind == frag.kind
        ):
            frags[-1].text_md = frags[-1].text_md + "\n\n" + text
        else:
            frags.append(frag)

    # Prime with an exposition fragment for everything before the first
    # real heading (chapter preamble, "Preview", etc.).
    cur = new_frag("exposition", chapter_title)
    hit_problem_zone = False
    prev_blank = True   # start-of-file counts as a paragraph boundary

    for line in md.splitlines():
        if hit_problem_zone:
            break
        m = H1_RE.match(line)
        if m is None:
            stripped = line.strip()
            # Detect inline example / solution intros at paragraph starts.
            # These are Morin-style "Example (Block on a plane):" that the
            # OCR left as plain text; we treat them the same as headings.
            if prev_blank and stripped:
                inline_title = match_inline_example(stripped)
                if inline_title is not None:
                    push(cur)
                    heading_line = stripped[:200]
                    cur = new_frag(
                        "example",
                        heading_line,
                        inline_title or heading_line,
                    )
                    cur.text_md += line + "\n"
                    prev_blank = False
                    continue
                if INLINE_SOLUTION_RE.match(stripped):
                    if cur.kind != "example":
                        push(cur)
                        cur = new_frag("example", stripped,
                                       example_title="(Solution)")
                    cur.text_md += line + "\n"
                    prev_blank = False
                    continue
            cur.text_md += line + "\n"
            prev_blank = (stripped == "")
            continue
        heading = m.group(1).strip()
        cls = classify_heading(heading)
        prev_blank = True  # after a heading the next content is a new para

        if cls.kind == "problem_zone":
            push(cur)
            hit_problem_zone = True
            continue
        if cls.kind == "noise":
            cur.text_md += line + "\n"
            continue
        if cls.kind == "solution":
            # Solution heading is part of the current example body when we
            # are inside one; otherwise upgrade the current fragment to
            # example-kind (some books put a stand-alone "# Solution:"
            # after an unnumbered example heading).
            if cur.kind != "example":
                push(cur)
                cur = new_frag("example", heading, example_title="(Solution)")
            cur.text_md += line + "\n"
            continue

        # section / subsection / example: flush current, then open a new one.
        push(cur)
        if cls.kind == "section":
            current_anchor = cls.anchor
            section_stack = [(1, cls.title)]
            cur = new_frag("exposition", cls.title)
        elif cls.kind == "subsection":
            section_stack = [t for t in section_stack if t[0] < 2] + [(2, cls.title)]
            cur = new_frag("exposition", cls.title)
        elif cls.kind == "example":
            cur = new_frag("example", cls.title, cls.example_title)

    if not hit_problem_zone:
        push(cur)

    # Subdivide oversized fragments, assign running IDs, and emit.
    running_idx = 0
    for frag in frags:
        for sub in _subdivide(frag):
            record = sub.finalize(running_idx)
            running_idx += 1
            yield record


# ---------------------------------------------------------------------------
# Fragment overflow splitter
# ---------------------------------------------------------------------------

def _subdivide(frag: Fragment) -> list[Fragment]:
    text = frag.text_md.strip()
    if len(text) <= MAX_CHARS:
        return [frag]
    # split on double-newline paragraph boundaries with overlap
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = (buf + "\n\n" + p).strip() if buf else p
        if len(candidate) <= TARGET_CHARS:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        # if a single paragraph is itself huge, hard-split on sentences
        if len(p) > MAX_CHARS:
            for hunk in _hard_wrap(p, TARGET_CHARS):
                chunks.append(hunk)
            buf = ""
        else:
            buf = p
    if buf:
        chunks.append(buf)

    # add overlaps: prepend the last OVERLAP_CHARS of the previous chunk
    with_overlap: list[str] = []
    for i, c in enumerate(chunks):
        if i == 0:
            with_overlap.append(c)
        else:
            prev = chunks[i - 1]
            overlap = prev[-OVERLAP_CHARS:] if len(prev) > OVERLAP_CHARS else prev
            with_overlap.append(overlap + "\n\n" + c)

    out: list[Fragment] = []
    for c in with_overlap:
        clone = Fragment(
            book=frag.book, book_display=frag.book_display, domain=frag.domain,
            is_solution_manual=frag.is_solution_manual,
            chapter=frag.chapter, chapter_title=frag.chapter_title,
            section_path=list(frag.section_path),
            heading=frag.heading, anchor=frag.anchor,
            kind=frag.kind, example_title=frag.example_title,
            text_md=c,
        )
        out.append(clone)
    return out


def _hard_wrap(text: str, target: int) -> list[str]:
    """Sentence-ish split for single paragraphs bigger than MAX_CHARS."""
    sentences = re.split(r"(?<=[\.\!\?])\s+", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + 1 + len(s) <= target:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks or [text]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_chapters(
    corpus: Path,
    include_solution_manuals: bool,
    only_books: Optional[set[str]],
) -> Iterator[tuple[Path, str, dict[str, str], bool]]:
    for book_folder in sorted(corpus.iterdir()):
        if not book_folder.is_dir():
            continue
        resolved = resolve_book(book_folder.name)
        if resolved is None:
            continue
        base_key, meta, is_sol = resolved
        if is_sol and not include_solution_manuals:
            continue
        if only_books is not None and base_key not in only_books:
            continue
        for chapter_folder in sorted(book_folder.iterdir()):
            if not chapter_folder.is_dir():
                continue
            full = chapter_folder / "full.md"
            if not full.exists():
                continue
            yield full, base_key, meta, is_sol


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument(
        "--exclude-solution-manuals", action="store_true",
        help="Skip *_solution_manual folders (default: include them, tagged kind=example).",
    )
    ap.add_argument(
        "--books", nargs="*", default=None,
        help="Restrict to these base book slugs (folder names without the "
             "_solution_manual suffix). Default: all known books.",
    )
    ap.add_argument(
        "--limit-chapters", type=int, default=None,
        help="For quick tests: process only the first N chapters total.",
    )
    args = ap.parse_args()

    if not args.corpus.exists():
        sys.exit(f"corpus not found: {args.corpus}")

    only_books = set(args.books) if args.books else None
    if only_books is not None:
        unknown = only_books - set(BOOKS)
        if unknown:
            sys.exit(f"unknown book slug(s): {sorted(unknown)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_suffix(args.out.suffix + ".tmp")

    stats_per_book: dict[str, dict[str, int]] = {}
    n_frags_total = 0
    n_chars_total = 0
    n_chapters = 0

    with tmp_path.open("w", encoding="utf-8") as out_f:
        for full_md, book_key, meta, is_sol in iter_chapters(
            args.corpus,
            include_solution_manuals=not args.exclude_solution_manuals,
            only_books=only_books,
        ):
            if args.limit_chapters is not None and n_chapters >= args.limit_chapters:
                break
            n_chapters += 1
            book_stats = stats_per_book.setdefault(
                book_key,
                {"chapters": 0, "fragments": 0, "exposition": 0, "example": 0,
                 "chars": 0, "solution_manual_chapters": 0},
            )
            book_stats["chapters"] += 1
            if is_sol:
                book_stats["solution_manual_chapters"] += 1
            for frag in parse_chapter(full_md, book_key, meta, is_sol):
                out_f.write(json.dumps(frag, ensure_ascii=False))
                out_f.write("\n")
                book_stats["fragments"] += 1
                book_stats[frag["kind"]] = book_stats.get(frag["kind"], 0) + 1
                book_stats["chars"] += frag["n_chars"]
                n_frags_total += 1
                n_chars_total += frag["n_chars"]

    tmp_path.replace(args.out)

    print(f"[build_textbook_fragments] wrote {n_frags_total:,} fragments "
          f"({n_chars_total:,} chars across {n_chapters} chapters) -> {args.out}")
    print("[build_textbook_fragments] per-book stats:")
    for k in sorted(stats_per_book):
        s = stats_per_book[k]
        print(f"  {k:40s}  chapters={s['chapters']:3d}  "
              f"frags={s['fragments']:5d}  "
              f"exposition={s.get('exposition',0):5d}  "
              f"example={s.get('example',0):5d}  "
              f"chars={s['chars']:9,d}"
              + (f"  [+SM: {s['solution_manual_chapters']}]" if s['solution_manual_chapters'] else ""))


if __name__ == "__main__":
    main()
