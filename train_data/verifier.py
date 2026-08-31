"""Self-contained, rule-based verifier for physics / math answer judging.

This module is the RL-friendly counterpart of :mod:`eval.judge`. It
contains *only* the rule-based judgment logic (no LLM judge, no
network, no class state) and exposes two free-standing entry points
that an external trainer (verl, TRL, OpenRLHF, ...) can call directly
as a reward function:

* :func:`compute_reward(pred, gold, precision=None)`
    Mirrors ``Judger.is_equal``: iterates over the six atomic answer
    types ``{MC, TF, NV, IN, EX, EQ}`` (TUP intentionally removed) and
    returns ``1.0`` as soon as **any** of them reports a match,
    otherwise ``0.0``. ``pred`` and ``gold`` are expected to already
    be the cleaned answer strings (i.e. what the per-type judges in
    ``judge.py`` get after ``extract_ans`` / ``normalize_answer``).

* :func:`compute_reward_by_answer_type(pred, gold, answer_type,
    type_sequence=None, precision=1e-8)`
    Mirrors ``Judger.auto_judge``: runs the per-type, schema-aware
    pipeline (boxed extraction + normalization + ``\\pm`` expansion +
    per-type comparator) and supports the composite types ``TUP``
    (position-strict, per-position inner type) and ``ALT`` (any pred
    box matches any gold box under the gold's declared inner type)
    via ``type_sequence``.

All helpers (boxed extractor, ``_strip_string`` normaliser, LaTeX
parse lock, ``\\pm`` expansion, ``remove_const``, assignment match,
per-type judges, etc.) live in this single file so it can be dropped
into a training repo without copying the rest of the study_bench
tree. The original module-level constants (``PHY_CONST``,
``NO_TRAILING_STRS``, ...) and behaviours are preserved verbatim so a
prediction that scored ``True`` under ``Judger`` continues to score
``1.0`` here.
"""

from __future__ import annotations

import atexit
import math
import multiprocessing as _mp
import os
import queue as _stdqueue
import re
import struct
import subprocess
import sys
import threading
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import sympy as sp
from sympy import Eq, Mul, N, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex as _raw_parse_latex

try:
    from latex2sympy2_extended import latex2sympy as _latex2sympy  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency in standalone use.
    _latex2sympy = None


# ===========================================================================
# Module constants (verbatim from eval/judge_utils.py)
# ===========================================================================

STRIP_STRS = [
    ":", ".", "/", ",", "#", "?", "$", '"', "'",
    # "ки" is the delimeter for Math-Shepherd
    "к", "и",
    # LaTeX
    "\\(", "\\)", "\\[", "\\]",
]

PHY_CONST = [
    "G", "N_A", "N_{A}", "R", "V_m", "V_{m}", "e",
    "m_e", "m_{e}", "m_p", "m_{p}", "m_n", "m_{n}",
    "\\varepsilon_0", "\\varepsilon_{0}",
    "\\epsilon_0", "\\epsilon_{0}",
    "\\mu_0", "\\mu_{0}", "\\mu_e", "\\mu_{e}", "\\mu_p", "\\mu_{p}",
    "a_0", "a_{0}",
    "\\mu_B", "\\mu_{B}", "\\mu_N", "\\mu_{N}",
    "\\hbar", "h",
    "\\alpha",
    "R_\\infty", "R_{\\infty}",
]

NO_TRAILING_STRS = ["(", "[", "{", "\\"] + STRIP_STRS

# Atomic, single-element inner types (composite TUP / ALT not included).
# Used by ``compute_reward_by_answer_type`` to validate ``type_sequence``.
_INNER_TYPES = ("NV", "EX", "EQ", "IN", "MC", "TF", "QL")

# ``compute_reward`` enumerates these six atomic types in order. This is
# the same list ``Judger.is_equal`` walked, with ``TUP`` removed (TUP was
# unreachable there anyway because ``judgment_methods`` had no ``TUP``
# entry, but we explicitly drop it as requested).
_COMPUTE_REWARD_TYPE_ORDER = ("MC", "TF", "NV", "IN", "EX", "EQ")


# ===========================================================================
# Thread-safe LaTeX parser
# ===========================================================================

# Hydra 1.3 requires antlr4-python3-runtime 4.9.*, while SymPy 1.14's
# ANTLR-backed LaTeX parser requires 4.11.*.  When ``ACE_ANTLR411_DIR`` is
# configured, parse requests run in a worker subprocess whose ``sys.path``
# puts a staged 4.11 runtime (including its dist-info metadata) first.  The
# parent process can therefore keep the runtime required by Hydra/omegaconf.
# Parsed expressions cross the process boundary as ``srepr`` and are rebuilt
# with evaluate=False wrappers so scientific-notation structure is preserved.
# Without the directory, parsing remains inline for standalone environments.
# When ``ACE_ANTLR411_DIR`` is set, its worker is preferred; the optional
# ``latex2sympy2_extended`` parser is the fallback.  Without that directory,
# ``latex2sympy2_extended`` is preferred over SymPy's native parser.  The lock
# also serializes access to the worker pipe.
_LATEX_LOCK = threading.Lock()

_ANTLR411_DIR = os.environ.get("ACE_ANTLR411_DIR", "/home/test/test1708/StudyBench/train_data/antlr411").strip()
_LATEX_WORKER_FLAG = "--latex-worker"


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly ``n`` bytes from a binary stream, or ``None`` on EOF."""
    chunks: List[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _build_unsrepr_ns() -> dict:
    """Build the restricted namespace used to restore parser srepr output."""
    ns = dict(sp.__dict__)
    for _name in ("Add", "Mul", "Pow"):
        _cls = getattr(sp, _name)
        ns[_name] = (lambda c: (lambda *a, **k: c(*a, evaluate=False)))(_cls)
    return ns


_UNSREPR_NS = _build_unsrepr_ns()


def _unsrepr(text: str):
    """Rebuild a SymPy expression from srepr while preserving its structure."""
    return eval(text, {"__builtins__": {}}, _UNSREPR_NS)  # noqa: S307


def _pdeathsig_preexec() -> None:  # pragma: no cover - Linux child-side only
    """Kill a parser worker if its parent process dies."""
    try:
        import ctypes
        import signal as _sig

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, _sig.SIGKILL)
    except Exception:
        pass


_PREEXEC = _pdeathsig_preexec if sys.platform.startswith("linux") else None


class _LatexWorker:
    """Long-lived client for parsing LaTeX with the staged ANTLR runtime."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), _LATEX_WORKER_FLAG],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env=os.environ.copy(),
            preexec_fn=_PREEXEC,
        )

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def _exchange(self, s: str):
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        data = s.encode("utf-8")
        self._proc.stdin.write(struct.pack(">I", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()
        status = _read_exact(self._proc.stdout, 1)
        if status is None:
            raise EOFError("latex worker closed the pipe")
        header = _read_exact(self._proc.stdout, 4)
        if header is None:
            raise EOFError("latex worker truncated response header")
        (n,) = struct.unpack(">I", header)
        payload = _read_exact(self._proc.stdout, n) if n else b""
        if payload is None:
            raise EOFError("latex worker truncated response payload")
        if status == b"O":
            return _unsrepr(payload.decode("utf-8"))
        raise ValueError(
            "latex worker failed to parse: "
            + payload.decode("utf-8", "replace")
        )

    def parse(self, s: str):
        """Parse once, respawning and retrying if the worker pipe breaks."""
        last_exc: Optional[BaseException] = None
        for _attempt in range(2):
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            try:
                return self._exchange(s)
            except (BrokenPipeError, EOFError, OSError) as exc:
                last_exc = exc
                self._kill()
        raise RuntimeError(f"latex worker unavailable: {last_exc!r}")

    def shutdown(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            if self._proc.poll() is None:
                self._proc.wait(timeout=2)
        except Exception:
            pass
        self._kill()


_LATEX_WORKER = _LatexWorker()
atexit.register(_LATEX_WORKER.shutdown)


def parse_latex(s):
    """Thread-safe LaTeX parser with optional ANTLR-version isolation."""
    with _LATEX_LOCK:
        # Prefer the explicitly selected ANTLR runtime when configured.
        # If its worker cannot parse the expression or is unavailable, use
        # latex2sympy2_extended before falling back to SymPy's native parser.
        if _ANTLR411_DIR:
            try:
                return _LATEX_WORKER.parse(s)
            except Exception:
                pass
        if _latex2sympy is not None:
            try:
                return _latex2sympy(s)
            except Exception:
                pass
        return _raw_parse_latex(s)


def _run_latex_worker() -> None:  # pragma: no cover - exercised in child
    """Serve framed LaTeX parse requests using the staged ANTLR runtime."""
    antlr_dir = os.environ.get("ACE_ANTLR411_DIR", "").strip()
    if antlr_dir:
        sys.path.insert(0, antlr_dir)

    raw_out_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)
    out = os.fdopen(raw_out_fd, "wb", buffering=0)
    inp = sys.stdin.buffer
    while True:
        header = _read_exact(inp, 4)
        if header is None:
            break
        (n,) = struct.unpack(">I", header)
        body = _read_exact(inp, n) if n else b""
        if body is None:
            break
        try:
            expr = _raw_parse_latex(body.decode("utf-8", "replace"))
            payload = sp.srepr(expr).encode("utf-8")
            status = b"O"
        except Exception as exc:
            payload = repr(exc).encode("utf-8", "replace")
            status = b"E"
        try:
            out.write(status)
            out.write(struct.pack(">I", len(payload)))
            out.write(payload)
            out.flush()
        except (BrokenPipeError, OSError):
            break


if __name__ == "__main__" and _LATEX_WORKER_FLAG in sys.argv:
    _run_latex_worker()
    sys.exit(0)


# ===========================================================================
# String normalization (verbatim from eval/math_equivalence.py::_strip_string)
# ===========================================================================

def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _contains_chinese(d) -> bool:
    def is_chinese_char(ch):
        return "\u4e00" <= ch <= "\u9fff"

    def check(value):
        if isinstance(value, str):
            return any(is_chinese_char(ch) for ch in value)
        if isinstance(value, dict):
            return any(check(v) for v in value.values())
        if isinstance(value, list):
            return any(check(item) for item in value)
        return False

    return check(d)


# ===========================================================================
# Boxed extraction (verbatim from eval/judge_utils.py)
# ===========================================================================

def _walk_boxed_close(string: str, idx: int) -> Optional[int]:
    """Find the index of the closing ``}`` that matches the ``{`` opened
    by ``\\boxed{`` / ``\\fbox{`` starting at ``idx``.

    Brace counter is LaTeX-aware: backslash-escaped braces (``\\{``,
    ``\\}``) and any other two-char escape (``\\\\`` etc.) are skipped.
    """
    n = len(string)
    i = idx
    depth = 0
    while i < n:
        ch = string[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    right = _walk_boxed_close(string, idx)
    return string[idx: right + 1] if right is not None else None


def _all_boxed_only_strings(string: str) -> List[str]:
    """Return every top-level ``\\boxed{...}`` / ``\\fbox{...}`` substring."""
    if not isinstance(string, str) or not string:
        return []
    results: List[str] = []
    cursor = 0
    n = len(string)
    while cursor < n:
        b = string.find("\\boxed", cursor)
        f = string.find("\\fbox", cursor)
        candidates = [x for x in (b, f) if x >= 0]
        if not candidates:
            break
        idx = min(candidates)
        right = _walk_boxed_close(string, idx)
        if right is None:
            break
        results.append(string[idx: right + 1])
        cursor = right + 1
    return results


def _remove_boxed(s: Optional[str]) -> Optional[str]:
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left): -1]
    except Exception:
        return None


# ===========================================================================
# Answer extraction / normalization (from Judger.normalize_answer,
# extract_boxed_answer, extract_ans, extract_explicit_ans, ...)
# ===========================================================================

def normalize_answer(final_answer: str) -> str:
    special_signal_map = {
        "\\left": "",
        "\\right": "",
        "\u2236": ":",
        "\uff0c": ",",
        "$": "",
        "\\approx": "=",
        "\\simeq": "=",
        "\\sim": "=",
        "^\\prime": "'",
        "^{\\prime}": "'",
        "^\\circ": "",
        "%": "",
    }
    for signal, repl in special_signal_map.items():
        final_answer = final_answer.replace(signal, repl)
    final_answer = re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", "\\1", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.strip()
    final_answer = final_answer.strip("$")
    final_answer = final_answer.strip()
    final_answer = final_answer.replace("\\,", " ")
    final_answer = _strip_string(final_answer)
    return final_answer.rstrip("\\")


def _clean_trailing(s: str) -> str:
    """Drop trailing punctuation (matches Judger.clean_trailing)."""
    s = str(s).strip()
    while s != "" and s[-1] in NO_TRAILING_STRS:
        s = s[:-1].strip()
    return s


def extract_boxed_answer(text: str) -> str:
    """Extract the *last* ``\\boxed{}`` content, normalised."""
    content = _remove_boxed(_last_boxed_only_string(text))
    if content is None:
        match = re.search(r"\\boxed{", text)
        if match:
            start_index = match.end()
            end_index = start_index
            stack = 1
            while stack > 0 and end_index < len(text):
                if text[end_index] == "{":
                    stack += 1
                elif text[end_index] == "}":
                    stack -= 1
                end_index += 1
            if stack == 0:
                content = text[start_index: end_index - 1]
                if not content:
                    return text
                return normalize_answer(content)
    if content is None:
        return text
    return normalize_answer(content)


def _extract_explicit_ans(resp_str: str, strict_extract: bool = True) -> Optional[str]:
    resp_str = _clean_trailing(resp_str)
    if "herefore" in resp_str:
        resp_str = resp_str.split("herefore")[-1].strip()

    if "oxed{" in resp_str:
        return extract_boxed_answer(resp_str)

    resp = resp_str
    if "is the ans" in resp:
        resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[-1].strip()
    elif "is our ans" in resp:
        resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[-1].strip()
    elif "answer is" in resp:
        resp = resp.split("answer is")[-1].strip()
    elif "answer:" in resp:
        resp = resp.split("answer:")[-1].strip()
    elif "answer :" in resp:
        resp = resp.split("answer :")[-1].strip()
    else:
        return None

    if resp.startswith("$") and resp.endswith("$"):
        resp = resp[1:-1]
    return resp


def extract_ans(resp_str: str, strict_extract: bool = True) -> str:
    """Extract a single answer segment from a free-form response."""
    ans = _extract_explicit_ans(resp_str, strict_extract=strict_extract)
    if ans is not None:
        return ans
    if not strict_extract:
        matches = re.findall(
            r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)", resp_str, re.DOTALL
        )
        if matches:
            return matches[-1]
        matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
        if matches:
            return matches[-1]
    return ""


def _split_by_comma(expr: str) -> List[str]:
    """Bracket-aware comma split (matches Judger.split_by_comma)."""
    in_bracket_num = 0
    splitted_expr: List[str] = []
    start_idx = 0
    for i, char in enumerate(expr):
        if char in ["(", "["]:
            in_bracket_num += 1
        elif char in [")", "]"]:
            in_bracket_num -= 1
        elif char == "," and in_bracket_num == 0:
            splitted_expr.append(expr[start_idx:i].strip())
            start_idx = i + 1
    if start_idx < len(expr):
        splitted_expr.append(expr[start_idx:].strip())
    if splitted_expr:
        splitted_expr = [item.strip("$").strip() for item in splitted_expr]
    return splitted_expr


def _extract_all_boxed_answers(text: str) -> List[str]:
    """Return normalized contents of every top-level ``\\boxed{}``."""
    contents: List[str] = []
    for raw in _all_boxed_only_strings(text):
        inner = _remove_boxed(raw)
        if inner is None or not inner:
            continue
        contents.append(normalize_answer(inner))
    return contents


def _extract_ans_as_list(resp_str: str, multi_boxed: str = "all") -> List[str]:
    """Flat list of atomic-token answer strings."""
    boxed_list = _extract_all_boxed_answers(resp_str)
    if boxed_list:
        if multi_boxed == "last":
            boxed_list = [boxed_list[-1]]
        flat: List[str] = []
        for content in boxed_list:
            for part in _split_by_comma(content):
                if part:
                    flat.append(part)
        return flat

    single = extract_ans(resp_str)
    if not single:
        return []
    return [p for p in _split_by_comma(single) if p]


def _normalized_raw_boxed_list(text: str) -> List[str]:
    """Raw boxed contents (no comma splitting), normalised."""
    boxed = _extract_all_boxed_answers(text)
    return [normalize_answer(b) if isinstance(b, str) else b for b in boxed]


def _trans_plus_minus_sign(expr_list: List[str]) -> List[str]:
    new_expr_list: List[str] = []
    for expr in expr_list:
        if isinstance(expr, str) and "\\pm" in expr:
            new_expr_list.append(expr.replace("\\pm", "+"))
            new_expr_list.append(expr.replace("\\pm", "-"))
        else:
            new_expr_list.append(expr)
    return new_expr_list


def _normalize_type_sequence(type_sequence) -> List[str]:
    """Coerce ``type_sequence`` into a list of upper-case inner-type tags."""
    if type_sequence is None:
        return []
    if isinstance(type_sequence, str):
        tokens = [t.strip() for t in type_sequence.split(",")]
    else:
        try:
            tokens = [str(t).strip() for t in type_sequence]
        except TypeError:
            return []
    return [t.upper() for t in tokens if t]


# ===========================================================================
# Sympy / constant helpers
# ===========================================================================

_PI_SYM = None
_PI_SYM_LOCK = threading.Lock()


def _get_pi_sym():
    global _PI_SYM
    if _PI_SYM is None:
        with _PI_SYM_LOCK:
            if _PI_SYM is None:
                _PI_SYM = parse_latex("\\pi")
    return _PI_SYM


def _sympy_sub_pi(expression_sympy):
    return expression_sympy.subs(_get_pi_sym(), math.pi)


def remove_const(input_string: str) -> str:
    for constant in PHY_CONST:
        input_string = input_string.replace(constant, "")
    if input_string.strip() == "":
        input_string = "1"
    return re.sub(r"\s+", " ", input_string).strip()


# ===========================================================================
# Per-type judges (verbatim from Judger.judge_*)
# ===========================================================================

def judge_MC(pred: str, gold: str, precision: Optional[float] = None) -> bool:
    del precision
    common_answer = [chr(i) for i in range(65, 91)]
    if pred == gold:
        return True
    if pred.startswith("[") and pred.endswith("]"):
        pred = pred.strip("[]")
    if pred and pred[0] in common_answer and (len(pred) > 1 and pred[1] == ":"):
        return pred[0] == gold
    return False


def judge_TF(pred: str, gold: str, precision: Optional[float] = None) -> bool:
    del precision
    if _contains_chinese(pred):
        if pred in ["\u662f", "\u5bf9", "\u6b63\u786e", "\u80fd"]:
            pred = "TRUE"
        elif pred in ["\u5426", "\u9519", "\u9519\u8bef", "\u4e0d\u80fd"]:
            pred = "FALSE"
    else:
        pred = pred.upper()
    answers = ["TRUE", "FALSE", "T", "F", "YES", "NO", "Y", "N"]
    gold = gold.upper()
    assert gold in answers
    if pred not in answers:
        return False
    if gold in ["TRUE", "YES", "T", "Y"]:
        gold = "TRUE"
    if gold in ["FALSE", "NO", "F", "N"]:
        gold = "FALSE"
    if pred in ["TRUE", "YES", "T", "Y"]:
        pred = "TRUE"
    if pred in ["FALSE", "NO", "F", "N"]:
        pred = "FALSE"
    return pred == gold


def judge_qualitative(pred: str, gold: str, precision: Optional[float] = None) -> bool:
    """Case-insensitive, whitespace-collapsed string equality for QL."""
    del precision
    if not isinstance(pred, str) or not isinstance(gold, str):
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return norm(pred) == norm(gold)


def judge_single_numerical_value(
    pred: str, gold: str, precision: Optional[float] = None,
) -> bool:
    precision = _DEFAULT_PRECISION if precision is None else precision

    def is_scientific_notation(expr):
        return (
            isinstance(expr, Mul)
            and isinstance(expr.args[1], Pow)
            and expr.args[1].args[0] == 10
        )

    def to_scientific_notation_latex(num):
        num_sci = f"{num:.2e}"
        base, exponent = num_sci.split("e")
        exponent = int(exponent)
        return f"{base}\\times 10^{{{exponent}}}"

    if pred == gold:
        return True
    try:
        pred_value = float(pred)
        gold_value = float(gold)
        gold_decimal_places = (
            len(str(gold_value).split(".")[1]) if "." in str(gold_value) else 0
        )
        pred_value = round(pred_value, gold_decimal_places)
        if abs((pred_value - gold_value) / gold_value) <= precision * 1.01:
            return True
        sgold = to_scientific_notation_latex(float(gold))
        exp_gold = parse_latex(sgold)
        spred = to_scientific_notation_latex(float(pred))
        exp_pred = parse_latex(spred)
        base_pred, exponent_pred = N(exp_pred.args[0]), N(exp_pred.args[1].args[1])
        base_gold, exponent_gold = N(exp_gold.args[0]), N(exp_gold.args[1].args[1])
        gold_decimal_places = (
            len(str(base_gold).split(".")[1]) if "." in str(base_gold) else 0
        )
        base_pred = round(base_pred, gold_decimal_places)
        if abs(base_pred - base_gold) <= 0.1 * 1.01:
            return True
    except Exception:
        pass

    try:
        exp_pred = parse_latex(pred)
        exp_gold = parse_latex(gold)
        n_pred = N(exp_pred)
        n_gold = N(exp_gold)
        gold_decimal_places = (
            len(str(n_gold).split(".")[1]) if "." in str(n_gold) else 0
        )
        n_pred = round(n_pred, gold_decimal_places)
        if abs((n_pred - n_gold) / n_gold) <= precision * 1.01:
            return True

        if is_scientific_notation(exp_pred) != is_scientific_notation(exp_gold):
            if is_scientific_notation(exp_pred):
                gold = to_scientific_notation_latex(float(gold))
                exp_gold = parse_latex(gold)
            else:
                pred = to_scientific_notation_latex(float(pred))
                exp_pred = parse_latex(pred)

        if is_scientific_notation(exp_pred) and is_scientific_notation(exp_gold):
            base_pred, exponent_pred = N(exp_pred.args[0]), N(exp_pred.args[1].args[1])
            base_gold, exponent_gold = N(exp_gold.args[0]), N(exp_gold.args[1].args[1])
            gold_decimal_places = (
                len(str(base_gold).split(".")[1]) if "." in str(base_gold) else 0
            )
            base_pred = round(base_pred, gold_decimal_places)
            if abs(base_pred - base_gold) <= 0.1 * 1.01:
                return True
        else:
            if N(exp_pred) == N(exp_gold):
                return True
    except Exception:
        pass

    return False


def judge_expression(
    pred: str, gold: str, precision: Optional[float] = None,
) -> bool:
    precision = _DEFAULT_PRECISION if precision is None else precision

    def extract_expression(expression: str) -> str:
        if "=" in expression:
            expression = expression.split("=")[1]
        return expression.strip()

    exp1 = extract_expression(pred)
    exp2 = extract_expression(gold)
    expr1_sym = sympify(parse_latex(exp1))
    expr2_sym = sympify(parse_latex(exp2))
    if expr1_sym == expr2_sym:
        return True
    expr1_sym = _sympy_sub_pi(expr1_sym)
    expr2_sym = _sympy_sub_pi(expr2_sym)
    if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
        not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)
    ):
        return False
    if not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
        try:
            return judge_single_numerical_value(
                expr1_sym, expr2_sym, precision=precision,
            )
        except Exception:
            return False
    try:
        simplified_expr = simplify(expr1_sym - expr2_sym)
        num_value = simplified_expr.evalf()
        flag = abs(num_value) < precision
        try:
            flag = bool(flag)
        except Exception:
            return False
        return flag
    except Exception:
        return False


def judge_interval(
    pred: str, gold: str, precision: Optional[float] = None,
) -> bool:
    precision = _DEFAULT_PRECISION if precision is None else precision

    def parse_interval(interval: str):
        parsed = []
        for part in interval.split("\\cup"):
            bounds, interval_type = part.strip(), ""
            interval_type += "open_left" if bounds.startswith("(") else "closed_left"
            interval_type += "_open_right" if bounds.endswith(")") else "_closed_right"
            numbers = bounds.strip("()[]").split(",")
            parsed.append((numbers, interval_type))
        return parsed

    def compare_intervals(intervals1, intervals2) -> bool:
        list1 = [(tuple(item[0]), item[1]) for item in intervals1]
        list2 = [(tuple(item[0]), item[1]) for item in intervals2]
        if len(list1) != len(list2):
            return False
        for interval1 in list1:
            interval_numbers1, interval_type1 = interval1
            matched = False
            for interval2 in list2:
                interval_numbers2, interval_type2 = interval2
                if interval_type1 == interval_type2:
                    bounds_match = judge_expression(
                        interval_numbers1[0], interval_numbers2[0], precision=precision,
                    ) and judge_expression(
                        interval_numbers1[1], interval_numbers2[1], precision=precision,
                    )
                    if bounds_match:
                        matched = True
                        list2.remove(interval2)
                        break
            if not matched:
                return False
        return True

    return compare_intervals(parse_interval(pred), parse_interval(gold))


def judge_equation(
    pred: str, gold: str, precision: Optional[float] = None,
) -> bool:
    del precision  # equation comparison is symbolic; no tolerance needed.

    def simplify_equation(latex_eq: str):
        lhs, rhs = latex_eq.split("=")
        lhs_expr = parse_latex(lhs)
        rhs_expr = parse_latex(rhs)
        equation = Eq(lhs_expr, rhs_expr)
        return simplify(equation.lhs - equation.rhs)

    try:
        expr1_sym = simplify_equation(pred)
        expr2_sym = simplify_equation(gold)
        difference = simplify(expr1_sym - expr2_sym)
        if difference == 0:
            return True
        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)
        if (division_result_1.is_Integer and division_result_1 != 0) or (
            division_result_2.is_Integer and division_result_2 != 0
        ):
            return True
        return False
    except Exception:
        return False


# Per-type comparator registry. ``TUP`` / ``ALT`` are NOT in this map: they
# are composite types dispatched per-position via ``type_sequence`` inside
# :func:`compute_reward_by_answer_type`.
_JUDGMENT_METHODS: dict = {
    "MC": judge_MC,
    "TF": judge_TF,
    "NV": judge_single_numerical_value,
    "IN": judge_interval,
    "EX": judge_expression,
    "EQ": judge_equation,
    "QL": judge_qualitative,
}

# Fallback numeric tolerance when callers don't pass ``precision=``.
_DEFAULT_PRECISION = 1e-8


# ===========================================================================
# Pairing / matching helpers (from Judger._match_pair, _assignment_match,
# _judge_tuple_with_types, _judge_alt_with_types, _run_with_timeout)
# ===========================================================================

def _run_with_timeout(func: Callable, timeout_seconds: float = 5):
    """Backwards-compatible shim that just calls ``func()`` directly.

    The original implementation spawned a daemon ``threading.Thread`` and
    abandoned it on timeout, but ``sympy.simplify`` /
    ``sympy.parsing.latex.parse_latex`` hold the GIL the whole time, so
    abandoned threads kept burning CPU and eventually GIL-starved the
    main loop into apparent deadlock. The hard timeout now lives one
    layer up, in the module-level :class:`_JudgeRunner` subprocess: it
    runs the *entire* ``compute_reward`` / ``compute_reward_by_answer_type``
    sweep in a long-lived child process and ``terminate()``-s the child
    on timeout, which actually kills sympy mid-call.

    ``timeout_seconds`` is intentionally ignored.
    """
    del timeout_seconds  # see docstring
    return func()


def _pm_variants(s):
    """Expand ``\\pm`` to ``+`` and ``-`` variants for one element."""
    if not isinstance(s, str) or "\\pm" not in s:
        return [s]
    return [s.replace("\\pm", "+"), s.replace("\\pm", "-")]


def _match_pair(method: Callable, pred, gold, *, precision: Optional[float] = None) -> bool:
    """Single pairwise judgment with timeout + ``\\pm`` expansion +
    const-stripping fallback."""
    if isinstance(pred, str) and isinstance(gold, str) and pred == gold:
        return True

    pred_variants = _pm_variants(pred)
    gold_variants = _pm_variants(gold)

    try:
        def check():
            for p in pred_variants:
                for g in gold_variants:
                    if method(p, g, precision=precision):
                        return True
                    if method(
                        remove_const(p) if isinstance(p, str) else p,
                        remove_const(g) if isinstance(g, str) else g,
                        precision=precision,
                    ):
                        return True
            return False

        return _run_with_timeout(check, timeout_seconds=5)
    except Exception:
        return False


def _assignment_match(
    method: Callable,
    preds: Sequence,
    golds: Sequence,
    precisions: Sequence[float],
) -> bool:
    """True iff there's a one-to-one assignment with every (gold, pred) matching."""
    n = len(golds)
    if n == 0:
        return True
    if n > 8:
        used = [False] * n
        for i, (g, prec) in enumerate(zip(golds, precisions)):
            for j, p in enumerate(preds):
                if used[j]:
                    continue
                if _match_pair(method, p, g, precision=prec):
                    used[j] = True
                    break
            else:
                return False
        return True

    ok = [[False] * n for _ in range(n)]
    for i, (g, prec) in enumerate(zip(golds, precisions)):
        for j, p in enumerate(preds):
            if _match_pair(method, p, g, precision=prec):
                ok[i][j] = True

    used = [False] * n

    def assign(i: int) -> bool:
        if i == n:
            return True
        for j in range(n):
            if not used[j] and ok[i][j]:
                used[j] = True
                if assign(i + 1):
                    return True
                used[j] = False
        return False

    return assign(0)


def _judge_tuple_with_types(
    gold_boxes: Sequence[str],
    pred_boxes: Sequence[str],
    type_sequence: Sequence[str],
    precisions: Sequence[float],
) -> bool:
    """README-compliant TUP: position-strict per-type dispatch."""
    n = len(gold_boxes)
    if n == 0 or len(pred_boxes) != n:
        return False
    if len(type_sequence) != n:
        return False
    for i, inner in enumerate(type_sequence):
        method = _JUDGMENT_METHODS.get(inner)
        if method is None:
            return False
        prec = precisions[i] if i < len(precisions) else precisions[-1]
        if not _match_pair(method, pred_boxes[i], gold_boxes[i], precision=prec):
            return False
    return True


def _judge_alt_with_types(
    gold_boxes: Sequence[str],
    pred_boxes: Sequence[str],
    type_sequence: Sequence[str],
    precisions: Sequence[float],
) -> bool:
    """README-compliant ALT: any (pred, gold) pair under gold's declared type wins."""
    if not gold_boxes or not pred_boxes:
        return False
    if len(type_sequence) != len(gold_boxes):
        return False
    gold_methods = []
    for j, inner in enumerate(type_sequence):
        method = _JUDGMENT_METHODS.get(inner)
        if method is None:
            return False
        prec = precisions[j] if j < len(precisions) else precisions[-1]
        gold_methods.append((method, prec))
    for pred_item in pred_boxes:
        for j, gold_item in enumerate(gold_boxes):
            method, prec = gold_methods[j]
            if _match_pair(method, pred_item, gold_item, precision=prec):
                return True
    return False


# ===========================================================================
# Extraction pipeline (matches Judger.extract_normalized_lists)
# ===========================================================================

def extract_normalized_lists(
    pred: str, gold: str, answer_type: str,
) -> tuple[List[str], List[str]]:
    """Run the same extraction + normalization pipeline ``auto_judge`` uses.

    Returns ``(extracted_pred_list, normalized_gold_list)``. For ``TUP`` /
    ``ALT`` the lists are RAW boxed contents (no inner comma-splitting),
    so per-position type dispatch stays aligned. For all other types
    the legacy comma-split behaviour is preserved so ``\\boxed{a, b}``
    still collapses to two atoms.
    """
    answer_type = str(answer_type).upper() if answer_type is not None else ""

    if answer_type in ("TUP", "ALT"):
        gold_list = _normalized_raw_boxed_list(gold)
        pred_list = _normalized_raw_boxed_list(pred)
        if answer_type == "TUP" and gold_list:
            k = len(gold_list)
            if len(pred_list) > k:
                pred_list = pred_list[-k:]
        extracted_pred = pred_list
    else:
        gold_list = _extract_ans_as_list(gold, multi_boxed="all")
        pred_all = _extract_ans_as_list(pred, multi_boxed="all")
        k = len(gold_list)
        extracted_pred = pred_all[-k:] if k and len(pred_all) > k else pred_all
        gold_list = _trans_plus_minus_sign(gold_list)
        extracted_pred = _trans_plus_minus_sign(extracted_pred)

    extracted_pred = [
        normalize_answer(item) if isinstance(item, str) else item
        for item in extracted_pred
    ]
    gold_list = [
        normalize_answer(item) if isinstance(item, str) else item
        for item in gold_list
    ]
    return extracted_pred, gold_list


# ===========================================================================
# Inner reward implementations (called in-process; may hang on bad input)
# ===========================================================================

def _compute_reward_impl(
    pred: str,
    gold: str,
    precision: Optional[float] = None,
) -> float:
    """Type-agnostic reward.

    Mirrors :meth:`Judger.is_equal`. Iterates over the six atomic answer
    types ``("MC", "TF", "NV", "IN", "EX", "EQ")`` and returns ``1.0``
    as soon as **any** of them reports a match between ``pred`` and
    ``gold``; otherwise returns ``0.0``.

    Each per-type judge is also retried on the constant-stripped form
    (``remove_const``), which lets physics expressions that differ only
    in symbolic constants (e.g. ``\\hbar``) still match. A 5 s wall-clock
    timeout per type guards against pathological LaTeX inputs.

    ``pred`` and ``gold`` are expected to already be the cleaned answer
    strings (the kind that ``judge.py``'s per-type judges receive after
    ``extract_ans`` / ``normalize_answer``). If you start from raw model
    completions with ``\\boxed{}`` wrappers, run them through
    :func:`extract_normalized_lists` first or use
    :func:`compute_reward_by_answer_type` which handles extraction.

    Note: TUP / ALT / QL are intentionally NOT in this enumeration; use
    :func:`compute_reward_by_answer_type` when ``answer_type`` is known.
    """
    for answer_type in _COMPUTE_REWARD_TYPE_ORDER:
        try:
            method = _JUDGMENT_METHODS[answer_type]

            def check_equivalence(method=method):
                return method(pred, gold, precision=precision) or method(
                    remove_const(pred) if isinstance(pred, str) else pred,
                    remove_const(gold) if isinstance(gold, str) else gold,
                    precision=precision,
                )

            if _run_with_timeout(check_equivalence, timeout_seconds=5):
                return 1.0
        except Exception:
            pass
    return 0.0


def _compute_reward_by_answer_type_impl(
    pred: str,
    gold: str,
    answer_type: str,
    type_sequence: Union[str, Sequence[str], None] = None,
    precision: Union[float, Sequence[float]] = 1e-8,
) -> float:
    """Type-aware reward.

    Mirrors :meth:`Judger.auto_judge`. Runs the full schema-aware
    pipeline (boxed extraction + normalization + ``\\pm`` expansion +
    per-type comparator) and returns ``1.0`` on match, ``0.0``
    otherwise.

    Parameters
    ----------
    pred, gold:
        Raw model output / curator gold. May contain one or many
        ``\\boxed{}`` wrappers and surrounding chain-of-thought; this
        function applies the same extraction policy as ``auto_judge``.
    answer_type:
        One of ``{"MC", "TF", "NV", "IN", "EX", "EQ", "QL", "TUP",
        "ALT"}``. Case-insensitive.
    type_sequence:
        Required for ``TUP`` / ``ALT``: per-position inner types
        declared by the schema. Accepted as either a comma-separated
        string (``"NV,EX,QL"``) or a Python list. Ignored for the
        atomic types.
    precision:
        Numeric tolerance. Either a scalar (applied to every position)
        or a list of scalars whose length matches ``gold``'s box count.

    Semantics
    ---------
    * **TUP** — position-strict. Each pair ``(pred_i, gold_i)`` is
      compared with the comparator declared by ``type_sequence[i]``.
      Lengths must agree.
    * **ALT** — any pred box that matches any gold box under that
      gold's declared inner type wins.
    * **Atomic types** — dispatch to the type's comparator and run a
      one-to-one assignment search between extracted pred boxes and
      extracted gold boxes (so tolerance-based matches still succeed
      when several pairings are individually valid).

    No mutable global state is touched; safe for concurrent use from
    RL data workers.
    """
    precision_list: List[float] = (
        list(precision) if isinstance(precision, (list, tuple)) else [precision]
    )
    answer_type = str(answer_type).upper() if answer_type is not None else ""

    if answer_type in ("TUP", "ALT"):
        ts = _normalize_type_sequence(type_sequence)
        extracted_pred, gold_list = extract_normalized_lists(pred, gold, answer_type)
        if not gold_list or not extracted_pred:
            return 0.0
        if len(ts) != len(gold_list):
            return 0.0
        if len(precision_list) < len(gold_list):
            precision_list = (precision_list * len(gold_list))[: len(gold_list)]
        if answer_type == "TUP":
            if len(extracted_pred) != len(gold_list):
                return 0.0
            ok = _judge_tuple_with_types(
                gold_list, extracted_pred, ts, precision_list,
            )
            return 1.0 if ok else 0.0
        ok = _judge_alt_with_types(
            gold_list, extracted_pred, ts, precision_list,
        )
        return 1.0 if ok else 0.0

    method = _JUDGMENT_METHODS.get(answer_type)
    if method is None:
        return 0.0

    extracted_pred, gold_list = extract_normalized_lists(pred, gold, answer_type)
    if not gold_list or not extracted_pred:
        return 0.0
    if len(extracted_pred) != len(gold_list):
        return 0.0

    if len(precision_list) <= 1:
        precision_list = precision_list * len(gold_list)

    ok = _assignment_match(method, extracted_pred, gold_list, precision_list)
    return 1.0 if ok else 0.0


# ===========================================================================
# Subprocess-based hard timeout
# ===========================================================================
#
# Why this exists
# ---------------
# ``_compute_reward_impl`` and ``_compute_reward_by_answer_type_impl``
# ultimately call ``sympy.parsing.latex.parse_latex`` and
# ``sympy.simplify``. Both can fall into pathological inputs that loop
# forever; both hold the GIL the whole time. Python signal-based
# timeouts can't interrupt C-extension work, and any thread-based
# abandonment (the previous ``_run_with_timeout``) just leaks CPU- and
# GIL-holding zombies.
#
# Fix
# ---
# A single long-lived child process owns sympy. Each public call puts a
# request on a ``multiprocessing.Queue`` and ``.get(timeout=…)`` the
# answer. On timeout the child is ``terminate()``-ed (SIGTERM, then
# SIGKILL if needed) and respawned on the next call. SIGKILL stops
# sympy instantly no matter how deep it was in C. Leakage stays bounded
# to one child process.
#
# Config (env vars)
# -----------------
# * ``STUDYBENCH_REWARD_TIMEOUT_S``        per-call wall-clock budget,
#                                          default 30.0
# * ``STUDYBENCH_REWARD_DISABLE_SUBPROC=1`` opt out, run inline (used by
#                                          unit tests, debugging, and by
#                                          outer wrappers that already
#                                          provide their own subprocess
#                                          isolation, e.g.
#                                          ``verl/utils/reward_score/
#                                          studybench_physics.py``).

_DEFAULT_TIMEOUT_S = float(os.environ.get("STUDYBENCH_REWARD_TIMEOUT_S", "30"))


def _is_subproc_disabled() -> bool:
    """Read the disable flag on every call so callers can flip it after
    forking (e.g. an outer ``_JudgeRunner`` that wants to skip our
    subprocess hop). Cheap: it's just an os.environ dict lookup."""
    return os.environ.get("STUDYBENCH_REWARD_DISABLE_SUBPROC", "") == "1"


# Sentinel ops dispatched to the worker process.
_OP_REWARD = "compute_reward"
_OP_REWARD_BY_TYPE = "compute_reward_by_answer_type"

_RESULT_OK = "ok"
_RESULT_ERR = "err"


def _judge_subproc_main(req_q, resp_q) -> None:  # pragma: no cover (child)
    """Long-lived child process: pull (idx, op, args) requests off
    ``req_q``, dispatch to the inner ``_*_impl`` function, push back
    ``(idx, _RESULT_OK, score)`` or ``(idx, _RESULT_ERR, repr(exc))``.

    ``None`` on req_q means shutdown.
    """
    # Inside the child we MUST run the inner impls directly -- if we
    # called the public ``compute_reward(...)`` we would recursively
    # spawn yet another grandchild. Set the disable flag defensively in
    # case some downstream code in the impl ever inspects it.
    os.environ["STUDYBENCH_REWARD_DISABLE_SUBPROC"] = "1"
    while True:
        try:
            item = req_q.get()
        except (KeyboardInterrupt, EOFError):
            return
        if item is None:
            return
        idx, op, args, kwargs = item
        try:
            if op == _OP_REWARD:
                score = _compute_reward_impl(*args, **kwargs)
            elif op == _OP_REWARD_BY_TYPE:
                score = _compute_reward_by_answer_type_impl(*args, **kwargs)
            else:
                resp_q.put((idx, _RESULT_ERR, f"unknown op: {op!r}"))
                continue
            resp_q.put((idx, _RESULT_OK, float(score)))
        except Exception as exc:
            resp_q.put((idx, _RESULT_ERR, repr(exc)))


class _JudgeRunner:
    """Single-worker process pool with hard per-call timeout + respawn.

    Concurrent calls (e.g. multi-threaded reward workers in an RL
    trainer) are serialized through one child by ``self._lock``: sympy
    is CPU-bound and holds the GIL anyway, so multi-thread parallelism
    across the same child doesn't help. Serializing also makes the
    kill-on-timeout / respawn semantics simple (no risk of killing an
    in-flight unrelated request).
    """

    def __init__(self, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._proc: Optional[_mp.Process] = None
        self._req_q = None
        self._resp_q = None
        self._counter = 0
        try:
            self._ctx = _mp.get_context("fork")
        except ValueError:  # pragma: no cover - non-POSIX fallback
            self._ctx = _mp.get_context()

    def _start(self) -> None:
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_judge_subproc_main,
            args=(self._req_q, self._resp_q),
            daemon=True,
        )
        self._proc.start()

    def _kill_and_clear(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=2)
                if self._proc.is_alive():
                    self._proc.kill()
                    self._proc.join(timeout=2)
            except Exception:
                pass
        self._proc = None
        self._req_q = None
        self._resp_q = None

    def shutdown(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                try:
                    self._req_q.put(None)
                except Exception:
                    pass
                self._proc.join(timeout=2)
            self._kill_and_clear()

    def run(self, op: str, args: Tuple[Any, ...], kwargs: dict) -> float:
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                self._start()
            self._counter += 1
            idx = self._counter
            payload = (idx, op, args, kwargs)
            try:
                self._req_q.put(payload)
            except Exception:
                self._kill_and_clear()
                self._start()
                self._req_q.put(payload)
            try:
                rid, status, value = self._resp_q.get(timeout=self.timeout_s)
            except _stdqueue.Empty:
                # Snapshot a short representation of the gold for
                # post-mortem -- args[1] is `gold` for both ops.
                gold_preview = ""
                if len(args) >= 2 and isinstance(args[1], str):
                    gold_preview = args[1][:80]
                print(
                    f"[verifier] reward worker hard timeout "
                    f"({self.timeout_s:.0f}s) on op={op} -- killing "
                    f"pid={self._proc.pid if self._proc else None}, "
                    f"sample scored 0.0. gold[:80]={gold_preview!r}",
                    file=sys.stderr, flush=True,
                )
                self._kill_and_clear()
                return 0.0
            if rid != idx:
                # Stale response from a previously killed child slipped
                # through; treat as miss and move on.
                return 0.0
            if status == _RESULT_OK:
                return float(value)
            # _RESULT_ERR: verifier raised inside the child. Mirror inline
            # behaviour (return 0).
            return 0.0


_JUDGE: Optional[_JudgeRunner] = None
_JUDGE_INIT_LOCK = threading.Lock()


def _get_judge() -> _JudgeRunner:
    global _JUDGE
    if _JUDGE is not None:
        return _JUDGE
    with _JUDGE_INIT_LOCK:
        if _JUDGE is None:
            _JUDGE = _JudgeRunner(timeout_s=_DEFAULT_TIMEOUT_S)
            atexit.register(_JUDGE.shutdown)
    return _JUDGE


# ===========================================================================
# Public reward entry points  (subprocess-protected by default)
# ===========================================================================

def compute_reward(
    pred: str,
    gold: str,
    precision: Optional[float] = None,
) -> float:
    """Type-agnostic reward, hard-bounded by ``STUDYBENCH_REWARD_TIMEOUT_S``
    (default 30s). Runs ``_compute_reward_impl`` in a long-lived child
    process; pathological sympy inputs return 0.0 instead of hanging.

    Set ``STUDYBENCH_REWARD_DISABLE_SUBPROC=1`` to bypass the subprocess
    hop and call ``_compute_reward_impl`` inline (useful when an outer
    layer already provides isolation, or for unit tests).

    See :func:`_compute_reward_impl` for the underlying scoring rule.
    """
    if _is_subproc_disabled():
        return _compute_reward_impl(pred, gold, precision)
    return _get_judge().run(_OP_REWARD, (pred, gold), {"precision": precision})


def compute_reward_by_answer_type(
    pred: str,
    gold: str,
    answer_type: str,
    type_sequence: Union[str, Sequence[str], None] = None,
    precision: Union[float, Sequence[float]] = 1e-8,
) -> float:
    """Type-aware reward, hard-bounded by ``STUDYBENCH_REWARD_TIMEOUT_S``
    (default 30s). Same subprocess-isolation contract as
    :func:`compute_reward`. See :func:`_compute_reward_by_answer_type_impl`
    for the underlying scoring rule.
    """
    if _is_subproc_disabled():
        return _compute_reward_by_answer_type_impl(
            pred, gold, answer_type,
            type_sequence=type_sequence, precision=precision,
        )
    return _get_judge().run(
        _OP_REWARD_BY_TYPE,
        (pred, gold, answer_type),
        {"type_sequence": type_sequence, "precision": precision},
    )


__all__ = [
    "compute_reward",
    "compute_reward_by_answer_type",
    # Lower-level building blocks intentionally re-exported so users
    # who need them in a custom reward shaping pipeline don't have to
    # reach into ``_`` names.
    "normalize_answer",
    "extract_ans",
    "extract_boxed_answer",
    "extract_normalized_lists",
    "remove_const",
    "judge_MC",
    "judge_TF",
    "judge_qualitative",
    "judge_single_numerical_value",
    "judge_expression",
    "judge_interval",
    "judge_equation",
]


if __name__ == "__main__":
    # Quick smoke tests, mirroring the cases in eval/judge.py's __main__.
    test_cases = [
        ("\\boxed{\\pi}", "3.14159265358979", "EX"),
        ("\\boxed{4.39 \\times 10^{3}}", "\\boxed{439}", "NV"),
        ("\\boxed{False}", "\\boxed{No}", "TF"),
        ("\\boxed{n \\frac{\\mu^2}{3k_B T}}",
         "\\boxed{\\frac{n \\mu^{2} \\mu_0}{3 k_B T}}", "EX"),
        ("\\boxed{\\sigma(\\omega)=\\frac{ne^{2}\\tau}{m(1-i\\omega\\tau)}}",
         "\\boxed{\\sigma(\\omega)=\\frac{ne^2\\tau}{m}\\frac{1}{1-i\\omega\\tau}}",
         "EQ"),
    ]
    for pred, gold, atype in test_cases:
        r1 = compute_reward_by_answer_type(pred, gold, atype)
        r2 = compute_reward(extract_ans(pred), extract_ans(gold))
        print(f"[{atype}] type-aware={r1}  type-agnostic={r2}  | pred={pred!r}")
