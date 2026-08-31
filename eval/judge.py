import sympy as sp
from sympy import simplify, Eq, sympify, Pow, N, Mul
from sympy.parsing.latex import parse_latex as _raw_parse_latex
import itertools
import re
import math
import multiprocessing as mp
import threading
from judge_utils import *
from math_equivalence import _strip_string
import importlib
import os
import sys
import time
from pathlib import Path
import httpx
import requests
from openai import OpenAI


# Single source of truth for the default LLM judge model. The CLI in
# ``run_ioaa_benchmark`` and the factory in ``eval.make_default_judger`` both
# reference this constant so we never have three diverging defaults.
DEFAULT_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"

# Default location of the local DeepSeek-V4-Flash model directory. Used by
# the ``dsv4`` judge backend to import ``encoding/encoding_dsv4.py`` for
# raw-prompt encoding/decoding (the model ships no Jinja chat template, so
# the OpenAI ``/v1/chat/completions`` route is not available unless the
# vLLM server is launched with ``--chat-template``). Override with
# ``JUDGE_DSV4_MODEL_DIR`` or the ``--judge_dsv4_model_dir`` CLI flag.
DEFAULT_DSV4_MODEL_DIR = "/home/tsinghua/cyh/models/DeepSeek-V4-Flash"


# ANTLR's LaTeX grammar (used by ``sympy.parsing.latex.parse_latex``) is NOT
# thread-safe: concurrent calls from worker threads in ``eval_file`` can
# raise spurious exceptions or deadlock inside the parser. We serialize all
# entries through a single lock. The lock is held only for the parse call
# itself; downstream sympy work (``simplify``, ``Eq``, ``sympify``) operates
# on immutable objects and is safe to run concurrently.
_LATEX_LOCK = threading.Lock()


def _timeout_process_worker(func, conn):
    """Run one rule-based comparison in an isolated, killable process.

    SymPy can spend unbounded CPU time constructing/rounding enormous
    numbers. A thread timeout cannot stop that work, so the timeout boundary
    is deliberately a process boundary. The fork child may inherit the
    parent's parser lock state; reset our wrapper lock before invoking the
    callback so a sibling evaluation thread holding it cannot deadlock the
    child.
    """
    global _LATEX_LOCK
    _LATEX_LOCK = threading.Lock()
    try:
        conn.send(("ok", bool(func())))
    except BaseException as exc:  # report only a pickle-safe representation
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        conn.close()


def parse_latex(s):
    """Thread-safe wrapper around ``sympy.parsing.latex.parse_latex``."""
    with _LATEX_LOCK:
        return _raw_parse_latex(s)


# Resolve ``judge_prompt.txt`` relative to this module so callers don't have
# to ``chdir`` into ``eval/`` for ``aux_judge`` to find the template.
_JUDGE_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "judge_prompt.txt"
)


# ============================================================================
# Pluggable judge backends
#
# ``aux_judge`` used to be hard-wired to an ``openai.OpenAI`` client talking
# to the public OpenAI API (or whatever ``OPENAI_BASE_URL`` aliased). We now
# route the LLM call through one of two backends:
#
#   * ``OpenAIChatBackend`` -- the original behaviour. POSTs to
#     ``/v1/chat/completions`` via the official ``openai`` SDK. Works with
#     any chat-compatible endpoint (OpenAI proper, vLLM with a chat template,
#     Together, DeepInfra, ...).
#
#   * ``DsV4CompletionsBackend`` -- talks to a local DeepSeek-V4-Flash vLLM
#     server. The default ``serve_deepseek_v4_flash.sh`` only exposes
#     ``/v1/completions`` (no Jinja chat template), so this backend builds
#     the prompt itself via the official ``encoding_dsv4`` module shipped in
#     the model dir, POSTs the raw prompt, then parses
#     ``reasoning_content`` / ``content`` back out via
#     ``parse_message_from_completion_text``.
#
# Selection rules (descending precedence):
#   1. Explicit ``backend=`` arg passed to ``Judger(...)`` /
#      ``make_default_judger(...)``.
#   2. ``JUDGE_BACKEND`` env var, one of {auto, openai, dsv4}.
#   3. ``auto``: pick ``dsv4`` if ``judge_model`` starts with
#      ``deepseek-v4`` (case-insensitive), else ``openai``.
#
# All other knobs follow the same explicit-arg > env-var > default ladder:
#   ``base_url``       JUDGE_BASE_URL    -> OPENAI_BASE_URL -> SDK default
#   ``api_key``        JUDGE_API_KEY     -> OPENAI_API_KEY  -> empty
#   ``dsv4_model_dir`` JUDGE_DSV4_MODEL_DIR -> DEFAULT_DSV4_MODEL_DIR
#   ``thinking_mode``  JUDGE_THINKING_MODE  -> "thinking"
#   ``temperature``    JUDGE_TEMPERATURE    -> 0.0
#   ``max_tokens``     JUDGE_MAX_TOKENS     -> 8192 (chat) / 16384 (dsv4)
#   ``timeout``        JUDGE_TIMEOUT_S      -> 600
# ============================================================================


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Recognize OpenAI-compatible rate-limit errors, including wrapped ones."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = (
            getattr(current, "status_code", None)
            or getattr(current, "http_status", None)
            or getattr(response, "status_code", None)
        )
        text = f"{type(current).__name__}: {current}".lower()
        if (
            status_code == 429
            or "ratelimit" in type(current).__name__.lower()
            or "rate limit" in text
            or "too many requests" in text
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class JudgeBackend:
    """Abstract LLM-judge call interface."""

    name: str = "<base>"

    def complete_chat(
        self,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> str:
        raise NotImplementedError


class OpenAIChatBackend(JudgeBackend):
    """``/v1/chat/completions`` over the official ``openai`` SDK."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client_lock = threading.Lock()
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            if self._client is None:
                kwargs: dict = {"http_client": httpx.Client(verify=False)}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                self._client = OpenAI(**kwargs)
            return self._client

    def complete_chat(self, messages, *, temperature, max_tokens, timeout):
        # ``timeout`` is a per-call hint; the SDK already honours its own
        # default unless we pass ``with_options``. Keep behaviour uniform
        # by setting it per call.
        client = self._get_client().with_options(timeout=timeout)
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
        )
        return resp.choices[0].message.content or ""


class DsV4CompletionsBackend(JudgeBackend):
    """Local DeepSeek-V4-Flash judge via raw ``/v1/completions``.

    Builds the prompt from a chat-style ``messages`` list using the
    official ``encoding_dsv4`` encoder shipped in the model directory,
    posts the raw prompt to ``base_url + '/completions'``, then runs the
    output through ``parse_message_from_completion_text`` so the caller
    sees just the assistant's ``content`` (the reasoning trace under
    ``reasoning_content`` is dropped, since the judge prompt expects a
    very specific TRUE/FALSE format that lives in the post-think
    ``content``).
    """

    name = "dsv4"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        model_dir: str = DEFAULT_DSV4_MODEL_DIR,
        thinking_mode: str = "thinking",
    ):
        if thinking_mode not in ("thinking", "chat"):
            raise ValueError(
                f"thinking_mode must be 'thinking' or 'chat', got {thinking_mode!r}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.thinking_mode = thinking_mode
        self.model_dir = model_dir
        # Lazily import the encoder so importing ``judge`` works on
        # machines where the DSv4 weights aren't present.
        self._encoder_lock = threading.Lock()
        self._dsv4 = None
        # Connection-pooled session for thread-safe POSTs across the
        # eval worker pool.
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64, pool_maxsize=64
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        if self.api_key:
            self._session.headers.update(
                {"Authorization": f"Bearer {self.api_key}"}
            )

    def _get_encoder(self):
        if self._dsv4 is not None:
            return self._dsv4
        with self._encoder_lock:
            if self._dsv4 is None:
                encoding_dir = Path(self.model_dir) / "encoding"
                if not encoding_dir.is_dir():
                    raise RuntimeError(
                        f"DSv4 encoder not found at {encoding_dir!s}. "
                        f"Set JUDGE_DSV4_MODEL_DIR to your model directory "
                        f"(or pass --judge_dsv4_model_dir)."
                    )
                if str(encoding_dir) not in sys.path:
                    sys.path.insert(0, str(encoding_dir))
                self._dsv4 = importlib.import_module("encoding_dsv4")
            return self._dsv4

    def complete_chat(self, messages, *, temperature, max_tokens, timeout):
        dsv4 = self._get_encoder()
        prompt = dsv4.encode_messages(
            messages, thinking_mode=self.thinking_mode
        )
        resp = self._session.post(
            f"{self.base_url}/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                # DeepSeek officially recommends temperature=1.0, top_p=1.0,
                # but for a judge we want near-deterministic output. The
                # caller-provided ``temperature`` is honoured verbatim;
                # default 0.0 -> greedy decoding which vLLM supports.
                "temperature": temperature,
                "top_p": 1.0,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["text"]
        if not raw.endswith(dsv4.eos_token):
            raw = raw + dsv4.eos_token
        parsed = dsv4.parse_message_from_completion_text(
            raw, thinking_mode=self.thinking_mode
        )
        # We deliberately discard ``reasoning_content`` and return only the
        # post-``</think>`` ``content``: the judge prompt's TRUE/FALSE
        # block lives there, and our downstream regex expects that exact
        # final-answer text without the chain-of-thought interleaved.
        return parsed.get("content") or ""


def make_judge_backend(
    judge_model: str,
    *,
    backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    dsv4_model_dir: str | None = None,
    thinking_mode: str | None = None,
) -> JudgeBackend:
    """Build a :class:`JudgeBackend` from an explicit-arg / env-var ladder.

    All keyword arguments accept ``None`` meaning "fall back to the
    matching env var, then the documented default". See the module-level
    backend overview for the exact precedence.
    """
    backend = (backend or _env_str("JUDGE_BACKEND") or "auto").lower()
    if backend == "auto":
        backend = "dsv4" if str(judge_model).lower().startswith("deepseek-v4") else "openai"

    base_url = base_url or _env_str("JUDGE_BASE_URL") or _env_str("OPENAI_BASE_URL")
    api_key = api_key or _env_str("JUDGE_API_KEY") or _env_str("OPENAI_API_KEY")

    if backend == "openai":
        return OpenAIChatBackend(model=judge_model, base_url=base_url, api_key=api_key)
    if backend == "dsv4":
        return DsV4CompletionsBackend(
            model=judge_model,
            base_url=base_url or "http://127.0.0.1:8000/v1",
            api_key=api_key,
            model_dir=dsv4_model_dir or _env_str("JUDGE_DSV4_MODEL_DIR")
                      or DEFAULT_DSV4_MODEL_DIR,
            thinking_mode=thinking_mode or _env_str("JUDGE_THINKING_MODE") or "thinking",
        )
    raise ValueError(
        f"Unknown judge backend {backend!r}; expected one of "
        f"{{'auto', 'openai', 'dsv4'}}."
    )


def initialize_client():
    """Backwards-compatible shim retained for any external caller.

    The real client is now built lazily on first ``aux_judge`` call;
    this is kept so old import sites don't break.
    """
    return None


_INNER_TYPES = ("NV", "EX", "EQ", "IN", "MC", "TF", "QL")


class Judger:
    def __init__(
        self,
        strict_extract=True,
        judge_model=DEFAULT_JUDGE_MODEL,
        *,
        judge_backend: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        judge_dsv4_model_dir: str | None = None,
        judge_thinking_mode: str | None = None,
    ):
        # Per-element judgment methods. TUP / ALT are NOT in this map: they
        # are not single-element types, and ``auto_judge`` dispatches them
        # via :meth:`_judge_tuple_with_types` / :meth:`_judge_alt_with_types`
        # which apply the per-position ``type_sequence``-driven inner
        # judgments described in
        # ``study_bench_dataset/problems/README.md``.
        self.judgment_methods = {
            "MC": self.judge_MC,
            "TF": self.judge_TF,
            "NV": self.judge_single_numerical_value,
            "IN": self.judge_interval,
            "EX": self.judge_expression,
            "EQ": self.judge_equation,
            # QL (qualitative) only does case-insensitive exact-string
            # match here; semantic equivalence (e.g. "tidal forces" vs
            # "tides") is delegated to ``aux_judge`` (LLM judge) when the
            # rule-based path fails.
            "QL": self.judge_qualitative,
        }
        self.strict_extract = strict_extract
        self.pi = parse_latex("\\pi")
        # ``self.precision`` is kept as the *fallback* tolerance used when a
        # caller does not supply one via the ``precision=`` kwarg of any
        # ``judge_*`` method. We deliberately never mutate this attribute on
        # the hot path: every numeric comparison resolves its tolerance from
        # the kwarg first, so a single ``Judger`` instance is safe to share
        # across worker threads.
        self.precision = 1e-8
        self.judge_model = judge_model
        # Build the LLM backend up front so any "model not found" or
        # "encoder dir missing" error surfaces at construction time, well
        # before the first ``aux_judge`` call. Backend choice + endpoint
        # follow the explicit-arg > env-var > default ladder documented
        # in ``make_judge_backend``.
        self._judge_backend = make_judge_backend(
            judge_model,
            backend=judge_backend,
            base_url=judge_base_url,
            api_key=judge_api_key,
            dsv4_model_dir=judge_dsv4_model_dir,
            thinking_mode=judge_thinking_mode,
        )

    def _resolve_precision(self, precision):
        """Pick the effective tolerance for a single comparison.

        Threads precision explicitly through every judgment call site so
        we no longer have to write to ``self.precision`` from concurrent
        workers (which used to race when ``eval_file`` ran the same
        ``Judger`` from many threads).
        """
        return self.precision if precision is None else precision

    def normalize_answer(self, final_answer):
        special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        for signal in special_signal_map:
            final_answer = final_answer.replace(signal, special_signal_map[signal])
        final_answer = re.sub(r'\\(?:mathrm|mathbf)\{~?([^}]*)\}', '\\1', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(
            r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
        final_answer = re.sub(
            r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
        final_answer = final_answer.strip()
        final_answer = final_answer.strip("$")
        final_answer = final_answer.strip()
        final_answer = final_answer.replace("\\,", " ")
        #print(final_answer)
        final_answer = _strip_string(final_answer)
        return final_answer.rstrip("\\")
    
    def extract_boxed_answer(self, text):
        # extract answer wrapped in \boxed{} from models' output
        # TODO: add other extraction pattern
        # last boxed only
        content = remove_boxed(last_boxed_only_string(text))
        if content == None:
            match = re.search(r'\\boxed{', text)
            if match:
                start_index = match.end()
                end_index = start_index
                stack = 1
                while stack > 0 and end_index < len(text):
                    if text[end_index] == '{':
                        stack += 1
                    elif text[end_index] == '}':
                        stack -= 1
                    end_index += 1
                if stack == 0:
                    content = text[start_index:end_index - 1]
                    if not content:
                        return text
                    else:
                        content = self.normalize_answer(content)
                        return content
        if content == None:
            return text
        content = self.normalize_answer(content)
        return content
    
    def extract_ans(self, resp_str: str) -> str:
        """Extract answer segment from complete `resp`."""
        ans = self.extract_explicit_ans(resp_str)
        if ans is not None:
            return ans
        elif not self.strict_extract:
            # Speculate with the last latex formula
            matches = re.findall(
                r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)", resp_str, re.DOTALL
            )
            if len(matches) > 0:
                return matches[-1]
            # Speculate with the last number
            matches = re.findall(r"-?\d*\.?\d+", resp_str.replace(",", ""))
            if len(matches) > 0:
                return matches[-1]
        return ""  # Empty str if no answer is found

    def extract_all_boxed_answers(self, text: str) -> list:
        """Return normalized contents of every top-level ``\\boxed{}`` in ``text``.

        This is the multi-answer counterpart to :meth:`extract_boxed_answer`:
        instead of collapsing to the last group, every match is returned in
        source order. Empty / unparseable boxes are skipped.
        """
        contents = []
        for raw in all_boxed_only_strings(text):
            inner = remove_boxed(raw)
            if inner is None or not inner:
                continue
            contents.append(self.normalize_answer(inner))
        return contents

    def extract_ans_as_list(self, resp_str: str, multi_boxed: str = "all") -> list:
        """Extract answer segments and return them as a flat list of tokens.

        - When ``resp_str`` contains one or more ``\\boxed{}`` groups:
            * ``multi_boxed='all'`` keeps every group's content in order
              (used for the gold side, where the curator may list several
              acceptable alternatives or several distinct answer slots).
            * ``multi_boxed='last'`` keeps only the trailing group (matches
              the legacy single-boxed extraction used for predictions of
              single-answer questions).
          Each kept group is then split by comma (bracket-aware), so that
          ``\\boxed{a, b}`` and ``\\boxed{a}, \\boxed{b}`` collapse to the
          same atomic-token list ``["a", "b"]``.
        - Otherwise the legacy :meth:`extract_ans` fallback runs and its
          (possibly speculative) result is comma-split into a 1+ element
          list. Returns ``[]`` if nothing could be extracted.
        """
        boxed_list = self.extract_all_boxed_answers(resp_str)
        if boxed_list:
            if multi_boxed == "last":
                boxed_list = [boxed_list[-1]]
            flat = []
            for content in boxed_list:
                for part in self.split_by_comma(content):
                    if part:
                        flat.append(part)
            return flat

        single = self.extract_ans(resp_str)
        if not single:
            return []
        return [p for p in self.split_by_comma(single) if p]


    def extract_explicit_ans(self, resp_str: str) -> str:
        resp_str = self.clean_trailing(resp_str)
        # might be answer only
        if "herefore" in resp_str:
            resp_str = resp_str.split("herefore")[-1].strip()

        if "oxed{" in resp_str:
            resp = self.extract_boxed_answer(resp_str)
        else:
            resp = resp_str

            # should be answer only
            if "is the ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is the ans")[-2].strip())[
                    -1
                ].strip()
            elif "is our ans" in resp:
                resp = re.split(r"(,|\.|\!\|?)", resp.split("is our ans")[-2].strip())[
                    -1
                ].strip()
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
    
        
    def split_by_comma(self, expr: str):
        # Splits expressions by commas outside of brackets
        # 用于处理逗号的嵌套情况
        # 例子: "f(x, y, z), g(a, b, c), h(i, j)"
        in_bracket_num = 0 # 这个值为0时，说明当前不在括号内部
        splitted_expr = []
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
    
    
    def judge_MC(self, pred, gold, precision=None):
        # ``precision`` is accepted for a uniform call signature across all
        # ``judgment_methods``; MC has no numeric tolerance to use it for.
        del precision
        common_answer = [chr(i) for i in range(65, 91)] # 'A'~'Z'
        if pred == gold:
            return True
        else:
            if pred.startswith("[") and pred.endswith("]"):
                pred = pred.strip("[]")
            if pred[0] in common_answer and (len(pred) > 1 and pred[1] == ":"):
                return pred[0] == gold
            else:
                return False
            
    def judge_TF(self, pred, gold, precision=None):
        del precision  # unused; see judge_MC.
        if contains_chinese(pred):
            if pred in ["是", "对", "正确", "能"]:
                pred = "TRUE"
            elif pred in ["否", "错", "错误", "不能"]:
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

    def judge_qualitative(self, pred, gold, precision=None):
        """Cheap exact-string match for QL (qualitative) answers.

        QL answers are short natural-language phrases like ``"tidal
        forces"`` or ``"general relativity"``. We only attempt a
        case-insensitive, whitespace-collapsed equality here; anything
        more lenient (synonyms, paraphrasing, partial credit) is
        deferred to the LLM-based ``aux_judge`` fallback in the eval
        pipeline. Keeping this method strict avoids accidentally
        accepting semantically-different short answers that happen to
        share substrings.
        """
        del precision  # unused; QL is exact string match only.
        if not isinstance(pred, str) or not isinstance(gold, str):
            return False
        norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
        return norm(pred) == norm(gold)
        
    def judge_single_numerical_value(self, pred, gold, precision=None):
        precision = self._resolve_precision(precision)
        def is_scientific_notation(expr):
            return isinstance(expr, Mul) and isinstance(expr.args[1], Pow) and expr.args[1].args[0] == 10

        def to_scientific_notation_latex(num):
            num_sci = f"{num:.2e}"
            base, exponent = num_sci.split('e')
            exponent = int(exponent)
            return f"{base}\\times 10^{{{exponent}}}"

        # pure value -> can be parsed by python
        if pred == gold: # exact the same
            return True
        try: # can be parsed by python directly
            pred_value = float(pred)
            gold_value = float(gold)
            gold_decimal_places = (
                len(str(gold_value).split(".")[1]) if "." in str(gold_value) else 0
            )

            # Round pred_value to match the number of decimal places in gold
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
            if (
                exponent_pred == exponent_gold
                and abs(base_pred - base_gold) <= 0.1 * 1.01
            ):
                return True
        except Exception as e:
            pass
        
        # cannot be parsed by python, use scipy expression to judge
        # like 2^5, \log _2 7
        try:
            exp_pred = parse_latex(pred)
            exp_gold = parse_latex(gold)
            n_pred = N(exp_pred)
            n_gold = N(exp_gold)
            gold_decimal_places = (
                len(str(n_gold).split(".")[1]) if "." in str(n_gold) else 0
            ) 
            n_pred = round(n_pred, gold_decimal_places)
            if abs((n_pred - n_gold)/n_gold) <= precision * 1.01:
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
                if (
                    exponent_pred == exponent_gold
                    and abs(base_pred - base_gold) <= 0.1 * 1.01
                ):
                    return True
            else:
                if N(exp_pred) == N(exp_gold):
                    return True
        except Exception as e:
            pass
        
        return False
    
    
    def judge_interval(self, pred, gold, precision=None):
        precision = self._resolve_precision(precision)
        def parse_interval(interval):
            # Parse the interval string and return a list of tuples. Each tuple contains the interval values and types.
            parsed = []
            for part in interval.split('\\cup'):
                bounds, interval_type = part.strip(), ''
                if bounds.startswith('('):
                    interval_type += 'open_left'
                else:
                    interval_type += 'closed_left'
                if bounds.endswith(')'):
                    interval_type += '_open_right'
                else:
                    interval_type += '_closed_right'
                # Remove the interval characters to just get the numbers
                numbers = bounds.strip('()[]').split(',')
                parsed.append((numbers, interval_type))
            return parsed
        
        def compare_intervals(intervals1, intervals2):
            list1 = [(tuple(item[0]), item[1]) for item in intervals1]
            list2 = [(tuple(item[0]), item[1]) for item in intervals2]

            if len(list1) != len(list2):
                return False

            # Compare each parsed interval from list1 against all in list2
            for interval1 in list1:
                interval_numbers1, interval_type1 = interval1
                matched = False
                for interval2 in list2:
                    interval_numbers2, interval_type2 = interval2
                    # First check if the types of intervals match
                    if interval_type1 == interval_type2:
                        # Then check if both bounds of the intervals are mathematically equal
                        bounds_match = self.judge_expression(
                                            interval_numbers1[0], interval_numbers2[0],
                                            precision=precision,
                                        ) and \
                                        self.judge_expression(
                                            interval_numbers1[1], interval_numbers2[1],
                                            precision=precision,
                                        )
                        if bounds_match:
                            matched = True
                            list2.remove(interval2)
                            break
                if not matched:
                    return False
            return True
        
        # Parse both interval expressions
        parsed_intervals1 = parse_interval(pred)
        parsed_intervals2 = parse_interval(gold)

        # Compare the parsed intervals
        return compare_intervals(parsed_intervals1, parsed_intervals2)

    
    def judge_expression(self, pred, gold, precision=None):
        precision = self._resolve_precision(precision)
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()
        exp1 = extract_expression(pred)
        exp2 = extract_expression(gold)
        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))
        #print(expr1_sym)
        #print(expr2_sym)
        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)
            #print(expr1_sym)
            #print(expr2_sym)
            # judge if the expression contains symbol(like x, y)
            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    return self.judge_single_numerical_value(
                        expr1_sym, expr2_sym, precision=precision,
                    )
                except Exception as e:
                    return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)
                    #print(simplified_expr)
                    num_value = simplified_expr.evalf()
                    #print(num_value)
                    flag = abs(num_value) < precision
                    try:
                        flag = bool(flag)
                    except:
                        return False
                    #assert type(flag) == bool
                    return flag
                except Exception as e:
                    return False
                
    def judge_equation(self, pred, gold, precision=None):
        del precision  # equation comparison is symbolic; no tolerance needed.
        def simplify_equation(latex_eq):
            lhs, rhs = latex_eq.split('=')
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)
            equation = Eq(lhs_expr, rhs_expr)
            simplified_eq = simplify(equation.lhs - equation.rhs)
            return simplified_eq
        try:
            expr1_sym = simplify_equation(pred)
            expr2_sym = simplify_equation(gold)
            difference = simplify(expr1_sym - expr2_sym)
            
            if difference == 0:
                return True
            else:
                division_result_1 = simplify(expr1_sym / expr2_sym)
                division_result_2 = simplify(expr2_sym / expr1_sym)
                if (division_result_1.is_Integer and division_result_1 != 0) or (division_result_2.is_Integer and division_result_2 != 0):
                    return True
                else:
                    return False
        except:
            return False
        
    def judge_tuple(self, pred, gold, precision=None):
        """Legacy tuple judge: assumes every element is a numerical value.

        Kept only for backward compatibility (e.g. :meth:`is_equal`'s
        type-enumeration loop). New code should go through
        :meth:`auto_judge` with ``answer_type='TUP'`` and a populated
        ``type_sequence`` so each element is dispatched per the README.
        """
        precision = self._resolve_precision(precision)
        pred_list = self.split_by_comma(pred.strip("()"))
        gold_list = self.split_by_comma(gold.strip("()"))
        if len(pred_list) != len(gold_list):
            return False
        for p, g in zip(pred_list, gold_list):
            if not self.judge_single_numerical_value(p, g, precision=precision):
                return False
        return True

    # --------------------------------------------------------------------
    # type_sequence helpers
    # --------------------------------------------------------------------

    def _normalize_type_sequence(self, type_sequence) -> list[str]:
        """Coerce ``type_sequence`` into a list of upper-case inner-type tags.

        Accepts the wire-format string ``"NV,EX,QL"`` and Python lists.
        Strips whitespace and discards empty tokens. Does NOT validate
        membership in :data:`_INNER_TYPES` here; that check happens in the
        callers so we can return a precise reason on bad input.
        """
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

    def _judge_tuple_with_types(self, gold_boxes, pred_boxes, type_sequence, precisions):
        """README-compliant TUP judgment: position-strict per-type dispatch.

        ``gold_boxes`` and ``pred_boxes`` are lists of normalised raw
        ``\\boxed{}`` contents (no comma-splitting). The pred list must
        already be trimmed to the trailing ``len(gold_boxes)`` boxes (so
        chain-of-thought intermediate boxes don't poison the comparison).
        """
        n = len(gold_boxes)
        if n == 0 or len(pred_boxes) != n:
            return False
        if len(type_sequence) != n:
            return False
        for i, inner in enumerate(type_sequence):
            method = self.judgment_methods.get(inner)
            if method is None:
                # Unknown / forbidden inner type (e.g. "TUP" or "ALT")
                # cannot be dispatched: fail loudly rather than silently
                # accept a malformed answer.
                return False
            prec = precisions[i] if i < len(precisions) else precisions[-1]
            if not self._match_pair(method, pred_boxes[i], gold_boxes[i], precision=prec):
                return False
        return True

    def _judge_alt_with_types(self, gold_boxes, pred_boxes, type_sequence, precisions):
        """README-compliant ALT judgment: any matched (pred, gold) pair wins.

        For each pair ``(pred_boxes[i], gold_boxes[j])``, ONLY the
        judgment method declared by ``type_sequence[j]`` is tried; we
        deliberately do NOT cycle through every per-element judge as the
        old ``_alt_methods`` cartesian-product did, because the new
        schema pins every gold alternative to a definite expected type.
        """
        if not gold_boxes or not pred_boxes:
            return False
        if len(type_sequence) != len(gold_boxes):
            return False
        # Cache method lookups so the inner loop stays cheap.
        gold_methods = []
        for j, inner in enumerate(type_sequence):
            method = self.judgment_methods.get(inner)
            if method is None:
                return False
            prec = precisions[j] if j < len(precisions) else precisions[-1]
            gold_methods.append((method, prec))
        for pred_item in pred_boxes:
            for j, gold_item in enumerate(gold_boxes):
                method, prec = gold_methods[j]
                if self._match_pair(method, pred_item, gold_item, precision=prec):
                    return True
        return False

    def _normalized_raw_boxed_list(self, text: str) -> list:
        """Return raw ``\\boxed{}`` contents (NO inner comma-splitting),
        normalised, with ``\\pm`` expansion applied. This is the gold /
        pred extraction that TUP and ALT consume per README schema:
        ``len(type_sequence) == number of \\boxed{} groups``.
        """
        boxed = self.extract_all_boxed_answers(text)
        # ``trans_plus_minus_sign`` may double the list when ``\pm`` is
        # present; that's only meaningful when comparing single-value
        # alternatives. For TUP we keep the raw (no ``\pm``-expansion)
        # list so positions stay aligned. For ALT we want the expansion
        # so both sides of a `\pm` count as candidates -- callers pick.
        return [self.normalize_answer(b) if isinstance(b, str) else b for b in boxed]

    def judge(self, answer_type, pred, gold, type_sequence=None, precision=1e-8):
        """
        Args:
            answer_type (str)
            pred (str): the model's complete response
            gold (str): the ground truth answer
            type_sequence (str or list of str, optional): for ``TUP`` and
                ``ALT``, the per-position inner types declared by the
                schema (see README). Required for those two types.

        Returns:
            bool: True/False
        """
        # For TUP / ALT we route through ``auto_judge`` so the
        # type_sequence-driven dispatch logic stays in one place.
        if str(answer_type).upper() in ("TUP", "ALT"):
            return self.auto_judge(
                pred, gold, answer_type=answer_type,
                precision=precision, type_sequence=type_sequence,
            )

        extracted_pred = self.extract_ans(pred)
        if not extracted_pred: # no boxed answer in model's output
            return False
        extracted_pred = self.normalize_answer(extracted_pred)
        gold = self.normalize_answer(gold) if type(gold) == str else gold

        # judge correctness according to different types. Precision is passed
        # through as a kwarg so we never mutate ``self.precision`` (which is
        # shared across worker threads in ``eval_file``).
        try:
            return self.judgment_methods[answer_type](
                extracted_pred, gold, precision=precision,
            )
        except Exception:
            return False
            
    def extract_normalized_lists(self, pred, gold, answer_type):
        """Run the same extraction + normalization pipeline ``auto_judge`` uses.

        Returns ``(extracted_pred_list, normalized_gold_list)``. Both lists
        are post-``trans_plus_minus_sign`` and post-``normalize_answer``,
        i.e. they are exactly the strings that the per-type judgment
        method compares. Either list may be empty (e.g. when the model
        produced no boxed answer or when the gold has no boxed answer);
        callers can use that to short-circuit their own logic.

        Extraction policy:
          * ``TUP`` / ``ALT`` (per the README schema, ``type_sequence``
            length == number of raw ``\\boxed{}`` groups): we return the
            RAW boxed-content lists with NO inner comma-splitting. The
            split-by-comma normalization the legacy code did would
            silently re-flatten ``\\boxed{(a,b)}`` into two atoms, which
            no longer matches the per-position type dispatch.
          * For all other types: legacy behaviour -- comma-split each
            box's content into atoms (so ``\\boxed{a, b}`` ~ ``\\boxed{a},
            \\boxed{b}``) so single-element types still see one token.
          * For non-ALT types we also keep only the trailing
            ``len(gold_list)`` predictions, preserving the long-standing
            "the final boxed is the answer" behavior for chain-of-thought
            outputs that emit intermediate boxed groups.

        Exposed as a method (rather than inlined in :meth:`auto_judge`) so
        the eval pipeline can record ``normalized_gt`` /
        ``extracted_answer`` on each completion without re-implementing
        the (subtle, ALT-vs-non-ALT) extraction policy here.
        """
        answer_type = str(answer_type).upper() if answer_type is not None else ""

        if answer_type in ("TUP", "ALT"):
            gold_list = self._normalized_raw_boxed_list(gold)
            pred_list = self._normalized_raw_boxed_list(pred)
            if answer_type == "TUP" and gold_list:
                # Trim chain-of-thought intermediate boxes: keep only the
                # trailing ``len(gold)`` boxes from pred. ALT keeps all
                # candidates because each pred box is a candidate answer.
                k = len(gold_list)
                if len(pred_list) > k:
                    pred_list = pred_list[-k:]
            # IMPORTANT: do NOT pass TUP / ALT through
            # ``trans_plus_minus_sign``. That helper *duplicates* any list
            # element containing ``\\pm``, which would inflate the list
            # length and break the per-position alignment that TUP /
            # type_sequence-driven ALT rely on. The ``\\pm``-aware match
            # logic for these types lives in :meth:`_match_pair` instead.
            extracted_pred = pred_list
        else:
            gold_list = self.extract_ans_as_list(gold, multi_boxed="all")
            pred_all = self.extract_ans_as_list(pred, multi_boxed="all")
            k = len(gold_list)
            extracted_pred = pred_all[-k:] if k and len(pred_all) > k else pred_all
            gold_list = self.trans_plus_minus_sign(gold_list)
            extracted_pred = self.trans_plus_minus_sign(extracted_pred)

        # `extract_ans_as_list` already normalizes boxed contents, but the
        # legacy fallback path (no boxed found) does not, and `\\pm` expansion
        # may also re-introduce whitespace, so normalize once more here.
        extracted_pred = [
            self.normalize_answer(item) if isinstance(item, str) else item
            for item in extracted_pred
        ]
        gold_list = [
            self.normalize_answer(item) if isinstance(item, str) else item
            for item in gold_list
        ]
        return extracted_pred, gold_list

    def auto_judge(self, pred, gold, answer_type, precision=1e-8, type_sequence=None):
        """Strict, type-aware judgment driven by ``answer_type`` (+ ``type_sequence``).

        Extraction policy:
          - For ``TUP`` / ``ALT`` we use raw ``\\boxed{}`` extraction (NO
            inner comma-splitting), because the README's schema pins
            ``len(type_sequence)`` to the number of ``\\boxed{}`` groups
            in ``gold``, not the number of comma-split atoms.
          - For all other types we keep the legacy comma-split behaviour
            so single-element answers like ``\\boxed{a, b}`` still
            collapse to two atoms.
          - ``pred`` keeps every ``\\boxed{}`` for ALT (each box is a
            candidate). For non-ALT types we keep only the trailing
            ``len(gold)`` predictions, which preserves the long-standing
            "the final boxed is the answer" behaviour for chain-of-
            thought outputs that emit intermediate boxed groups.

        Comparison policy (matches
        ``study_bench_dataset/problems/README.md``):
          - **TUP**: position-strict. Each pair ``(pred_i, gold_i)`` is
            compared with ``judgment_methods[type_sequence[i]]`` and ALL
            must pass. Lengths must agree.
          - **ALT**: cartesian product over (pred boxes, gold boxes); for
            pair ``(i, j)``, ONLY the comparator declared by
            ``type_sequence[j]`` is tried. Any hit wins.
          - **Other types**: dispatch to
            ``self.judgment_methods[answer_type]`` and a true assignment
            search between pred list and gold list (a greedy walk could
            wrongly fail when tolerance-based matches admit multiple
            valid pairings).

        Thread safety: precision is passed all the way down to each
        judgment method via the ``precision=`` kwarg. We *never* mutate
        ``self.precision`` here, so a single ``Judger`` instance is safe
        to share across the worker threads in ``eval_file``.
        """

        precision_list = list(precision) if isinstance(precision, list) else [precision]
        answer_type = str(answer_type).upper() if answer_type is not None else ""

        # ----- TUP / ALT: per-position dispatch via type_sequence ---------
        if answer_type in ("TUP", "ALT"):
            ts = self._normalize_type_sequence(type_sequence)
            extracted_pred, gold_list = self.extract_normalized_lists(
                pred, gold, answer_type,
            )
            if not gold_list or not extracted_pred:
                return False
            # If the schema entry is missing / malformed type_sequence,
            # rule-based judging cannot proceed; the eval pipeline will
            # then fall back to ``aux_judge``.
            if len(ts) != len(gold_list):
                return False
            if len(precision_list) < len(gold_list):
                precision_list = (precision_list * len(gold_list))[: len(gold_list)]
            if answer_type == "TUP":
                if len(extracted_pred) != len(gold_list):
                    return False
                return self._judge_tuple_with_types(
                    gold_list, extracted_pred, ts, precision_list,
                )
            # ALT
            return self._judge_alt_with_types(
                gold_list, extracted_pred, ts, precision_list,
            )

        # ----- Single-element types --------------------------------------
        method = self.judgment_methods.get(answer_type)
        if method is None:
            # Unknown / missing answer_type: refuse to guess. The caller
            # (e.g. eval pipeline) can fall back to a model-based judge.
            return False

        extracted_pred, gold_list = self.extract_normalized_lists(pred, gold, answer_type)
        if not gold_list or not extracted_pred:
            return False
        if len(extracted_pred) != len(gold_list):
            return False

        if len(precision_list) <= 1:
            precision_list = precision_list * len(gold_list)

        return self._assignment_match(
            method, extracted_pred, gold_list, precision_list,
        )

    def _assignment_match(self, method, preds, golds, precisions):
        """True iff there's a one-to-one assignment with every (gold, pred) matching.

        Tolerance-based comparators (e.g. ``judge_single_numerical_value``
        with rounding) make this a real assignment problem: the same
        prediction can plausibly match more than one gold, and a greedy
        first-match-wins walk can fail when a valid assignment exists. We
        use DFS over a precomputed match matrix; that's O(n!) in the
        worst case but fine for the n <= ~6 lists IOAA / multi-slot
        TUP answers actually carry. For safety we cap n at 8 and fall
        back to a single greedy attempt for larger lists, which is
        unlikely to come up in practice.
        """
        n = len(golds)
        if n == 0:
            return True
        if n > 8:
            # Defensive cap; greedy fallback for absurdly large lists.
            used = [False] * n
            for i, (g, prec) in enumerate(zip(golds, precisions)):
                for j, p in enumerate(preds):
                    if used[j]:
                        continue
                    if self._match_pair(method, p, g, precision=prec):
                        used[j] = True
                        break
                else:
                    return False
            return True

        # Build the n*n match matrix once so the DFS doesn't re-invoke the
        # (expensive, latex-parsing) judgment method on the same pair.
        ok = [[False] * n for _ in range(n)]
        for i, (g, prec) in enumerate(zip(golds, precisions)):
            for j, p in enumerate(preds):
                if self._match_pair(method, p, g, precision=prec):
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

    def _match_pair(self, method, pred, gold, *, precision=None):
        """Run a single pairwise judgment with timeout and two cheap fallbacks.

        Three layers of comparison are attempted:

          1. Trivial-equality short-circuit. After all the upstream
             extraction / normalization passes, if ``pred`` and ``gold``
             collapse to byte-identical strings they are by definition
             equivalent, regardless of which per-type judge would have
             been called. This matches what the legacy
             ``judge_single_numerical_value`` did at the top of itself
             (and which everything else inherited via ``judge_tuple``);
             without it, type-specific judges (e.g. ``judge_equation``
             with single-letter LHS) can spuriously fail on inputs they
             can't parse, even when both sides are textually identical.

          2. The declared per-type ``method`` itself.

          3. ``method`` again on the const-stripped strings -- helps with
             physics expressions that differ only in symbolic constants
             like ``\\hbar``.

        ``precision`` is forwarded to the judgment method as a kwarg, so
        callers must not pre-set ``self.precision`` (and indeed we never
        do that any more).
        """
        if isinstance(pred, str) and isinstance(gold, str) and pred == gold:
            return True

        # ``\\pm`` expansion happens here (per-element) instead of
        # globally on the list, so we can use it for TUP / ALT without
        # breaking position alignment. We try every (pred-variant,
        # gold-variant) combination; any match wins.
        pred_variants = self._pm_variants(pred)
        gold_variants = self._pm_variants(gold)

        try:
            def check():
                for p in pred_variants:
                    for g in gold_variants:
                        if method(p, g, precision=precision):
                            return True
                        if method(
                            self.remove_const(p),
                            self.remove_const(g),
                            precision=precision,
                        ):
                            return True
                return False
            return self._run_with_timeout(check, timeout_seconds=5)
        except Exception:
            return False

    def _pm_variants(self, s):
        """Expand ``\\pm`` to ``+`` and ``-`` variants for one element.

        Returns ``[s]`` unchanged when ``s`` has no ``\\pm`` (the common
        case, no extra work). When ``\\pm`` is present we return both
        sign-substituted forms so TUP / ALT comparisons accept either.
        """
        if not isinstance(s, str) or "\\pm" not in s:
            return [s]
        return [s.replace("\\pm", "+"), s.replace("\\pm", "-")]
    
    def aux_judge(self, pred, gold, question, reference, stem=""):
        """LLM-based equivalence judge.

        Parameters
        ----------
        pred, gold
            Raw model completion / curator gold (usually wrapped in
            ``\\boxed{}``).
        question
            The current sub-question text.
        reference
            The curator's reference solution for the sub-question.
        stem
            The parent problem's shared setup, sourced by callers from
            ``parent['problem']`` (the schema no longer has a separate
            ``stem`` field; ``parent['problem']`` now stores the shared
            preamble for multi-sub parents and the full text for solo
            parents). Multi-part IOAA problems put the variable
            definitions in this preamble and only put the actual ask in
            the sub-question, so the LLM judge cannot grade equivalence
            without it; we compose ``stem + sub-question`` here when
            ``stem`` is non-empty. The parameter name stays ``stem`` for
            backward compatibility — it is just a string, not a JSON
            field reference.

        Returns
        -------
        (correctness, message)
            ``correctness`` is a bool. ``message`` is the raw judge
            response on success, or an ``[aux_judge ...]`` error string
            on failure. Even when ``pred`` contains no ``\\boxed{}`` we
            still call the judge -- a model that wrote a correct answer
            in plain prose deserves the chance to be graded -- and
            simply tell the judge that no boxed answer was extracted.
        """
        # The LLM judge gets the *raw* boxed answers (no ``\pm`` expansion)
        # so it sees the original notation; the rule-based judge already
        # consumed the ``\pm``-expanded form upstream of this call.
        gold_list = self.extract_ans_as_list(gold, multi_boxed="all")
        extracted_pred = self.extract_ans_as_list(pred, multi_boxed="all")

        ra_text = ", ".join(gold_list) if gold_list else gold
        if extracted_pred:
            sa_text = ", ".join(extracted_pred)
        else:
            sa_text = "(no \\boxed{} answer found in student's solution)"

        full_question = (
            f"{stem.strip()}\n\n{question.strip()}"
            if stem and stem.strip()
            else question
        )

        # Rate limits are retried after a fixed two-second delay. Other
        # failures are surfaced immediately so the eval pipeline keeps moving
        # and the failure remains attributable per sub-question.
        try:
            with open(_JUDGE_PROMPT_PATH, 'r', encoding='utf-8') as file:
                judge_prompt = file.read()
            judge_prompt = (
                judge_prompt.replace("{{problem}}", full_question)
                .replace("{{RS}}", reference)
                .replace("{{RA}}", ra_text)
                .replace("{{SS}}", pred)
                .replace("{{SA}}", sa_text)
            )
            # The DSv4 backend builds a sensible default ``max_tokens``;
            # for the OpenAI backend we use a generous 8K so even very
            # long judge replies don't get truncated mid-TRUE/FALSE.
            default_max_tokens = (
                16384 if self._judge_backend.name == "dsv4" else 8192
            )
            temperature = float(_env_str("JUDGE_TEMPERATURE", "0.0"))
            max_tokens = int(_env_str("JUDGE_MAX_TOKENS", str(default_max_tokens)))
            timeout_s = float(_env_str("JUDGE_TIMEOUT_S", "600"))
            max_rate_limit_retries = int(
                _env_str("JUDGE_RATE_LIMIT_MAX_RETRIES", "10")
            )
            rate_limit_retries = 0
            while True:
                try:
                    res = self._judge_backend.complete_chat(
                        [{"role": "user", "content": judge_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout_s,
                    )
                    break
                except Exception as exc:
                    if (
                        not _is_rate_limit_error(exc)
                        or rate_limit_retries >= max_rate_limit_retries
                    ):
                        raise
                    rate_limit_retries += 1
                    print(
                        "[aux_judge] rate limited; waiting 2s before retry "
                        f"{rate_limit_retries}/{max_rate_limit_retries}.",
                        flush=True,
                    )
                    time.sleep(2.0)
        except Exception as e:
            err = f"[aux_judge error] {type(e).__name__}: {e}"
            print(err, flush=True)
            return False, err

        # Robust TRUE/FALSE parse: prefer the "## Equivalence Judgement"
        # section, but fall back to any standalone TRUE / FALSE token in
        # the response. Some LLMs drop the second header or wrap the
        # response in code fences; we don't want a cosmetic format
        # deviation to silently downgrade a correct judgement to wrong.
        match = re.search(
            r"##\s*Equivalence\s*Judgement\s*[\r\n]+\s*(TRUE|FALSE)\b",
            res or "",
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper() == "TRUE", res
        match = re.search(r"\b(TRUE|FALSE)\b", res or "", re.IGNORECASE)
        if match:
            return (
                match.group(1).upper() == "TRUE",
                f"{res}\n[aux_judge: parsed via lenient TRUE/FALSE regex]",
            )
        return (
            False,
            f"[aux_judge parse error: no TRUE/FALSE found]\n--- raw judge response ---\n{res}",
        )

    def _run_with_timeout(self, func, timeout_seconds=5):
        """Run a rule-based comparison with a hard CPU/time limit.

        This used to run ``func`` in a daemon thread. Python has no safe API
        to terminate a running thread, so a timed-out SymPy call continued
        consuming CPU and could hold ``_LATEX_LOCK`` forever. A forked child
        gives us a real cancellation boundary; on timeout it is terminated
        and escalated to SIGKILL if needed.
        """
        try:
            ctx = mp.get_context("fork")
        except ValueError:  # pragma: no cover - Linux provides fork
            ctx = None

        if ctx is None:
            # Defensive fallback for platforms without fork. This preserves
            # the old API, but callers should expect that a timed-out thread
            # cannot be force-killed there.
            result = {}

            def target():
                try:
                    result["value"] = func()
                except Exception as exc:
                    result["exception"] = exc

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout_seconds)
            if thread.is_alive():
                return False
            if "exception" in result:
                raise result["exception"]
            return result.get("value", False)

        parent_conn, child_conn = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_timeout_process_worker,
            args=(func, child_conn),
            daemon=True,
        )
        process.start()
        child_conn.close()
        try:
            if not parent_conn.poll(timeout_seconds):
                process.terminate()
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                return False

            status, payload = parent_conn.recv()
            process.join(timeout=1.0)
            if status == "error":
                raise RuntimeError(payload)
            return bool(payload)
        finally:
            parent_conn.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            process.close()

    def is_equal(self, gold, pred, precision=None):
        answer_type_list = ["MC", "TF", "NV", "IN", "EX", "EQ"]
        for answer_type in answer_type_list:
            try:
                def check_equivalence(answer_type=answer_type):
                    method = self.judgment_methods[answer_type]
                    return (
                        method(pred, gold, precision=precision)
                        or method(
                            self.remove_const(pred),
                            self.remove_const(gold),
                            precision=precision,
                        )
                    )

                if self._run_with_timeout(check_equivalence, timeout_seconds=5):
                    return True
            except Exception as e:
                #print(e)
                pass
        return False

    def remove_const(self, input_string):
        for constant in PHY_CONST:
            # Escape any special characters in the constant for safe regex replacement
            #escaped_constant = re.escape(constant)
            #input_string = re.sub(r"\b{escaped_constant}\b", "1", input_string)
            input_string = input_string.replace(constant, "")
        if input_string.strip() == "":
            input_string = "1"
        # Remove extra spaces or unwanted artifacts from the string
        return re.sub(r"\s+", " ", input_string).strip()

    def sympy_sub_pi(self, expression_sympy):
        return expression_sympy.subs(self.pi, math.pi)
    
    def trans_plus_minus_sign(self, expr_list: list):
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list
    
    def clean_trailing(
        self,
        s: str,  # The input string.
    ) -> str:  # The cleaned string with trailing punctuation marks removed.
        """Removes trailing punctuation marks from a string."""
        s = str(s).strip()
        while s != "" and s[-1] in NO_TRAILING_STRS:
            s = s[:-1].strip()
        return s
    
if __name__ == "__main__":
    judger = Judger()
    pred = "\\boxed{\\pi}"
    gold = "3.14159265358979"
    pred = "\\boxed{0.496}"
    gold = "\\boxed{0.496}"
    gold = "\\boxed{No}"
    pred = "\\boxed{False}"
    gold = "\\boxed{439}"
    pred = "\\boxed{4.39 \\times 10^{3}}"
    gold = "\\boxed{\\hbar}"
    pred = "\\boxed{1}"
    gold = "\\boxed{\\frac{n \\mu^{2} \\mu_0}{3 k_B T}}"
    pred = "\\boxed{n \\frac{\\mu^2}{3k_B T}}"
    gold = "\\boxed{\\frac{\\tau I^{2} s^{2}}{2 \\pi^{2} l_{B}^{2} l_{A}^{2} \\hbar^{2} \\omega^{2}}[1+\\frac{1}{2} \\cos \\frac{\\Delta \\omega}{c}(l_{A}-l_{B})}"
    pred = "\\boxed{(\\frac{I s}{4 \\pi \\hbar \\omega})^2 \\frac{\\tau}{l_A^2 l_B^2}}"
    gold = "\\boxed{\\ddot{\\lambda}-(\\lambda+1) \\dot{\\varphi}^{2}+\\omega_{s}^{2} \\lambda+\\omega_{p}^{2}(1-\\cos \\varphi)=0}"
    pred = "\\boxed{r_0^2 \\ddot{\\lambda} = \\omega_s^2 r_0^2 \\lambda - \\omega_p^2 r_0^2 \\cos(\\varphi)}"
    gold = "\\boxed{\\dfrac{\\hbar\\,\\pi\\,n}{8\\,m\\,v}}"
    pred = "\\boxed{\\dfrac{\\hbar\\pi n}{8 m v}}"
    gold = "\\boxed{\\dfrac{\\hbar\\,\\pi\\,n}{8\\,m\\,v}}"
    pred = "\\boxed{\\frac{\\hbar\\pi n}{8 m v}}"
    gold = "\\boxed{\\sigma(\\omega)=\\frac{ne^2\\tau}{m}\\frac{1}{1-i\\omega\\tau}}"
    pred = "\\boxed{\\sigma(\\omega)=\\frac{ne^{2}\\tau}{m(1-i\\omega\\tau)}}"
    print(judger.auto_judge(pred, gold, answer_type="EQ"))
