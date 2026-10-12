r"""Deciding whether a written LaTeX answer means the reference answer (MATH-500).

MATH-500 answers are LaTeX: 311 of the 500 are bare integers, and the rest are
fractions, surds, tuples (``\left( 3, \frac{\pi}{2} \right)``), intervals
(``(3,4]``), unions, column vectors (``\begin{pmatrix} -1/3 \\ 2/3 \end{pmatrix}``)
and ten ``\text{...}`` words. A model writes those in its own house style --
``\tfrac\pi2``, ``(3, \pi/2)``, ``x = 6``, ``5 \text{ cm}``, ``\sqrt{12}`` for
``2\sqrt3`` -- and scoring any of them wrong would measure our comparison rather
than the model.

So the comparison is an **ordered, named ladder**, and the level that decided is
recorded on the trace. In order:

1. ``exact`` -- the strings are identical.
2. ``normalized`` -- identical after :func:`normalize`, which only ever rewrites
   *how* an expression is written (see :func:`normalize`).
3. ``unit_suffix`` -- identical once a trailing unit annotation (``\text{ cents}``,
   ``degrees``) is dropped from both sides.
4. ``*_elementwise`` -- both sides are the same *kind* of structure (a union, a
   matrix, a bracketed comma list) with the same number of parts, and every part
   matches. The brackets are part of the comparison: ``(3,4]`` is not ``[3,4)``.
5. ``numeric`` -- both sides evaluate to the same real number (1e-9 relative).
6. ``symbolic`` -- ``simplify(a - b)`` is zero, in a killable subprocess with a
   timeout, because ``simplify`` on a hostile expression does not return.

Anything else is a mismatch, and the reason is recorded. Four traps are
deliberately avoided, each of which silently mis-grades:

* **The shared LaTeX canonicaliser is value-destroying here.**
  :func:`llm.core.parser.math_parser.canonicalize_latex` rewrites ``5!`` to ``5``
  and drops a trailing constant. Both are right for an antiderivative and
  catastrophic for an answer that *is* a factorial. This module parses with that
  hook disabled.
* **An unevaluated ``Integral``/``Sum`` has ``.is_number == True``.** A model
  that restates the problem would otherwise be evaluated into the right answer.
  Those nodes are refused before anything is evaluated.
* **Structure is compared as structure.** Splitting ``(3,4]`` into numbers and
  comparing them element-wise while ignoring the brackets scores an open
  interval as a closed one. The delimiters have to agree first.
* **A ``\text{}`` answer is a word, not an expression.** ``\text{even}`` and
  ``\text{odd}`` must never reach a numeric comparison that reads both as
  nothing and calls them equal; they are compared as normalised strings and
  nothing else.
"""
from __future__ import annotations

import concurrent.futures
import functools
import multiprocessing
import re
from typing import Any

import sympy as sp

from .comparison import Match

_TEXT_GROUP = re.compile(r"\\(?:text|mbox|mathrm|textbf|textit|mathbf)\s*\{([^{}]*)\}")
_SPACING = re.compile(
    r"\\(?:left|right|displaystyle|textstyle|limits|nolimits|quad|qquad|,|;|:|!|\s)"
)
_DEGREES = re.compile(r"\^\s*\{?\s*\\(?:circ|degree)\s*\}?|\u00b0")
_FRAC_ALIAS = re.compile(r"\\[dt]frac")
_SQRT_BARE = re.compile(r"\\sqrt\s*(\d|[a-zA-Z])(?![\w{])")
_SQRT_CMD = re.compile(r"\\sqrt\s*(\\[a-zA-Z]+)(?![\w{])")
_FRAC_BARE = re.compile(r"\\frac\s*(\d|[a-zA-Z])\s*(\d|[a-zA-Z])(?![\w{])")
_FRAC_BARE1 = re.compile(r"\\frac\s*\{([^{}]*)\}\s*(\d|[a-zA-Z])(?![\w{])")
_FRAC_CMD = re.compile(r"\\frac\s*(\\[a-zA-Z]+)\s*(\\[a-zA-Z]+|\{[^{}]*\}|\d|[a-zA-Z])")
_FRAC_CMD2 = re.compile(r"\\frac\s*(\d|[a-zA-Z]|\{[^{}]*\})\s*(\\[a-zA-Z]+)")
# `\frac9{19}`: a bare numerator followed by a braced denominator. The reference
# answers use this spelling and a model almost never does, so without it a right
# answer is compared against a string that parses differently.
_FRAC_BARE2 = re.compile(r"\\frac\s*(\d|[a-zA-Z])\s*(\{[^{}]*\})")
# A currency marker, LaTeX-escaped or bare. Stripped BEFORE the math delimiters,
# because `$` alone turns `\$32` into a stray backslash rather than into `32`.
_CURRENCY = re.compile(r"\\\$|(?<![\\])\$")
# `12^{\mathrm{th}}`, `12^{\text{st}}`, `3^{rd}` -- an ordinal's superscript, which
# says how the number is written and never what it is.
_ORDINAL_SUP = re.compile(r"\^\s*\{?\s*(?:\\(?:text|mathrm|rm)\s*\{)?\s*(?:st|nd|rd|th)\s*\}?\s*\}?")
# A thousands separator as LaTeX spells it, and only that: `10,\!080`, `1{,}200`.
# A bare comma is left alone -- it separates the parts of `1,-2` and `(6,31,-1)`.
# `10,\!080`, `1{,}200`, and `11,\! 111,\! 100` -- the space after the separator
# is written by hand in several references, so it is allowed for.
_THOUSANDS = re.compile(r"(?<=\d)(?:,\\!|\{,\}|\\,)\s*(?=\d{3})")
# `x = 6` when the reference is `6`. Both sides go through it, so a reference
# that names its variable keeps it only if the candidate does too.
# `x = 6` and `x \in [-2,7]`: the variable the answer is *about*, which the
# reference sometimes names and the model usually does not.
_LEAD_VAR = re.compile(r"^[a-zA-Z]\s*(?:=|\\in\b|\\to\b)\s*")
_TRAIL_PUNCT = re.compile(r"[.\s]+$")
_LEADING_PLUS = re.compile(r"(?<![\w)\]}])\+(?=[\\\d(])")
# A trailing unit annotation: `\text{ cents}`, ` degrees`. Removed only from the
# *end*, and only as a whole group or a run of words, so `\text{even}` -- which
# is the entire answer -- keeps its word.
# A trailing unit annotation. The optional exponent after the group is what
# makes `15\mbox{ cm}^2` and `864 \mbox{ inches}^2` come off: the unit is not the
# last thing in the string, its square is.
_UNIT_TAIL = re.compile(
    r"(?:\\(?:text|mbox|mathrm)\s*\{[^{}]*\}|\s[A-Za-z]{2,}[A-Za-z\s]*)"
    r"(?:\s*\^\s*\{?\s*\d+\s*\}?)?\s*$"
)
# The matrix row separator, which the spacing rule would otherwise eat: `\\` is
# two escapes, and `\` followed by whitespace is a spacing command.
_ROW_SENTINEL = "\x00ROW\x00"

_CLOSERS = {"(": ")", "[": "]", "\\{": "\\}", "\\langle": "\\rangle", "<": ">"}
_MATRIX = re.compile(r"\\begin\{([bp]matrix\*?)\}(.*)\\end\{\1\}", re.S)
_ROW_SPLIT = re.compile(r"\\\\|&")

# Nodes that mean "the model restated the problem" rather than "the model
# answered".
_UNEVALUATED = (sp.Integral, sp.Sum, sp.Product, sp.Limit, sp.Derivative)
_NUMERIC_RTOL = 1e-9
_DEFAULT_TIMEOUT_S = 5.0


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #

def normalize(text: str, rules: list[str] | None = None) -> str:
    """Rewrite *how* an expression is written, never what it says.

    Every rule below is presentational. None of them removes a term, evaluates
    anything, or widens what counts as an answer -- the ladder's later levels do
    the mathematics, and keeping this function value-preserving is what makes
    ``normalized`` a trustworthy verdict on its own.
    """
    from llm.core.parser.math_parser import strip_bpe_sentinels

    collected = rules if rules is not None else []
    out, restored = strip_bpe_sentinels(str(text).strip())
    if restored:
        collected.append("restore_bpe_whitespace")

    def rule(name: str, value: str) -> str:
        nonlocal out
        if value != out:
            collected.append(name)
        return value

    out = rule("strip_currency", _CURRENCY.sub("", out))
    out = rule("strip_math_delimiters", out.replace("$", ""))
    out = out.replace("\\\\", _ROW_SENTINEL)
    out = rule("thousands_separator", _THOUSANDS.sub("", out))
    out = rule("unwrap_text", _TEXT_GROUP.sub(r" \1 ", out))
    out = rule("drop_spacing_commands", _SPACING.sub(" ", out))
    out = rule("frac_aliases", _FRAC_ALIAS.sub(r"\\frac", out))
    out = rule("degrees", _DEGREES.sub("", out))
    out = rule("percent", out.replace("\\%", "").replace("%", ""))
    out = rule("brace_bare_sqrt", _SQRT_BARE.sub(r"\\sqrt{\1}", out))
    out = rule("brace_bare_frac",
               _FRAC_BARE2.sub(r"\\frac{\1}\2",
                               _FRAC_BARE1.sub(r"\\frac{\1}{\2}",
                                               _FRAC_BARE.sub(r"\\frac{\1}{\2}", out))))
    out = rule("brace_command_frac",
               _FRAC_CMD2.sub(r"\\frac{\1}{\2}", _FRAC_CMD.sub(r"\\frac{\1}{\2}", out)))
    out = rule("brace_command_sqrt", _SQRT_CMD.sub(r"\\sqrt{\1}", out))
    out = rule("ordinal_superscript", _ORDINAL_SUP.sub("", out))
    out = rule("drop_leading_variable", _LEAD_VAR.sub("", out))
    out = rule("trailing_punctuation", _TRAIL_PUNCT.sub("", out))
    out = rule("leading_plus", _LEADING_PLUS.sub("", out))
    out = rule("collapse_space", " ".join(out.split()))
    return out.replace(_ROW_SENTINEL, "\\\\")


def compact(text: str) -> str:
    """The normalised form with every space removed, case-folded.

    What the ``normalized`` level compares. Spacing inside LaTeX carries no
    meaning, and case matters in no MATH-500 answer -- the ten word answers are
    proper nouns and parities, and neither is ambiguous in lower case.
    """
    return re.sub(r"\s+", "", normalize(text)).casefold()


def strip_unit_suffix(text: str) -> str:
    """Drop a trailing unit annotation: ``5.4 \\text{ cents}`` -> ``5.4``.

    Repeated until nothing more comes off, but never down to nothing: an answer
    that is *only* a word (``\\text{Evelyn}``) keeps it, because there is
    nothing else for the unit to annotate.
    """
    out = str(text).strip()
    while True:
        stripped = _UNIT_TAIL.sub("", out).strip()
        if stripped == out or not stripped:
            return out
        out = stripped


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #

def _bracketed(text: str) -> tuple[str, str, str] | None:
    """(open, close, inner) if the whole string is one bracketed group, else None."""
    stripped = text.strip()
    for opener, closer in _CLOSERS.items():
        if stripped.startswith(opener) and stripped.endswith(closer):
            inner = stripped[len(opener):len(stripped) - len(closer)]
            depth = 0
            for char in inner:
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth -= 1
                if depth < 0:
                    return None          # `(1,2)\cup(3,4)`: not one group
            return opener, closer, inner
    return None


def _split_depth0(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    current = ""
    depth = 0
    index = 0
    while index < len(text):
        if depth == 0 and text.startswith(separator, index):
            parts.append(current)
            current = ""
            index += len(separator)
            continue
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        current += char
        index += 1
    parts.append(current)
    return [part.strip() for part in parts]


def split_tuple(text: str) -> tuple[str, str, list[str]] | None:
    """``(open, close, parts)`` of a bracketed comma list, else ``None``.

    The brackets are returned rather than discarded: an interval's delimiters
    are part of its meaning, and comparing ``(3,4]`` with ``[3,4)`` by their
    endpoints alone calls two different sets equal.
    """
    group = _bracketed(normalize(text))
    if group is None:
        return None
    opener, closer, inner = group
    parts = _split_depth0(inner, ",")
    return (opener, closer, parts) if len(parts) > 1 else None


def split_union(text: str) -> list[str] | None:
    parts = _split_depth0(normalize(text), "\\cup")
    return parts if len(parts) > 1 else None


def matrix_rows(text: str) -> list[str] | None:
    match = _MATRIX.search(normalize(text))
    if match is None:
        return None
    return [cell.strip() for cell in _ROW_SPLIT.split(match.group(2)) if cell.strip()]


def split_bare_list(text: str) -> list[str] | None:
    """An *unbracketed* comma list -- ``1,-2``, ``3, 5, 7`` -- else ``None``.

    Kept apart from :func:`split_tuple` because the two mean different things.
    A bracketed list is a point or an interval and its order and delimiters are
    part of the answer; a bare list is "the solutions are", and a model that
    writes the same roots in the other order has answered the question. So this
    one is compared as an unordered multiset and the bracketed one is not.
    """
    normalised = normalize(text)
    if _bracketed(normalised) is not None or _MATRIX.search(normalised):
        return None
    parts = _split_depth0(normalised, ",")
    return parts if len(parts) > 1 and all(parts) else None


def expand_pm(text: str) -> list[str] | None:
    r"""``1 \pm \sqrt{19}`` -> ``["1 + \sqrt{19}", "1 - \sqrt{19}"]``, else ``None``.

    Several MATH-500 references write a pair of roots this way and a model
    almost always writes them out. Expanding is not a loosening of the answer
    key: the two spellings denote the same two numbers, and the comparison that
    follows still has to match both of them.

    Only a single top-level ``\pm`` is expanded. Two of them would denote four
    values under one reading and two under another, and guessing which is not
    something a grader should do.
    """
    normalised = normalize(text)
    parts = _split_depth0(normalised, "\\pm")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    head, tail = parts
    return [f"{head} + {tail}", f"{head} - {tail}"]


def vector_parts(text: str) -> list[str] | None:
    r"""The components of a vector, however it is spelled, else ``None``.

    A ``pmatrix`` column, ``(a, b, c)``, ``<a, b, c>`` and
    ``\langle a, b, c \rangle`` are one vector written four ways, and MATH-500
    references use the first while models use the others. This is deliberately
    *not* routed through the bracketed-tuple comparison, which compares
    delimiters and must keep doing so -- ``(3,4]`` is not ``[3,4)``. It is
    reached only when one side is unambiguously a matrix, where no interval
    reading is possible.
    """
    rows = matrix_rows(text)
    if rows is not None:
        return rows
    tuple_parts = split_tuple(text)
    return list(tuple_parts[2]) if tuple_parts is not None else None


# --------------------------------------------------------------------------- #
# parsing and evaluation
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _parser() -> Any:
    """A parser that does **not** apply the antiderivative canonicaliser.

    ``MathLatexParser.parse`` calls ``canonicalize_latex`` itself, so declining
    to call it here is not enough -- the parser would apply it anyway and ``5!``
    would arrive as ``5``. Overriding the one hook is the narrowest fix.
    """
    from llm.core.parser.math_parser import MathLatexParser

    class _AnswerLatexParser(MathLatexParser):
        def _canonicalize_latex(self, latex: str) -> str:
            return latex

    return _AnswerLatexParser()


def parse_expression(text: str) -> Any | None:
    """The sympy expression a normalised answer names, or ``None``.

    ``None`` covers both "latex2sympy raised" and "this is a restatement of the
    problem": neither can be compared, and treating a parse failure as a
    mismatch is the honest reading -- the later levels simply do not run.
    """
    try:
        expr = _parser().parse(normalize(text))
    except Exception:
        return None
    if expr is None:
        return None
    try:
        if expr.has(*_UNEVALUATED):
            return None
    except Exception:
        return None
    return expr


def numeric_value(expr: Any) -> float | None:
    """The real number an expression evaluates to, or ``None``."""
    if expr is None:
        return None
    try:
        if expr.free_symbols or not expr.is_number:
            return None
        value = complex(sp.N(expr, 30))
    except Exception:
        return None
    if value != value or abs(value.real) == float("inf"):    # nan, oo
        return None
    if abs(value.imag) > 1e-12 * max(abs(value.real), 1.0):
        return None
    return value.real


def _symbolic_worker(left: str, right: str) -> bool:
    """Run in a subprocess: is ``left - right`` zero?

    Takes ``srepr`` strings rather than expressions, because that is what
    survives pickling to a forkserver child cleanly -- the same contract
    :mod:`llm.core.math.verification` uses.
    """
    difference = sp.simplify(sp.sympify(left) - sp.sympify(right))
    if difference == 0:
        return True
    try:
        return bool(difference.is_number and abs(complex(sp.N(difference, 25))) < 1e-12)
    except Exception:
        return False


_MP_CONTEXT = multiprocessing.get_context("forkserver")
_POOL: concurrent.futures.ProcessPoolExecutor | None = None


def symbolically_equal(left: Any, right: Any, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                       use_subprocess: bool = True) -> bool:
    """``simplify(left - right) == 0``, with a timeout that actually fires.

    A hard timeout needs a killable process: SIGALRM is main-thread-only and
    cannot interrupt a C-level SymPy loop, and lm-eval's API backend leaves
    worker threads alive, so the context is ``forkserver`` rather than ``fork``.
    This is the same machinery, and the same reasoning, as
    :func:`llm.core.math.verification._symbolic_confirm`.

    A timeout is reported as "not equal". That is the conservative direction:
    the levels above have already failed, so the alternative is calling two
    expressions equal because comparing them was slow.
    """
    global _POOL
    if left is None or right is None:
        return False
    try:
        payload = (sp.srepr(left), sp.srepr(right))
    except Exception:
        return False

    if not use_subprocess:
        try:
            return _symbolic_worker(*payload)
        except Exception:
            return False

    if _POOL is None:
        _POOL = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=_MP_CONTEXT)
    try:
        return bool(_POOL.submit(_symbolic_worker, *payload).result(timeout=timeout_s))
    except concurrent.futures.TimeoutError:
        # cancel_futures only drops pending work; the worker that timed out is
        # still executing and must be killed, or its manager thread blocks
        # forever joining it -- at the latest, at interpreter exit.
        for process in _POOL._processes.values():
            if process.is_alive():
                process.kill()
        _POOL.shutdown(wait=False, cancel_futures=True)
        _POOL = None
        return False
    except Exception:
        _POOL = None
        return False


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #

def _scalar_level(candidate: str, reference: str, *, timeout_s: float,
                  use_subprocess: bool) -> str | None:
    """The level at which two non-structured answers agree, or ``None``."""
    if candidate.strip() == reference.strip():
        return "exact"
    if compact(candidate) == compact(reference):
        return "normalized"
    bare_candidate, bare_reference = strip_unit_suffix(candidate), strip_unit_suffix(reference)
    if compact(bare_candidate) == compact(bare_reference):
        return "unit_suffix"

    # The originals first. A unit strip is a guess about what the trailing words
    # mean, and letting it feed the parser would put that guess in front of
    # every numeric and symbolic comparison rather than behind them.
    left, right = parse_expression(candidate), parse_expression(reference)
    if left is None or right is None:
        left = left if left is not None else parse_expression(bare_candidate)
        right = right if right is not None else parse_expression(bare_reference)
    left_value, right_value = numeric_value(left), numeric_value(right)
    if left_value is not None and right_value is not None:
        tolerance = _NUMERIC_RTOL * max(1.0, abs(right_value))
        return "numeric" if abs(left_value - right_value) <= tolerance else None
    if left_value is not None or right_value is not None:
        # One side is a number and the other is not: `x+1` is not 4, and a
        # symbolic comparison of the two would only be a slower way of saying so.
        return None
    if symbolically_equal(left, right, timeout_s=timeout_s, use_subprocess=use_subprocess):
        return "symbolic"
    return None


def _unordered_match(left: list[str], right: list[str], *, timeout_s: float,
                     use_subprocess: bool) -> bool:
    """Do two lists name the same multiset of values?

    Greedy one-to-one pairing rather than a sort, because "equal" here is the
    whole scalar ladder and not string equality -- ``0.5`` and ``\\frac{1}{2}``
    have to be able to pair with each other.
    """
    if len(left) != len(right):
        return False
    remaining = list(right)
    for value in left:
        for index, other in enumerate(remaining):
            if _scalar_level(value, other, timeout_s=timeout_s,
                             use_subprocess=use_subprocess):
                remaining.pop(index)
                break
        else:
            return False
    return not remaining


def match_level(candidate: str, reference: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S,
                use_subprocess: bool = True) -> str | None:
    """The level at which a written answer agrees with one reference, or ``None``.

    Structure is decided before value: two answers of *different* kinds -- a
    tuple against a scalar, a union against an interval -- are a mismatch
    outright rather than something to evaluate, because evaluating one of them
    would compare a number against whichever part of the other happened to parse.
    """
    if candidate.strip() == reference.strip():
        return "exact"
    if compact(candidate) == compact(reference):
        return "normalized"

    # A vector against the same vector spelled differently, checked before the
    # matrix splitter below -- that one refuses a pair where only one side is a
    # matrix, which is exactly this case. Narrow on purpose: it needs one side
    # to be unambiguously a matrix, so no interval reading is possible and the
    # delimiter-strict tuple comparison further down keeps owning `(3,4]`.
    candidate_matrix, reference_matrix = matrix_rows(candidate), matrix_rows(reference)
    if (candidate_matrix is None) != (reference_matrix is None):
        left_parts, right_parts = vector_parts(candidate), vector_parts(reference)
        if left_parts is None or right_parts is None or len(left_parts) != len(right_parts):
            return None
        if all(_scalar_level(a, b, timeout_s=timeout_s, use_subprocess=use_subprocess)
               for a, b in zip(left_parts, right_parts)):
            return "vector_elementwise"
        return None

    for splitter, name in ((split_union, "union"), (matrix_rows, "matrix")):
        parts, reference_parts = splitter(candidate), splitter(reference)
        if (parts is None) != (reference_parts is None):
            return None
        if parts is not None:
            if len(parts) != len(reference_parts):
                return None
            if all(match_level(part, other, timeout_s=timeout_s,
                               use_subprocess=use_subprocess)
                   for part, other in zip(parts, reference_parts)):
                return f"{name}_elementwise"
            return None

    # `1 \pm \sqrt{19}` against the two roots written out, and a bare comma list
    # of roots against the same roots in another order. Both are unordered: a
    # bare list is "the solutions are", not a point.
    pm_candidate, pm_reference = expand_pm(candidate), expand_pm(reference)
    if pm_candidate is not None or pm_reference is not None:
        left = pm_candidate or split_bare_list(candidate) or [candidate]
        right = pm_reference or split_bare_list(reference) or [reference]
        return ("pm_elementwise"
                if _unordered_match(left, right, timeout_s=timeout_s,
                                    use_subprocess=use_subprocess) else None)

    bare_candidate, bare_reference = split_bare_list(candidate), split_bare_list(reference)
    if bare_candidate is not None and bare_reference is not None:
        return ("set_elementwise"
                if _unordered_match(bare_candidate, bare_reference, timeout_s=timeout_s,
                                    use_subprocess=use_subprocess) else None)
    if (bare_candidate is None) != (bare_reference is None):
        return None

    tuple_parts, reference_tuple = split_tuple(candidate), split_tuple(reference)
    if (tuple_parts is None) != (reference_tuple is None):
        return None
    if tuple_parts is not None:
        if (tuple_parts[0] != reference_tuple[0] or tuple_parts[1] != reference_tuple[1]
                or len(tuple_parts[2]) != len(reference_tuple[2])):
            return None
        if all(_scalar_level(part, other, timeout_s=timeout_s,
                             use_subprocess=use_subprocess)
               for part, other in zip(tuple_parts[2], reference_tuple[2])):
            return "tuple_elementwise"
        return None

    return _scalar_level(candidate, reference, timeout_s=timeout_s,
                         use_subprocess=use_subprocess)


def answer_matches(candidate: str | None, accepted: object, *,
                   timeout_s: float = _DEFAULT_TIMEOUT_S,
                   use_subprocess: bool = True) -> Match:
    """Compare one written answer against a problem's accepted answers.

    ``accepted`` is a list because the shared schema says so; MATH-500 ships one
    answer per problem, and any of them counting is the same rule every other
    tree applies.
    """
    variants = [str(a) for a in (accepted if isinstance(accepted, (list, tuple))
                                 else [accepted])]
    raw = "" if candidate is None else str(candidate).strip()
    rules: list[str] = []
    normalized = normalize(raw, rules) if raw else ""
    if not raw:
        return Match(False, None, "", "", detail="the answer was empty")

    for variant in variants:
        level = match_level(raw, variant, timeout_s=timeout_s,
                            use_subprocess=use_subprocess)
        if level:
            return Match(True, variant, level, normalized, rules=tuple(rules))
    return Match(False, None, "", normalized, rules=tuple(rules))
