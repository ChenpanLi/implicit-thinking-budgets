r"""Deciding whether a written answer is the number the reference names (GSM8K).

Every GSM8K answer is an integer: the dataset states it after ``####`` and
nothing else appears there. What a *model* writes for that integer is another
matter -- ``$100``, ``100 dollars``, ``\boxed{100}``, ``1,623``, ``18.00``, and
occasionally ``eighteen`` -- and scoring any of those wrong would measure our
comparison rather than the model. So both sides go through the same **ordered,
named** rule list and are compared as numbers afterwards.

The stance is :mod:`llm.core.oneword.normalization`'s: each rule is about *how a
number is written*, never about what it means. Nothing here rescues an answer
that names a different number, and nothing here reads "a number somewhere in the
text" -- the whole string has to reduce to one literal, because a reader that
takes any number would read ``\mathbf{K}(0.25)`` as ``0.25``. A line of prose
with a number in it is not an answer; the
extractor already refused to call it one.

The last rule is a deliberate reuse rather than a copy:
:func:`llm.core.oneword.normalization.normalize_answer` already turns
``eighteen`` and ``twenty-one`` into digits, and running both sides through it
costs nothing and cannot make a wrong answer right -- both sides are
transformed the same way.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .comparison import Match


_UNICODE_MINUS = {"\u2212": "-", "\u2013": "-", "\u2014": "-"}
_SPACING = re.compile(
    r"\\(?:left|right|displaystyle|textstyle|quad|qquad|,|;|:|!|\s)|[~$]"
)
_TEXT_GROUP = re.compile(r"\\(?:text|mbox|mathrm|textbf|textit|mathbf)\s*\{([^{}]*)\}")
_CURRENCY = re.compile(r"[$\u20ac\u00a3\u00a5]|\\\$|\bUSD\b|\bEUR\b", re.IGNORECASE)
_PERCENT = re.compile(r"\\%|%")
# A thousands separator, and only that: a comma or a thin space between a digit
# and exactly three more digits. `1,623` is one number; `1,-2` is not a number
# at all and falls through unchanged.
_THOUSANDS = re.compile(r"(?<=\d)[,\u202f\u2009](?=\d{3}(?!\d))")
# A trailing unit, which GSM8K questions invite ("How many dollars...?"): a run
# of words after the number. Dropped rather than rejected -- the prompt asked
# for a bare number, and a model that adds "eggs" has still answered.
_UNIT_TAIL = re.compile(r"(?<=[\d)}\]])\s*[A-Za-z][A-Za-z\s.\-]*$")
_LEADING_WORDS = re.compile(r"^(?:the\s+|a\s+|an\s+|answer\s*[:=]?\s*)+", re.IGNORECASE)
# A reproduced problem number, which the answer-line cleaner normally takes off
# first -- this is for a caller that hands over a raw line. The trailing space is
# what keeps `3.14` a number rather than the third answer.
_ENUMERATION = re.compile(r"^\s*(?:[-\u2013\u2014\u2022*>]\s*)*"
                          r"(?:Q(?:uestion)?|P(?:roblem)?\s*)?\d+\s*[.):\-]\s+")
_TRAILING_PUNCT = re.compile(r"[.,;:!?\s]+$")
_LITERAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")


def reads_as_integer(text: str | None) -> bool:
    """Whether ``text``, as written, names an integer.

    The answer alphabet of the two trees that compare numbers: GSM8K states its
    answer after ``####`` and AIME's is an integer from 0 to 999, so no accepted
    answer in either pool is anything else. That is what lets
    :func:`llm.core.shortmath.answers.strip_glued_index` read ``1.18`` as the
    first answer being 18 -- 1.18 is not an answer these trees can have -- while
    leaving ``1.00`` alone, which is one.
    """
    value, _rules = read_number(text)
    return value is not None and value == value.to_integral_value()


def read_number(text: str | None) -> tuple[Decimal | None, list[str]]:
    """The number a written answer names, plus the rules that got there.

    ``None`` when the whole string does not reduce to one numeric literal. That
    is a refusal, not a zero: the caller reports it as a mismatch with a reason
    rather than inventing a value.
    """
    from llm.core.parser.math_parser import strip_bpe_sentinels

    from .answers import strip_control_tokens, unwrap_boxed

    if text is None:
        return None, []
    rules: list[str] = []
    out, restored = strip_bpe_sentinels(str(text).strip())
    if restored:
        rules.append("restore_bpe_whitespace")
    out = strip_control_tokens(out)

    def rule(name: str, value: str) -> str:
        nonlocal out
        if value != out:
            rules.append(name)
        return value

    out = rule("strip_enumeration", _ENUMERATION.sub("", out))
    out = rule("unwrap_boxed", unwrap_boxed(out).strip())
    for bad, good in _UNICODE_MINUS.items():
        out = out.replace(bad, good)
    out = rule("strip_latex_decoration",
               _SPACING.sub(" ", _TEXT_GROUP.sub(r" \1 ", out)))
    out = rule("strip_currency", _CURRENCY.sub(" ", out))
    out = rule("strip_percent", _PERCENT.sub(" ", out))
    out = " ".join(out.split())
    out = _LEADING_WORDS.sub("", out).strip()
    out = rule("strip_thousands_separator", _THOUSANDS.sub("", out))
    out = rule("strip_unit_words", _UNIT_TAIL.sub("", out).strip())
    out = rule("strip_trailing_punctuation", _TRAILING_PUNCT.sub("", out).strip())
    compact = out.replace(" ", "")

    if _LITERAL.fullmatch(compact):
        try:
            return Decimal(compact), rules
        except InvalidOperation:      # unreachable for a matched literal
            return None, rules

    # The words. Both sides go through this, so it cannot make a wrong answer
    # right -- it can only stop `eighteen` and `18` from being two answers.
    from llm.core.oneword.normalization import normalize_answer

    spelled = normalize_answer(out)
    if spelled and _LITERAL.fullmatch(spelled):
        rules.append("words_to_number")
        try:
            return Decimal(spelled), rules
        except InvalidOperation:
            return None, rules
    return None, rules


def answer_matches(candidate: str | None, accepted: object) -> Match:
    """Compare one written answer against a problem's accepted answers.

    ``accepted`` is a list because the shared schema says so; GSM8K writes one
    entry per problem, and any of them counting is the same rule the one-word
    tree applies.
    """
    variants = [str(a) for a in (accepted if isinstance(accepted, (list, tuple))
                                 else [accepted])]
    raw = "" if candidate is None else str(candidate).strip()
    for variant in variants:
        if raw and raw == variant.strip():
            return Match(True, variant, "exact", raw)

    value, rules = read_number(raw)
    if value is None:
        return Match(False, None, "", raw, rules=tuple(rules),
                     detail="the answer does not reduce to a single number")
    for variant in variants:
        reference, _ = read_number(variant)
        if reference is not None and value == reference:
            return Match(True, variant, "numeric", str(value), rules=tuple(rules))
    return Match(False, None, "", str(value), rules=tuple(rules))
