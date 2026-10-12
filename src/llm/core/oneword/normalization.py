"""Deciding whether a written answer means the same thing as the reference.

The multiple-choice experiment does not need this module at all: there, an
answer is a capital letter and equality is equality. Here an answer is a word or
a number that a model wrote in its own house style, and the reference is one
string from the pool (``datasets/onewordqa/atomic.jsonl``). ``Sao Tome`` and
``São Tomé`` are the same city; ``four`` and ``4`` are the same number of legs; ``+213`` and
``213`` are the same dialling code. Scoring any of those wrong would measure our
comparison rather than the model.

So both sides go through the same **ordered, named** rule list and are compared
after it. The rules are named because a comparison nobody can read back is a
comparison nobody can audit: :func:`answer_matches` records *which* accepted
variant matched and *at what level* (``exact`` before normalisation, or
``normalized`` after it), and that lands on the trace record.

What this module deliberately does **not** do is alias. ``Mammalia`` is not
accepted for ``Mammal`` and ``Saint John's`` is not accepted for ``St. John's``:
the accepted set is whatever ``correct_answers`` holds, and widening it is an
edit to the dataset, where it is visible and reviewable, not a rule in here where
it would silently loosen every future question of that shape. Each rule below is
about *how a string is written*, never about what it means.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from llm.core.parser.math_parser import strip_bpe_sentinels

# Any control sequence, not a list of the expected ones -- see
# `llm.core.mcq.answers`, where matching `\text` by name would turn `\textbf{C}`
# into the token "BF". No answer is ever a LaTeX command, so nothing is lost.
_LATEX = re.compile(r"\\[a-zA-Z]+\s*|\\[\s\S]?")
_MATH_DELIMS = re.compile(r"[~$]")
_MARKDOWN = re.compile(r"\*\*|__|[`*_]")
# Per-item numbering and bullets a model reproduces from the prompt: "1. Paris",
# "2) 4", "Q3: yellow", "- Paris".
_ENUMERATION = re.compile(r"^\s*(?:[-\u2013\u2014\u2022>]\s*)*(?:Q(?:uestion)?\s*)?\d+\s*[.):\-]\s+",
                          re.IGNORECASE)
_BULLET = re.compile(r"^\s*[-\u2013\u2014\u2022*>]\s+")
# Curly quotes and dashes that NFKD leaves alone.
_UNICODE_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
}
_WRAPPING = "()[]{}<>\"'`.,;:!?*_ \t"
# Deleted rather than turned into a space, unlike the rest of the punctuation:
# "St. Johns" and "St. John's" are one name written two ways, and a space would
# make them two answers ("st john s" vs "st johns").
_APOSTROPHE = re.compile(r"['\u2019]")
_NUMERIC = re.compile(r"[+-]?\d+(?:\.\d+)?")
_ORDINAL_DIGITS = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
_ARTICLE = re.compile(r"^(?:the|a|an)\s+")
_PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000}
# The ordinal forms, because `planet_position` asks for a position "expressed as
# a number" and a model will happily write "first".
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
    "sixtieth": 60, "seventieth": 70, "eightieth": 80, "ninetieth": 90,
    "hundredth": 100, "thousandth": 1000,
}


def _canonical_number(text: str) -> str | None:
    """``+213`` -> ``213``, ``4.0`` -> ``4``, ``1,000`` -> ``1000``, ``3rd`` -> ``3``.

    Fires only when the *whole* string is one numeric literal. "a number
    somewhere in the text" would read `K(0.25)` as `0.25`, and the same
    restraint applies to a one-word answer.
    """
    compact = text.replace(",", "").replace(" ", "").replace("_", "")
    ordinal = _ORDINAL_DIGITS.match(compact)
    if ordinal:
        compact = ordinal.group(1)
    if not _NUMERIC.fullmatch(compact):
        return None
    negative = compact.startswith("-")
    body = compact.lstrip("+-")
    if "." in body:
        whole, _, fraction = body.partition(".")
        fraction = fraction.rstrip("0")
        body = f"{whole}.{fraction}" if fraction else whole
    body = body.lstrip("0") or "0"
    return f"-{body}" if negative and body != "0" else body


def _words_to_number(text: str) -> str | None:
    """``twenty-one`` -> ``21``, ``first`` -> ``1``. The whole string or nothing.

    Both sides of a comparison go through this, so the conversion cannot make a
    wrong answer right: it can only stop ``four`` and ``4`` from being two
    different answers to "how many legs".
    """
    tokens = [t for t in re.split(r"[\s-]+", text) if t and t != "and"]
    if not tokens or not all(t in _UNITS or t in _TENS or t in _SCALES or t in _ORDINALS
                             for t in tokens):
        return None
    total = 0
    current = 0
    seen = False
    for token in tokens:
        if token in _SCALES:
            scale = _SCALES[token]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            seen = True
            continue
        value = _UNITS.get(token, _TENS.get(token, _ORDINALS.get(token)))
        if value is None:                        # unreachable given the guard above
            return None
        current += value
        seen = True
    return str(total + current) if seen else None


def normalize_answer(text: str | None) -> str:
    """Reduce a written answer to the form both sides are compared in."""
    if text is None:
        return ""
    out, _ = strip_bpe_sentinels(str(text))
    out = _LATEX.sub(" ", out)
    out = _MATH_DELIMS.sub(" ", out).replace("{", " ").replace("}", " ")
    out = _MARKDOWN.sub("", out)
    out = _ENUMERATION.sub("", out)
    out = _BULLET.sub("", out)
    out = unicodedata.normalize("NFKD", out)
    out = "".join(ch for ch in out if not unicodedata.combining(ch))
    for source, target in _UNICODE_FOLD.items():
        out = out.replace(source, target)
    out = " ".join(out.split()).strip(_WRAPPING)
    out = out.casefold()
    if not out:
        return ""

    number = _canonical_number(out)
    if number is not None:
        return number

    out = _ARTICLE.sub("", out).strip()
    words = _words_to_number(out)
    if words is not None:
        return words

    out = _APOSTROPHE.sub("", out)
    out = _PUNCTUATION.sub(" ", out)
    return " ".join(out.split())


@dataclass(frozen=True)
class Match:
    """Why a written answer did or did not count as the reference."""

    matched: bool
    variant: str | None      # the accepted answer it matched, if any
    level: str               # "exact" | "normalized" | ""
    normalized: str          # the candidate, normalised

    def to_data(self) -> dict[str, object]:
        return {"matched": self.matched, "variant": self.variant,
                "level": self.level, "normalized": self.normalized}


def answer_matches(candidate: str | None, accepted: object) -> Match:
    """Compare one written answer against a question's accepted variants.

    ``accepted`` is that question's ``correct_answers`` -- a list, because four
    questions in the pool genuinely have two right answers (Bolivia has two
    capitals, an ace is worth 1 or 11). Any one of them counts.
    """
    variants = [str(a) for a in (accepted if isinstance(accepted, (list, tuple)) else [accepted])]
    raw = "" if candidate is None else str(candidate).strip()
    for variant in variants:
        if raw and raw == variant.strip():
            return Match(True, variant, "exact", normalize_answer(raw))

    normalized = normalize_answer(raw)
    if normalized:
        for variant in variants:
            if normalized == normalize_answer(variant):
                return Match(True, variant, "normalized", normalized)
    return Match(False, None, "", normalized)
