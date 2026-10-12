"""Parsing a model's answer to a list of multiple-choice questions.

The composed task asks for one letter per question, in order, as a
comma-separated list on an ``Answer:`` line. Finding that list in a raw response
is :func:`llm.core.mcq.extraction.extract_answer`; turning what it found into k
letters is this module.

The design mirrors :mod:`llm.core.parser.extraction`: *outcomes are
first-class*. "the response was empty", "the box held prose", "the box held
three letters where twenty were asked for" and "the box named option F of four"
are four different failures with four different fixes, and collapsing them into
one parse-error counter would hide which of them happened.

Nothing here invents an answer. A list that is *short* is not padded, a list
with one invalid token is not trimmed to its valid half, and a letter outside
the offered range is not clamped: k-1 answers cannot be attributed to k
positions, and guessing which position was skipped would describe us rather than
the model.

A list that is *too long* is the one exception, and it is a truncation rather
than a repair: the first ``expected`` letters are graded and the rest dropped
(``truncate=True``, the default). The asymmetry is not a compromise, it is the
same rule as everywhere else -- an answer is only ever read from what the model
actually wrote for a position. Position i of a 21-letter list has a letter, and
it is the i-th one, so nothing is guessed by grading it; position 21 of a
19-letter list has none, and nothing can be. What the over-long lists look like
on record is a model that cycled its own list one letter too far, or restated it
with a duplicate, and reading none of it would score a written-out answer as
unparseable. ``n_found`` keeps the original count and ``to_data`` sets
``truncated``, so a run can still be filtered on it afterwards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class AnswerListStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"                    # nothing to parse at all
    NO_LETTERS = "no_letters"          # text present, but no option letter in it
    WRONG_LENGTH = "wrong_length"      # letters found, but not as many as asked for
    INVALID_LETTER = "invalid_letter"  # a token naming an option that was not offered


@dataclass
class AnswerList:
    status: AnswerListStatus
    letters: list[str] = field(default_factory=list)
    n_expected: int = 0
    n_found: int = 0
    detail: str = ""
    normalized: str = ""
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status is AnswerListStatus.OK

    def to_data(self) -> dict[str, Any]:
        """The payload recorded on the parse stage of a trace record."""
        return {
            "reason": self.status.value,
            "letters": list(self.letters),
            "n_expected": self.n_expected,
            "n_found": self.n_found,
            "truncated": self.truncated,
            "detail": self.detail,
            "normalized": self.normalized,
        }


# LaTeX decoration a model puts around a letter when it is already writing
# inside \boxed{}. Stripped rather than rejected: `\text{B}` is a correctly
# formatted answer wearing a hat, and scoring it wrong would measure our parser.
#
# *Any* control sequence is dropped, not a list of the expected ones. Matching
# a list by name is how `\textbf{C}` becomes the token "BF" -- the alternation
# matches `\text` and leaves `bf` behind -- and no answer letter is ever a
# LaTeX command, so there is nothing to lose by being general. The second arm
# takes the punctuation-named ones and LaTeX's escaped space: `\,`, `\;`,
# `\!`, `\\` and `\ ` alike, plus a trailing lone backslash.
_LATEX = re.compile(r"\\[a-zA-Z]+\s*|\\[\s\S]?")
_SPACING = re.compile(r"[~$]")
# Per-item numbering the model reproduces from the prompt: "1. B", "2) C", "Q3: A".
_NUMBERING = re.compile(r"(?:^|(?<=[\s,;]))(?:Q(?:uestion)?\s*)?\d+\s*[.):\-]\s*", re.IGNORECASE)
# Everything that can legitimately separate two answers.
_SEPARATORS = re.compile(r"[,;/|]|\band\b|\s+", re.IGNORECASE)
# Punctuation and delimiters that can wrap a single token: "(B)", "[B]", "B.",
# and the bullets a model puts in front of a list it is proud of: "- B", "* B",
# "\u2022 B". Stripped rather than rejected, for the same reason the LaTeX
# decoration is: a bullet is not a competing answer, and a token that reduces to
# nothing is dropped by `_tokens`.
_TRIM = "()[]{}<>.:;'\"`*_-\u2013\u2014\u2022 \t"


def _normalize(text: str) -> str:
    """Reduce LaTeX decoration and prompt echo to bare, separated tokens."""
    out = _LATEX.sub(" ", text)
    out = _SPACING.sub(" ", out)
    # Braces survive the wrapper strip (``\text{B}`` -> `` {B}``); drop them so
    # the token trimmer does not have to.
    out = out.replace("{", " ").replace("}", " ")
    out = _NUMBERING.sub(" ", out)
    return " ".join(out.split())


def _tokens(normalized: str) -> list[str]:
    return [t for t in (part.strip(_TRIM) for part in _SEPARATORS.split(normalized)) if t]


def parse_answer_list(
    text: str | None,
    *,
    expected: int,
    n_choices: int = 4,
    truncate: bool = True,
) -> AnswerList:
    """Turn a response's answer segment into ``expected`` option letters.

    ``n_choices`` bounds which letters are legal, so naming option F of four is
    an error rather than a silently accepted answer that can never match.

    With ``truncate`` (the default) a segment naming *more* than ``expected``
    letters is graded on its first ``expected`` -- see the module docstring for
    why that is a reading and not a repair. ``truncate=False`` reports it as a
    WRONG_LENGTH instead, and is what a caller that wants every letter a segment
    names passes: :func:`llm.core.mcq.extraction._letters_only` reads whole
    lines through this function and must get the whole line back.
    """
    if expected < 1:
        raise ValueError(f"expected must be at least 1, got {expected}")
    valid = set(LETTERS[:n_choices])

    if text is None or not str(text).strip():
        return AnswerList(AnswerListStatus.EMPTY, n_expected=expected,
                          detail="nothing inside the box")

    normalized = _normalize(str(text))
    tokens = _tokens(normalized)

    # A separator-free run ("BDA") is a legitimate way to answer. It is split
    # only when it is the *whole* answer and every one of its characters is an
    # option that was actually offered: that second condition is what keeps
    # prose out. "the" would otherwise split into T, H, E, and "answer" into
    # six letters, turning a formatting failure into a wrong-length one.
    # Whether the run's length matches is deliberately not part of the test --
    # a short run is a wrong-length answer, which is a more useful diagnosis
    # than "no letters", and reporting its length invents nothing.
    if len(tokens) == 1:
        single = tokens[0].upper()
        if single and all(c in valid for c in single):
            tokens = list(single)

    letters = [t.upper() for t in tokens]

    if not letters:
        return AnswerList(AnswerListStatus.NO_LETTERS, n_expected=expected,
                          normalized=normalized, detail="no option letters in the box")

    bad = [t for t in letters if t not in valid]
    if bad:
        named = ", ".join(repr(t) for t in bad[:5])
        # Prose in the box and a wrong option letter are different failures with
        # different fixes -- the first is a formatting problem, the second is an
        # answer to a question nobody asked -- so they are not one bucket. The
        # split is on shape, not on count: a lone "F" against four options is a
        # named option, while "the answer is" is not, however many tokens it has.
        letter_shaped = [t for t in bad if len(t) == 1 and t in LETTERS]
        if not letter_shaped and not any(t in valid for t in letters):
            return AnswerList(
                AnswerListStatus.NO_LETTERS, n_expected=expected,
                n_found=0, normalized=normalized,
                detail=f"no option letter in the box; found {named}",
            )
        offered = f"{LETTERS[0]}-{LETTERS[n_choices - 1]}"
        return AnswerList(
            AnswerListStatus.INVALID_LETTER, letters=letters, n_expected=expected,
            n_found=len(letters), normalized=normalized,
            detail=f"not one of the offered options ({offered}): {named}",
        )

    if len(letters) > expected and truncate:
        return AnswerList(
            AnswerListStatus.OK, letters=letters[:expected], n_expected=expected,
            n_found=len(letters), normalized=normalized, truncated=True,
            detail=f"{len(letters)} answers for {expected} questions; "
                   f"graded on the first {expected}",
        )

    if len(letters) != expected:
        return AnswerList(
            AnswerListStatus.WRONG_LENGTH, letters=letters, n_expected=expected,
            n_found=len(letters), normalized=normalized,
            detail=f"{len(letters)} answers for {expected} questions",
        )

    return AnswerList(AnswerListStatus.OK, letters=letters, n_expected=expected,
                      n_found=len(letters), normalized=normalized,
                      detail="one letter per question")
