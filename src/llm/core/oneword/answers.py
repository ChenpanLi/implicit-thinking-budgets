"""Turning a located answer segment into k written answers.

The counterpart of :mod:`llm.core.mcq.answers`, and the same design: *outcomes
are first-class*. "the response was empty", "the block held prose", "the block
held three answers where twenty were asked for" are three different failures
with three different fixes, and one parse-error counter hides all of them.

The one structural difference from the multiple-choice parser is the whole
reason this module exists: **answers are split on newlines and on nothing
else.** A multiple-choice answer is one character, so ``BDA`` can be split apart
and ``A, C, B`` can be split on commas. A one-word answer is not one word: 39 of
the pool's answers contain a space (``Addis Ababa``, ``pounds per square inch``,
``Sri Jayawardenepura Kotte``), so splitting on whitespace would shred them and
splitting on commas would be a guess. One line, one answer.

Nothing here repairs. A block of the wrong length is not padded and not trimmed:
k-1 answers cannot be attributed to k questions, and choosing which question was
skipped would describe us rather than the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerLineStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"                # nothing to parse at all
    # Named for the *lines*, so it cannot be confused with the extractor's
    # `no_answer` (which means the block was never located at all).
    NO_ANSWERS = "no_answer_lines"  # text present, but no answer-bearing line in it
    WRONG_LENGTH = "wrong_length"  # answers found, but not as many as asked for


@dataclass
class AnswerLines:
    status: AnswerLineStatus
    answers: list[str] = field(default_factory=list)
    n_expected: int = 0
    n_found: int = 0
    detail: str = ""
    normalized: str = ""
    split_rule: str = ""

    @property
    def ok(self) -> bool:
        return self.status is AnswerLineStatus.OK

    def to_data(self) -> dict[str, Any]:
        """The payload recorded on the parse stage of a trace record."""
        return {
            "reason": self.status.value,
            "answers": list(self.answers),
            "n_expected": self.n_expected,
            "n_found": self.n_found,
            "detail": self.detail,
            "normalized": self.normalized,
            "split_rule": self.split_rule,
        }


# Decoration around one answer line, stripped rather than rejected: a bullet, a
# reproduced question number and a bold marker are not competing answers.
#
# The space after the number is optional in one case: `1.yes` / `1)yes`, a `.` or
# `)` with a *letter* straight after it. DeepSeek-R1-0528-Qwen3-8B writes its
# StrategyQA block that way on up to half of its answers, and requiring the space
# would compare `1 yes` against `yes` and score a right answer wrong. A digit after
# the dot still ends the match, so `3.14` stays a number, and `-`/`:` still need
# the space, so `3-D` is not read as question 3 answering `D`.
_NUMBER_SEPARATOR = r"(?:[.):\-]\s+|[.)](?=[^\W\d_]))"
_ENUMERATION = re.compile(r"^\s*(?:[-–—•*>]\s*)*(?:Q(?:uestion)?\s*)?\d+\s*" + _NUMBER_SEPARATOR,
                          re.IGNORECASE)
_BULLET = re.compile(r"^\s*[-–—•*>]\s+")
_MARKDOWN = re.compile(r"\*\*|__")
# A line that is only scaffolding: a code fence, a horizontal rule, or the
# marker word on a line of its own ("Answer:" above the block).
_SCAFFOLD = re.compile(r"^\s*(?:`{3,}[a-zA-Z]*|[-=_*~]{3,}|#{1,6})\s*$")
_MARKER_ONLY = re.compile(r"^[\s*_#>]*(?:final\s+|the\s+)?answers?[\s*_:.\-]*$", re.IGNORECASE)
_TRIM = " \t\"'`*_.,;:!?‘’“”"

# Chat-template control tokens, which arrive glued to the last answer of a
# generation that ran to its end of turn: `Southwest<|im_end|>`,
# `TG<｜end▁of▁sentence｜>`, `Paris</think>`. They are the generation's
# frame rather than anything the model wrote as an answer -- 17% of the answers
# on record carry one -- and leaving one in place is not a cosmetic problem:
# `normalize_answer` turns `TG<｜end▁of▁sentence｜>` into
# `tg end of sentence`, so a right answer is scored wrong.
#
# What is stripped is *control tokens*, deliberately not "any angle-bracketed
# run". A model that fills the prompt's `<your answer to question 3>`
# placeholder in as `<4>` or `<green>` has copied the example rather than
# answered it; unwrapping those would erase the evidence and score the copy as
# an answer.
_CONTROL_TOKEN = re.compile(
    r"""<(?: [^<>\s]*[|｜][^<>\s]*            # anything holding a pipe: <|im_end|>
           | /[A-Za-z][^<>\s]*                    # a closing tag: </s>, </think>
           | (?: bos|eos|pad|unk|s|eot|eot_id|endoftext|end_of_text
               | end_of_turn|start_of_turn|im_start|im_end|think|thought )
        )>""",
    re.IGNORECASE | re.VERBOSE,
)
# The same token, cut off mid-way by the generation limit: `Southwest<｜end▁of`.
# Only the piped forms are recognised truncated, since `<t` is not evidence of
# anything while `<|` and `<｜` open no answer anybody writes.
_TRUNCATED_CONTROL_TOKEN = re.compile(r"<[|｜][^<>\s]*$")

# An answer block written on one line, the newlines the prompt asked for simply
# missing: `Answer: 1. AK 2. 12 3. Baton Rouge`. The numbering is what makes
# that readable at all -- without it a run of words is not k answers -- so it is
# held to the same standard as the numbered *trailing* block: 1..k, complete and
# in order. The trailing space is required unless a letter follows (`1.AK`), so
# that `3.14` is a number rather than the third answer.
_INLINE_NUMBER = re.compile(r"(?:(?<=^)|(?<=\s))(\d+)\s*[.)](?:\s+|(?=[^\W\d_]))")


def strip_control_tokens(text: str) -> str:
    """``text`` with every chat-template control token removed.

    Replaced by a space rather than by nothing, so a token that arrived *between*
    two words (`Baton<|im_end|>Rouge`) does not silently join them.
    """
    out = _CONTROL_TOKEN.sub(" ", text)
    out = _TRUNCATED_CONTROL_TOKEN.sub(" ", out)
    return out.strip()


def split_inline_numbered(text: str, *, expected: int) -> list[str]:
    """``1. AK 2. 12 3. Baton Rouge`` as ``expected`` answers, else ``[]``.

    Bounded three times over, because this is the one place a line of prose
    could be mistaken for a block: the numbers must run ``1..expected`` with no
    gaps, nothing may precede the first of them, and no field may come out
    empty. Anything else is not this shape, and the later readings get their
    turn rather than a partial one being invented here.
    """
    if expected < 2:
        return []
    matches = list(_INLINE_NUMBER.finditer(text))
    if len(matches) != expected:
        return []
    if [int(match.group(1)) for match in matches] != list(range(1, expected + 1)):
        return []
    if text[:matches[0].start()].strip().strip(_TRIM).strip():
        return []
    answers: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        field = text[match.end():end].strip().strip(_TRIM).strip()
        if not field:
            return []
        answers.append(field)
    return answers


def clean_answer_line(line: str) -> str:
    """One raw line reduced to the answer it carries, or "" if it carries none.

    Only decoration is removed -- and a chat-template control token that the
    generation left glued to the answer counts as decoration: the model did not
    write `<|im_end|>`, the serving stack did. The answer text itself is handed
    on exactly as the model wrote it -- comparing it is
    :func:`llm.core.oneword.normalization.answer_matches`'s job, and doing any
    of that work here would leave the trace showing a string nobody wrote.
    """
    line = strip_control_tokens(line)
    if _SCAFFOLD.match(line) or _MARKER_ONLY.match(line):
        return ""
    out = _ENUMERATION.sub("", line)
    out = _BULLET.sub("", out)
    out = _MARKDOWN.sub("", out)
    return out.strip().strip(_TRIM).strip()


def answer_lines(text: str) -> list[str]:
    """Every answer-bearing line of a segment, in order."""
    return [cleaned for cleaned in (clean_answer_line(line) for line in text.split("\n"))
            if cleaned]


def parse_answer_lines(text: str | None, *, expected: int) -> AnswerLines:
    """Split a response's answer segment into ``expected`` written answers."""
    if expected < 1:
        raise ValueError(f"expected must be at least 1, got {expected}")

    if text is None or not str(text).strip():
        return AnswerLines(AnswerLineStatus.EMPTY, n_expected=expected,
                           detail="nothing in the answer block")

    lines = answer_lines(str(text))
    rule = "lines"

    # The one fallback, and it is bounded twice over: the segment must be a
    # single line, and the split must land on exactly the expected count. No
    # answer in the pool contains a comma, so a comma inside a one-line block
    # really is a separator -- but only when the arithmetic agrees, because a
    # count that does not match is a wrong-length answer and saying so is more
    # useful than inventing an alignment.
    if len(lines) == 1 and expected > 1:
        parts = [part.strip().strip(_TRIM).strip() for part in lines[0].split(",")]
        parts = [part for part in parts if part]
        if len(parts) == expected:
            lines, rule = parts, "comma_fallback"

    if not lines:
        return AnswerLines(AnswerLineStatus.NO_ANSWERS, n_expected=expected,
                           normalized=str(text).strip(), split_rule=rule,
                           detail="no answer-bearing line in the block")

    normalized = "\n".join(lines)
    if len(lines) != expected:
        return AnswerLines(
            AnswerLineStatus.WRONG_LENGTH, answers=lines, n_expected=expected,
            n_found=len(lines), normalized=normalized, split_rule=rule,
            detail=f"{len(lines)} answers for {expected} questions",
        )

    return AnswerLines(AnswerLineStatus.OK, answers=lines, n_expected=expected,
                       n_found=len(lines), normalized=normalized, split_rule=rule,
                       detail="one answer per question")
