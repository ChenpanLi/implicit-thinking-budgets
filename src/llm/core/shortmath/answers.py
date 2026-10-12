"""Turning a located answer segment into k written answers.

The counterpart of :mod:`llm.core.oneword.answers`, and the same design:
*outcomes are first-class*. "the response was empty", "the block held prose",
"the block held three answers where twenty were asked for" are three different
failures with three different fixes, and one parse-error counter hides all of
them. Nothing here repairs: a block of the wrong length is not padded and not
trimmed, because k-1 answers cannot be attributed to k problems and choosing
which problem was skipped would describe us rather than the model.

Three rules differ from the one-word module, and every one of them is forced by
what a mathematical answer may contain:

* **There is no comma fallback.** The one-word parser may, for a single-line
  block, split on commas, because no answer in that pool contains one. Here
  ``(6,31,-1)``, ``1,-2`` and ``3, 5, 7`` are single MATH-500 answers, and
  ``10{,}080`` is one number. Splitting on commas would shred them.
* **Lines inside an unclosed group are joined.** A model that writes
  ``\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}`` across three lines wrote *one*
  answer, and reading three would score a right answer UNPARSEABLE for a reason
  that is ours. :func:`answer_lines` therefore joins a line to the next while a
  LaTeX environment or a brace group is still open. The join is bounded (see
  ``_MAX_JOIN``) so a response that never closes a brace cannot swallow the
  whole block.
* **``\\boxed{...}`` is unwrapped.** Both trees ask for boxed answers (MATH-500
  by the prompt, GSM8K because models write them anyway), and the box is a
  wrapper around the answer rather than part of it. The unwrap is balanced, so
  ``\\boxed{\\frac{1}{2}}`` survives.

The one repair in this module is :func:`strip_glued_index`, and it is not a
repair of an *answer* -- it is a reading of the prompt's own scaffolding. See
its docstring for the three conditions that have to hold at once before it
fires.
"""
from __future__ import annotations

import re
from collections.abc import Callable
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
# reproduced problem number and a bold marker are not competing answers.
_ENUMERATION = re.compile(r"^\s*(?:[-\u2013\u2014\u2022*>]\s*)*"
                          r"(?:Q(?:uestion)?|P(?:roblem)?\s*)?\d+\s*[.):\-]\s+",
                          re.IGNORECASE)
_BULLET = re.compile(r"^\s*[-\u2013\u2014\u2022*>]\s+")
_MARKDOWN = re.compile(r"\*\*|__")
# A line that is only scaffolding: a code fence, a horizontal rule, or the
# marker word on a line of its own ("Answer:" above the block).
_SCAFFOLD = re.compile(r"^\s*(?:`{3,}[a-zA-Z]*|[-=_*~]{3,}|#{1,6})\s*$")
_MARKER_ONLY = re.compile(r"^[\s*_#>]*(?:final\s+|the\s+)?answers?[\s*_:.\-]*$", re.IGNORECASE)
# Deliberately short of the one-word module's list: a comma, a period and a
# closing bracket all carry meaning in a mathematical answer. `1,-2` must keep
# its comma and `(3,4]` its bracket; only quotes, markdown and whitespace go.
_TRIM = " \t\"'`*_"
# A trailing sentence period, which `5.` and `\frac12.` both end in and neither
# means. Only a period preceded by something that is not a digit, or a period at
# the very end after a closing brace/bracket -- `3.14` keeps its point.
_TRAILING_PERIOD = re.compile(r"(?<=[^\s.])\.$")

# Chat-template control tokens, which arrive glued to the last answer of a
# generation that ran to its end of turn: `18<|im_end|>`,
# `\boxed{6}<｜end▁of▁sentence｜>`. They are the generation's frame rather than
# anything the model wrote, and leaving one in place turns a right answer into
# an unparseable one. Copied from `llm.core.oneword.answers` rather than
# imported: that module belongs to another tree's grading, and a change made
# there for a one-word reason must not silently move a MATH-500 score.
_CONTROL_TOKEN = re.compile(
    r"""<(?: [^<>\s]*[|\uff5c][^<>\s]*            # anything holding a pipe: <|im_end|>
           | /[A-Za-z][^<>\s]*                    # a closing tag: </s>, </think>
           | (?: bos|eos|pad|unk|s|eot|eot_id|endoftext|end_of_text
               | end_of_turn|start_of_turn|im_start|im_end|think|thought )
        )>""",
    re.IGNORECASE | re.VERBOSE,
)
# The same token, cut off mid-way by the generation limit: `18<｜end▁of`.
_TRUNCATED_CONTROL_TOKEN = re.compile(r"<[|\uff5c][^<>\s]*$")

# An answer block written on one line, the newlines the prompt asked for simply
# missing: `Answer: 1. 100 2. 30 3. 623`. The numbering is what makes that
# readable at all, so it is held to the same standard as the numbered trailing
# block: 1..k, complete and in order. The trailing space is required so that
# `3.14` is a number rather than the third answer.
_INLINE_NUMBER = re.compile(r"(?:(?<=^)|(?<=\s))(\d+)\s*[.)]\s+")

# The same enumeration with the space dropped: `1.18` where the prompt's
# template showed `1. <your answer>`. Read literally it is the decimal 1.18, and
# that is why `_ENUMERATION` insists on the space -- but a model that glues the
# two together has still numbered its answers, and scoring the whole block wrong
# measures our reading rather than its arithmetic. `strip_glued_index` is what
# resolves the ambiguity, and only when the domain says it can be resolved.
_GLUED_INDEX = re.compile(r"^(\d+)\.(\S.*)$")

_BOXED = re.compile(r"\\boxed\s*\{")
# How many following lines one unclosed group may swallow. A pmatrix answer runs
# to a handful of lines; a response that opened a brace and never closed it is a
# truncation, and joining the rest of the block into it would turn a diagnosable
# wrong-length into an unreadable single answer.
_MAX_JOIN = 8


def strip_control_tokens(text: str) -> str:
    """``text`` with every chat-template control token removed.

    Replaced by a space rather than by nothing, so a token that arrived *between*
    two tokens of an expression does not silently join them.
    """
    out = _CONTROL_TOKEN.sub(" ", text)
    out = _TRUNCATED_CONTROL_TOKEN.sub(" ", out)
    return out.strip()


def _depth(text: str) -> int:
    """How many brace groups and LaTeX environments this text leaves open.

    Backslash-escaped braces do not count, and neither does the ``\\\\`` row
    separator, which is two escapes rather than an escaped brace.
    """
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2                      # an escape consumes the next character
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    depth += len(re.findall(r"\\begin\s*\{", text))
    depth -= len(re.findall(r"\\end\s*\{", text))
    return depth


def unwrap_boxed(text: str) -> str:
    r"""``\boxed{x}`` -> ``x``, balanced, or ``text`` unchanged.

    Only when the box is the whole answer. ``\boxed{1}+\boxed{2}`` is not a box
    around an answer, it is two boxes, and rewriting it would invent an answer
    nobody wrote.
    """
    match = _BOXED.search(text)
    if match is None or text[:match.start()].strip().strip(_TRIM):
        return text
    depth = 0
    for index in range(match.end() - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                if text[index + 1:].strip().strip(_TRIM).strip("$."):
                    return text          # something followed the box: not a bare box
                return text[match.end():index]
    return text                          # unbalanced: a truncated generation


def clean_answer_line(line: str) -> str:
    """One raw line reduced to the answer it carries, or "" if it carries none.

    Only decoration is removed -- the enumeration, a bullet, markdown, a chat
    control token the serving stack glued on, the ``\\boxed{}`` wrapper and a
    sentence-ending period. The answer text itself is handed on exactly as the
    model wrote it; deciding whether it *means* the reference is the
    comparator's job, and doing any of that work here would leave the trace
    showing a string nobody wrote.
    """
    line = strip_control_tokens(line)
    if _SCAFFOLD.match(line) or _MARKER_ONLY.match(line):
        return ""
    out = _ENUMERATION.sub("", line)
    out = _BULLET.sub("", out)
    out = _MARKDOWN.sub("", out)
    out = out.strip().strip(_TRIM).strip()
    out = unwrap_boxed(out).strip().strip(_TRIM).strip()
    return _TRAILING_PERIOD.sub("", out).strip()


def join_open_groups(lines: list[str]) -> list[str]:
    """Join each line to the next while it leaves a group open.

    A ``\\begin{pmatrix}`` answer written across lines is one answer. The join
    is bounded by ``_MAX_JOIN``, so an unbalanced brace in a truncated response
    cannot swallow the rest of the block.
    """
    joined: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        consumed = 0
        while (_depth(current) > 0 and index + 1 < len(lines) and consumed < _MAX_JOIN):
            index += 1
            consumed += 1
            current = f"{current} {lines[index].strip()}"
        joined.append(current)
        index += 1
    return joined


def answer_lines(text: str) -> list[str]:
    """Every answer-bearing line of a segment, in order."""
    raw = join_open_groups(str(text).split("\n"))
    return [cleaned for cleaned in (clean_answer_line(line) for line in raw) if cleaned]


def split_inline_numbered(text: str, *, expected: int) -> list[str]:
    """``1. 100 2. 30 3. 623`` as ``expected`` answers, else ``[]``.

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
        field_text = clean_answer_line(text[match.end():end])
        if not field_text:
            return []
        answers.append(field_text)
    return answers


def strip_glued_index(lines: list[str], *,
                      readable: Callable[[str], bool]) -> list[str] | None:
    """``["1.18"]`` -> ``["18"]`` when that is the only reading, else ``None``.

    The prompt numbers the answer lines at every k -- even a k=1 block is shown
    as ``1. <your answer>`` -- so a leading ``i.`` is the template's scaffolding
    and not part of the answer. :func:`clean_answer_line` already takes it off
    when the model leaves the space in. Some models do not: DeepSeek-R1-0528-
    Qwen3-8B writes ``1.18``, and read literally that is the decimal 1.18.

    Three conditions have to hold at once before the index is read as an index,
    and together they make this incapable of changing a line that would already
    be read correctly:

    * **Every** line carries **its own** position: line *i* starts ``i.``, from
      1 to the length of the block, with something non-empty after the dot. A
      block numbered any other way is not this shape and is left alone.
    * **No** line is readable as an answer as written -- ``readable`` is the
      domain's own alphabet, asked line by line. GSM8K and AIME answers are
      integers, so ``1.18`` cannot be one and the glued reading is the only one
      left; a line like ``1.00`` *is* an integer, so the block stays as written
      and nothing is repaired. MATH-500 passes no predicate at all (see
      :func:`parse_answer_lines`), because a decimal is a legal answer there and
      the two readings genuinely cannot be told apart.
    * The block is read as a whole: either every index comes off or none does,
      because a block that is numbered is numbered throughout, and mixing the
      two readings would produce answers nobody wrote.

    Only the newline-split block is treated this way. ``split_inline_numbered``
    is left alone deliberately: with no line breaks to corroborate them, glued
    numbers in a run of prose have nothing to distinguish them from a decimal.
    """
    if not lines:
        return None
    matches = [_GLUED_INDEX.match(line) for line in lines]
    if not all(matches):
        return None
    if [int(m.group(1)) for m in matches] != list(range(1, len(lines) + 1)):
        return None
    if any(readable(line) for line in lines):
        return None
    stripped = [clean_answer_line(m.group(2)) for m in matches]
    if not all(stripped):
        return None
    return stripped


def parse_answer_lines(text: str | None, *, expected: int,
                       readable: Callable[[str], bool] | None = None) -> AnswerLines:
    """Split a response's answer segment into ``expected`` written answers.

    ``readable`` says whether one written line could be an answer in this
    domain at all, and is what lets :func:`strip_glued_index` resolve
    ``1.18``. A tree whose answers can look like a glued index -- MATH-500,
    where ``1.5`` is a legal answer to problem 1 -- passes nothing, and the
    block is read as written.
    """
    if expected < 1:
        raise ValueError(f"expected must be at least 1, got {expected}")

    if text is None or not str(text).strip():
        return AnswerLines(AnswerLineStatus.EMPTY, n_expected=expected,
                           detail="nothing in the answer block")

    lines = answer_lines(str(text))
    rule = "lines"

    if readable is not None and len(lines) == expected:
        unglued = strip_glued_index(lines, readable=readable)
        if unglued is not None:
            lines = unglued
            rule = "lines+glued_index"

    if not lines:
        return AnswerLines(AnswerLineStatus.NO_ANSWERS, n_expected=expected,
                           normalized=str(text).strip(), split_rule=rule,
                           detail="no answer-bearing line in the block")

    normalized = "\n".join(lines)
    if len(lines) != expected:
        return AnswerLines(
            AnswerLineStatus.WRONG_LENGTH, answers=lines, n_expected=expected,
            n_found=len(lines), normalized=normalized, split_rule=rule,
            detail=f"{len(lines)} answers for {expected} problems",
        )

    return AnswerLines(AnswerLineStatus.OK, answers=lines, n_expected=expected,
                       n_found=len(lines), normalized=normalized, split_rule=rule,
                       detail="one answer per problem")
