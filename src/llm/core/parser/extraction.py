"""Answer extraction from raw model responses.

Separated from :mod:`pattern_parser` (which is a generic brace scanner) so that
extraction *outcomes* are first-class: "the API returned nothing", "the model
never wrote a box", and "the box was truncated mid-brace" are kept apart rather
than collapsed into one undifferentiated parse-error counter, which would hide
whether a failure was a client timeout or bad model output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .pattern_parser import PatternParseError, parse_command_braces

# Tolerant of the whitespace mangling seen in byte-BPE-corrupted outputs, where
# "FINAL ANSWER" arrives as "FINALĠANSWER" or with the space removed entirely.
_FINAL_ANSWER = re.compile(r"FINAL\W{0,4}ANSWER", re.IGNORECASE)


class ExtractionStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"              # API returned nothing (timeout, retries exhausted)
    NO_BOXED = "no_boxed"        # text produced, but no \boxed{} anywhere
    UNBALANCED = "unbalanced"    # \boxed{ opened and never closed (truncation)
    # Produced only by llm.core.mcq.extraction, whose contract is an answer
    # *line* rather than a box: text was produced, but nothing in it -- no
    # marker, no box, no option letter at the end -- named an option. It lives
    # here rather than in a second enum so the trace writer, the sentinel tables
    # and every `frac_<reason>` metric keep switching on one type.
    NO_ANSWER = "no_answer"


@dataclass
class ExtractResult:
    status: ExtractionStatus
    value: str = ""
    n_matches: int = 0
    chosen_index: int | None = None
    detail: str = ""


def extract_boxed(text: str | None, *, prefer_final_answer: bool = False) -> ExtractResult:
    """Pull the answer out of a raw model response.

    By default returns the **last** ``\\boxed{}``, which is what makes this
    robust to reasoning traces that box intermediate results.

    With ``prefer_final_answer``, the first box appearing after a
    ``FINAL ANSWER`` marker wins, falling back to the last box when there is no
    marker. Running both strategies as separate lm-eval filter ensembles scores
    them side by side, which is how the cost of template-bleed is measured.
    """
    if text is None or not str(text).strip():
        return ExtractResult(ExtractionStatus.EMPTY, detail="response was empty or None")

    text = str(text)

    if prefer_final_answer:
        marker = None
        for marker in _FINAL_ANSWER.finditer(text):
            pass  # keep the last marker; models often restate the instruction first
        if marker is not None:
            tail = text[marker.end():]
            try:
                boxes = parse_command_braces(tail, command="boxed")
            except PatternParseError:
                boxes = []
            if boxes:
                return ExtractResult(ExtractionStatus.OK, boxes[0].strip(),
                                     n_matches=len(boxes), chosen_index=0,
                                     detail="first box after FINAL ANSWER")

    try:
        boxes = parse_command_braces(text, command="boxed")
    except PatternParseError as exc:
        return ExtractResult(ExtractionStatus.UNBALANCED, detail=str(exc))

    if not boxes:
        if "\\boxed" in text:
            # The command is present but never opened a brace group.
            return ExtractResult(ExtractionStatus.UNBALANCED,
                                 detail="\\boxed present but no brace group followed")
        return ExtractResult(ExtractionStatus.NO_BOXED, detail="no \\boxed{} in response")

    return ExtractResult(ExtractionStatus.OK, boxes[-1].strip(),
                         n_matches=len(boxes), chosen_index=len(boxes) - 1,
                         detail="last box")
