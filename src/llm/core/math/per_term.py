r"""Asking for one ``\boxed{}`` per term of a sum, and grading each of them.

The indefinite-integrals counterpart of the answer contract the QA experiments
use: the model is handed the same k-term integral, but instead of one
antiderivative of the whole sum it is asked for **k antiderivatives, one per
line, each in its own ``\boxed{}``** -- and every one of them is graded on its
own.

Why the mode exists. A joint answer confounds two failures that a research
question about compositional generalization has to keep apart: *not being able
to integrate a term* and *not being able to carry k antiderivatives through one
answer*. A wrong sum says only that something went wrong somewhere. Per-term
answers say which terms were right, so ``acc`` (every term right) and
``acc_terms`` (the share of terms right) can disagree -- and their disagreement
is the measurement. Asking the model for the split directly is exact and free,
exactly as ``term_correct`` is in the one-word QA and MMLU trees.

Two things this module does not do, both deliberate:

* **It does not repair.** A block of k-1 boxes is a ``WRONG_LENGTH``, never k-1
  answers attributed to k terms with one position guessed at. Same stance as
  :mod:`llm.core.oneword.answers`, and for the same reason: choosing which term
  was skipped would describe us rather than the model.
* **It does not re-decide a term.** Each box goes through
  :func:`llm.core.math.verification.check_antiderivative` -- the same checker,
  the same stage list, the same verdicts as every other integrals run -- against
  that term's own integrand. Nothing here reimplements verification.

**Position is prompt order, never ``component_integrands`` order.** SymPy
re-sorts an ``Add`` when it renders the integral, so the two disagree on almost
every ``random`` composition; a grader trusting the list order would compare
every answer with its neighbour's integrand and report a plausible, wrong
figure. The prompt order comes from :mod:`llm.core.analysis.prompt_spans`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from llm.core.analysis.prompt_spans import SpanError, locate_terms
from llm.core.math.verification import (
    CheckResult,
    StageRecord,
    Verdict,
    check_antiderivative,
)
from llm.core.parser.math_parser import strip_bpe_sentinels

# Tolerant of byte-BPE whitespace mangling, the same way
# `llm.core.parser.extraction` is: "FINAL ANSWER" arrives as "FINALĠANSWER" often
# enough that an exact match would miss the marker on a whole class of runs.
_FINAL_ANSWER = re.compile(r"FINAL\W{0,4}ANSWER", re.IGNORECASE)

# The line label a numbered answer block carries: "1. \boxed{...}", "(2) ...",
# "3) ...". Only used as evidence *the model itself* supplied about which answer
# belongs to which term -- never to invent one that is missing.
_LINE_NUMBER = re.compile(r"^\s*(?:[-–—•*>]\s*)*\(?(\d+)\s*[.):\-]\s*")

# The same evidence, read leniently: the label may carry markdown emphasis, a
# bullet, a heading mark, or the word the model chose for a term, and the answer
# need not follow it immediately on the line ("1. The antiderivative of 3x^6 is
# \boxed{...}"). Only ever used to *group and order* boxes the model itself
# numbered -- `_LINE_NUMBER` stays the strict reading used for reordering a
# tight block, where a trailing word would mean the line is prose rather than a
# label.
#
# The 1..999 bound and the required terminator keep it off ordinary sentences
# that open with a figure ("2 x^2 is the antiderivative"): a label must be
# punctuated as one.
_LINE_LABEL = re.compile(
    r"""^\s*(?:[-–—•*>\#]\s*)*        # bullet or heading marks
        (?:\*{1,3}\s*)?                # markdown bold/italic opener
        (?:
            \(?(?P<num>\d{1,3})\s*[.):\]]      #  "3."  "(3)"  "3)"  "3]"
          | (?:term|answer|antiderivative|result|part|integral)
            \s*\#?\s*(?P<word>\d{1,3})\s*[.:)\]]?   #  "Term 3:"  "**Answer 3**"
        )""",
    re.IGNORECASE | re.VERBOSE,
)

# Lines that carry no content of their own and so must not break a block: blank
# lines, display-math fences, markdown rules, and stray closing delimiters. The
# single most common answer block on record is a numbered list with one blank
# line between entries, which a strictly-consecutive reading splits into k
# blocks of one box each -- and then reports as WRONG_LENGTH 1/k.
_STRUCTURAL_LINE = re.compile(
    r"^\s*(?:\$\$|\\\[|\\\]|\\\(|\\\)|\$|-{3,}|\*{3,}|_{3,}|<br\s*/?>|\\\\)*\s*$"
)

# A plain (unboxed) numbered answer line: "1. e^x + C", "3) \frac{x^7}{7}".
# The fallback of last resort, used only when no reading of the \boxed{} in the
# response yields k answers -- see `_plain_numbered_list`.
_PLAIN_NUMBERED = re.compile(r"^\s*(?:[-–—•*>\#]\s*)*\**\(?(\d{1,3})\s*[.):]\s*(.+?)\s*\**$")

# Wrappers a model puts round the mathematics of such a line, stripped only when
# they enclose the whole of it.
_MATH_WRAPPERS = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))

# Prose the model puts before the mathematics on a numbered line, e.g.
# "1. The antiderivative of 3x^6 is \frac{3x^7}{7}". Only stripped when what
# follows still looks like mathematics; see `_strip_line_prose`.
_LINE_PROSE = re.compile(
    r"^(?:the\s+)?(?:antiderivative|integral|answer|result)\b[^:=]*?"
    r"(?:\bis\b|[:=])\s*",
    re.IGNORECASE,
)

_BOXED = "\\boxed"


class BoxedListStatus(str, Enum):
    """Why a response did or did not yield k boxed answers.

    The first four values deliberately repeat the strings of
    :class:`llm.core.parser.extraction.ExtractionStatus`, so the ``frac_empty`` /
    ``frac_no_boxed`` / ``frac_unbalanced`` metrics keep meaning what they mean
    in every other integrals task and nothing downstream has to learn a second
    vocabulary. ``WRONG_LENGTH`` is the one this mode adds, and it is the whole
    reason a separate enum exists: "the model boxed six answers where twenty
    terms were asked for" is not "the model wrote no box".
    """

    OK = "ok"
    EMPTY = "empty"                # API returned nothing (timeout, retries exhausted)
    NO_BOXED = "no_boxed"          # text produced, but no \boxed{} anywhere
    UNBALANCED = "unbalanced"      # \boxed{ opened and never closed (truncation)
    WRONG_LENGTH = "wrong_length"  # boxes found, but not as many as terms asked for


@dataclass
class BoxedList:
    """One response's answer block, as k boxed answers in term order."""

    status: BoxedListStatus
    answers: list[str] = field(default_factory=list)
    n_expected: int = 0
    n_found: int = 0
    n_boxes: int = 0            # boxes anywhere in the response, block or not
    detail: str = ""
    strategy: str = ""
    line_numbers: list[int] = field(default_factory=list)
    reordered: bool = False
    bpe_restored: bool = False

    @property
    def ok(self) -> bool:
        return self.status is BoxedListStatus.OK

    def to_data(self) -> dict[str, Any]:
        """The payload recorded on the parse stage of a trace record."""
        return {
            "reason": self.status.value,
            "answers": list(self.answers),
            "n_expected": self.n_expected,
            "n_found": self.n_found,
            "n_boxes": self.n_boxes,
            "detail": self.detail,
            "strategy": self.strategy,
            "line_numbers": list(self.line_numbers),
            "reordered": self.reordered,
            "bpe_restored": self.bpe_restored,
        }

    def rendered(self) -> str:
        r"""What lm-eval shows as this response's filtered extraction.

        The answers rejoined with newlines -- the format the prompt asked for --
        so ``filtered_resps`` reads as the answer block rather than as a Python
        repr, and a failed reading surfaces as its own sentinel instead of as an
        empty string.
        """
        if self.status is BoxedListStatus.OK:
            return "\n".join(self.answers)
        if self.status is BoxedListStatus.WRONG_LENGTH and self.answers:
            # The short (or long) reading is shown, not hidden: it is what the
            # WRONG_LENGTH is *about*, and seeing it is how a prompt problem is
            # told from a model problem.
            return (f"[[WRONG_LENGTH {self.n_found}/{self.n_expected}]]\n"
                    + "\n".join(self.answers))
        return f"[[{self.status.value.upper()}]]"


# --------------------------------------------------------------------------- #
# reading the boxes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Box:
    start: int   # index of the backslash of \boxed
    end: int     # one past the closing brace
    content: str


def boxed_spans(text: str) -> tuple[list[_Box], bool]:
    r"""Every ``\boxed{...}`` in `text`, with its extent, and whether one dangled.

    A positional scan rather than
    :func:`llm.core.parser.pattern_parser.parse_command_braces`, which returns
    contents only: this mode has to know which *line* each box sits on, since
    the newline is what separates one term's answer from the next. The brace
    walk is otherwise the same, nesting included.
    """
    boxes: list[_Box] = []
    unbalanced = False
    cursor = 0
    length = len(text)
    while True:
        index = text.find(_BOXED, cursor)
        if index < 0:
            break
        probe = index + len(_BOXED)
        while probe < length and text[probe].isspace():
            probe += 1
        if probe >= length or text[probe] != "{":
            # `\boxed` with no brace group: either truncation or prose about the
            # command. Recorded as a dangling box so the caller can tell "no box
            # at all" from "a box that never closed".
            unbalanced = True
            cursor = index + len(_BOXED)
            continue
        depth = 0
        walk = probe
        closed = False
        while walk < length:
            char = text[walk]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    boxes.append(_Box(index, walk + 1, text[probe + 1:walk]))
                    cursor = walk + 1
                    closed = True
                    break
            walk += 1
        if not closed:
            unbalanced = True
            break
    return boxes, unbalanced


def _blocks(text: str, boxes: Sequence[_Box],
            *, skip_structural: bool = False) -> list[list[_Box]]:
    """Group boxes into maximal runs of consecutive box-bearing lines.

    A block is what the prompt asked for: the answers, one per line, with
    nothing but answers between them. Grouping by line rather than taking the
    last k boxes outright is what keeps a reasoning trace that boxes an
    intermediate result three lines above the answer block from contributing to
    it. A line holding several boxes is still one line of the block -- the
    newline is how the format is *asked* for, not a claim about how a model that
    ignored it must be read.

    With ``skip_structural``, lines that carry no content of their own -- blank
    lines, ``$$`` and ``\\[`` fences, markdown rules -- do not break a block
    either. This is the same grouping rule applied to how models actually lay a
    list out: "1. \\boxed{...}", a blank line, "2. \\boxed{...}" is one answer
    block written with breathing room, not k blocks of one answer each, and
    reading it as the latter would report it as WRONG_LENGTH 1/k. Anything with
    words in it still breaks the block, so a reasoning paragraph between two
    boxes separates them.
    """
    blocks: list[list[_Box]] = []
    current: list[_Box] = []
    lines = text.split("\n") if skip_structural else []
    previous_line: int | None = None
    for box in boxes:
        line = text.count("\n", 0, box.start)
        joined = previous_line is None or line - previous_line <= 1
        if not joined and skip_structural:
            joined = all(_STRUCTURAL_LINE.match(lines[between]) is not None
                         for between in range(previous_line + 1, line))
        if joined:
            current.append(box)
        else:
            blocks.append(current)
            current = [box]
        previous_line = line
    if current:
        blocks.append(current)
    return blocks


def _line_start(text: str, index: int) -> int:
    start = text.rfind("\n", 0, index)
    return 0 if start < 0 else start + 1


def _numbering(text: str, block: Sequence[_Box]) -> list[int]:
    """The line labels of a block, one per box, or ``[]`` if any is missing.

    Only a *complete* labelling counts: a block where the model numbered three
    of twenty lines says nothing about the other seventeen, and a partial
    mapping is exactly the kind of guess this module refuses to make.
    """
    numbers: list[int] = []
    for box in block:
        prefix = text[_line_start(text, box.start):box.start]
        match = _LINE_NUMBER.match(prefix)
        if match is None or prefix[match.end():].strip():
            return []
        numbers.append(int(match.group(1)))
    return numbers


def _label_of(text: str, lines: Sequence[str], box: _Box) -> int | None:
    """The number the model itself put on this box's answer, if it put one.

    Read from the box's own line first. Failing that, from the nearest earlier
    line that is not purely structural -- a box written as its own display
    equation sits between a ``$$`` fence and the line that numbered it:

        3. **Term 3**: $\\int -3x^{-9}\\,dx$
        $$
        \\boxed{\\dfrac{3}{8x^{8}}}
        $$

    The walk back stops at the first line with content, so a box separated from
    a numbered line by a sentence is unlabelled rather than credited with that
    line's number.
    """
    line = text.count("\n", 0, box.start)
    prefix = text[_line_start(text, box.start):box.start]
    match = _LINE_LABEL.match(prefix)
    if match is not None:
        return int(match.group("num") or match.group("word"))
    for earlier in range(line - 1, -1, -1):
        if _STRUCTURAL_LINE.match(lines[earlier]) is not None:
            continue
        match = _LINE_LABEL.match(lines[earlier])
        return int(match.group("num") or match.group("word")) if match else None
    return None


def _numbered_run(text: str, boxes: Sequence[_Box], expected: int) -> list[_Box]:
    """The last run of boxes the model numbered ``1..expected`` in order.

    The strongest evidence available short of the format being obeyed exactly:
    the model said which answer is which. It reads three shapes the recorded
    runs are full of and consecutive-line grouping cannot -- a numbered list
    broken up by prose ("1. The antiderivative of 3x^6 is \\boxed{...}"), one
    written as display equations, and one the model wrote out twice (where
    grouping yields a single block of 2k boxes, and the *last* complete run is
    the model's final word).

    A run is only accepted when the labels are exactly 1, 2, ... expected with
    no gaps and no repeats, so a partially numbered block still yields nothing
    rather than a guess about the unlabelled entries.
    """
    lines = text.split("\n")
    labels = [_label_of(text, lines, box) for box in boxes]
    best: list[_Box] = []
    run: list[_Box] = []
    for box, label in zip(boxes, labels):
        if label == 1:
            run = [box]
        elif run and label == len(run) + 1:
            run.append(box)
        else:
            run = []
            continue
        if len(run) == expected:
            best = list(run)
            run = []
    return best


def _strip_math_wrappers(text: str) -> str:
    """Unwrap ``$...$`` / ``\\[...\\]`` when they enclose the whole line."""
    text = text.strip()
    for opener, closer in _MATH_WRAPPERS:
        while (len(text) > len(opener) + len(closer)
               and text.startswith(opener) and text.endswith(closer)):
            text = text[len(opener):-len(closer)].strip()
    return text


def _strip_line_prose(text: str) -> str:
    """Drop a leading "The antiderivative of ... is" from a numbered line.

    Only the fixed lead-ins above, and only when something is left after them:
    the point is to reach the mathematics a model buried in a sentence, not to
    hunt for mathematics inside arbitrary prose.
    """
    match = _LINE_PROSE.match(text.strip())
    if match is None:
        return text
    rest = text.strip()[match.end():].strip().rstrip(".")
    return rest or text


def _plain_numbered_list(text: str, expected: int) -> list[str]:
    r"""The last unboxed ``1.``…``k.`` list of answers in the response.

    Reached only when no reading of the response's ``\boxed{}`` yields k
    answers -- usually because there are none at all. Models that talk
    themselves out of the format still tend to close with the list the prompt
    asked for, just written as "Final Answers:\n1. e^x + C\n2. ...", and
    scoring that as "the model wrote nothing" measures our contract rather than
    its mathematics.

    Held to the same standard as `_numbered_run`: a complete 1..k run, in
    order, or nothing. Each entry is unwrapped from its ``$...$`` and stripped
    of a lead-in sentence; whatever is left is handed to the ordinary checker,
    which is free to find it unparseable.
    """
    best: list[str] = []
    run: list[str] = []
    for line in text.split("\n"):
        match = _PLAIN_NUMBERED.match(line)
        if match is None:
            if line.strip() and run:
                run = []          # prose inside the list ends it
            continue
        number, body = int(match.group(1)), match.group(2)
        body = _strip_line_prose(_strip_math_wrappers(body))
        if number == 1:
            run = [body]
        elif run and number == len(run) + 1:
            run.append(body)
        else:
            run = []
            continue
        if len(run) == expected:
            best = list(run)
            run = []
    return best


def _order_by_numbering(answers: list[str], numbers: list[int]) -> tuple[list[str], bool]:
    """Answers put in the model's own numbering, when it named every position.

    Not a repair, and the distinction matters: a block labelled ``1.``…``k.`` is
    the model saying which answer belongs to which term. Ignoring that in favour
    of line order would score a model that wrote its answers out of order on a
    problem it did not answer. It applies **only** to a complete permutation of
    ``1..k`` -- anything less is left exactly as written.
    """
    if len(numbers) != len(answers) or sorted(numbers) != list(range(1, len(answers) + 1)):
        return answers, False
    if numbers == list(range(1, len(answers) + 1)):
        return answers, False
    ordered = [""] * len(answers)
    for answer, number in zip(answers, numbers):
        ordered[number - 1] = answer
    return ordered, True


def extract_boxed_list(text: str | None, *, expected: int,
                       prefer_final_answer: bool = False) -> BoxedList:
    r"""Read k boxed answers out of one raw response.

    The reading rule is *the last reading that is the right length wins*, which
    is what a reasoning model needs: it boxes candidates while it works and only
    its closing list commits. Several readings of "the list" are tried, ordered
    by how much of the model's own evidence each one rests on, and the first to
    yield exactly `expected` answers is taken:

    1. **The last block of consecutive box-bearing lines** -- the format as
       asked for, and the only reading that needs no interpretation at all.
    2. **The last such block once blank lines and display-math fences are
       allowed inside it** (`_blocks(skip_structural=True)`). A list written
       with a blank line between entries is the commonest answer block on
       record, and the strict reading turns every one of them into k blocks of
       one box.
    3. **The last run of boxes the model numbered ``1..k``** (`_numbered_run`).
       This is the model saying which answer is which, and it reads three
       shapes the line-grouping cannot: a list broken up by prose, one written
       as display equations, and one written out twice.
    4. **Every box in the response, when there are exactly k of them.** No
       choice is being made here: the model boxed k things for k terms.
    5. **The last k boxes**, when there are more. The weakest of the boxed
       readings -- a reasoning trace that boxes intermediate results can end on
       k of them by luck -- so it is tried last and named in the trace.
    6. **The last unboxed ``1.``…``k.`` list** (`_plain_numbered_list`), for a
       response that used no ``\boxed{}`` at all.

    ``prefer_final_answer`` runs the same ladder over the region after the last
    FINAL ANSWER marker first; that is the second ensemble, and it is the same
    distinction ``boxed_last`` / ``final_answer`` draws for a single answer.

    Nothing here repairs a reading: a ladder that never lands on `expected`
    answers reports ``WRONG_LENGTH`` with what the last block did hold, so "the
    model lost count" stays distinguishable from "the model wrote nothing".

    Whitespace is restored first (:func:`strip_bpe_sentinels`): around half of
    the malformed generations on record arrive with every space as ``Ġ`` and
    every newline as ``Ċ``, which collapses the response to a single line -- and
    every step here reads lines.
    """
    if expected < 1:
        raise ValueError(f"a document must ask for at least one answer, got {expected}")
    if text is None or not str(text).strip():
        return BoxedList(BoxedListStatus.EMPTY, n_expected=expected,
                         detail="response was empty or None")

    # `strip_bpe_sentinels` reports whether it had to do anything; that is worth
    # keeping, since a response whose newlines arrived as `Ċ` would have been
    # unreadable as a block before the restore.
    restored, bpe_restored = strip_bpe_sentinels(str(text))
    boxes, unbalanced = boxed_spans(restored)

    def _result(block: Sequence[_Box], strategy: str) -> BoxedList:
        answers = [box.content.strip() for box in block]
        numbers = _numbering(restored, block)
        answers, reordered = _order_by_numbering(answers, numbers)
        return _listed(answers, strategy, numbers=numbers, reordered=reordered)

    def _listed(answers: list[str], strategy: str, *, numbers: list[int] | None = None,
                reordered: bool = False) -> BoxedList:
        status = (BoxedListStatus.OK if len(answers) == expected
                  else BoxedListStatus.WRONG_LENGTH)
        detail = strategy if status is BoxedListStatus.OK else (
            f"{strategy}; {len(answers)} boxed answers for {expected} terms")
        return BoxedList(status, answers=answers, n_expected=expected,
                         n_found=len(answers), n_boxes=len(boxes), detail=detail,
                         strategy=strategy, line_numbers=list(numbers or []),
                         reordered=reordered, bpe_restored=bpe_restored)

    def _ladder(region: str, candidates: Sequence[_Box], suffix: str,
                *, first: bool = False) -> BoxedList | None:
        """The reading ladder, over the whole response or one region of it.

        ``first`` reverses which block wins when several are the right length.
        Over a whole response the last one does -- a reasoning model boxes
        candidates as it goes and commits only at the end. Over the region after
        a FINAL ANSWER marker the *first* does, because the marker has already
        said where the answer starts and what follows it is the answer.
        """
        if not candidates:
            return None
        for skip_structural, name in ((False, "block of the expected length"),
                                      (True, "spaced block of the expected length")):
            blocks = _blocks(region, candidates, skip_structural=skip_structural)
            for block in (blocks if first else reversed(blocks)):
                if len(block) == expected:
                    return _result(block, ("first " if first else "last ") + name + suffix)
        numbered = _numbered_run(region, candidates, expected)
        if numbered:
            return _result(numbered, "numbered 1..k run" + suffix)
        if len(candidates) == expected:
            return _result(candidates, "every box in the response" + suffix)
        if len(candidates) > expected:
            return _result(candidates[-expected:], "last k boxes" + suffix)
        return None

    if boxes and prefer_final_answer:
        marker = None
        for marker in _FINAL_ANSWER.finditer(restored):
            pass  # the last marker: models restate the instruction before obeying it
        if marker is not None:
            after = [box for box in boxes if box.start >= marker.end()]
            found = _ladder(restored, after, " after FINAL ANSWER", first=True)
            if found is not None:
                return found

    if boxes:
        found = _ladder(restored, boxes, "")
        if found is not None:
            return found

    # No boxed reading landed on k answers. Before giving up, the unboxed list:
    # a response that ignored `\boxed{}` and simply numbered its answers has
    # said everything needed to grade it.
    plain = _plain_numbered_list(restored, expected)
    if plain:
        return _listed(plain, "unboxed numbered 1..k list")

    if not boxes:
        if unbalanced:
            return BoxedList(BoxedListStatus.UNBALANCED, n_expected=expected,
                             bpe_restored=bpe_restored,
                             detail="\\boxed present but no closed brace group followed")
        return BoxedList(BoxedListStatus.NO_BOXED, n_expected=expected,
                         bpe_restored=bpe_restored,
                         detail="no \\boxed{} in response")

    return _result(_blocks(restored, boxes)[-1], "last block")


# --------------------------------------------------------------------------- #
# term order
# --------------------------------------------------------------------------- #
def term_order(doc: dict[str, Any]) -> list[int]:
    """Component-list indices, ordered as the rendered integral shows them.

    Read off ``integrand_latex_integral`` itself rather than off the whole
    prompt, because the prompt is built *from* this ordering -- the terms are
    numbered to the model in it -- and asking `doc_to_text` for it here would
    recurse. It is the same region `prompt_spans` narrows to anyway
    (``locate_terms`` finds the rendered integral inside the prompt and works
    only within it), so the two orderings are the same by construction.
    """
    integral = str(doc.get("integrand_latex_integral") or "")
    if not integral:
        raise SpanError(
            f"{doc.get('id')!r}: no integrand_latex_integral, so the order the terms "
            "are shown in cannot be read and the answers could not be attributed."
        )
    located = locate_terms(integral, doc)
    return [span.index for span in sorted(located.spans, key=lambda span: span.start)]


# --------------------------------------------------------------------------- #
# grading
# --------------------------------------------------------------------------- #
# Which failure decides an attempt when its terms disagree. A term nobody could
# decide must not be reported as a wrong answer -- the stance
# `verification.py` takes on timeouts, applied to a list of verdicts -- so an
# undecidable term outranks a wrong one. `term_verdicts` is on every record, so
# any other reading is recoverable without re-running anything.
_PRECEDENCE = (Verdict.CHECKER_ERROR, Verdict.TIMEOUT, Verdict.UNPARSEABLE, Verdict.WRONG)


def _result_for_failed_extraction(parsed: BoxedList, terms: Sequence[str]) -> CheckResult:
    """A block that could not be read is UNPARSEABLE, and grades no term at all.

    Deliberately not "k wrong answers": k-1 boxes cannot be attributed to k
    terms, and counting an unreadable block as a row of wrong terms would put a
    formatting failure into ``acc_terms`` as if it were mathematics.
    """
    stage = StageRecord("candidate_boxed_list_parse", status="fail", data=parsed.to_data())
    return CheckResult(
        Verdict.UNPARSEABLE, "candidate_boxed_list_parse", [stage],
        {
            "reason": parsed.status.value,
            "message": parsed.detail,
            "candidate_answers": list(parsed.answers),
            "term_correct": [],
            "term_verdicts": [],
            "n_correct": 0,
            "n_terms": len(terms),
        },
    )


def check_term_answers(parsed: BoxedList, terms: Sequence[str], *,
                       variable: str = "x", timeout_s: float = 10.0) -> CheckResult:
    """Grade one boxed answer per term, in prompt order.

    ``terms`` is the composition's integrands **in the order the prompt shows
    them** (:func:`term_order`), which is what makes answer *i* comparable
    with term *i*. Each answer goes through
    :func:`llm.core.math.verification.check_antiderivative` against its own
    term, so no reference antiderivative is consulted and grading a term here is
    grading an ordinary one-term problem.

    The returned :class:`CheckResult` has the shape every other integrals task
    produces -- an ordered stage list, one ``term_<i>_check`` stage per term
    carrying that term's own verdict -- and the full per-term stage lists travel
    in ``detail["term_results"]``, so a per-term failure is as debuggable as a
    joint one.

    **The one place this deviates from the trace contract, deliberately:** every
    term is checked, so the stage list does not stop at the first failure and
    ``stages[-1]`` is simply the last term rather than the stage that decided.
    Grading only up to the first wrong term would throw away exactly the
    per-term outcomes this mode exists to produce. ``verdict_stage`` names the
    term that decided instead, and ``term_verdicts`` carries all of them.
    """
    started = time.perf_counter()
    if not parsed.ok:
        return _result_for_failed_extraction(parsed, terms)

    parse_stage = StageRecord(
        "candidate_boxed_list_parse",
        ms=(time.perf_counter() - started) * 1000.0,
        data=parsed.to_data(),
    )

    stages = [parse_stage]
    term_results: list[dict[str, Any]] = []
    term_correct: list[bool] = []
    term_verdicts: list[str] = []
    for position, (answer, integrand) in enumerate(zip(parsed.answers, terms), start=1):
        began = time.perf_counter()
        result = check_antiderivative(answer, integrand, symbol=variable,
                                      input_kind="latex", timeout_s=timeout_s)
        correct = result.verdict is Verdict.CORRECT
        term_correct.append(correct)
        term_verdicts.append(result.verdict.value)
        term_results.append({
            "position": position,
            "integrand": integrand,
            "answer": answer,
            "verdict": result.verdict.value,
            "verdict_stage": result.stage,
            "confidence": result.confidence,
            "reason": result.detail.get("reason"),
            "stages": [stage.to_dict() for stage in result.stages],
        })
        stages.append(StageRecord(
            f"term_{position}_check",
            status="ok" if correct else "fail",
            ms=(time.perf_counter() - began) * 1000.0,
            data={"position": position, "integrand": integrand, "answer": answer,
                  "verdict": result.verdict.value, "verdict_stage": result.stage,
                  "reason": result.detail.get("reason")},
        ))

    n_correct = sum(term_correct)
    all_correct = n_correct == len(terms)
    verdict = Verdict.CORRECT
    if not all_correct:
        seen = set(term_verdicts)
        verdict = next((candidate for candidate in _PRECEDENCE if candidate.value in seen),
                       Verdict.WRONG)
    # The stage that decided: the first term carrying the deciding verdict, so
    # `stages[-1]` is not silently the last term when an earlier one is what
    # settled the attempt.
    deciding = next((entry for entry in term_results
                     if not all_correct and entry["verdict"] == verdict.value), None)
    stage_name = (f"term_{deciding['position']}_check" if deciding is not None
                  else f"term_{max(len(terms), 1)}_check")
    reason = "all_correct" if all_correct else ((deciding or {}).get("reason")
                                                or "term_mismatch")
    # Numeric-only agreement on any term is numeric-only agreement on the
    # attempt: the confidence describes the weakest link, as it does for a sum.
    confidence = "numeric_only" if any(
        entry["confidence"] == "numeric_only" for entry in term_results) else "high"

    return CheckResult(
        verdict, stage_name, stages,
        {
            "reason": reason,
            "candidate_answers": list(parsed.answers),
            "term_integrands": [str(term) for term in terms],
            "term_correct": term_correct,
            "term_verdicts": term_verdicts,
            "term_results": term_results,
            "n_correct": n_correct,
            "n_terms": len(terms),
            # 1-based, to match the numbering the prompt shows the model.
            "wrong_positions": [i + 1 for i, ok in enumerate(term_correct) if not ok],
            "n_unparseable_terms": sum(v == Verdict.UNPARSEABLE.value for v in term_verdicts),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        confidence=confidence,
    )
