r"""Finding the answer *block* in a raw response to a short-answer maths prompt.

The sibling of :mod:`llm.core.oneword.extraction`, with the same stance: a
strict format measures the model's house style rather than the model. So the
contract is the weakest one that is still unambiguous -- **k answers on the
last k lines** -- and everything models actually write around that is accepted:
a numbered block, a bare block, an ``Answer:`` marker, ``**Answer**:``, a
``\boxed{}`` per line, one marked answer per line.

Two things this module does *not* do, both deliberate:

* It does not decide how many answers a block holds, or what they are. That is
  :func:`llm.core.shortmath.answers.parse_answer_lines`, whose outcomes stay
  first-class -- a block of three answers where twenty were asked for is still a
  WRONG_LENGTH, not a silent partial answer. This module decides only *where in
  the response the answer block is*.
* It does not repair. When nothing yields a block the result is
  :data:`ExtractionStatus.NO_ANSWER`, a recorded outcome like any other.

The reading rule is *the last thing that looks like an answer block wins*, which
is what a reasoning model needs: it weighs candidates aloud and only its closing
lines commit. In order: a numbered trailing block, then every ``answer`` marker
and every ``\boxed{`` from the end backwards, then a bare trailing block, then
one marked answer per line, and finally -- ``lenient`` only -- the closing lines
whatever their shape. The gap between ``acc,lenient`` and ``acc,explicit``
prices that last guess in the score's own units.

What differs from the one-word extractor is :class:`Shape`, and it exists
because "a line short enough to be an answer" is not the same sentence in the
two domains. A one-word answer is at most four words and 25 characters; a
MATH-500 answer may be ``\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}`` (45
characters, nine whitespace-separated tokens, not one of them an English word).
So the shape test counts **plain alphabetic words** rather than tokens, and each
tree supplies its own caps:

* GSM8K: short, and a line with no digit in it is not a number
  (:data:`NUMERIC_SHAPE`).
* MATH-500: generous in characters and in tokens, strict in English words
  (:data:`LATEX_SHAPE`).

Lines that leave a LaTeX group open are joined to the next before any of this
(:func:`llm.core.shortmath.answers.join_open_groups`), so a matrix written
across lines is one answer-shaped line rather than three unreadable ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from llm.core.parser.extraction import ExtractResult, ExtractionStatus
from llm.core.parser.math_parser import strip_bpe_sentinels

from .answers import (
    clean_answer_line,
    join_open_groups,
    split_inline_numbered,
    strip_control_tokens,
)


@dataclass(frozen=True)
class Shape:
    """What a line has to look like to be read as one answer.

    ``max_words`` counts **plain English words** -- whitespace tokens that are
    entirely alphabetic -- and not tokens in general, because a mathematical
    answer is mostly tokens that are not words. ``require_digit`` says a line
    with no digit cannot be an answer, which is true of GSM8K (every answer is a
    number) and false of MATH-500 (``\\text{Evelyn}``, ``x+1``).
    """

    max_chars: int
    max_words: int
    require_digit: bool = False


#: GSM8K: every answer is a number, so a line is short and holds a digit.
NUMERIC_SHAPE = Shape(max_chars=80, max_words=4, require_digit=True)
#: MATH-500: an expression, an interval, a tuple or a small matrix.
LATEX_SHAPE = Shape(max_chars=220, max_words=6, require_digit=False)

# The marker word, tolerant of what surrounds it: markdown (`**Answer**`),
# headings (`### Answer`), LaTeX (`\textbf{Answer}`) and BPE mangling are all
# just non-letter characters, so the boundaries do the work.
_MARKER = re.compile(
    r"(?<![A-Za-z])(?:final|the)?\W{0,4}answers?(?![A-Za-z])",
    re.IGNORECASE,
)

# Everything that can sit between the marker and the first answer: punctuation,
# markdown, LaTeX decoration, and the copulas a model writes instead of a colon.
_LEAD = re.compile(
    r"^(?:[\s:;*_~`>|=,.\-\u2013\u2014){\]}]|\\[a-zA-Z]+|\bis\b|\bare\b|\bwas\b|\bwere\b"
    r"|\bwould\s+be\b|\bshould\s+be\b|\bmust\s+be\b)+",
    re.IGNORECASE,
)

_BOXED = re.compile(r"\\boxed\s*\{?")
_NUMBERED = re.compile(
    r"^\s*(?:[-\u2013\u2014\u2022*>]\s*)*(?:Q(?:uestion)?|P(?:roblem)?\s*)?(\d+)\s*[.):\-]\s+(.*\S)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_WORD = re.compile(r"^[A-Za-z]+$")
_DIGIT = re.compile(r"\d")


def _lines_of(text: str) -> list[str]:
    """The response's lines, with an unclosed LaTeX group joined to the next.

    Applied here rather than to the whole response once, because joining is what
    a *reader of answers* needs and not what a reader of reasoning wants: a
    paragraph that opens a brace and never closes it would otherwise absorb the
    lines after it everywhere in the module.
    """
    return join_open_groups(text.split("\n"))


def _is_answer_shaped(line: str, shape: Shape) -> bool:
    """Could this line be one answer, rather than a sentence about one?

    A heuristic, and the only one in the module -- which is why every step that
    uses it also insists on the *count* being exactly right. A block of k
    answer-shaped lines at the end of a response that was asked for k answers is
    the answer block; k-1 of them is not three-quarters of one.
    """
    cleaned = clean_answer_line(line)
    if not cleaned:
        return False
    if len(cleaned) > shape.max_chars:
        return False
    words = sum(1 for token in cleaned.split() if _WORD.fullmatch(token))
    if words > shape.max_words:
        return False
    if shape.require_digit and not _DIGIT.search(cleaned):
        return False
    # A line carrying its own marker is another problem's answer line, not a
    # plain answer: reading it as one is how "P2 ... so Answer: 4" ends up
    # recorded as the answer to problem 2. The per-line step is what reads that
    # shape. A `\boxed{}` is *not* disqualifying here, unlike in the one-word
    # extractor: this prompt asks for one box per line, so a boxed line is the
    # answer block's normal form.
    if _MARKER.search(cleaned):
        return False
    return not cleaned.endswith(":")


def _numbered_trailing_block(text: str, *, expected: int) -> list[str]:
    """The closing lines numbered 1..expected, in order.

    Numbering is the one thing in a free-text block that says which answer
    belongs to which problem, so it is checked rather than assumed: the numbers
    must run 1..expected with no gaps, or the block is not this shape and the
    later steps get their turn.
    """
    collected: list[tuple[int, str]] = []
    for line in reversed(_lines_of(text)):
        if not line.strip():
            if collected:
                break
            continue
        match = _NUMBERED.match(line)
        if match is None:
            break
        collected.insert(0, (int(match.group(1)), match.group(2).strip()))
        if len(collected) >= expected:
            break
    if len(collected) != expected:
        return []
    if [number for number, _ in collected] != list(range(1, expected + 1)):
        return []
    return [answer for _, answer in collected]


def _trailing_block(text: str, *, expected: int, shape: Shape, shaped: bool) -> list[str]:
    """The closing ``expected`` lines of the response, as written.

    With ``shaped`` they must each look like an answer, which is the standard of
    every step but the last; without it they are taken whatever they are, which
    is the lenient guess. Either way the count must be exact: attributing k-1
    closing lines to k problems is the repair this module does not do.
    """
    collected: list[str] = []
    for line in reversed(_lines_of(text)):
        if not line.strip():
            if collected:
                break
            continue
        if shaped and not _is_answer_shaped(line, shape):
            break
        cleaned = clean_answer_line(line) if shaped else strip_control_tokens(line)
        if not cleaned:
            break
        collected.insert(0, cleaned)
        if len(collected) >= expected:
            break
    return collected if len(collected) == expected else []


def _brace_group(text: str, open_index: int) -> str | None:
    """The balanced contents of the group opening at ``open_index``, else None.

    A box is delimited, so it is read as the delimiters say rather than as "to
    the end of the line". An unbalanced group (a truncated response) returns
    None and the caller falls back to the line.
    """
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:index]
    return None


def _anchors(text: str) -> list[tuple[int, int, str]]:
    r"""Every marker and every ``\boxed{`` in the text, as (start, end, kind)."""
    found = [(m.start(), m.end(), "marker") for m in _MARKER.finditer(text)]
    found += [(m.start(), m.end(), "boxed") for m in _BOXED.finditer(text)]
    found.sort()
    return found


def _answers_after(segment: str, *, expected: int, shape: Shape) -> list[str]:
    """Read answers from an anchor's segment: its own line, then the next ones.

    The marker's own line is tried as a whole block first, for the response that
    writes ``Answer: 1. 100 2. 30 3. 623`` -- the block the prompt asked for,
    with the newlines missing.
    """
    lines = _lines_of(segment)
    inline = split_inline_numbered(lines[0], expected=expected)
    if inline:
        return inline

    answers: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            if answers:
                break        # a blank line ends the answer block
            continue         # the marker was alone on its line: "**Answer**\n 6"
        if not _is_answer_shaped(line, shape):
            if index == 0 and not answers and expected == 1:
                # Prose on the marker's own line ("Answer: the height is 5 cm").
                # For a single problem that line is the model's answer, wrong
                # form and all; guessing which of its words is the answer would
                # be inventing one. For several problems it says nothing about
                # which problem it answers, so it is dropped.
                cleaned = clean_answer_line(line)
                if cleaned:
                    answers.append(cleaned)
            break
        cleaned = clean_answer_line(line)
        if not cleaned:
            if answers:
                break        # scaffolding after the block ends it
            continue         # scaffolding before it (a code fence, "Answer:")
        answers.append(cleaned)
        if len(answers) >= expected:
            break
    return answers


def _anchor_answers_in_line(line: str, *, shape: Shape,
                            expected: int = 1) -> list[str] | None:
    r"""The answers the *last* anchor on one line names, or None if it has none.

    Line-scoped on purpose: this is the reading of a response that answers each
    problem where it works it out, so one answer must not run on into the next
    problem's block. Taking the last anchor of the line collapses the two kinds
    that overlap -- ``Answer: \boxed{6}`` is one answer, not two.
    """
    anchors = _anchors(line)
    if not anchors:
        return None
    _, end, kind = anchors[-1]
    segment = None
    if kind == "boxed" and end and line[end - 1] == "{":
        segment = _brace_group(line, end - 1)
    if segment is None:
        segment = _LEAD.sub("", line[end:])
    return _answers_after(segment, expected=expected, shape=shape)


def _per_question_answers(text: str, *, expected: int, shape: Shape) -> list[str]:
    """One marked answer per line, for a response that answers in turn.

    The shape a reasoning model falls into on a k-problem prompt when it does not
    gather its answers at the end: a block of working per problem, each ending
    ``Answer: 100`` or ``\\boxed{100}``. Read from the last such line backwards,
    taking the closing ``expected`` of them, and accepted only at exactly that
    many.

    What is assumed is that the model answered the problems *in order* -- the
    same assumption any list of answers is read under. What is not assumed is
    the count: a short collection is discarded whole rather than aligned.
    """
    answers: list[str] = []
    for line in reversed(_lines_of(text)):
        found = _anchor_answers_in_line(line, shape=shape)
        if found is None:
            continue          # reasoning, not an answer line
        if len(found) != 1:
            break             # ambiguous line: stop rather than pick from it
        answers.insert(0, found[0])
        if len(answers) == expected:
            break
    return answers if len(answers) == expected else []


def extract_answers(
    text: str | None,
    *,
    expected: int,
    shape: Shape,
    tail_fallback: bool = True,
) -> ExtractResult:
    """Locate the answer block in a raw response and return it as text.

    ``expected`` is how many problems were asked; it bounds the search rather
    than shaping its output. The returned ``value`` is the answers joined by
    **newlines**, which is how :func:`parse_answer_lines` splits them apart
    again -- never by commas, which appear inside answers here.

    With ``tail_fallback=False`` only blocks the model wrote *as* its answer are
    honoured, which is the ``explicit`` ensemble and the control for what the
    closing-lines guess bought.
    """
    if expected < 1:
        raise ValueError(f"expected must be at least 1, got {expected}")

    if text is None or not str(text).strip():
        return ExtractResult(ExtractionStatus.EMPTY, detail="response was empty or None")

    # Whitespace first: a response whose spaces and newlines arrived as byte-BPE
    # sentinels is one long line, and every step below reads lines.
    text, restored = strip_bpe_sentinels(str(text))
    note = "; whitespace restored from BPE sentinels" if restored else ""
    anchors = _anchors(text)

    def found(answers: list[str], detail: str, index: int | None = None) -> ExtractResult:
        return ExtractResult(ExtractionStatus.OK, "\n".join(answers),
                             n_matches=len(anchors), chosen_index=index,
                             detail=f"{detail}{note}")

    answers = _numbered_trailing_block(text, expected=expected)
    if answers:
        return found(answers, "numbered trailing block")

    # The latest anchor that named the right number of answers wins outright. An
    # anchor that named *some* is remembered rather than returned, so a response
    # whose real block sits further down is still read as a block first.
    partial: ExtractResult | None = None
    for index in range(len(anchors) - 1, -1, -1):
        start, end, kind = anchors[index]
        segment = None
        if kind == "boxed" and end and text[end - 1] == "{":
            segment = _brace_group(text, end - 1)
        if segment is None:
            segment = _LEAD.sub("", text[end:])
        answers = _answers_after(segment, expected=expected, shape=shape)
        if len(answers) == expected:
            return found(answers, f"{kind} anchor at char {start}", index)
        if answers and partial is None:
            partial = found(answers, f"{kind} anchor at char {start}", index)

    answers = _trailing_block(text, expected=expected, shape=shape, shaped=True)
    if answers:
        return found(answers, "bare trailing lines")

    if expected > 1:
        answers = _per_question_answers(text, expected=expected, shape=shape)
        if answers:
            return found(answers, f"one marked answer per line, {expected} of them")

    # Nothing read a full block, so the short one is handed on as it stands:
    # `parse_answer_lines` reports it as a WRONG_LENGTH, which says where the
    # model lost count, and is a more useful diagnosis than "no answer".
    if partial is not None:
        return partial

    if tail_fallback:
        answers = _trailing_block(text, expected=expected, shape=shape, shaped=False)
        if answers:
            return found(answers, "no answer block written as such; the closing lines")

    if anchors:
        detail = f"{len(anchors)} answer marker(s), none followed by an answer"
    else:
        detail = "no answer marker and no answer block at the end of the response"
    return ExtractResult(ExtractionStatus.NO_ANSWER, n_matches=len(anchors),
                         detail=detail + note)
