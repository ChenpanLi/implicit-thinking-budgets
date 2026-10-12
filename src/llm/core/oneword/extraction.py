r"""Finding the answer *block* in a raw one-word-QA response.

The sibling of :mod:`llm.core.mcq.extraction`, with the same stance: a strict
format measures the model's house style rather than the model, and demanding an
exact block shape of a short written answer would do exactly that. So the
contract is the weakest one that is still unambiguous -- **k answers on the
last k lines** -- and everything models actually write around that is accepted:
a numbered block, a bare block, an ``Answer:`` marker, ``**Answer**:``, a
``\boxed{}``, one marked answer per line.

Two things this module does *not* do, both deliberate:

* It does not decide how many answers a block holds, or what they are. That is
  :func:`llm.core.oneword.answers.parse_answer_lines`, whose outcomes stay
  first-class -- a block of three answers where twenty were asked for is still a
  WRONG_LENGTH, not a silent partial answer. This module decides only *where in
  the response the answer block is*.
* It does not repair. When nothing yields a block the result is
  :data:`ExtractionStatus.NO_ANSWER`, a recorded outcome like any other.

The reading rule is *the last thing that looks like an answer block wins*, which
is what a reasoning model needs: it weighs candidates aloud and only its closing
lines commit. In order:

1. **A numbered trailing block** -- the closing lines numbered ``1.``…``k.`` in
   order. The strongest evidence available, because the numbering itself says
   which answer belongs to which question, so it is tried first and accepted
   only at exactly the expected count.
2. **Anchors, latest first** -- every ``answer``-word marker and every
   ``\boxed{`` in the response, tried from the end backwards. The block runs
   from the marker to the end of its line and onward through the following
   answer-shaped lines, so ``Answer: Paris`` and ``Answer:\n Paris\n 4`` are
   both one block; collection stops at the expected count, at a blank line, or
   at prose. The newlines are not required either: a marker line holding the
   whole numbered block, ``Answer: 1. AK 2. 12 3. Baton Rouge``, is read as one.
3. **A bare trailing block** -- the closing consecutive answer-shaped lines,
   with no marker at all. This is the format the user asked for read literally:
   k answers separated by newlines at the end of the response, and nothing
   announcing them.
4. **One marked answer per line** (k > 1 only) -- a response that answers each
   question where it discusses it and never gathers the answers at the end.
   Read from the last such line backwards and accepted only at exactly k.
5. **The closing lines whatever their shape** (``lenient`` only) -- the guess.

Steps 1 to 4 are shared by both ensembles, because in none of them is the model
doing anything but writing its answers. Step 5 is the guess, and it is what
``lenient`` has and ``explicit`` does not; the gap between ``acc,lenient`` and
``acc,explicit`` prices it in the score's own units.

When no step reads the expected number of answers but an anchor read *fewer*,
that short reading comes back, so the grader reports a WRONG_LENGTH -- which
says the model lost count -- rather than "no answer at all".

Chat-template control tokens are decoration, not answers: a generation that ran
to its end of turn leaves ``TG<｜end▁of▁sentence｜>`` or ``Paris<|im_end|>``, and
:func:`llm.core.oneword.answers.strip_control_tokens` takes them off every line
this module reads -- before the answer-shaped test, since a token also inflates
the length it measures. Left in place they are not cosmetic: normalisation turns
``TG<｜end▁of▁sentence｜>`` into ``tg end of sentence`` and scores a right answer
wrong. What is *not* stripped is an angle-bracketed run that is not a control
token (``<4>``, ``<green>``), which is the prompt's own placeholder filled in, a
copied example rather than an answer.

Whitespace is restored first (:func:`strip_bpe_sentinels`). Around half of the
malformed generations on record arrive with every space as ``Ġ`` and every
newline as ``Ċ``, which collapses the response to one line -- and every step
here reads lines.
"""
from __future__ import annotations

import re

from llm.core.parser.extraction import ExtractResult, ExtractionStatus
from llm.core.parser.math_parser import strip_bpe_sentinels

from .answers import _NUMBER_SEPARATOR, clean_answer_line, split_inline_numbered, strip_control_tokens

# An answer is short. The cap is what separates "the closing lines are the
# answer block" from "the response ends in a sentence", and it is deliberately
# generous: the longest answer in the pool, `pounds per square inch`, is four
# words, and `Sri Jayawardenepura Kotte` is 25 characters.
MAX_ANSWER_WORDS = 6
MAX_ANSWER_CHARS = 80

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
# The separator is the line cleaner's (`llm.core.oneword.answers`), so `1.yes`
# counts as a numbered line here exactly when the cleaner strips its number.
_NUMBERED = re.compile(r"^\s*(?:[-\u2013\u2014\u2022*>]\s*)*(?:Q(?:uestion)?\s*)?(\d+)\s*"
                       + _NUMBER_SEPARATOR + r"(.*\S)\s*$",
                       re.IGNORECASE)


def _is_answer_shaped(line: str) -> bool:
    """Could this line be one answer, rather than a sentence about one?

    A heuristic, and the only one in the module -- which is why every step that
    uses it also insists on the *count* being exactly right. A block of k
    answer-shaped lines at the end of a response that was asked for k answers is
    the answer block; k-1 of them is not three-quarters of one.
    """
    cleaned = clean_answer_line(line)
    if not cleaned:
        return False
    if len(cleaned) > MAX_ANSWER_CHARS or len(cleaned.split()) > MAX_ANSWER_WORDS:
        return False
    # A line carrying its own marker is another question's answer line, not a
    # plain answer: reading it as one is how "Q2 ... so Answer: 4" ends up
    # recorded as the answer to question 2. Step 4 is what reads that shape.
    if _MARKER.search(cleaned) or _BOXED.search(cleaned):
        return False
    return not cleaned.endswith(":")


def _numbered_trailing_block(text: str, *, expected: int) -> list[str]:
    """The closing lines numbered 1..expected, in order.

    Numbering is the one thing in a free-text block that says which answer
    belongs to which question, so it is checked rather than assumed: the numbers
    must run 1..expected with no gaps, or the block is not this shape and the
    later steps get their turn.
    """
    collected: list[tuple[int, str]] = []
    for line in reversed(text.split("\n")):
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


def _trailing_block(text: str, *, expected: int, shaped: bool) -> list[str]:
    """The closing ``expected`` lines of the response, as written.

    With ``shaped`` they must each look like an answer, which is steps 1-3's
    standard; without it they are taken whatever they are, which is step 5's
    guess. Either way the count must be exact: attributing k-1 closing lines to
    k questions is the repair this module does not do.
    """
    collected: list[str] = []
    for line in reversed(text.split("\n")):
        if not line.strip():
            if collected:
                break
            continue
        if shaped and not _is_answer_shaped(line):
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


def _answers_after(segment: str, *, expected: int) -> list[str]:
    """Read answers from an anchor's segment: its own line, then the next ones.

    The marker's own line is tried as a whole block first, for the response that
    writes ``Answer: 1. AK 2. 12 3. Baton Rouge`` -- the block the prompt asked
    for, with the newlines missing. Only the numbered form is read that way (see
    :func:`split_inline_numbered`); a marker line holding one answer needs no
    such reading and is handled by the loop below.
    """
    lines = segment.split("\n")
    inline = split_inline_numbered(lines[0], expected=expected)
    if inline:
        return inline

    answers: list[str] = []
    for index, line in enumerate(lines):
        if not line.strip():
            if answers:
                break        # a blank line ends the answer block
            continue         # the marker was alone on its line: "**Answer**\n Paris"
        if not _is_answer_shaped(line):
            if index == 0 and not answers and expected == 1:
                # Prose on the marker's own line ("Answer: the capital is Paris").
                # For a single question that line is the model's answer, wrong
                # form and all; guessing which of its words is the answer would
                # be inventing one. For several questions it says nothing about
                # which question it answers, so it is dropped.
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


def _anchor_answers_in_line(line: str, *, expected: int = 1) -> list[str] | None:
    """The answers the *last* anchor on one line names, or None if it has none.

    Line-scoped on purpose: this is the reading of a response that answers each
    question where it discusses it, so one answer must not run on into the next
    question's block. Taking the last anchor of the line collapses the two kinds
    that overlap -- ``Answer: \boxed{Paris}`` is one answer, not two.
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
    return _answers_after(segment, expected=expected)


def _per_question_answers(text: str, *, expected: int) -> list[str]:
    """One marked answer per line, for a response that answers in turn.

    The shape a reasoning model falls into on a k-question prompt when it does
    not gather its answers at the end: a block of working per question, each
    ending ``Answer: Paris``. Read from the last such line backwards, taking the
    closing ``expected`` of them, and accepted only at exactly that many.

    What is assumed is that the model answered the questions *in order* -- the
    same assumption any list of answers is read under. What is not assumed is
    the count: a short collection is discarded whole rather than aligned.
    """
    answers: list[str] = []
    for line in reversed(text.split("\n")):
        found = _anchor_answers_in_line(line)
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
    tail_fallback: bool = True,
) -> ExtractResult:
    """Locate the answer block in a raw response and return it as text.

    ``expected`` is how many questions were asked; it bounds the search rather
    than shaping its output. The returned ``value`` is the answers joined by
    **newlines** -- not by commas as in the multiple-choice extractor, because a
    written answer may contain a comma-free space and must survive being split
    apart again by :func:`parse_answer_lines`.

    With ``tail_fallback=False`` only blocks the model wrote *as* its answer are
    honoured -- steps 1 to 4 -- which is the ``explicit`` ensemble and the
    control for what the closing-lines guess bought.
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
        answers = _answers_after(segment, expected=expected)
        if len(answers) == expected:
            return found(answers, f"{kind} anchor at char {start}", index)
        if answers and partial is None:
            partial = found(answers, f"{kind} anchor at char {start}", index)

    answers = _trailing_block(text, expected=expected, shaped=True)
    if answers:
        return found(answers, "bare trailing lines")

    if expected > 1:
        answers = _per_question_answers(text, expected=expected)
        if answers:
            return found(answers, f"one marked answer per line, {expected} of them")

    # Nothing read a full block, so the short one is handed on as it stands:
    # `parse_answer_lines` reports it as a WRONG_LENGTH, which says where the
    # model lost count, and is a more useful diagnosis than "no answer".
    if partial is not None:
        return partial

    if tail_fallback:
        answers = _trailing_block(text, expected=expected, shaped=False)
        if answers:
            return found(answers, "no answer block written as such; the closing lines")

    if anchors:
        detail = f"{len(anchors)} answer marker(s), none followed by an answer"
    else:
        detail = "no answer marker and no answer block at the end of the response"
    return ExtractResult(ExtractionStatus.NO_ANSWER, n_matches=len(anchors),
                         detail=detail + note)
