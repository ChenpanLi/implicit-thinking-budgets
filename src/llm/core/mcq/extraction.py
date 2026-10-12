"""Finding the answer *segment* in a raw multiple-choice response.

The integrals experiment can insist on ``FINAL ANSWER: \\boxed{...}``, because a
symbolic antiderivative has to be delimited somehow. A multiple-choice answer
does not: it is one capital letter, and every model already has a house style
for presenting one. Insisting on the box would measure the house style instead
of the model.

So the contract is the weakest one that is still unambiguous: **a line that
looks like an answer**. Anything of the shape ``Answer: B``, ``**Answer**: B``,
``ANSWER: B``, ``Final answer - B``, ``The answers are: A, C, B`` is accepted,
and so is a ``\\boxed{}`` for the models that keep writing one.

Two things this module does *not* do, both deliberate:

* It does not decide *which letters* an answer segment names. That is
  :func:`llm.core.mcq.answers.parse_answer_list`, whose outcomes stay
  first-class -- a segment holding three letters where twenty were asked for is
  still a WRONG_LENGTH, not a silent partial answer. This module only decides
  *where in the response the answer is*, and hands that text on.
* It does not repair. When no anchor and no tail yields letters the result is
  :data:`ExtractionStatus.NO_ANSWER`, which is a recorded outcome in its own
  right.

Leniency has a price, and the price is measured rather than assumed: the last
step below reads letters out of closing prose, which in a truncated reasoning
trace was never an answer at all, and a wrong guess scores 1/n_choices by luck
where an unparseable one scores 0. That is why that step is a *flag*
(``tail_fallback``) and the MMLU task runs two ensembles over every response --
``lenient`` with it and ``explicit`` without. The gap between ``acc,lenient``
and ``acc,explicit`` is what the guess bought, in the same units as the score.

The reading rule is *the last thing that looks like an answer wins*, which is
what a reasoning model needs: it will weigh several options aloud, and only its
closing lines commit. In order:

1. **Every reading, latest first.** Four kinds of thing look like an answer, and
   they are gathered into one list ordered by position rather than tried one
   kind at a time, so that the rule above decides between them:

   * an **anchor** -- every ``answer``-word marker and every ``\\boxed{`` in the
     response. Its letters run to the end of its line and onward through
     following letter-only lines, so ``Answer: A, C, B`` and
     ``Answer:\\n1. A\\n2. C\\n3. B`` are both one segment; collection stops at
     the expected count, at a blank line, or at prose.
   * a **letter-only block** -- a run of consecutive lines holding nothing but
     option letters, ``A,C,D`` or a column of them or ``1. A\\n2. C\\n3. B``,
     wherever in the response it sits.
   * **one marked answer per line** (k > 1 only) -- a response that answers each
     question where it discusses it and never gathers the letters at the end
     (``Q1 ... Answer: B``, ``Q2 ... Answer: D``, ...). Without it that response
     is one letter for k questions, a whole answer discarded for its layout.
   * **one numbered answer per line** (k > 1 only) -- the same shape with the
     line number doing the anchoring: ``3. **A** - Gamma rays have the ...``.

   The last two are positioned at their *final* answer line, which is what makes
   them comparable with the first two: a model that opens with a summary list
   and then works through the questions one by one has committed to the answers,
   and on the runs on record those two disagree often enough to matter.

   The latest reading naming exactly k letters wins. So a model that muses "the
   answer is A" and then boxes B is read as B; one that boxes an intermediate
   result and then writes ``ANSWER: B`` is read as B; and one that lists its k
   letters and then writes a closing sentence -- or repeats the format
   instruction back -- is read as that list.
2. **Isolated letters at the end** (``lenient`` only): the capital option
   letters of the last non-empty line, else of the trailing block -- "So I'll go
   with B." Never the whole response: "the answer is at the end" is the
   assumption this step makes, and scanning a 20k-token reasoning trace for
   capital letters would always find some.

Step 1 is shared by both ensembles because none of its readings guesses -- the
model wrote those letters *as* its answer. Step 2 is the guess, and is what
``lenient`` has and ``explicit`` does not. The two are not equivalent even where
both find something: a model that writes ``A,B,A`` and then explains its choices
in prose naming B, A, A has contradicted itself, and step 1 reads the list it
wrote while step 2 alone would read the prose.

When no reading names exactly k letters, the latest one naming *more* comes
back, and :func:`parse_answer_list` grades its first k (see that module on why
truncating is a reading and not a repair). Failing that the latest naming
*fewer* comes back, so the grader reports a WRONG_LENGTH -- which says the model
lost count -- rather than "no answer at all".

Whitespace is restored first (:func:`strip_bpe_sentinels`). Around half of the
malformed generations on record arrive with every space as ``Ġ`` and every
newline as ``Ċ``, which turns the whole response into one line: without the
restore every line-wise reading above has no end to look at, and a marker's
segment is the entire remaining response.
"""
from __future__ import annotations

import re

from llm.core.parser.extraction import ExtractResult, ExtractionStatus
from llm.core.parser.math_parser import strip_bpe_sentinels

from .answers import LETTERS, AnswerListStatus, parse_answer_list

# The marker word. Tolerant of what surrounds it -- markdown (`**Answer**`),
# headings (`### Answer`), LaTeX (`\textbf{Answer}`) and the byte-BPE whitespace
# mangling that `_FINAL_ANSWER` in the integrals extractor already handles are
# all just non-letter characters, so the boundaries do the work. "final" and
# "the" are consumed when present only so the segment starts after them; they
# are never required.
_MARKER = re.compile(
    r"(?<![A-Za-z])(?:final|the)?\W{0,4}answers?(?![A-Za-z])",
    re.IGNORECASE,
)

# Everything that can sit between the marker and the first letter: punctuation,
# markdown, LaTeX decoration, and the copulas a model writes instead of a colon
# ("the answer is B", "the answers are A, B"). Repeated, so "answer** is: B"
# reduces in one pass. A word not in this set (say "option") ends the strip and
# the segment is then read as prose, which is a survivable outcome -- the letters
# are still found by the isolated-letter scan.
_LEAD = re.compile(
    r"^(?:[\s:;*_~`>|=,.\-\u2013\u2014){\]}]|\\[a-zA-Z]+|\bis\b|\bare\b|\bwas\b|\bwere\b"
    r"|\bwould\s+be\b|\bshould\s+be\b|\bmust\s+be\b)+",
    re.IGNORECASE,
)

_BOXED = re.compile(r"\\boxed\s*\{?")

# The position an answer line claims for itself: "1.", "2)", "Q3:", "**4.**".
_LINE_NUMBER = re.compile(
    r"^\s*(?:\*{1,2}|_{1,2})?\s*(?:Q(?:uestion)?\s*)?(\d{1,3})\s*[.):\-\]]\s*"
    r"(?:\*{1,2}|_{1,2})?\s*",
    re.IGNORECASE,
)


def _valid_letters(n_choices: int) -> str:
    if not 1 <= n_choices <= len(LETTERS):
        raise ValueError(f"n_choices must be in 1..{len(LETTERS)}, got {n_choices}")
    return LETTERS[:n_choices]


def _isolated_pattern(n_choices: int) -> re.Pattern[str]:
    """Option letters standing alone in prose: ``(B)``, ``**B**``, ``B.``, ``is C``.

    Case-sensitive on purpose. A lower-case "a" is the English article and a
    lower-case "b" is rarely an answer written in prose, so requiring the
    capital costs almost nothing and keeps the fallback from harvesting words.
    """
    letters = _valid_letters(n_choices)
    return re.compile(rf"(?<![A-Za-z0-9])([{letters[0]}-{letters[-1]}])(?![A-Za-z0-9])")


def _letters_only(line: str, n_choices: int) -> list[str] | None:
    """The option letters of a line that holds *nothing but* option letters.

    ``None`` when the line has any other content, which is what stops a segment
    from running on into the prose after it. The judgement is delegated to
    :func:`parse_answer_list` rather than re-implemented, so the numbering echo,
    LaTeX decoration and separator handling are the same here as at grading
    time; ``expected=1`` is a placeholder, and a length mismatch (the usual
    case, a line of k letters) is therefore not a rejection. ``truncate=False``
    for that same reason: this asks what the line says, and the whole line is
    the answer, so the first letter of it must not be mistaken for one.
    """
    parsed = parse_answer_list(line, expected=1, n_choices=n_choices, truncate=False)
    if parsed.status in (AnswerListStatus.OK, AnswerListStatus.WRONG_LENGTH):
        return list(parsed.letters)
    return None


def _letters_after(segment: str, *, expected: int, n_choices: int) -> list[str]:
    """Read option letters from an anchor's segment: its line, then the next ones."""
    letters: list[str] = []
    for index, line in enumerate(segment.split("\n")):
        if not line.strip():
            if letters:
                break        # a blank line ends the answer block
            continue         # the marker was alone on its line: "**Answer**\n B"
        found = _letters_only(line, n_choices)
        if found is None:
            if index == 0 and not letters:
                # Prose on the marker's own line ("Answer: option B is correct").
                # Its isolated letters are the answer; the lines after it are
                # not, or we would be reading the discussion as well.
                letters.extend(_isolated_pattern(n_choices).findall(line))
            break
        letters.extend(found)
        if len(letters) >= expected:
            break
    return letters


def _brace_group(text: str, open_index: int) -> str | None:
    """The balanced contents of the group opening at ``open_index``, else None.

    A box is delimited, so its contents should be read as the delimiters say
    rather than as "to the end of the line": ``\boxed{B} because A is wrong``
    would otherwise hand two letters to a one-question grading. An unbalanced
    group (a truncated response) returns None and the caller falls back to the
    line, which is where the tail steps can still find something.
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
    """Every marker and every ``\\boxed{`` in the text, as (start, end, kind)."""
    found = [(m.start(), m.end(), "marker") for m in _MARKER.finditer(text)]
    found += [(m.start(), m.end(), "boxed") for m in _BOXED.finditer(text)]
    found.sort()
    return found


def _letter_line_blocks(text: str, n_choices: int) -> list[tuple[int, list[str]]]:
    """Every run of consecutive letter-only lines, as (char offset, letters).

    The answer a model writes when it has stopped narrating: ``A,C,D`` on a line
    of its own, ``A\\nB\\nB\\nC\\nD\\nD`` down a column, ``1. A\\n2. C\\n3. B``
    numbered. Each such run is *a* candidate answer, positioned so the caller can
    apply the module's one reading rule -- the latest wins -- across these and
    the anchors alike. Reading only a run at the very end of the response
    would miss the answer of a model that listed its k letters and then added
    a closing sentence ("The correct answer for each question is C, as ...", or
    the format instruction repeated back).

    A blank line ends a run rather than being skipped over: two letters, a gap
    and three more are two things the model wrote, and joining them across the
    gap is how a fragment of the reasoning gets glued to the answer.
    """
    blocks: list[tuple[int, list[str]]] = []
    run: list[str] = []
    run_start = 0
    offset = 0
    for line in text.split("\n"):
        found = _letters_only(line, n_choices) if line.strip() else None
        if found:
            if not run:
                run_start = offset
            run.extend(found)
        elif run:
            blocks.append((run_start, run))
            run = []
        offset += len(line) + 1
    if run:
        blocks.append((run_start, run))
    return blocks


def _anchor_letters_in_line(line: str, n_choices: int) -> list[str] | None:
    """The letters the *last* anchor on one line names, or None if it has none.

    Line-scoped on purpose: this is the reading of a response that answers each
    question where it discusses it, so an answer must not run on into the next
    question's block. Taking the last anchor of the line collapses the two kinds
    that overlap -- ``Answer: \\boxed{B}`` is one answer, not two.
    """
    anchors = _anchors(line)
    if not anchors:
        return None
    start, end, kind = anchors[-1]
    segment = None
    if kind == "boxed" and end and line[end - 1] == "{":
        segment = _brace_group(line, end - 1)
    if segment is None:
        segment = _LEAD.sub("", line[end:])
    return _letters_after(segment, expected=1, n_choices=n_choices)


def _line_offsets(lines: list[str]) -> list[int]:
    """The character offset each line starts at."""
    offsets = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1
    return offsets


def _per_question_answers(text: str, *, expected: int,
                          n_choices: int) -> tuple[int, list[str]] | None:
    """One marked answer per line, for a response that answers in turn.

    The shape a reasoning model falls into on a k-question prompt when it does
    not gather its answers at the end: a block of working per question, each
    ending ``Answer: B``. Read from the last such line backwards, taking the
    closing ``expected`` of them, and accepted only at exactly that many.

    Two guards keep this from becoming the alignment guess the module refuses
    to make. A line naming more than one option ends the collection rather than
    contributing -- it is a sentence weighing options, not one question's
    answer -- and a short collection is discarded whole. What is still assumed
    is that the model answered the questions *in order*, which is the same
    assumption every comma-separated list is read under, and is why the letters
    are taken from the end of the response rather than from wherever in the
    reasoning they first appear.

    Returned with the offset of its *last* answer line, because that is where
    this reading sits for the module's latest-wins rule: a model that opens with
    a summary list and then works through the questions one by one has committed
    to the per-question answers, and on the runs on record those two disagree.
    """
    lines = text.split("\n")
    offsets = _line_offsets(lines)
    letters: list[str] = []
    end = 0
    for index in range(len(lines) - 1, -1, -1):
        found = _anchor_letters_in_line(lines[index], n_choices)
        if found is None:
            continue          # reasoning, not an answer line
        if len(found) != 1:
            break             # ambiguous line: stop rather than pick from it
        if not letters:
            end = offsets[index]
        letters.insert(0, found[0])
        if len(letters) == expected:
            break
    return (end, letters) if len(letters) == expected else None


def _numbered_answers(text: str, *, expected: int,
                      n_choices: int) -> tuple[int, list[str]] | None:
    """One numbered answer per line, each followed by its own explanation.

    ``3. **A** - Gamma rays have the highest frequency ...``, for every question
    in turn. It differs from :func:`_per_question_answers` in having no answer
    marker to anchor on -- the line number *is* the marker -- and from a
    letter-only block in that the letter is followed by prose, so the line is not
    a bare list item.

    Two guards do all the work of keeping the reasoning out. The letter must be
    the first thing after the number and must not be followed by a word, so the
    ``1. A different reading of the question ...`` that a model writes while
    thinking aloud is not an answer; and the numbers found must cover 1..k
    exactly, so a stray numbered line contributes nothing on its own. Where a
    number appears more than once -- the model worked through the list twice --
    the last occurrence is the answer, which is this module's rule everywhere.

    Returned with the offset of the last line it read, so that this reading
    competes with the others by position like every other kind.
    """
    valid = _valid_letters(n_choices)
    letter = re.compile(
        rf"^(?:\*{{1,2}}|_{{1,2}}|\(|\[|\\boxed\s*\{{)?\s*([{valid[0]}-{valid[-1]}])"
        r"\s*(?:\}|\)|\]|\*{1,2}|_{1,2})?"
    )
    lines = text.split("\n")
    offsets = _line_offsets(lines)
    found: dict[int, str] = {}
    end = 0
    for index, line in enumerate(lines):
        numbered = _LINE_NUMBER.match(line)
        if not numbered:
            continue
        position = int(numbered.group(1))
        if not 1 <= position <= expected:
            continue
        hit = letter.match(line[numbered.end():])
        if not hit:
            continue
        after = line[numbered.end() + hit.end():].lstrip()
        if after and after[0].isalnum():
            continue          # "1. A different reading ..." -- prose, not an answer
        found[position] = hit.group(1)
        end = offsets[index]
    if len(found) != expected:
        return None
    return end, [found[position] for position in range(1, expected + 1)]


def _isolated_at_end(text: str, *, expected: int, n_choices: int) -> tuple[list[str], str]:
    """The option letters of the response's closing prose -- the guess.

    Two readings of "at the end", each bounded to the end of the response and
    never its middle. For one question the *last* isolated letter wins: a
    sentence weighing several options ends on the one it settles for. For more
    than one, only an exact-length reading is accepted, since two letters in a
    closing sentence say nothing about which of three questions they answer.
    """
    isolated = _isolated_pattern(n_choices)

    last_line = next((line for line in reversed(text.split("\n")) if line.strip()), "")
    found = isolated.findall(last_line)
    if expected == 1 and found:
        return [found[-1]], "last option letter of the final line"
    if len(found) == expected:
        return found, "option letters of the final line"

    tail = text.rsplit("\n\n", 1)[-1]
    found = isolated.findall(tail)
    if expected == 1 and found:
        return [found[-1]], "last option letter of the final block"
    if len(found) == expected:
        return found, "option letters of the final block"

    return [], ""


def extract_answer(
    text: str | None,
    *,
    expected: int,
    n_choices: int = 4,
    tail_fallback: bool = True,
) -> ExtractResult:
    """Locate the answer in a raw response and return it as text.

    ``expected`` is how many questions were asked and ``n_choices`` how many
    options each offered; both bound the search rather than shape its output.
    The returned ``value`` is a comma-separated list of the letters found, which
    :func:`parse_answer_list` then judges -- so a segment of the wrong length
    reaches the grader as a WRONG_LENGTH and not as an answer.

    With ``tail_fallback=False`` only answers the model wrote *as* answers are
    honoured -- step 1 -- which is the ``explicit`` ensemble and the control for
    what the closing-prose guess bought.
    """
    if expected < 1:
        raise ValueError(f"expected must be at least 1, got {expected}")
    _valid_letters(n_choices)

    if text is None or not str(text).strip():
        return ExtractResult(ExtractionStatus.EMPTY, detail="response was empty or None")

    # Whitespace first: a response whose spaces and newlines arrived as byte-BPE
    # sentinels is one long line, and every step below reads lines.
    text, restored = strip_bpe_sentinels(str(text))
    note = "; whitespace restored from BPE sentinels" if restored else ""
    anchors = _anchors(text)

    # Everything that might be a reading, as (position, anchor index or None,
    # letters or None). Anchors and letter-only blocks go into one list rather
    # than being tried in two passes, because they answer to the same rule --
    # the latest thing that looks like an answer wins -- and a rule applied by
    # two passes is two rules. A block that a marker's segment already ran into
    # appears twice over, at the marker's position and at its own; both readings
    # are the same letters, so which one wins changes only the label. An
    # anchor's letters are left unread until the scan below reaches it: a
    # reasoning trace can hold dozens of anchors, and reading every one of them
    # to answer a question the last one already settled would cost 7x here.
    candidates: list[tuple[int, str, int | None, list[str] | None]] = [
        (start, "anchor", index, None) for index, (start, _, _) in enumerate(anchors)
    ]
    for start, letters in _letter_line_blocks(text, n_choices):
        # A *short* block is not a reading at all, where a short anchor is one.
        # An anchor is a claim -- the model wrote "Answer:" and then lost count,
        # which is worth reporting as the WRONG_LENGTH it is. Bare letters
        # closing a line are not: two of them above a k=3 answer are a fragment
        # of the reasoning, and there is nothing to say about them.
        if len(letters) >= expected:
            candidates.append((start, f"letter-only lines at char {start}", None, letters))
    if expected > 1:
        # The per-question readings, each at the offset of its last answer line.
        # They are gathered here rather than tried after the loop below so that
        # position decides between them and a list the model wrote earlier: a
        # response that opens with a summary and then answers one by one has
        # committed to the answers, and those two disagree often enough on the
        # runs on record to matter.
        for reader, label in ((_per_question_answers, "one marked answer per line"),
                              (_numbered_answers, "one numbered answer per line")):
            found = reader(text, expected=expected, n_choices=n_choices)
            if found is not None:
                candidates.append((
                    found[0],
                    f"{label}, {expected} of them, last at char {found[0]}",
                    None, found[1],
                ))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    def read(index: int) -> list[str]:
        """The letters an anchor names -- computed only if it is reached."""
        start, end, kind = anchors[index]
        segment = None
        if kind == "boxed" and end and text[end - 1] == "{":
            segment = _brace_group(text, end - 1)
        if segment is None:
            segment = _LEAD.sub("", text[end:])
        return _letters_after(segment, expected=expected, n_choices=n_choices)

    def result(start: int, how: str, index: int | None,
               letters: list[str]) -> ExtractResult:
        if index is not None:
            how = f"{anchors[index][2]} anchor at char {start}"
        return ExtractResult(
            ExtractionStatus.OK, ", ".join(letters), n_matches=len(anchors),
            chosen_index=index, detail=f"{how}{note}",
        )

    # Exactly k letters is an answer to the k questions. Nothing else is, yet:
    # a reading that named more is preferred to one that named fewer only once
    # no exact reading exists at all, because "C,C,...,C" twenty times over
    # followed by a 24-letter restatement should be read as the twenty.
    readings: list[tuple[int, str, int | None, list[str]]] = []
    for start, how, index, letters in candidates:
        if letters is None:
            letters = read(index)
        if len(letters) == expected:
            return result(start, how, index, letters)
        readings.append((start, how, index, letters))

    # An over-long reading before a short one: `parse_answer_list` grades the
    # first k letters of the first (every position has a letter the model wrote
    # for it) and can do nothing at all with the second, which it reports as the
    # WRONG_LENGTH it is -- where the model lost count, rather than "no answer".
    # `readings` is already ordered latest first, so the first match is the last
    # such thing the model wrote.
    for start, how, index, letters in readings:
        if len(letters) > expected:
            return result(start, how, index, letters)
    for start, how, index, letters in readings:
        if letters:
            return result(start, how, index, letters)

    if tail_fallback:
        letters, how = _isolated_at_end(text, expected=expected, n_choices=n_choices)
        if letters:
            return ExtractResult(
                ExtractionStatus.OK, ", ".join(letters), n_matches=len(anchors),
                detail=f"no answer written as such; {how}{note}",
            )

    if anchors:
        detail = f"{len(anchors)} answer marker(s), none followed by an option letter"
    else:
        detail = "no answer marker and no option letter at the end of the response"
    return ExtractResult(ExtractionStatus.NO_ANSWER, n_matches=len(anchors),
                         detail=detail + note)
