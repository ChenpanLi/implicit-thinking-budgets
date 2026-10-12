"""Cutting a thinking trace into per-question sections at every label mention.

Under ``--section-markers`` the prompt labels the questions Q01, Q02, ... and asks
the model to name a question by its label whenever it works on it.
:func:`parse_label_sections` reads every label mention as the start of that
question's section, and :func:`token_breakdown` sums the tokens each question cost.

Labels are read in any spelling a question number is written in: ``Q3``, ``Q03``,
``q 3``, ``Q-03``, ``Q.3``, ``Qn3``, ``Question 3``, ``question3``,
``QUESTION #3``, ``Question No. 3``, because a model writes the label however it
likes and a missed label silently moves tokens into the wrong bucket.

Everything inside the thinking but before the first label is ``outside``, and
everything after ``</think>`` is ``answer``.

The module is stdlib only and knows nothing about tokenizers: the parse works in
characters, and :func:`token_breakdown` takes the token offsets a caller computed
with whatever tokenizer is right.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# A question label. The group is the number; leading zeros are allowed and
# dropped. `(?<![A-Za-z0-9])` keeps `FAQ3` and `seq3` from reading as labels.
_LABEL = (
    r"(?<![A-Za-z0-9])"
    r"(?:question|ques|qn|q)"
    r"[ \t]*[#.:_\-]?[ \t]*"
    r"(?:(?:no\.?|num\.?|number)[ \t]*[#.:]?[ \t]*)?"
    r"0*(?P<{name}>\d{{1,3}})(?!\d)"
)
# A section with less content than this (after its own label) is flagged `empty`.
EMPTY_SECTION_CHARS = 20


@dataclass(frozen=True)
class Marker:
    kind: str          # always "label"
    label: int         # 1-based question number as written
    start: int         # char offsets of the matched marker text
    end: int
    text: str


@dataclass(frozen=True)
class Section:
    label: int
    start: int          # from the label's first char ...
    end: int            # ... to the next label's first char, or `</think>`
    closed_by: str      # "next_label" | "think_end"
    content_chars: int  # non-whitespace chars after the section's own label

    @property
    def empty(self) -> bool:
        return self.content_chars < EMPTY_SECTION_CHARS


@dataclass
class SectionParse:
    text: str
    k: int
    think_start: int
    think_end: int                 # index of `</think>`, or len(text) when absent
    think_closed: bool
    markers: list[Marker] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    # Never filled by a label parse; `summary` writes them as empty lists.
    orphan_ends: list[Marker] = field(default_factory=list)
    mismatched_ends: list[Marker] = field(default_factory=list)
    out_of_range: list[Marker] = field(default_factory=list)     # label 0 or > k
    answer_markers: list[Marker] = field(default_factory=list)   # after </think>, ignored

    def labels_never_marked(self) -> list[int]:
        seen = {s.label for s in self.sections}
        return [i for i in range(1, self.k + 1) if i not in seen]

    def char_breakdown(self) -> dict:
        """Characters per question, `outside`, `preamble` and `answer`; they sum to len(text)."""
        per = {i: 0 for i in range(1, self.k + 1)}
        for s in self.sections:
            per[s.label] += s.end - s.start
        return {
            "per_question": per,
            "outside": (self.think_end - self.think_start) - sum(per.values()),
            "preamble": self.think_start,
            "answer": len(self.text) - self.think_end,
        }


_LABEL_RE = re.compile(_LABEL.format(name="n"), re.IGNORECASE)


def parse_label_sections(text: str, k: int) -> SectionParse:
    """Sections of `text`'s thinking cut at every mention of a question label.

    Any label in any spelling `_LABEL` accepts switches
    the thinking to that question, and it stays there until a *different*
    label is mentioned. A repeated mention of the current label continues its
    section, so a visit is counted only on a switch. Labels outside 1..k are
    logged in `out_of_range` and change nothing. Thinking before the first
    label is `outside`; the last section closes at `</think>`.

    A recap such as `Q01: B, Q02: A, Q03: C` becomes a run of tiny sections --
    that is the rule applied literally, and it costs a handful of tokens per
    question; the markers are all in `parse.markers`, so it stays visible.
    """
    open_at = text.find(THINK_OPEN)
    think_start = open_at + len(THINK_OPEN) if open_at != -1 else 0
    close_at = text.find(THINK_CLOSE, think_start)
    think_closed = close_at != -1
    think_end = close_at if think_closed else len(text)

    parse = SectionParse(text=text, k=k, think_start=think_start,
                         think_end=think_end, think_closed=think_closed)
    current: Marker | None = None

    def close(end: int, how: str) -> None:
        body = text[current.end:end]
        parse.sections.append(Section(
            label=current.label, start=current.start, end=end, closed_by=how,
            content_chars=len(re.sub(r"\s", "", body)),
        ))

    for m in _LABEL_RE.finditer(text):
        marker = Marker("label", int(m.group("n")), m.start(), m.end(), m.group(0))
        if marker.start < think_start:
            continue
        if marker.start >= think_end:
            parse.answer_markers.append(marker)
            continue
        parse.markers.append(marker)
        if not 1 <= marker.label <= k:
            parse.out_of_range.append(marker)
            continue
        if current is not None and marker.label == current.label:
            continue
        if current is not None:
            close(marker.start, "next_label")
        current = marker

    if current is not None:
        close(think_end, "think_end")
    return parse


def token_breakdown(parse: SectionParse, offsets: Sequence[tuple[int, int]]) -> dict:
    """Tokens per question, `outside`, `preamble` and `answer`, from a tokenizer's offsets.

    `offsets` are the (start, end) character offsets of every token of
    `parse.text`, in order -- a Hugging Face fast tokenizer's
    `return_offsets_mapping`. A token belongs to the region its first character
    falls in, so the buckets partition the tokens and sum to `len(offsets)`.
    Marker tokens count inside their section: writing the marker is part of
    working on that question.
    """
    bounds = sorted((s.start, s.end, s.label) for s in parse.sections)
    per = {i: 0 for i in range(1, parse.k + 1)}
    visits = {i: 0 for i in range(1, parse.k + 1)}
    for s in parse.sections:
        visits[s.label] += 1
    outside = preamble = answer = 0
    j = 0
    for start, _ in offsets:
        if start >= parse.think_end:
            answer += 1
            continue
        if start < parse.think_start:
            preamble += 1
            continue
        while j < len(bounds) and bounds[j][1] <= start:
            j += 1
        if j < len(bounds) and bounds[j][0] <= start:
            per[bounds[j][2]] += 1
        else:
            outside += 1
    return {
        "per_question": per,
        "visits": visits,
        "outside": outside,
        "preamble": preamble,
        "answer": answer,
        "total": len(offsets),
    }


def summary(parse: SectionParse) -> dict:
    """Everything about a parse except its token counts, as plain data."""
    def m(x: Marker) -> dict:
        return {"kind": x.kind, "label": x.label, "start": x.start, "text": x.text}

    return {
        "k": parse.k,
        "think_closed": parse.think_closed,
        "n_markers": len(parse.markers),
        "n_sections": len(parse.sections),
        "sections": [
            {"label": s.label, "start": s.start, "end": s.end,
             "closed_by": s.closed_by, "empty": s.empty}
            for s in parse.sections
        ],
        "chars": parse.char_breakdown(),
        "labels_never_marked": parse.labels_never_marked(),
        "orphan_ends": [m(x) for x in parse.orphan_ends],
        "mismatched_ends": [m(x) for x in parse.mismatched_ends],
        "out_of_range": [m(x) for x in parse.out_of_range],
        "n_answer_markers": len(parse.answer_markers),
        "closed_by": {how: sum(s.closed_by == how for s in parse.sections)
                      for how in sorted({s.closed_by for s in parse.sections})},
    }
