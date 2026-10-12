"""Rendering questions into the text the model reads.

One module, so the dataset builder, the lm-eval prompt and the span locator can
never disagree about what a question looks like. In the integrals experiment the
equivalent disagreement is real and expensive: SymPy re-sorts an ``Add`` when it
renders it, so ``component_integrands`` order and prompt order differ on almost
every composition, and a grader that trusts the list order labels every term
with its neighbour's position. Here prompt order *is* list order -- but only
because this function is the single renderer, and the dataset builder asserts it
by re-rendering what it wrote.

A one-word question is one line, unlike a multiple-choice question with its
block of lettered options, so the separator is a single newline and the block is
a plain numbered list.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

# One question per line. The blank line the multiple-choice renderer uses is
# what separates two *blocks*; here there are no blocks to separate.
QUESTION_SEPARATOR = "\n"


def number_label(index: int) -> str:
    """The default label: ``3.``."""
    return f"{index}."


def marker_label(index: int, width: int = 2) -> str:
    """The section-marker label: ``Q03``.

    Zero-padded so every label of a composition has the same shape. Used by
    ``--section-markers`` runs, whose thinking is cut at every label mention
    (``llm.core.analysis.question_sections.parse_label_sections``). No question
    of this answer shape's pools contains ``Q<digit>``.
    """
    return f"Q{index:0{width}d}"


def render_question(
    index: int,
    question: str,
    label: Callable[[int], str] = number_label,
) -> str:
    """One numbered question.

    ``index`` is 1-based, matching the numbering the prompt asks the model to
    answer in.
    """
    text = " ".join(str(question).split())
    if not text:
        raise ValueError(f"question {index} is empty")
    return f"{label(index)} {text}"


def render_block(
    questions: Sequence[str],
    label: Callable[[int], str] = number_label,
) -> tuple[str, list[tuple[int, int]]]:
    """The whole question block, plus each question's character span within it.

    The spans are returned rather than searched for afterwards, because a
    substring search is exactly the mistake this module exists to prevent: two
    questions in one composition may share a prefix.
    """
    if not questions:
        raise ValueError("a composition must hold at least one question")

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for position, question in enumerate(questions, start=1):
        text = render_question(position, question, label)
        if parts:
            cursor += len(QUESTION_SEPARATOR)
        spans.append((cursor, cursor + len(text)))
        cursor += len(text)
        parts.append(text)
    return QUESTION_SEPARATOR.join(parts), spans
