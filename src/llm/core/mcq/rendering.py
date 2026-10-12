"""Rendering questions into the text the model reads.

One module, so the dataset builder, the lm-eval prompt and the span locator can
never disagree about what a question looks like. In the integrals experiment the
equivalent disagreement is real and expensive: SymPy re-sorts an ``Add`` when it
renders it, so ``component_integrands`` order and prompt order differ on almost
every composition, and a grader that trusts the list order labels every term
with its neighbour's position. Here prompt order *is* list order -- but only
because this function is the single renderer, and the dataset builder asserts it
by re-rendering what it wrote.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from .answers import LETTERS

# One blank line between questions. Load-bearing twice over: it is what makes a
# k=20 prompt readable, and it is the boundary the span locator cuts on.
QUESTION_SEPARATOR = "\n\n"


def number_label(index: int) -> str:
    """The default label: ``3.``."""
    return f"{index}."


def marker_label(index: int, width: int = 2) -> str:
    """The section-marker label: ``Q03``.

    Zero-padded so every label of a composition has the same shape, and chosen
    because no question of the MMLU-Redux pool contains ``Q<digit>`` -- a
    marker the model writes can then never be confused with question text it
    quotes.
    """
    return f"Q{index:0{width}d}"


def render_question(
    index: int,
    question: str,
    choices: Sequence[str],
    label: Callable[[int], str] = number_label,
) -> str:
    """One labelled question with its lettered options.

    ``index`` is 1-based, matching the numbering the prompt asks the model to
    answer in. ``label`` turns it into the text before the question; the
    default ``3.`` is what every dataset on disk was rendered with.
    """
    if not choices:
        raise ValueError(f"question {index} has no choices")
    if len(choices) > len(LETTERS):
        raise ValueError(f"question {index} has more choices than there are letters")
    lines = [f"{label(index)} {str(question).strip()}"]
    lines += [f"{LETTERS[i]}. {str(choice).strip()}" for i, choice in enumerate(choices)]
    return "\n".join(lines)


def render_block(
    questions: Sequence[str],
    choices_list: Sequence[Sequence[str]],
    label: Callable[[int], str] = number_label,
) -> tuple[str, list[tuple[int, int]]]:
    """The whole question block, plus each question's character span within it.

    The spans are returned rather than searched for afterwards, because a
    substring search is exactly the mistake this module exists to prevent: two
    questions in one composition may share a prefix.
    """
    if len(questions) != len(choices_list):
        raise ValueError(
            f"{len(questions)} questions but {len(choices_list)} choice lists; "
            "the per-question fields must be parallel"
        )
    if not questions:
        raise ValueError("a composition must hold at least one question")

    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for position, (question, choices) in enumerate(zip(questions, choices_list), start=1):
        text = render_question(position, question, choices, label)
        if parts:
            cursor += len(QUESTION_SEPARATOR)
        spans.append((cursor, cursor + len(text)))
        cursor += len(text)
        parts.append(text)
    return QUESTION_SEPARATOR.join(parts), spans
