"""Rendering problems into the text the model reads.

One module, so the dataset builder, the lm-eval prompt and the span locator can
never disagree about what a problem looks like. In the integrals experiment the
equivalent disagreement is real and expensive: SymPy re-sorts an ``Add`` when it
renders it, so ``component_integrands`` order and prompt order differ on almost
every composition, and a grader that trusts the list order labels every term
with its neighbour's position. Here prompt order *is* list order -- but only
because this function is the single renderer, and the dataset builder asserts it
by re-rendering what it wrote.

Two choices differ from :mod:`llm.core.oneword.rendering`, and both are forced
by how much longer these problems are:

* **The separator is a blank line**, as in the multiple-choice renderer, not a
  single newline. A GSM8K word problem is several sentences and a MATH-500
  problem can be a paragraph plus a displayed equation plus an Asymptote
  figure; run twenty of them together on consecutive lines and the numbering is
  the only thing telling them apart.
* **The problem text is kept as written, except that blank lines inside it are
  collapsed** (:func:`clean_problem`). The one-word renderer collapses *all*
  whitespace into single spaces, which would destroy a displayed equation and an
  ``[asy]`` block. But 42 MATH-500 problems ship an Asymptote figure whose source
  contains blank lines, and a blank line inside a problem in a prompt whose
  problems are separated by blank lines is an ambiguous boundary. Collapsing
  only the blank lines keeps the figure readable -- Asymptote does not care --
  and keeps "a blank line starts the next problem" true.

The cleaning happens at pool-build time, so the ``question`` field stored in the
dataset is exactly the text that is rendered; nothing downstream has to
re-derive it.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# One blank line between problems. Load-bearing twice over: it is what makes a
# k=20 prompt readable, and it is the boundary a reader (human or model) cuts on.
QUESTION_SEPARATOR = "\n\n"

# A run of blank lines inside one problem, collapsed to a single newline. See
# the module docstring: the blank line is the *boundary*, so it must not also
# appear inside a problem.
_BLANK_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")


def clean_problem(text: str) -> str:
    """One problem's text as it is stored and as it is rendered.

    Trailing whitespace goes from every line, blank lines inside the problem
    collapse, and nothing else changes: the LaTeX, the line breaks of a
    displayed equation and the Asymptote source all survive byte for byte.
    """
    lines = [line.rstrip() for line in str(text).strip().split("\n")]
    return _BLANK_RUN.sub("\n", "\n".join(lines)).strip()


def number_label(index: int) -> str:
    """The default label: ``3.``."""
    return f"{index}."


def marker_label(index: int, width: int = 2) -> str:
    """The section-marker label: ``Q03``.

    Zero-padded so every label of a composition has the same shape. Used by
    ``--section-markers`` runs, whose thinking is cut at every label mention
    (``llm.core.analysis.question_sections.parse_label_sections``). No problem
    of this answer shape's pools contains ``Q<digit>``.
    """
    return f"Q{index:0{width}d}"


def render_question(
    index: int,
    question: str,
    label: Callable[[int], str] = number_label,
) -> str:
    """One numbered problem.

    ``index`` is 1-based, matching the numbering the prompt asks the model to
    answer in. A multi-line problem keeps its line breaks; only the first line
    carries the number, which is the format every reader of these prompts --
    and the numbered-block answer reader -- already expects.
    """
    text = clean_problem(question)
    if not text:
        raise ValueError(f"problem {index} is empty")
    return f"{label(index)} {text}"


def render_block(
    questions: Sequence[str],
    label: Callable[[int], str] = number_label,
) -> tuple[str, list[tuple[int, int]]]:
    """The whole problem block, plus each problem's character span within it.

    The spans are returned rather than searched for afterwards, because a
    substring search is exactly the mistake this module exists to prevent: two
    problems in one composition may share a prefix.
    """
    if not questions:
        raise ValueError("a composition must hold at least one problem")

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
