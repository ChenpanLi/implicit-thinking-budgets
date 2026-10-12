"""Short-answer mathematics: the core shared by the GSM8K, MATH-500 and AIME trees.

Three task layers import this package and nothing else of each other's, the
same arrangement :mod:`llm.core.mcq` has with the MMLU and GPQA task layers. They ask
the same question -- does a model that solves k problems one at a time solve
them in one prompt? -- of problems with the same *shape*: a paragraph of text,
one short answer, k of them numbered in a block.

Everything here is shared between the three but one thing, and the exception is
the point of the split: what it means for a written answer to *be* the reference
answer. GSM8K and AIME compare numbers (:mod:`llm.core.shortmath.numeric`);
MATH-500 compares LaTeX through an ordered ladder ending in a symbolic check
(:mod:`llm.core.shortmath.latex`). Both satisfy
:class:`llm.core.shortmath.comparison.Comparator`, and the grader takes one as
an argument rather than branching on a tree's name.

The pool modules -- what each corpus is, and what counts as a usable row -- are
:mod:`llm.core.shortmath.gsm8k`, :mod:`llm.core.shortmath.math500` and
:mod:`llm.core.shortmath.aime`, one per dataset.
"""
