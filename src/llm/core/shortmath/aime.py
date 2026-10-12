r"""The AIME pool: where its two halves come from, and what counts as usable.

The counterpart of :mod:`llm.core.shortmath.gsm8k` and
:mod:`llm.core.shortmath.math500`, and the one pool in this repository built
from **two** Hub datasets rather than one: the 2024 and 2025 American Invitational
Mathematics Examinations, taken together, which is 60 problems (four contests of
15). Nobody publishes the two years in one repo, and evaluating on 2024 alone is
a 30-problem corpus -- small enough that one item is 3.3 accuracy points.

* 2024 -- ``HuggingFaceH4/aime_2024`` (AIME I and II, 30 problems)
* 2025 -- ``yentinglin/aime_2025`` (AIME I and II, 30 problems)

The two were chosen over the other uploads of the same contests because their
columns are the *same* columns -- ``problem``, ``answer``, ``solution``, ``url``,
``year`` -- so merging them needs no per-source field map, only a per-source
revision in the manifest.

Three properties of the pool shape everything downstream.

**Every answer is an integer from 0 to 999.** That is the AIME's own answer
format, not a convention of ours, and it is *checked* rather than assumed: a row
whose answer is not one is dropped and counted. The comparison is therefore
GSM8K's (:mod:`llm.core.shortmath.numeric`) and not MATH-500's LaTeX ladder --
there is no fraction, surd or interval to compare.

**The category is the contest, and the level is the problem number.** The AIME
ships no subject label, but it ships something better: a problem's *position*.
An AIME paper runs from 1 (easiest) to 15 (hardest) by construction, so
``level`` is the problem number and carries real difficulty information, the way
MATH-500's 1-5 band does. ``category`` is the contest the problem came from
(``2024 AIME I``, ``2025 AIME II``, ...) -- four of them, 15 problems each, so a
category-balanced draw is balanced across the four papers instead of degenerating
to a uniform draw the way GSM8K's single category does.

**The pool is small and the problems are hard.** 60 items is a tenth of MATH-500
and a twentieth of GSM8K, and a 4B model solves few of them three times out of
three. That is a fact about the corpus, not about this module, but it decides two
things: a k=20 composed set may be impossible to draw for a weak model (the AIME
composed sets stop at k=10), and a batching pass has only ``60/k`` prompts per
seed, so its accuracy is noisier than any other dataset's at the same k.
"""
from __future__ import annotations

import re
from typing import Any

#: One source per year, in the order the pool is laid out. ``(dataset, config,
#: split)``, so a caller can load them with one loop and pin one revision each.
SOURCES: tuple[tuple[str, str | None, str], ...] = (
    ("HuggingFaceH4/aime_2024", None, "train"),
    ("yentinglin/aime_2025", None, "train"),
)

QUESTION_FIELD = "problem"
ANSWER_FIELD = "answer"
URL_FIELD = "url"

BROAD_CATEGORY = "math"

#: A paper's problem numbers, 1 (easiest) to 15 (hardest). Used as ``level``.
LEVELS: tuple[int, ...] = tuple(range(1, 16))

#: The AIME's answer space: an integer from 0 to 999 inclusive.
ANSWER_MIN, ANSWER_MAX = 0, 999

# `.../wiki/index.php/2024_AIME_I_Problems/Problem_7` -- the AoPS URL both
# sources carry, and the only field that names the contest and the problem
# number. `year` is a string in one source and an integer in the other, and
# neither carries the half, so this is read instead of either.
_URL = re.compile(r"(\d{4})_AIME_(I{1,2})_Problems/Problem_(\d+)", re.IGNORECASE)
# The answer as the files write it: a bare integer, occasionally padded (`016`)
# or wrapped in a box by an upload that copied it out of a solution.
_ANSWER = re.compile(r"[+-]?\d+")
_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def _url_parts(row: dict[str, Any]) -> tuple[str, str, int] | None:
    """``(year, half, problem number)`` from the row's AoPS URL, or ``None``."""
    match = _URL.search(str(row.get(URL_FIELD, "")))
    if match is None:
        return None
    return match.group(1), match.group(2).upper(), int(match.group(3))


def contest_of(row: dict[str, Any]) -> str | None:
    """The paper this problem is from: ``2025 AIME II``. The record's category."""
    parts = _url_parts(row)
    return None if parts is None else f"{parts[0]} AIME {parts[1]}"


def year_of(row: dict[str, Any]) -> int | None:
    """The contest year, read from the URL rather than from ``year``.

    The 2024 source stores the year as a string and the 2025 source as an
    integer; reading the URL gives one type for both and needs no coercion that
    a future third source could quietly fail.
    """
    parts = _url_parts(row)
    return None if parts is None else int(parts[0])


def problem_number_of(row: dict[str, Any]) -> int | None:
    """The problem's position in its paper, 1 to 15. The record's ``level``.

    An AIME paper is ordered by difficulty by construction, so this is a real
    difficulty band and not a serial number: problem 15 of any paper is meant to
    be the hardest thing on it.
    """
    parts = _url_parts(row)
    if parts is None or parts[2] not in LEVELS:
        return None
    return parts[2]


def final_answer(row: dict[str, Any]) -> str | None:
    """The canonical answer of one raw row, or ``None`` if it has none.

    The whole field has to reduce to a single integer in ``[0, 999]``. "A number
    somewhere in the text" is deliberately not accepted -- it would read
    ``K(0.25)`` as ``0.25`` -- but a ``\boxed{...}``
    wrapper is unwrapped first, because an upload that copied its answer out of a
    solution has still written the answer down.
    """
    raw = str(row.get(ANSWER_FIELD, "")).strip()
    if not raw:
        return None
    boxed = _BOXED.fullmatch(raw)
    if boxed is not None:
        raw = boxed.group(1).strip()
    raw = raw.replace(",", "").replace("$", "").strip()
    if not _ANSWER.fullmatch(raw):
        return None
    value = int(raw)
    if not ANSWER_MIN <= value <= ANSWER_MAX:
        return None
    # Normalised, so `016` and `16` are one reference rather than two: the
    # comparison is numeric and would cope either way, but a reference that
    # itself needs lenient parsing is a reference nobody can read at a glance.
    return str(value)


def is_clean(row: dict[str, Any]) -> bool:
    """Is this row usable as a problem?

    Four things are checkable and all four are checked: that a problem was
    asked, that an answer inside the AIME's answer space was written down for it,
    that the URL names the contest (which is the record's category) and that it
    names a problem number in 1..15 (which is the record's level). A row failing
    any of them is dropped and *counted* by the generator rather than silently
    skipped -- with 60 problems, one dropped in silence is 1.7% of the corpus.
    """
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return False
    if final_answer(row) is None:
        return False
    return contest_of(row) is not None and problem_number_of(row) is not None


def reject_reason(row: dict[str, Any]) -> str:
    """Why a row is not usable, for the manifest's rejection counts."""
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return "empty_problem"
    if not str(row.get(ANSWER_FIELD, "")).strip():
        return "empty_answer"
    if final_answer(row) is None:
        return "answer_is_not_an_integer_0_999"
    if contest_of(row) is None:
        return "url_does_not_name_the_contest"
    return "url_does_not_name_a_problem_number"


def sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    """Chronological order: year, then paper I before II, then problem number.

    The pool is written in this order, so ``q-0000`` is 2024 AIME I problem 1 and
    ``q-0059`` is 2025 AIME II problem 15 -- an id that can be read by eye, and
    an order that does not depend on how either upload happened to shuffle its
    rows (the 2024 one is not in contest order).
    """
    parts = _url_parts(row)
    if parts is None:
        return (9999, 9, 99)
    year, half, number = parts
    return (int(year), len(half), number)
