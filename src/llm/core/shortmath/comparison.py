"""What a comparison of one written answer against one reference reports.

GSM8K, MATH-500 and AIME share :mod:`llm.core.shortmath` and differ in exactly
one thing: what it means for a written answer to be the reference answer. GSM8K
and AIME compare numbers; MATH-500 compares LaTeX. Everything else -- the rendering, the block
location, the line splitting, the grading loop, the trace record -- is the same
code reading the same fields.

That one difference is a :class:`Comparator`: a callable taking the candidate
and the accepted variants and returning a :class:`Match`. The match records
*how* it was reached and not only whether it was, because a comparison nobody
can read back is a comparison nobody can audit: ``exact`` and ``symbolic`` are
both "correct", but a run whose correct answers are mostly ``symbolic`` is a run
whose grader is doing a lot of work, and that is worth being able to see in the
trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Match:
    """Why a written answer did or did not count as the reference."""

    matched: bool
    variant: str | None = None      # the accepted answer it matched, if any
    level: str = ""                 # the rule that decided: "exact", "numeric", ...
    normalized: str = ""            # the candidate, normalised
    detail: str = ""                # why a comparison could not be made at all
    rules: tuple[str, ...] = field(default_factory=tuple)

    def to_data(self) -> dict[str, object]:
        data: dict[str, object] = {
            "matched": self.matched, "variant": self.variant,
            "level": self.level, "normalized": self.normalized,
        }
        if self.rules:
            data["rules"] = list(self.rules)
        if self.detail:
            data["detail"] = self.detail
        return data


class Comparator(Protocol):
    """Compare one written answer against a problem's accepted variants."""

    def __call__(self, candidate: str | None, accepted: object) -> Match:
        ...
