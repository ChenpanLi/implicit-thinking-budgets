"""Where each term of a composed integral sits inside its rendered integral.

The integrals task grades one answer per term in the order the prompt shows the
terms, and SymPy reorders the summands when it renders the sum, so that order has
to be read off the rendered LaTeX (:func:`llm.core.math.per_term.term_order`).

Locating a term is not a substring search, because SymPy renders a term differently
alone than inside a sum: ``-sin(13*x - 15)`` alone is
``- \\sin{\\left(13 x - 15 \\right)}``, and inside an ``Add`` the sign is factored
out into the join. So each term is matched on its *unsigned body*, the sign is taken
back afterwards, and matches are claimed one-to-one: a composition may sum the same
atom more than once and each copy must get its own occurrence. Longer bodies are
matched first so a short term ("x") cannot steal the occurrence that belongs to a
longer one ("x e^{4 x + 5}").

Anything that cannot be located is a hard error naming the document. A silently
skipped term would leave a k-term composition graded as a (k-1)-term one.
"""
from __future__ import annotations

import functools
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable

import sympy as sp


class SpanError(RuntimeError):
    """A prompt whose terms could not be located."""


@dataclass(frozen=True)
class CharSpan:
    """One term's extent in the prompt, in characters."""

    index: int          # position in the composition's component list
    term: str           # the sympy source of the term
    latex: str          # how it is rendered inside the sum
    start: int
    end: int


@dataclass(frozen=True)
class PromptSpans:
    doc_id: str
    prompt: str
    spans: tuple[CharSpan, ...]


# --------------------------------------------------------------------------- #
# Locating the terms
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=4096)
def term_latex(term: str) -> str:
    try:
        return sp.latex(sp.sympify(term))
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise SpanError(f"could not render term {term!r}: {exc}") from exc


def unsigned_body(latex: str) -> str:
    """The term's rendering without the sign an `Add` would factor out."""
    stripped = latex.lstrip()
    if stripped.startswith("-"):
        return stripped[1:].lstrip()
    if stripped.startswith("+"):
        return stripped[1:].lstrip()
    return stripped


# A term's rendering must sit on its own in the sum, not glued to a longer name
# or number. Without this the term `-x` matches the `x` of the trailing `dx`
# (and `5 x^{9}` matches inside `15 x^{9}`), which reads a span belonging to
# something else entirely, and can make two terms claim the same occurrence.
_LEFT_BLOCKS = set(string.ascii_letters + string.digits + "\\")
_RIGHT_BLOCKS = set(string.ascii_letters + string.digits + "^_{")


def _on_boundary(haystack: str, start: int, end: int) -> bool:
    if start > 0 and haystack[start - 1] in _LEFT_BLOCKS:
        return False
    if end < len(haystack) and haystack[end] in _RIGHT_BLOCKS:
        return False
    return True


def _claim_occurrence(haystack: str, needle: str, taken: list[tuple[int, int]],
                      *, window: tuple[int, int] | None = None) -> tuple[int, int]:
    """First standalone occurrence of `needle` that overlaps nothing already claimed."""
    low, high = window or (0, len(haystack))
    start = haystack.find(needle, low, high)
    while start != -1:
        end = start + len(needle)
        overlaps = any(start < claimed_end and end > claimed_start
                       for claimed_start, claimed_end in taken)
        if not overlaps and _on_boundary(haystack, start, end):
            return start, end
        start = haystack.find(needle, start + 1, high)
    raise LookupError(needle)


def _extend_over_sign(prompt: str, start: int, limit: int) -> int:
    """Pull a leading `+`/`-` (and its spaces) into the span."""
    cursor = start
    while cursor > limit and prompt[cursor - 1] == " ":
        cursor -= 1
    if cursor > limit and prompt[cursor - 1] in "+-":
        cursor -= 1
        while cursor > limit and prompt[cursor - 1] == " ":
            cursor -= 1
        return cursor
    return start


# Everything a k-term integral may legitimately still contain once every term's
# span has been cut out of it: the integral notation, the separators, the
# differential, whitespace. Anything else means a span landed somewhere it does
# not belong -- and that is a wrong number, not a warning.
_SCAFFOLD = re.compile(r"^(\\int|\\,|\\left\(|\\right\)|d[a-zA-Z]|[\s+\-])*$")


def _verify_residue(prompt: str, region: tuple[int, int],
                    spans: Iterable[tuple[int, int]], doc_id: str) -> None:
    kept, cursor = [], region[0]
    for start, end in sorted(spans):
        kept.append(prompt[cursor:start])
        cursor = max(cursor, end)
    kept.append(prompt[cursor:region[1]])
    residue = "".join(kept)
    if not _SCAFFOLD.match(residue):
        raise SpanError(
            f"{doc_id}: after removing every term span, the integral still reads "
            f"{residue!r}. A term was matched somewhere it does not belong, so the "
            f"term order would be wrong.")


def locate_terms(prompt: str, record: dict[str, Any]) -> PromptSpans:
    """Character spans of every component term inside `prompt`."""
    doc_id = str(record.get("id", "?"))
    terms = [str(term) for term in record.get("component_integrands") or []]
    if not terms:
        raise SpanError(f"{doc_id}: record carries no component_integrands")

    integral = str(record.get("integrand_latex_integral") or "")
    region_start = prompt.find(integral) if integral else -1
    if region_start < 0:
        raise SpanError(
            f"{doc_id}: the record's integrand_latex_integral does not appear in the "
            f"prompt. The dataset and the prompt template have drifted apart; the "
            f"spans would be guesses.")
    region = (region_start, region_start + len(integral))

    rendered = [term_latex(term) for term in terms]
    bodies = [unsigned_body(latex) for latex in rendered]
    # Longest first: "x" would otherwise claim the occurrence inside "x e^{4 x}".
    order = sorted(range(len(terms)), key=lambda index: -len(bodies[index]))

    taken: list[tuple[int, int]] = []
    found: dict[int, tuple[int, int]] = {}
    for index in order:
        try:
            start, end = _claim_occurrence(prompt, bodies[index], taken, window=region)
        except LookupError:
            raise SpanError(
                f"{doc_id}: term {index} ({terms[index]!r}) renders as "
                f"{bodies[index]!r}, which does not appear in the prompt's integral. "
                f"Refusing to score a {len(terms)}-term composition from "
                f"{len(taken)} located terms.") from None
        taken.append((start, end))
        found[index] = (_extend_over_sign(prompt, start, region[0]), end)

    _verify_residue(prompt, region, found.values(), doc_id)

    spans = tuple(CharSpan(index=index, term=terms[index], latex=rendered[index],
                           start=found[index][0], end=found[index][1])
                  for index in range(len(terms)))
    return PromptSpans(doc_id=doc_id, prompt=prompt, spans=spans)

