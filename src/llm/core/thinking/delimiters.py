"""Where a model's thinking block starts and ends, per tokenizer.

Budget forcing has to interrupt a reasoning model *inside* its thinking block:
it stops generation when the model tries to close the block, and it closes the
block itself once the budget is spent. Both halves need the delimiters as
tokens, not as prose, so this module resolves them from the tokenizer rather
than from a table keyed by model name -- a checkpoint that renames itself, or a
new distill of a known family, then needs no edit here.

Two families are recognised today:

``think-tags``
    ``<think>`` / ``</think>``, both single special tokens. Qwen3-Thinking, all
    three R1 distills and Phi-4-reasoning are this family.
``gemma-channel``
    Gemma 4's channel syntax: thinking opens with ``<|channel>thought\\n`` and
    closes with ``\\n<channel|>``. The close is two tokens, of which only the
    last (``<channel|>``) is the delimiter proper -- which is why
    :attr:`ThinkDelimiters.stop_token_id` is the *last* token of the close text
    rather than the whole of it.

A model whose tokenizer matches neither is a hard error naming both overrides,
never a guess: forcing generation to stop on a token the model does not use
would produce a run whose "thinking budget" measured nothing, and it would look
entirely normal in the figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# (family name, text that opens thinking, text that closes it). Order is the
# match order; each entry is accepted only when every marker it needs is a
# single token in the tokenizer's vocabulary.
FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("think-tags", "<think>", "</think>"),
    ("gemma-channel", "<|channel>thought\n", "\n<channel|>"),
)


class DelimiterError(RuntimeError):
    """The thinking delimiters could not be resolved for this tokenizer."""


@dataclass(frozen=True)
class ThinkDelimiters:
    """The two markers, as text and as tokens, for one tokenizer."""

    family: str
    open_text: str
    close_text: str
    open_token_ids: tuple[int, ...]
    close_token_ids: tuple[int, ...]

    @property
    def stop_token_id(self) -> int:
        """The token whose appearance means "the model ended its thinking".

        The last token of the close text: for ``</think>`` that is the whole of
        it, and for Gemma's ``\\n<channel|>`` it is ``<channel|>`` -- the leading
        newline is layout, and stopping on it would cut the model off at its
        first line break.
        """
        return self.close_token_ids[-1]

    def opens_thinking(self, prompt_text: str) -> bool:
        """Does this rendered chat prompt already leave thinking open?

        Several chat templates (Qwen3-Thinking, the R1 distills) end their
        generation prompt with the open marker, and several do not. The question
        is not "does the marker appear" but "which marker appears last": Gemma's
        non-thinking template emits an *empty* thought channel, opening and
        closing it, and appending a second opener there is what makes the block
        forceable at all.
        """
        return prompt_text.rfind(self.open_text) > prompt_text.rfind(self.close_text)


def _single_token(tokenizer: Any, text: str) -> int | None:
    """The id of `text` when it encodes to exactly one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if len(ids) == 1 else None


def resolve_delimiters(
    tokenizer: Any,
    *,
    open_text: str | None = None,
    close_text: str | None = None,
) -> ThinkDelimiters:
    """The thinking delimiters for `tokenizer`, or an explicit override.

    An override skips detection entirely and is checked only for encodability,
    since the point of it is to name markers this module does not know about.
    """
    if (open_text is None) != (close_text is None):
        raise DelimiterError(
            "give both --think-open and --think-close or neither: a block that "
            "is opened with one family's marker and closed with another's is "
            "not a thinking block"
        )
    if open_text is not None and close_text is not None:
        return _build(tokenizer, "override", open_text, close_text)

    for family, opener, closer in FAMILIES:
        # The close marker's own last token is what generation stops on, so it
        # is the one that must exist as a token; requiring the opener too keeps
        # a tokenizer that merely happens to spell `</think>` from matching.
        if _single_token(tokenizer, closer.strip()) is None:
            continue
        if _single_token(tokenizer, opener.strip()) is None:
            continue
        return _build(tokenizer, family, opener, closer)

    known = ", ".join(name for name, _, _ in FAMILIES)
    raise DelimiterError(
        f"this tokenizer has no recognised thinking delimiters (tried: {known}). "
        "If the model has them under other names, pass --think-open/--think-close; "
        "if it is not a reasoning model, budget forcing does not apply to it."
    )


def _build(tokenizer: Any, family: str, open_text: str, close_text: str) -> ThinkDelimiters:
    open_ids = tuple(int(i) for i in tokenizer.encode(open_text, add_special_tokens=False))
    close_ids = tuple(int(i) for i in tokenizer.encode(close_text, add_special_tokens=False))
    if not open_ids or not close_ids:
        raise DelimiterError(
            f"the {family} delimiters encode to nothing "
            f"({open_text!r} -> {open_ids}, {close_text!r} -> {close_ids})"
        )
    return ThinkDelimiters(
        family=family,
        open_text=open_text,
        close_text=close_text,
        open_token_ids=open_ids,
        close_token_ids=close_ids,
    )
