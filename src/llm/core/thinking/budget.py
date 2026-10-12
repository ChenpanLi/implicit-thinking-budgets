"""Budget forcing: making a reasoning model think for a chosen number of tokens.

The method is the one from the test-time-scaling literature (Muennighoff et al.,
*s1: Simple test-time scaling*, 2025), and it has exactly two halves:

* **too little thinking** -- when the model tries to close its thinking block
  before the budget is spent, the closing delimiter is suppressed, the string
  ``Wait`` is appended to the trace, and generation continues. The model then
  second-guesses itself instead of stopping.
* **too much thinking** -- when the budget is spent, generation is cut off
  mid-thought, the closing delimiter is appended *by us*, and so is an
  ``Answer:`` cue. The model is then decoding an answer, not a continuation.

The two halves are independent, and :class:`BudgetMode` is which of them apply:

* ``EXACT`` -- both, and the default. The realized thinking length lands on the
  budget.
* ``LOWER`` -- the floor alone. The model must spend at least the budget, and
  is then left to think for as long as it likes; the only ceiling is the
  context window, minus the room its own answer needs.
* ``UPPER`` -- the cap alone. The model may stop whenever it wants to, but is
  cut off at the budget if it has not. Nothing is ever injected mid-thought,
  so this is the one mode in which the thinking is entirely the model's own.

``margin`` relaxes the *floor* only, in tokens: a model that closes its block
within ``margin`` of the budget is taken at its word rather than being pushed
over the line. It is 0 by default. There is deliberately no
matching relaxation of the cap: a cap that could be exceeded is not a cap, and
the quantity a figure plots on its x-axis would no longer bound the condition.

Everything here is domain-free: it takes token ids in and gives text out, and
knows nothing about integrals, questions or grading. :mod:`llm.core.thinking.runner`
is what joins it to an experiment.

Why this cannot be done through lm-eval
---------------------------------------
lm-eval issues one generation per request and hands back its text. Budget
forcing is a *loop* over the same request -- generate, inspect, inject, generate
again -- so the request has to be re-issued with the previous rounds' tokens
prepended. That is a property of the harness, not of the task, which is why the
budget-forced runs live in their own directory and are produced by their own
runner (``python -m llm.budget_forcing``) rather than by a flag on ``llm.run``.

Two accounting rules, because both are easy to get quietly wrong:

* **Injected tokens count against the budget.** A ``Wait`` is thinking the model
  did not write but did read, and a run whose x-axis said 4096 while the model
  saw 4096 plus one token per continuation would be mislabelled by an amount
  that grows with how often the model tried to stop.
* **The thinking budget is never silently reduced.** A prompt that cannot fit
  its budget and an answer inside the context window is a hard error naming the
  documents, not a quietly shorter budget -- the budget is the independent
  variable of the whole experiment, and a run in which it was silently cut
  would look completely normal in the figures.
"""
from __future__ import annotations

import enum
import sys
from dataclasses import dataclass, field
from typing import Any, Sequence

# A forced answer needs somewhere to go. Fewer tokens than this left in the
# context window is treated as "this budget does not fit this model", rather
# than as an answer that comes back empty and is scored as a wrong one -- the
# same stance the checker takes on a timeout.
MIN_ANSWER_TOKENS = 32


class BudgetMode(str, enum.Enum):
    """Which half of the forcing applies -- see this module's docstring.

    A ``str`` enum so the value travels into ``run.json``, ``generations.jsonl``
    and a run id without a converter, and so a mode read back out of one of
    those files compares equal to the enum member.
    """

    LOWER = "lower"      # a floor: at least `budget` thinking tokens
    UPPER = "upper"      # a cap: at most `budget` thinking tokens
    EXACT = "exact"      # both: a floor and a cap at `budget`

    @property
    def has_floor(self) -> bool:
        return self in (BudgetMode.LOWER, BudgetMode.EXACT)

    @property
    def has_cap(self) -> bool:
        return self in (BudgetMode.UPPER, BudgetMode.EXACT)


@dataclass
class ForcedGeneration:
    """One budget-forced response, and how it was produced."""

    text: str
    prompt_tokens: int
    # Tokens spent inside the thinking block, injected continuation cues
    # included. This is the quantity the budget caps and the figures plot.
    thinking_tokens: int
    thinking_budget: int
    # How often the model tried to stop thinking and was made to continue.
    continuations: int
    answer_tokens: int
    rounds: int
    # True when this generation met the condition it was given: the floor was
    # reached under LOWER and EXACT, and trivially true under UPPER, which has
    # no floor to fall short of. False means the continuation cap ended the
    # loop first.
    budget_reached: bool
    # Which halves of the forcing were in effect, and the tolerance the floor
    # was allowed. On every generation, so a `generations.jsonl` row says what
    # produced it without a lookup into the run's `run.json`.
    budget_mode: str = BudgetMode.EXACT.value
    budget_margin: int = 0
    # How the thinking block actually ended: the model closed it itself (only
    # possible where the mode left it that freedom), or we cut it off at the
    # cap. Exactly one of the two is true.
    closed_by_model: bool = False
    cap_hit: bool = False
    # Always 0 and None here: the budget counts from the first thinking token.
    # Kept so that every `generations.jsonl` row has the same columns.
    free_tokens: int = 0
    count_started: bool | None = None
    # vLLM's finish reason for the *answer* generation, so an answer truncated
    # by --answer-tokens is visible rather than being read as a bad answer.
    answer_finish_reason: str | None = None
    # Set when the continuation cap ended the loop early; the run is still
    # usable, but this document's thinking is shorter than the budget says.
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "thinking_tokens": self.thinking_tokens,
            "thinking_budget": self.thinking_budget,
            "continuations": self.continuations,
            "answer_tokens": self.answer_tokens,
            "rounds": self.rounds,
            "budget_reached": self.budget_reached,
            "budget_mode": self.budget_mode,
            "budget_margin": self.budget_margin,
            "closed_by_model": self.closed_by_model,
            "cap_hit": self.cap_hit,
            "answer_finish_reason": self.answer_finish_reason,
            "note": self.note,
            "free_tokens": self.free_tokens,
            "count_started": self.count_started,
        }


@dataclass
class _State:
    """One request's running state across the rounds of the loop."""

    tokens: list[int]
    prompt_tokens: int
    # This request's own budget: budget forcing caps each item at its own
    # per-term share, so every document carries one.
    budget: int = 0
    thinking_tokens: int = 0
    continuations: int = 0
    rounds: int = 0
    done: bool = False
    note: str | None = None
    answer_tokens: int = 0
    answer_finish_reason: str | None = None
    injected: list[int] = field(default_factory=list)
    # How the block ended. `closed_by_model` is set where the mode allowed the
    # model to stop; `cap_hit` where we took the decision away from it.
    closed_by_model: bool = False
    cap_hit: bool = False


def per_prompt_budgets(budget: int | Sequence[int], n: int) -> list[int]:
    """One budget per prompt: a scalar repeated, or a sequence checked for length.

    A length mismatch is an error rather than a zip that quietly drops the tail,
    because a budget attached to the wrong prompt looks normal in every output.
    """
    if isinstance(budget, int):
        return [budget] * n
    budgets = [int(b) for b in budget]
    if len(budgets) != n:
        raise BudgetError(f"{len(budgets)} per-prompt budgets for {n} prompts")
    return budgets


class BudgetError(RuntimeError):
    """A budget that cannot be run as asked (context window, or a bad budget)."""


class BudgetForcer:
    """Drives an in-process vLLM engine through the forcing loop.

    The *mode* is a property of the instance rather than of a call, because one
    invocation of the runner is one condition: two modes at one budget are two
    runs, not two calls.
    """

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        delimiters: Any,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_model_len: int = 32768,
        answer_tokens: int = 2048,
        continue_cue: str = "\nWait",
        answer_cue: str = "\n\nAnswer:",
        continue_min_tokens: int = 32,
        max_continuations: int = 512,
        mode: BudgetMode | str = BudgetMode.EXACT,
        margin: int = 0,
        use_tqdm: bool = False,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.delimiters = delimiters
        self.mode = BudgetMode(mode)
        self.margin = int(margin)
        if self.margin < 0:
            raise BudgetError(f"--budget-margin must be >= 0, got {self.margin}")
        if self.margin and not self.mode.has_floor:
            # Not silently ignored: the margin relaxes the floor, and a mode
            # with no floor has nothing for it to relax. Accepting it would
            # write a number into `run.json` that described nothing.
            raise BudgetError(
                f"--budget-margin is a tolerance on the *floor*, and mode "
                f"{self.mode.value!r} has none. Use --budget-mode lower or exact, "
                "or drop the margin."
            )
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_model_len = int(max_model_len)
        self.answer_tokens = int(answer_tokens)
        self.continue_cue = continue_cue
        self.answer_cue = answer_cue
        self.continue_min_tokens = int(continue_min_tokens)
        self.max_continuations = int(max_continuations)
        self.use_tqdm = use_tqdm

        self.continue_ids = self._encode(continue_cue)
        self.answer_cue_ids = self._encode(answer_cue)
        if not self.continue_ids:
            raise BudgetError(f"the continuation cue {continue_cue!r} encodes to nothing")
        # What the thinking block must leave behind it, whatever the mode: the
        # closing delimiter, the answer cue, and somewhere for the answer to go.
        # Only `LOWER` reads it -- it is the mode with no budget-shaped ceiling,
        # so the context window is the ceiling and this is what it must not eat.
        self.reserved_tokens = (
            len(self.delimiters.close_token_ids)
            + len(self.answer_cue_ids)
            + MIN_ANSWER_TOKENS
        )

    # ------------------------------------------------------------------ #
    # prompts
    # ------------------------------------------------------------------ #
    def _encode(self, text: str) -> list[int]:
        return [int(i) for i in self.tokenizer.encode(text, add_special_tokens=False)]

    def prepare_prompt(self, messages: list[dict[str, str]]) -> list[int]:
        """Chat-template a conversation and leave the thinking block open.

        ``add_special_tokens=False`` on the re-encode is load-bearing: chat
        templates write their own BOS into the text, and encoding it again with
        special tokens on gives the model two of them.

        A template that already opens thinking (Qwen3-Thinking, the R1 distills)
        is left alone; one that does not (Phi-4-reasoning, R1-0528-Qwen3-8B) has
        the opener appended, because a budget can only be enforced on a block
        that was entered.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not self.delimiters.opens_thinking(text):
            text += self.delimiters.open_text
            if not self.delimiters.open_text.endswith("\n"):
                text += "\n"
        return self._encode(text)

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    def floor_target(self, budget: int) -> int:
        """The thinking length below which the model is not allowed to stop.

        Zero when the mode has no floor, which makes every "has the floor been
        met" test in the loop unconditional rather than mode-dependent.
        """
        if not self.mode.has_floor:
            return 0
        return max(0, budget - self.margin)

    def check_fits(self, prompts: list[list[int]], budget: int | Sequence[int],
                   labels: list[Any]) -> None:
        """Refuse a budget that does not leave room for an answer.

        Checked for the whole batch up front, so a run fails before generating
        anything rather than an hour in.

        The arithmetic is the same in all three modes, and that is not an
        oversight: under ``LOWER`` the budget is a floor, so the model spends
        *at least* that much and the room left over is at most what this
        computes. The mode changes only what the message calls the number.
        """
        budgets = per_prompt_budgets(budget, len(prompts))
        if min(budgets, default=0) < 0:
            bad = next(label for label, b in zip(labels, budgets) if b < 0)
            raise BudgetError(f"a thinking budget must be >= 0, got {min(budgets)} ({bad})")
        room = [
            (label, self.max_model_len - len(prompt) - b - len(self.answer_cue_ids)
             - len(self.delimiters.close_token_ids))
            for label, prompt, b in zip(labels, prompts, budgets)
        ]
        # Messages name the scalar when there is one, the largest when there is not.
        if not isinstance(budget, int):
            budget = f"per-document (max {max(budgets, default=0)})"
        squeezed = [(label, left) for label, left in room
                    if MIN_ANSWER_TOKENS <= left < self.answer_tokens]
        if squeezed:
            worst = min(squeezed, key=lambda entry: entry[1])
            # Above the floor, so the run goes ahead -- but an answer capped by
            # the context window rather than by --answer-tokens is a truncated
            # answer scored as a wrong one, and `n_answer_truncated` in the
            # summary is where it shows up afterwards.
            print(f"warning: budget {budget} leaves {len(squeezed)} of {len(prompts)} "
                  f"documents less than --answer-tokens ({self.answer_tokens}) for the "
                  f"answer; the tightest is {worst[0]} with {worst[1]}", file=sys.stderr)

        short = [(label, left) for label, left in room if left < MIN_ANSWER_TOKENS]
        if short:
            worst = min(short, key=lambda entry: entry[1])
            bound = "minimum" if self.mode is BudgetMode.LOWER else "maximum"
            raise BudgetError(
                f"thinking budget {budget} ({self.mode.value}, i.e. a {bound} of "
                f"{budget} thinking tokens) leaves {worst[1]} tokens for the answer on "
                f"{len(short)} of {len(prompts)} documents (worst: {worst[0]}), below the "
                f"{MIN_ANSWER_TOKENS}-token floor. The model's context window is "
                f"{self.max_model_len}. Lower the budget, or drop the larger k."
            )

    def run(self, prompts: list[list[int]], budget: int | Sequence[int], *,
            labels: list[Any] | None = None, progress: str = "") -> list[ForcedGeneration]:
        """Generate one budget-forced response per prompt, batched throughout.

        Every round issues one `LLM.generate` call for every request still
        thinking, so the engine batches across documents and trials; prefix
        caching is what keeps re-sending the growing prefix cheap.

        What ``budget`` *means* is the instance's :class:`BudgetMode`: a floor,
        a cap, or both. It is one number for every prompt, or a sequence with
        one budget per prompt; nothing else about the loop changes.
        """
        labels = list(labels or range(len(prompts)))
        self.check_fits(prompts, budget, labels)

        states = [_State(tokens=list(prompt), prompt_tokens=len(prompt), budget=b)
                  for prompt, b in zip(prompts, per_prompt_budgets(budget, len(prompts)))]
        for state in states:
            if state.budget == 0 and self.mode.has_cap:
                # A zero budget is a real condition -- "answer with no thinking at
                # all" -- and it is the loop's natural base case, not a special
                # one: nothing is generated inside the block before it is closed.
                # Only where the budget caps, though: a zero *floor* is no
                # constraint at all, and the model still thinks for as long as
                # it likes.
                state.done = True
                state.cap_hit = True

        round_index = 0
        while any(not state.done for state in states):
            round_index += 1
            pending = [state for state in states if not state.done]
            params = [self._thinking_params(state.budget - state.thinking_tokens,
                                            first=state.rounds == 0,
                                            used=len(state.tokens))
                      for state in pending]
            outputs = self._generate([state.tokens for state in pending], params)
            still = 0
            for state, output in zip(pending, outputs):
                state.rounds += 1
                self._absorb(state, output)
                still += 0 if state.done else 1
            if progress:
                print(f"{progress} thinking round {round_index}: "
                      f"{still}/{len(states)} still under budget", file=sys.stderr)

        return self._answer(states)

    def _thinking_params(self, remaining: int, *, first: bool, used: int):
        from vllm import SamplingParams

        delims = self.delimiters
        # Where the budget caps, a round may not produce more than is left of it.
        # Where it does not (`LOWER`), the ceiling is the context window less the
        # room the close, the cue and the answer will need -- so the model may
        # think past the budget, but never into its own answer's space.
        if self.mode.has_cap:
            allowed = max(1, remaining)
        else:
            allowed = max(1, self.max_model_len - used - self.reserved_tokens)
        # `min_tokens` applies only to a continuation. The first round measures
        # what the model would do unaided, which is the quantity a natural-length
        # baseline is read off; on later rounds it stops a model that wants to
        # finish from closing the block again immediately, which would otherwise
        # spend the budget one injected `Wait` at a time over thousands of rounds.
        # Only a mode with a floor continues a model that wanted to stop.
        floor = (0 if first or not self.mode.has_floor
                 else max(0, min(self.continue_min_tokens, allowed - 1)))
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=allowed,
            min_tokens=floor,
            # Both forms of "the model closed its thinking": the delimiter as a
            # token, and -- for a tokenizer where it is not special -- as text.
            stop_token_ids=[delims.stop_token_id],
            stop=[delims.close_text.strip()],
            include_stop_str_in_output=False,
            skip_special_tokens=False,
            # No per-request seed: one here would make every repeat of a document
            # return byte-identical text, the trap the task templates' `seed: null`
            # avoids on the HTTP path.
            seed=None,
        )

    def _generate(self, token_lists: list[list[int]], params: list[Any]) -> list[Any]:
        from vllm import TokensPrompt

        requests = [TokensPrompt(prompt_token_ids=tokens) for tokens in token_lists]
        outputs = self.llm.generate(requests, params, use_tqdm=self.use_tqdm)
        return [output.outputs[0] for output in outputs]

    def _absorb(self, state: _State, output: Any) -> None:
        """Fold one round's output into a request's state, and decide what next.

        The whole mode distinction lives here. Two questions are asked of every
        round, and the answers are independent:

        * did the model *want* to stop? `finish_reason == "length"` means it ran
          into the ceiling we set and had more to say; anything else -- the
          closing delimiter, EOS, a stop string -- means it was finished.
        * is it *allowed* to? Only once the floor is met, and only a mode with a
          floor has one to meet.
        """
        produced = self._strip_close([int(t) for t in output.token_ids])
        state.tokens.extend(produced)
        state.thinking_tokens += len(produced)
        wanted_to_stop = output.finish_reason != "length"
        budget = state.budget
        floor_target = self.floor_target(budget)

        if self.mode.has_cap and state.thinking_tokens >= budget:
            # The budget is spent. The block is closed by force in `_answer`,
            # whether or not the model was ready to close it itself.
            state.done = True
            state.cap_hit = True
            return
        if wanted_to_stop and state.thinking_tokens >= floor_target:
            # It is finished and it is allowed to be. Under UPPER this is every
            # generation that came in under the cap; under LOWER and EXACT it is
            # the model closing its own block on or above the floor, which the
            # margin is what makes reachable.
            state.done = True
            state.closed_by_model = True
            return
        if not wanted_to_stop:
            # It hit the ceiling mid-thought with the floor still unmet, which
            # only a capless mode can reach: carry on generating from where it
            # left off, without injecting anything. Nothing was suppressed here,
            # so this is not a continuation and is not counted as one.
            return
        if state.continuations >= self.max_continuations:
            # Recorded, not silently accepted: this document's thinking is
            # shorter than the run's budget claims, and the summary counts it.
            state.note = (f"stopped after {state.continuations} continuations with "
                          f"{state.thinking_tokens}/{budget} thinking tokens")
            state.done = True
            state.closed_by_model = True
            return

        # The model stopped early -- on the closing delimiter, on EOS, or on
        # nothing at all. Every one of those is "it wanted to be finished", and
        # the answer is the same: suppress the ending and make it go on.
        state.tokens.extend(self.continue_ids)
        state.thinking_tokens += len(self.continue_ids)
        state.injected.extend(self.continue_ids)
        state.continuations += 1
        if self.mode.has_cap and state.thinking_tokens >= budget:
            state.done = True
            state.cap_hit = True

    def _strip_close(self, ids: list[int]) -> list[int]:
        """Drop a trailing closing delimiter from a round's tokens.

        vLLM may or may not include the token it stopped on, depending on
        whether it stopped on a stop *string* or a stop *token id*. Leaving one
        in would close the thinking block inside text we are about to continue,
        so the block would read as closed twice and the injected `Wait` would sit
        outside it.
        """
        close = list(self.delimiters.close_token_ids)
        while ids and ids[-1] in {self.delimiters.stop_token_id, close[-1]}:
            ids = ids[:-1]
        if close[:-1] and ids[-len(close) + 1:] == close[:-1]:
            ids = ids[: -len(close) + 1]
        return ids

    def _answer(self, states: list[_State]) -> list[ForcedGeneration]:
        """Close every thinking block, inject the answer cue, and decode once.

        Identical in all three modes, deliberately: whether the model closed its
        own block or we cut it off, `_strip_close` has already removed any
        closing delimiter it wrote, so appending one here is a close and never a
        double close. The answer is then decoded under the same conditions in
        every mode, which is what makes the three comparable at all.
        """
        from vllm import SamplingParams

        close_ids = list(self.delimiters.close_token_ids)
        for state in states:
            state.tokens.extend(close_ids)
            state.tokens.extend(self.answer_cue_ids)
            state.injected.extend(close_ids + self.answer_cue_ids)

        params = [
            SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max(
                    MIN_ANSWER_TOKENS,
                    min(self.answer_tokens, self.max_model_len - len(state.tokens)),
                ),
                # The answer must not be cut short by the thinking delimiter,
                # which the model has every reason to emit again after being
                # closed by force.
                skip_special_tokens=False,
                seed=None,
            )
            for state in states
        ]
        outputs = self._generate([state.tokens for state in states], params)

        results = []
        for state, output in zip(states, outputs):
            answer_ids = [int(t) for t in output.token_ids]
            state.answer_tokens = len(answer_ids)
            state.answer_finish_reason = output.finish_reason
            state.tokens.extend(answer_ids)
            text = self.tokenizer.decode(
                state.tokens[state.prompt_tokens:], skip_special_tokens=False
            )
            results.append(ForcedGeneration(
                text=text,
                prompt_tokens=state.prompt_tokens,
                thinking_tokens=state.thinking_tokens,
                thinking_budget=state.budget,
                continuations=state.continuations,
                answer_tokens=state.answer_tokens,
                rounds=state.rounds,
                # "Did this generation meet the condition it was given." Under
                # UPPER the floor target is 0, so it is trivially true -- there
                # is no such thing as falling short of a cap.
                budget_reached=state.thinking_tokens >= self.floor_target(state.budget),
                budget_mode=self.mode.value,
                budget_margin=self.margin,
                closed_by_model=state.closed_by_model,
                cap_hit=state.cap_hit,
                answer_finish_reason=state.answer_finish_reason,
                note=state.note,
            ))
        return results
