"""An lm-eval backend that runs vLLM **in this process**: ``--model vllm-inprocess``.

The default path for local weights. There is no ``vllm serve``, no port and no
HTTP client: ``simple_evaluate`` drives a :class:`vllm.LLM` living in the same
process, which is what makes a run self-contained: two runs of the same model
have nothing shared to collide on.

lm-eval already ships an in-process vLLM backend (``@register_model("vllm")``).
It is subclassed rather than used directly for one reason, and that reason would
otherwise corrupt every long prompt silently:

**Upstream truncates the prompt, not the generation.** ``VLLM.generate_until``
calls ``maybe_truncate(...)`` with its default ``shrink_gen_toks=False``, so when
``len(prompt) + max_gen_toks > max_model_len`` it *left-truncates the prompt* and
keeps the generation cap intact. With this repository's configs -- Qwen3-4B asks
for ``max_tokens: 48128`` inside ``max_model_len: 49152`` -- that leaves about a
thousand tokens for the prompt, so every k=6 sum and beyond would have its
integrals chopped off the front and the model would be scored on a mutilated
question. Nothing downstream could see it: the trace records the answer to a
question nobody asked.

The HTTP path never does this. lm-eval's API backend only length-checks when
``tokenized_requests`` is on, and the hosted runs turn it off
("Tokenized requests are disabled. Context + generation length is not checked."),
so the full prompt goes to the server and the *server* caps the generation at
``max_model_len - prompt_tokens``.

:class:`InProcessVLLM` therefore restores it: before handing a batch to upstream
it rewrites each request's ``max_gen_toks`` to
``min(max_gen_toks, max_model_len - prompt_tokens)``. ``maybe_truncate`` then hits
its "everything fits" case and becomes a no-op, so the prompt is never touched
and in-process and hosted runs remain comparable document for document.

Importing this module registers the backend; :mod:`llm.core.serving.providers`
names it and ``llm.run`` imports it for a local model config.
"""
from __future__ import annotations

import copy
import logging

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM

from llm.core.serving import token_stash

eval_logger = logging.getLogger(__name__)

# Below this many tokens a generation cannot say anything useful, so a prompt
# that leaves less room than this is a hard error rather than a request that
# comes back empty and is scored as a wrong answer.
MIN_GEN_TOKS = 16


class ClampError(RuntimeError):
    """A prompt that leaves no room to answer within the context window."""


def free_engine(lm: object) -> None:
    """Release the GPU an in-process engine holds. Safe to call on anything.

    lm-eval never frees it -- ``evaluator.py`` gets as far as a
    ``# TODO: del model here, maybe`` -- and the run is not over when generation
    is: the results still have to be written. In-process the weights are held
    until the interpreter exits, so a caller that wants the GPU back before then
    has to ask.
    """
    import gc

    engine = getattr(lm, "model", None)
    if engine is None:
        return
    try:
        del engine
        lm.model = None  # type: ignore[attr-defined]
        gc.collect()
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception as exc:  # noqa: BLE001 - teardown must never fail a finished run
        eval_logger.warning(f"could not fully release the vLLM engine: {exc}")


@register_model("vllm-inprocess")
class InProcessVLLM(VLLM):
    """vLLM in-process, with the server's generation-cap semantics.

    Identical to lm-eval's ``vllm`` backend except that the generation cap, not
    the prompt, absorbs a context window that is too small -- see the module
    docstring for why that difference is load-bearing rather than cosmetic.
    """

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generate, then attribute each response's exact token counts to its doc.

        vLLM's `RequestOutput.outputs[0].token_ids` is the model's own native
        tokenization of what it wrote, available nowhere past this call --
        upstream `generate_until` decodes it to text and returns only that. This
        wraps `_model_generate`, the one place both the tokens and the prompt
        that produced them are in scope together, to record the count against
        its exact prompt string in `token_stash.by_context`; below, once
        upstream's own internal reordering (see the module docstring) has
        settled back into the caller's original request order, that is
        re-keyed onto `Instance.doc["id"]` in `token_stash.by_doc`, which is
        what `lm_tasks/*/utils.py` reads back in `process_results`.

        Matching is by the prompt's *exact token ids*, not its text, because
        that is what `_model_generate` actually receives; encoding it here with
        the same tokenizer that produced the request is exact so long as the
        prompt was never truncated, which this class exists to guarantee (see
        the module docstring) -- so this matching is safe specifically because
        of that guarantee, not despite its absence.

        The prompt's own length is recorded alongside the completion's, from the
        very same token id list the match is made on. It costs nothing here and
        it is the larger half of what a batched prompt spends: "how many tokens
        does one pass over this dataset cost" is a question about prompts at
        least as much as about completions, and no later stage can recover the
        prompt's *native* tokenization without re-tokenizing it by hand.
        """
        clamped = self._clamp_requests(requests)
        contexts = [req.args[0] for req in requests]
        unique_contexts = list(dict.fromkeys(contexts))
        encoded = self.tok_encode(unique_contexts)
        context_by_tokens = {tuple(ids): ctx for ctx, ids in zip(unique_contexts, encoded)}

        original_model_generate = self._model_generate

        def _capturing_model_generate(requests, **kwargs):
            # Parameter named to match upstream's own keyword call
            # (`self._model_generate(requests=..., ...)`) -- these are tokenized
            # prompts, not the `Instance` list the enclosing method is named for.
            token_id_requests = requests
            outputs = original_model_generate(token_id_requests, **kwargs)
            for token_ids, output in zip(token_id_requests, outputs, strict=True):
                context = context_by_tokens.get(tuple(token_ids))
                if context is not None:
                    token_stash.by_context.record(context, {
                        "vllm_prompt_tokens": len(token_ids),
                        "vllm_output_tokens": len(output.outputs[0].token_ids),
                    })
            return outputs

        self._model_generate = _capturing_model_generate
        try:
            texts = super().generate_until(clamped, disable_tqdm=disable_tqdm)
        finally:
            self._model_generate = original_model_generate

        for req in requests:
            usage = token_stash.by_context.drain_one(req.args[0])
            if usage is not None and isinstance(req.doc, dict) and req.doc.get("id") is not None:
                token_stash.by_doc.record(req.doc["id"], usage)
        return texts

    def _clamp_requests(self, requests):
        """Copies of `requests` whose `max_gen_toks` fits the context window.

        Copies, never in-place edits: a task hands the *same* ``generation_kwargs``
        dict to every document, and ``repeats`` puts the *same* ``Instance`` object
        in the list N times, so mutating either would leak one document's cap onto
        all the others.
        """
        window = self.max_length
        if not window:
            return requests

        contexts = [req.args[0] for req in requests]
        if not contexts:
            return requests
        lengths = [len(enc) for enc in self.tok_encode(contexts)]

        clamped, n_clamped, tightest = [], 0, None
        for req, prompt_tokens in zip(requests, lengths, strict=True):
            context, gen_kwargs = req.args[0], dict(req.args[1])
            asked = int(gen_kwargs.get("max_gen_toks") or self.max_gen_toks)
            room = window - prompt_tokens
            if room < MIN_GEN_TOKS:
                raise ClampError(
                    f"a prompt of {prompt_tokens} tokens leaves only {room} of the "
                    f"{window}-token context window to answer in, which is below the "
                    f"{MIN_GEN_TOKS}-token floor. Raise max_model_len in the model "
                    f"config, or drop the longest documents from the dataset."
                )
            if room >= asked:
                clamped.append(req)
                continue

            gen_kwargs["max_gen_toks"] = room
            capped = copy.copy(req)
            capped.arguments = (context, gen_kwargs)
            clamped.append(capped)
            n_clamped += 1
            tightest = room if tightest is None else min(tightest, room)

        if n_clamped:
            # Loud, because a capped generation is the documented cause of a
            # truncated answer.
            eval_logger.warning(
                f"{n_clamped}/{len(requests)} prompts leave less room than the "
                f"requested generation cap; max_gen_toks was reduced to fit "
                f"(tightest: {tightest} tokens). The prompts themselves are intact."
            )
        return clamped
