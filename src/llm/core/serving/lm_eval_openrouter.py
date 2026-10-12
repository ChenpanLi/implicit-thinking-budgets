"""An lm-eval backend for OpenRouter: ``--model openrouter-chat-completions``.

OpenRouter speaks OpenAI's chat-completions protocol, so almost nothing has to
change -- lm-eval's own ``local-chat-completions`` would very nearly work. It is
subclassed rather than reused for three reasons, each of which otherwise shows
up as a *quiet* failure at the far end of an expensive run:

* **The key.** The base class reads ``OPENAI_API_KEY``; ours lives in ``.env`` as
  ``OPENROUTER_API_KEY``. Taking it from the environment here, rather than
  passing an ``Authorization`` header in through ``model_args``, is also what
  keeps it out of ``run.json``, which records the model arguments verbatim.

* **Reasoning-only replies.** Some providers return an empty ``content`` with the
  whole answer in the ``reasoning`` field. The base class reads ``content`` and
  would score a perfectly good answer as an empty generation.

* **In-band errors.** OpenRouter reports upstream failures with HTTP 200 and no
  ``choices``, which the base class turns into ``""`` behind a generic warning.
  Here the provider's own message is logged, because "the model returned
  nothing" and "the provider refused the request" are different problems and the
  trace records both as an empty generation.

Importing this module registers the backend; :mod:`llm.core.serving.providers`
names it and ``llm.run`` imports it when a model config selects a hosted provider.
"""
from __future__ import annotations

import copy
import logging
from functools import cached_property
from typing import Dict, List, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER
from lm_eval.models.openai_completions import LocalChatCompletion

from llm.core.serving import token_stash
from llm.core.serving.providers import OPENROUTER_CHAT_URL, read_api_key

eval_logger = logging.getLogger(__name__)


@register_model("openrouter-chat-completions")
class OpenRouterChatCompletion(LocalChatCompletion):
    """OpenAI-compatible chat completions against OpenRouter."""

    def __init__(
        self,
        base_url: str = OPENROUTER_CHAT_URL,
        tokenizer_backend=None,
        tokenized_requests: bool = False,
        extra_headers: Dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        # Non-secret attribution headers (HTTP-Referer / X-Title). Kept separate
        # from `header`, which would replace the Authorization header wholesale.
        self._extra_headers = dict(extra_headers or {})
        super().__init__(base_url=base_url, tokenizer_backend=tokenizer_backend,
                         tokenized_requests=tokenized_requests, **kwargs)

    @cached_property
    def api_key(self) -> str:
        return read_api_key("openrouter")

    @cached_property
    def header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", **self._extra_headers}

    def _create_payload(self, *args, **kwargs) -> dict:
        """As the base class, plus asking OpenRouter for its fullest usage accounting.

        `usage: {include: true}` is what makes OpenRouter add provider-native
        token counts (and cost) alongside the GPT-4-tokenizer-normalized ones
        it reports by default. Every field it sends back is kept verbatim in
        the trace (`token_usage`, see `amodel_call`/`generate_until` below)
        rather than picked apart here, so which field is *the* exact native
        count can be decided once, by inspection, rather than guessed now.
        """
        payload = super()._create_payload(*args, **kwargs)
        payload["usage"] = {"include": True}
        return payload

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        if not isinstance(outputs, list):
            outputs = [outputs]
        generations: List[str] = []
        for out in outputs:
            choices = (out or {}).get("choices")
            if not choices:
                # 200 OK with no choices: an upstream error reported in-band.
                error = (out or {}).get("error") or out
                eval_logger.warning(f"OpenRouter returned no choices: {error}")
                generations.append("")
                continue
            texts = [""] * len(choices)
            for position, choice in enumerate(choices):
                message = choice.get("message") or {}
                text = message.get("content") or ""
                if not str(text).strip():
                    # Recovered from the reasoning channel rather than discarded;
                    # an answer that only ever existed there is still an answer.
                    text = message.get("reasoning") or ""
                index = choice.get("index", position)
                index = index if isinstance(index, int) and 0 <= index < len(texts) else position
                texts[index] = str(text)
            generations += texts
        return generations

    async def amodel_call(
        self,
        session,
        sem,
        messages,
        *,
        generate: bool = True,
        cache_keys: list | None = None,
        ctxlens: list | None = None,
        gen_kwargs: dict | None = None,
        **kwargs,
    ):
        """As `TemplateAPI.amodel_call`, plus stashing OpenRouter's own `usage` object.

        Reimplemented rather than wrapped: `num_concurrent` defaults to 16 in
        every run config this repo actually uses (see `llm.run`), which routes generation through this async
        path, not the synchronous `model_call` the base class also has. `usage`
        lives on the raw JSON response and is gone the moment
        `parse_generations` has pulled the text out of it -- nothing downstream
        of this method ever sees it again, so it has to be captured here or not
        at all. Kept line-for-line identical to
        `lm_eval.models.api_models.TemplateAPI.amodel_call` except for the
        block marked below; diverge from upstream only there.

        `cache_keys` is `[(context, gen_kwargs)]` -- length 1 in every run this
        repo makes, since a chat-completions endpoint answers one conversation
        per call regardless of the `batch_size` requested (`llm.run` passes
        batch size 1 for a hosted model). The context string is the stash
        key `generate_until` (below) re-keys onto `Instance.doc["id"]`.
        """
        gen_kwargs = copy.deepcopy(gen_kwargs)
        payload = self._create_payload(
            self.create_message(messages), generate=generate, gen_kwargs=gen_kwargs,
            seed=self._seed, **kwargs,
        )
        cache_method = "generate_until" if generate else "loglikelihood"
        acquired = await sem.acquire()
        try:
            async with session.post(self.base_url, json=payload, headers=self.header) as response:
                if not response.ok:
                    error_text = await response.text()
                    eval_logger.warning(
                        f"API request failed! Status code: {response.status}, "
                        f"Response text: {error_text}. Retrying..."
                    )
                response.raise_for_status()
                outputs = await response.json()

            # --- token usage: stash the whole `usage` object before it is
            # discarded (see the docstring above and `token_stash`). ---
            if cache_keys:
                usage = (outputs or {}).get("usage")
                if usage:
                    context = cache_keys[0][0] if isinstance(cache_keys[0], tuple) else None
                    if context is not None:
                        token_stash.by_context.record(context, dict(usage))

            tmp_answers = (
                self.parse_generations(outputs=outputs) if generate
                else self.parse_logprobs(outputs=outputs, tokens=messages, ctxlens=ctxlens)
            )
            answers = []
            for a in tmp_answers:
                if a is None:
                    eval_logger.warning(
                        "API returned null content. Content filled with "
                        f"`LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER = {LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER}`. "
                        "Check reasoning_content field or generation limits."
                    )
                    answers.append(LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER)
                else:
                    answers.append(a)
            if cache_keys:
                for res, cache in zip(answers, cache_keys, strict=False):
                    self.cache_hook.add_partial(cache_method, cache, res)
            return answers
        except BaseException as e:  # noqa: BLE001 - upstream re-raises to trigger tenacity's retry
            eval_logger.error(f"Exception:{e!r}, {locals().get('outputs', '(no outputs)')}, retrying.")
            raise
        finally:
            if acquired:
                sem.release()

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """Generate, then re-key each response's stashed `usage` onto its doc.

        `amodel_call` above only ever sees a context string, never the
        `Instance` (and so never its `doc`) that string came from -- lm-eval's
        own batching unpacks `req.args` before either method is reached. This
        recovers that identity the same way `InProcessVLLM.generate_until`
        does: drain `token_stash.by_context` (keyed by the exact prompt string,
        set above) once superclass generation has returned, and re-record under
        `Instance.doc["id"]` in `token_stash.by_doc`, which
        `lm_tasks/*/utils.py` reads back in `process_results`.
        """
        texts = super().generate_until(requests, disable_tqdm=disable_tqdm)
        for req in requests:
            usage = token_stash.by_context.drain_one(req.args[0])
            if usage is not None and isinstance(req.doc, dict) and req.doc.get("id") is not None:
                token_stash.by_doc.record(req.doc["id"], usage)
        return texts
