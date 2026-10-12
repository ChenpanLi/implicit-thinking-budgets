"""Which backend generates for a model: vLLM in this process, or a hosted API.

A model YAML without a ``provider`` key names local weights, which are loaded
into the running process (:mod:`llm.core.serving.lm_eval_vllm`). A hosted model
is the same YAML with one extra line::

    provider: openrouter
    model_id: "anthropic/claude-opus-5"

Everything downstream -- the tasks, the filters, the checker, the tracing -- is
unchanged, so a hosted model and a local one differ only in who generated the text.

The API key is deliberately **not** part of the endpoint. The lm-eval backend
reads it from the environment itself (see :mod:`llm.core.serving.lm_eval_openrouter`),
so a key cannot reach ``run.json`` or a log line by being carried around inside a
dict of model arguments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm.core.serving.vllm_server import REPO_ROOT, ServerError, build_inprocess_model_args

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

LOCAL_PROVIDER = "vllm"

# lm-eval backend that constructs vLLM in this process. See
# :mod:`llm.core.serving.lm_eval_vllm`.
INPROCESS_BACKEND = "vllm-inprocess"

# provider -> (default chat-completions URL, environment variable holding the key,
#              lm-eval backend that talks to it)
HOSTED_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "openrouter": (OPENROUTER_CHAT_URL, "OPENROUTER_API_KEY", "openrouter-chat-completions"),
}

PROVIDERS = (LOCAL_PROVIDER, *sorted(HOSTED_PROVIDERS))


@dataclass
class Endpoint:
    """Where generations come from, and how lm-eval should ask for them."""

    provider: str
    chat_url: str | None
    model_id: str
    lm_eval_backend: str
    hosted: bool
    # True when the weights are loaded into this process; `chat_url` is then None.
    in_process: bool = False
    # Non-secret extras only (OpenRouter's attribution headers).
    headers: dict[str, str] = field(default_factory=dict)
    config_path: Path | None = None
    # Only for the in-process backend: the resolved lm-eval model arguments, so
    # `run.json` records the engine that was actually built.
    model_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """The record written into ``run.json``. Contains no credentials."""
        return {
            "provider": self.provider,
            "chat_url": self.chat_url,
            "model_id": self.model_id,
            "lm_eval_backend": self.lm_eval_backend,
            "hosted": self.hosted,
            "in_process": self.in_process,
            "headers": dict(self.headers),
            "config_path": str(self.config_path) if self.config_path else None,
        }


def provider_of(config: dict) -> str:
    """The provider a model config names, defaulting to the local one."""
    provider = str(config.get("provider", LOCAL_PROVIDER) or LOCAL_PROVIDER).strip().lower()
    if provider not in PROVIDERS:
        raise ServerError(
            f"unknown provider {provider!r} in model config. Known: {', '.join(PROVIDERS)}")
    return provider


def is_hosted(config: dict) -> bool:
    return provider_of(config) != LOCAL_PROVIDER


def register_backend(endpoint: Endpoint) -> None:
    """Import the module that registers `endpoint`'s backend with lm-eval.

    Deferred to call time: importing ``lm_eval`` and vLLM is slow, and a dry run
    should not pay for it.
    """
    if endpoint.lm_eval_backend == "openrouter-chat-completions":
        from llm.core.serving import lm_eval_openrouter  # noqa: F401
    elif endpoint.lm_eval_backend == INPROCESS_BACKEND:
        from llm.core.serving import lm_eval_vllm  # noqa: F401


def api_key_env(provider: str) -> str:
    """Name of the environment variable holding `provider`'s key."""
    try:
        return HOSTED_PROVIDERS[provider][1]
    except KeyError:
        raise ServerError(f"{provider!r} is not a hosted provider") from None


def read_api_key(provider: str, *, env_file: Path | None = None) -> str:
    """The key for a hosted provider, from the environment or from ``.env``."""
    name = api_key_env(provider)
    key = os.environ.get(name)
    if key and key.strip():
        return key.strip()

    key = _read_env_file(name, env_file if env_file is not None else REPO_ROOT / ".env")
    if key:
        return key

    raise ServerError(
        f"{name} is not set, so the {provider} model cannot be reached. "
        f"Add it to {REPO_ROOT / '.env'} ({name}=...) or export it."
    )


def _read_env_file(name: str, path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def resolve_endpoint(config: dict, *, config_path: Path | None = None,
                     seed: int | None = None) -> Endpoint:
    """The endpoint that will serve `config`: in-process vLLM, or a hosted API.

    Raises :class:`ServerError` -- including for a hosted provider whose key is
    missing, which is checked *here* so the run dies before tasks are loaded
    rather than after every request has failed.
    """
    provider = provider_of(config)
    model_id = str(config.get("model_id") or "").strip()
    if not model_id:
        raise ServerError("model config must define 'model_id'")

    if provider == LOCAL_PROVIDER:
        return Endpoint(provider=provider, chat_url=None, model_id=model_id,
                        lm_eval_backend=INPROCESS_BACKEND, hosted=False,
                        in_process=True, config_path=config_path,
                        model_args=build_inprocess_model_args(config, seed=seed))

    default_url, _, backend = HOSTED_PROVIDERS[provider]
    read_api_key(provider)  # fail now, loudly, rather than 500 requests from now
    headers = {str(k): str(v) for k, v in dict(config.get("headers") or {}).items()}
    return Endpoint(provider=provider, chat_url=str(config.get("base_url") or default_url),
                    model_id=model_id, lm_eval_backend=backend, hosted=True,
                    headers=headers, config_path=config_path)


def require_local(config: dict, config_path: Path | None, what: str) -> None:
    """Refuse a hosted model where local weights are the whole point."""
    provider = provider_of(config)
    if provider == LOCAL_PROVIDER:
        return
    where = f" ({config_path})" if config_path else ""
    raise ServerError(
        f"{what} needs weights served locally, but this model config{where} sets "
        f"provider: {provider}, which is a hosted API. "
        f"Pick a model config with the default provider ({LOCAL_PROVIDER})."
    )
