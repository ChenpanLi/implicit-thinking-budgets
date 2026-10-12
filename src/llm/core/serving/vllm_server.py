"""Model configs, and the arguments that build a vLLM engine from one.

Every model is one YAML under ``src/llm/config/models``. A YAML may carry a
``variants`` block: named partial overrides for the few runs of the paper that
were made with different serving settings (a YaRN window, a shorter context).
``load_model_config(name, variant=...)`` merges one of them in; without a
variant the block is dropped, so the config a run records is exactly the one
it was served with.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_DIR = REPO_ROOT / "src" / "llm" / "config" / "models"


class ServerError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict:
    """A model config as plain Python containers.

    `to_container` rather than `dict()`: nested blocks (`hf_overrides`,
    `extra_body`) would otherwise stay `DictConfig`, which is not JSON
    serializable and so cannot go into a request body or into `run.json`.
    """
    return dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def _with_variant(config: dict, variant: str | None, path: Path) -> dict:
    variants = config.pop("variants", None) or {}
    if variant is None:
        return config
    if variant not in variants:
        known = ", ".join(sorted(variants)) or "none"
        raise ServerError(f"{path.stem} has no variant {variant!r} (known: {known})")
    return {**config, **variants[variant]}


def load_model_config(name_or_path: str, config_dir: Path = DEFAULT_CONFIG_DIR, *,
                      variant: str | None = None) -> tuple[dict, Path]:
    """Resolve a model alias to its YAML config, with `variant` merged in if given."""
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        return _with_variant(_load_yaml(path), variant, path), path

    candidate = config_dir / f"{name_or_path}.yaml"
    if candidate.exists():
        return _with_variant(_load_yaml(candidate), variant, candidate), candidate

    available = ", ".join(sorted(p.stem for p in config_dir.glob("*.yaml")))
    raise ServerError(f"no model config for {name_or_path!r}. Available: {available}")


def build_inprocess_model_args(config: dict, *, seed: int | None = None) -> dict[str, Any]:
    """The model config as arguments for lm-eval's in-process vLLM backend.

    ``pretrained``, ``tensor_parallel_size``, ``max_model_len`` and friends are
    named parameters of lm-eval's ``VLLM``; ``gpu_memory_utilization`` and
    ``hf_overrides`` reach ``vllm.LLM`` through its ``**kwargs``. Values are real
    Python types, because lm-eval only coerces a ``model_args`` *string*.

    ``batch_size`` is not set here. lm-eval injects it from ``simple_evaluate``,
    and the only correct value is ``"auto"``: any integer chunks the requests and
    calls ``LLM.generate`` once per chunk instead of letting vLLM batch
    continuously.
    """
    model_id = config.get("model_id")
    if not model_id:
        raise ServerError("model config must define 'model_id'")

    model_args: dict[str, Any] = {"pretrained": str(model_id)}
    for key, cast in [("tensor_parallel_size", int),
                      ("gpu_memory_utilization", float),
                      ("max_model_len", int),
                      ("max_num_seqs", int),
                      ("dtype", str)]:
        if config.get(key) is not None:
            model_args[key] = cast(config[key])
    if config.get("trust_remote_code"):
        model_args["trust_remote_code"] = True
    if config.get("enforce_eager"):
        model_args["enforce_eager"] = True
    if config.get("hf_overrides"):
        # A deep copy: vLLM sets a dict-valued override on the HF config object
        # as is, and the config dict also goes into run.json.
        model_args["hf_overrides"] = json.loads(json.dumps(config["hf_overrides"]))
    if config.get("max_tokens") is not None:
        model_args["max_gen_toks"] = int(config["max_tokens"])
    if seed is not None:
        # The engine's own RNG seed. Distinct from a per-request sampling seed,
        # which must stay unset -- one there makes every trial return the same text.
        model_args["seed"] = int(seed)
    return model_args
