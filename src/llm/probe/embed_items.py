"""Embed every item on its own with Qwen3-Embedding-8B (GPU).

    python -m llm.probe.embed_items                       # every batching dataset
    python -m llm.probe.embed_items --dataset gsm8k
    python -m llm.probe.embed_items --dataset gsm8k --dry-run   # texts and their digest only

What is embedded is the item alone (see :func:`llm.probe.common.embed_text`): the
integral in LaTeX, the question with its lettered options, the bare question; never
the prompt around it. Pooling is the model's own recipe: the hidden state at an EOS
token appended to every text, L2-normalised. One file per dataset, shared by every
model: ``$RUN_DIR/probe/embeddings/<ds>/qwen--qwen3-embedding-8b.safetensors``, with a
``.json`` sidecar holding the item ids (row order) and a SHA-256 of the texts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np

from llm import specs
from llm.probe import common

MAX_LENGTH = 8192


def texts(dataset: str) -> tuple[list[str], list[str]]:
    pool = common.records(dataset)
    return [str(r["id"]) for r in pool], [common.embed_text(dataset, r) for r in pool]


def digest(items: list[str]) -> str:
    return hashlib.sha256("\x00".join(items).encode()).hexdigest()


def embed(items: list[str], *, batch_size: int) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device.type == "cuda" else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(common.EMBED_MODEL, padding_side="right")
    model = AutoModel.from_pretrained(common.EMBED_MODEL, dtype=dtype).to(device).eval()
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    vectors, truncated = [], 0
    for start in range(0, len(items), batch_size):
        batch = tokenizer(items[start:start + batch_size], padding=True, truncation=True,
                          max_length=MAX_LENGTH - 1, return_tensors="pt")
        ids, mask = batch["input_ids"], batch["attention_mask"]
        lengths = mask.sum(dim=1)
        truncated += int((lengths >= MAX_LENGTH - 1).sum())
        # Append EOS after each sequence's last real token (right padding).
        ids = torch.cat([ids, torch.full((ids.size(0), 1), pad)], dim=1)
        mask = torch.cat([mask, torch.zeros(mask.size(0), 1, dtype=mask.dtype)], dim=1)
        for row, length in enumerate(lengths.tolist()):
            ids[row, length] = eos
            mask[row, length] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.inference_mode():
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            last = mask.sum(dim=1) - 1
            rows = torch.arange(hidden.size(0), device=device)
            if not bool((ids[rows, last] == eos).all()):
                raise SystemExit("a pooled position is not the EOS token")
            pooled = torch.nn.functional.normalize(hidden[rows, last].float(), p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
        print(f"  embedded {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    info = {"n_truncated": truncated, "max_length": MAX_LENGTH, "device": str(device),
            "dtype": str(dtype).replace("torch.", "")}
    return np.concatenate(vectors, axis=0), info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", default=specs.BATCHING_DATASETS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the first texts and the digest; load no model")
    args = parser.parse_args(argv)

    for dataset in args.dataset:
        ids, items = texts(dataset)
        out = common.embeddings_file(dataset)
        print(f"[{dataset}] {len(ids)} items, texts sha256 {digest(items)} -> {out}")
        if args.dry_run:
            for item_id, text in list(zip(ids, items))[:2]:
                print(f"  {item_id}: {text!r}")
            continue
        from safetensors.numpy import save_file

        matrix, info = embed(items, batch_size=args.batch_size)
        meta = {
            "schema": 1,
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": "item-embeddings", "domain": dataset, "embed_model": common.EMBED_MODEL,
            "pooling": "last-token(eos) + l2-normalised", "context": "isolated",
            "n_items": len(ids), "item_ids": ids, "texts_sha256": digest(items),
            "hidden_size": int(matrix.shape[1]), **info,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeddings": matrix}, str(out), metadata={"prediction": json.dumps(meta)})
        out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  wrote {matrix.shape} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
