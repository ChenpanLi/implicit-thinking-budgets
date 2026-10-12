"""Build one item pool into ``$RUN_DIR/datasets/<dataset>.jsonl``.

    python -m llm.pools gsm8k
    python -m llm.pools integrals          # copies the shipped pool

An existing pool that matches the rebuild byte for byte is left alone. One that
differs is never overwritten without ``--force``: every batch and run refers to the
pool's item ids, so a silently changed pool would change what they mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm import paths, specs

from . import HUB_BUILDERS, build


def write_if_changed(path: Path, text: str, force: bool) -> str:
    """Write ``text`` to ``path`` unless it already holds it; return what happened."""
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return "unchanged"
        if not force:
            raise SystemExit(
                f"{path} exists and differs from the rebuild. Batches and runs refer "
                "to its item ids; pass --force only if you mean to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "written"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", choices=sorted(specs.DATASETS))
    parser.add_argument("--force", action="store_true",
                        help="replace an existing pool that differs from the rebuild")
    args = parser.parse_args(argv)

    out = paths.pool(args.dataset)
    manifest_out = out.with_suffix(".manifest.json")
    if args.dataset in HUB_BUILDERS:
        records, manifest = build(args.dataset)
        manifest["argv"] = ["python", "-m", "llm.pools", *sys.argv[1:]]
        pool_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        manifest_text = json.dumps(manifest, indent=2) + "\n"
    else:
        shipped = paths.shipped_pool(args.dataset)
        pool_text = shipped.read_text(encoding="utf-8")
        manifest_text = shipped.with_suffix(".manifest.json").read_text(encoding="utf-8")

    status = write_if_changed(out, pool_text, args.force)
    n = pool_text.count("\n")
    print(f"[pool] {args.dataset}: {n} items -> {out} ({status})")
    if status == "written" or not manifest_out.exists():
        manifest_out.write_text(manifest_text, encoding="utf-8")
        print(f"[pool] manifest -> {manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
