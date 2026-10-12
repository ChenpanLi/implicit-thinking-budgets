"""The statistics tables a batching strategy may condition on.

A strategy never opens a file itself: every table comes through a
:class:`StatsBundle`, passed on the command line as ``--stats NAME=PATH.csv``, so
the manifest can record exactly which tables a build read. Of the strategies the
paper uses only ``predicted-calibration`` reads one (the per-item k selection,
under the name ``predicted-largest-k``).
"""
from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """One CSV as a list of row dicts, in file order."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class StatsBundle:
    """Named CSV tables, loaded lazily and recorded once read."""

    def __init__(self, *, tables: Mapping[str, Path] | None = None) -> None:
        self._table_paths: dict[str, Path] = {
            name: Path(p) for name, p in (tables or {}).items()
        }
        self._cache: dict[str, list[dict[str, str]]] = {}
        # Every table actually read, for the manifest.
        self.read_tables: dict[str, str] = {}

    def table(self, name: str) -> list[dict[str, str]]:
        """One ``--stats name=path.csv`` table, as a list of row dicts."""
        if name in self._cache:
            return self._cache[name]
        path = self._table_paths.get(name)
        if path is None:
            known = ", ".join(sorted(self._table_paths)) or "(none)"
            raise SystemExit(
                f"this strategy asked for the statistics table {name!r}, which "
                f"was not supplied. Pass it with --stats {name}=<path.csv>. "
                f"Supplied: {known}"
            )
        if not path.exists():
            raise SystemExit(f"statistics table {name!r} not found: {path}")
        rows = read_csv(path)
        self._cache[name] = rows
        self.read_tables[name] = str(path)
        return rows
