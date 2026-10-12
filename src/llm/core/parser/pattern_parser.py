"""A brace scanner: the contents of every balanced ``\\boxed{...}`` (or other command).

Handles nested braces and multiple hits, and raises :class:`PatternParseError`
when a command's braces are never closed.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


class PatternParseError(Exception):
    """Raised when a matched command's braces cannot be balanced/parsed."""


def _normalize_command(cmd: str) -> str:
    if not cmd:
        raise ValueError("command must be a non-empty string")
    return cmd if cmd.startswith("\\") else "\\" + cmd


def parse_command_braces(text: str, command: str = "\\boxed") -> List[str]:
    """Extract contents inside top-level balanced braces after a LaTeX command.

    Args:
        text: source string to search in.
        command: command name, e.g. "\\boxed" or "boxed".

    Returns:
        List of strings (contents found inside each command's outermost braces).

    Behavior notes:
    - Finds literal occurrences of the command followed by a brace, e.g. ``\\boxed{...}``.
    - Handles nested braces inside the content correctly.
    - Skips occurrences where a matching closing brace cannot be found; if any such
      unmatched occurrence is found, a PatternParseError is raised.
    """
    cmd = _normalize_command(command)
    results: List[str] = []
    unmatched_positions: List[int] = []

    i = 0
    L = len(text)
    while True:
        idx = text.find(cmd, i)
        if idx == -1:
            break
        # Find the first '{' after the command
        j = idx + len(cmd)
        # Skip any whitespace between command and brace
        while j < L and text[j].isspace():
            j += 1
        if j >= L or text[j] != '{':
            # Not a braced usage; move past this command
            logger.debug("Found %s at %d but no opening brace after it", cmd, idx)
            i = idx + len(cmd)
            continue

        # Now parse balanced braces from j (which points to '{')
        depth = 0
        k = j
        while k < L:
            ch = text[k]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    # content between j+1 and k
                    content = text[j + 1 : k]
                    results.append(content)
                    i = k + 1
                    break
            k += 1
        else:
            # Reached end without closing all braces
            unmatched_positions.append(idx)
            # debug, not error: truncation mid-box is an expected model failure
            # that callers classify as an `unbalanced` extraction outcome.
            logger.debug("Unmatched braces after %s at position %d", cmd, idx)
            i = idx + len(cmd)

    if unmatched_positions:
        raise PatternParseError(f"Unmatched braces after {command} at positions: {unmatched_positions}")

    logger.info("Found %d occurrences of %s", len(results), cmd)
    return results
