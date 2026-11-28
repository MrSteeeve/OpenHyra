"""Compare Python tensor-operator revisions without fitting clones."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import StringIO
import tokenize


@dataclass(frozen=True)
class DuplicateMatch:
    cand_id: str
    similarity: float


_IGNORED = {
    tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
    tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING,
}


def _is_float(value: str) -> bool:
    lower = value.lower()
    if lower.startswith(("0x", "0b", "0o")) or lower.endswith("j"):
        return False
    return "." in lower or "e" in lower


def normalize_source(source: str) -> tuple[str, ...]:
    """Return ordered lexical tokens, excluding layout/comments and float values.

    This is a similarity heuristic, not a validity checker or proof of
    semantic identity. Names, operators, strings and integers remain distinct.
    An incomplete token stream raises TokenError rather than being treated as
    a novel valid program.
    """
    result = []
    for item in tokenize.generate_tokens(StringIO(source).readline):
        if item.type in _IGNORED:
            continue
        if item.type == tokenize.ERRORTOKEN and item.string.isspace():
            continue
        value = item.string
        if item.type == tokenize.NUMBER and _is_float(value):
            value = "<float>"
        result.append(value)
    return tuple(result)


def find_duplicate(source: str, previous: Mapping[str, str], threshold: float = 0.94) -> DuplicateMatch | None:
    """Return the strongest prior lexical match if ratio is strictly > threshold.

    Ratio = 2 * matched tokens / (left length + right length), using the
    standard-library SequenceMatcher (autojunk disabled). Same-score ties use
    lexical candidate ID, independent of dict insertion order. Empty programs
    are left for the model contract gate, not silently considered valid.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity threshold must be in [0, 1]")
    tokens = normalize_source(source)
    if not tokens:
        return None
    best = None
    for cand_id, prior_source in sorted(previous.items()):
        try:
            prior_tokens = normalize_source(prior_source)
        except (tokenize.TokenError, IndentationError):
            continue  # A malformed previous proposal has no lexical sequence.
        score = SequenceMatcher(None, tokens, prior_tokens, autojunk=False).ratio()
        if score > threshold and (best is None or score > best.similarity):
            best = DuplicateMatch(cand_id, score)
    return best
