"""Shared helpers for the grounded assistant."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence


def tokenize(text: str) -> List[str]:
    """Tokenize natural language into lowercase alphanumeric tokens."""

    tokens = [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if token]
    expanded: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        for variant in _token_variants(token):
            if variant and variant not in seen:
                seen.add(variant)
                expanded.append(variant)
    return expanded


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    """Return whether the lowercase text contains any of the given phrases."""

    normalized = str(text).lower()
    return any(str(phrase).lower() in normalized for phrase in phrases)


def overlap_score(query_tokens: Iterable[str], candidate_tokens: Iterable[str]) -> float:
    """Compute a simple token-overlap score between query and candidate."""

    query = set(str(token).lower() for token in query_tokens if str(token).strip())
    candidate = set(str(token).lower() for token in candidate_tokens if str(token).strip())
    if not query or not candidate:
        return 0.0
    return float(len(query & candidate)) / float(len(candidate))


def _token_variants(token: str) -> List[str]:
    """Return conservative singular variants for matching part names."""

    text = str(token).lower().strip()
    variants = [text]
    if len(text) > 4 and text.endswith("ies"):
        variants.append(text[:-3] + "y")
    if len(text) > 4 and text.endswith("es") and not text.endswith("ses"):
        variants.append(text[:-2])
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        variants.append(text[:-1])
    return variants
