"""Ranking metrics used by DeepEntry release checks."""
from __future__ import annotations

from collections.abc import Iterable


def mean_reciprocal_rank(ranks: Iterable[int | float]) -> float:
    """Compute mean reciprocal rank from 1-based positive ranks."""
    vals = [float(r) for r in ranks if r and float(r) > 0]
    if not vals:
        return 0.0
    return sum(1.0 / r for r in vals) / len(vals)


def recall_at_k(ranks: Iterable[int | float], k: int) -> float:
    """Compute Recall@K from 1-based positive ranks."""
    vals = [float(r) for r in ranks if r and float(r) > 0]
    if not vals:
        return 0.0
    return sum(1 for r in vals if r <= k) / len(vals)
