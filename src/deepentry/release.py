"""Helpers for reading DeepEntry release artifacts."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a tab-separated release table."""
    return pd.read_csv(Path(path), sep="\t")


def expected_headline_metrics() -> dict[str, float]:
    """Return manuscript headline metrics for the 38-virus benchmark."""
    return {"MRR": 0.4011, "Recall@10": 0.6491, "Recall@20": 0.7368}
