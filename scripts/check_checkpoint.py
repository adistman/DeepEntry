#!/usr/bin/env python3
"""Verify that a released DeepEntry checkpoint loads with the public model code."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepentry.model import load_receptor_ranker_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    model, metadata = load_receptor_ranker_checkpoint(args.checkpoint, map_location="cpu")
    dims = metadata.get("model_dimensions", {})
    n_params = sum(param.numel() for param in model.parameters())
    print("Checkpoint load passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"parameters: {n_params}")
    for key in ["esm_dim", "proj_dim", "hidden_dim"]:
        if key in dims:
            print(f"{key}: {dims[key]}")
    for key in ["fold_idx", "test_viral_id", "best_mrr"]:
        if key in metadata:
            print(f"{key}: {metadata[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
