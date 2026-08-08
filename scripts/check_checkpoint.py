#!/usr/bin/env python3
"""Verify that a released DeepEntry checkpoint loads with the public model code."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.low_rank_model import LowRankInteractionModel  # noqa: E402


def infer_dimensions(state: dict) -> tuple[int, int, int]:
    """Infer (esm_dim, proj_dim, hidden_dim) from the state dict shapes."""
    esm_dim = state["viral_proj.weight"].shape[1]
    proj_dim = state["viral_proj.weight"].shape[0]
    hidden_dim = state["classifier.0.weight"].shape[0]
    return esm_dim, proj_dim, hidden_dim


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    esm_dim, proj_dim, hidden_dim = infer_dimensions(state)

    model = LowRankInteractionModel(
        esm_dim=esm_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        dropout=0.5,
    )
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    incompatible = [k for k in state if k not in compatible]
    model_state.update(compatible)
    model.load_state_dict(model_state)

    n_params = sum(param.numel() for param in model.parameters())
    print("Checkpoint load passed.")
    print(f"checkpoint: {args.checkpoint}")
    print(f"parameters: {n_params}")
    print(f"esm_dim: {esm_dim}")
    print(f"proj_dim: {proj_dim}")
    print(f"hidden_dim: {hidden_dim}")
    print(f"layers loaded: {len(compatible)}/{len(state)}")
    if incompatible:
        print(f"skipped (shape mismatch): {incompatible}")
    for key in ["epoch", "val_p_at_1", "val_recall_at_10", "val_mrr"]:
        if key in ckpt:
            print(f"{key}: {ckpt[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
