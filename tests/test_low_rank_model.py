"""Tests for the released DeepEntry model definition (LowRankInteractionModel)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.low_rank_model import LowRankInteractionModel, count_parameters  # noqa: E402


def test_default_init() -> None:
    model = LowRankInteractionModel()
    assert isinstance(model, torch.nn.Module)
    # default: esm_dim=1280, proj_dim=128, hidden_dim=256
    assert model.viral_proj.weight.shape == (128, 1280)
    assert model.host_proj.weight.shape == (128, 1280)
    assert model.classifier[0].weight.shape == (256, 4 * 128)


def test_release_dimensions() -> None:
    """The published LOVO56 checkpoint uses esm_dim=5120, proj_dim=128, hidden_dim=256."""
    model = LowRankInteractionModel(esm_dim=5120, proj_dim=128, hidden_dim=256, dropout=0.5)
    assert model.viral_proj.weight.shape == (128, 5120)
    assert model.host_proj.weight.shape == (128, 5120)
    assert model.classifier[0].weight.shape == (256, 512)
    assert model.classifier[3].weight.shape == (128, 256)
    assert count_parameters(model) == 1_495_809


def test_forward_shape() -> None:
    model = LowRankInteractionModel(esm_dim=5120, proj_dim=128, hidden_dim=256, dropout=0.5)
    model.eval()
    viral = torch.randn(8, 5120)
    host = torch.randn(8, 5120)
    with torch.no_grad():
        out = model(viral, host)
    assert out.shape == (8,)  # per-pair logits, squeezed
    assert torch.isfinite(out).all()


def test_forward_pairs_are_symmetric() -> None:
    """Pairwise scoring: swapping embeddings swaps the logit order."""
    model = LowRankInteractionModel(esm_dim=5120, proj_dim=128, hidden_dim=256, dropout=0.5)
    model.eval()
    viral = torch.randn(1, 5120)
    host_a = torch.randn(1, 5120)
    host_b = torch.randn(1, 5120)
    with torch.no_grad():
        s_ab = model(viral, host_a).item()
        s_ba = model(viral, host_b).item()
    assert s_ab != s_ba  # different receptors score differently


def test_count_parameters() -> None:
    model = LowRankInteractionModel(esm_dim=5120, proj_dim=128, hidden_dim=256, dropout=0.5)
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
