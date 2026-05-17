"""Public model definition for released DeepEntry receptor-ranker checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError(
        "deepentry.model requires PyTorch. Install with `pip install -e .[model]`."
    ) from exc


class LowRankInteractionModel(nn.Module):
    """Low-rank pairwise interaction model used by the released checkpoints.

    The model scores one viral protein embedding against one host receptor
    embedding. The output is an uncalibrated logit used for candidate ranking.
    """

    def __init__(
        self,
        esm_dim: int = 5120,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.viral_norm = nn.LayerNorm(esm_dim)
        self.host_norm = nn.LayerNorm(esm_dim)
        self.viral_proj = nn.Linear(esm_dim, proj_dim)
        self.host_proj = nn.Linear(esm_dim, proj_dim)
        interaction_dim = 4 * proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, viral_emb: torch.Tensor, host_emb: torch.Tensor) -> torch.Tensor:
        viral_emb = self.viral_norm(viral_emb)
        host_emb = self.host_norm(host_emb)
        v_proj = self.viral_proj(viral_emb)
        h_proj = self.host_proj(host_emb)
        interaction = torch.cat(
            [v_proj, h_proj, v_proj * h_proj, torch.abs(v_proj - h_proj)], dim=1
        )
        return self.classifier(interaction).squeeze(-1)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def infer_model_dimensions(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    """Infer model dimensions from a released checkpoint state dict."""
    required = ["viral_norm.weight", "viral_proj.weight", "classifier.0.weight"]
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise ValueError(f"State dict is missing required keys: {missing}")
    return {
        "esm_dim": int(state_dict["viral_norm.weight"].numel()),
        "proj_dim": int(state_dict["viral_proj.weight"].shape[0]),
        "hidden_dim": int(state_dict["classifier.0.weight"].shape[0]),
    }


def load_receptor_ranker_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    dropout: float = 0.0,
) -> tuple[LowRankInteractionModel, dict[str, Any]]:
    """Load a released receptor-ranker checkpoint.

    Returns the model in evaluation mode and a metadata dictionary containing
    non-weight fields stored in the checkpoint.
    """
    checkpoint = _torch_load(Path(checkpoint_path), map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain a 'model_state_dict' entry.")
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint 'model_state_dict' is not a dictionary.")
    dims = infer_model_dimensions(state_dict)
    model = LowRankInteractionModel(**dims, dropout=dropout)
    model.load_state_dict(state_dict, strict=True)
    model.to(map_location)
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    metadata["model_dimensions"] = dims
    return model, metadata
