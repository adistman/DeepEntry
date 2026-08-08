"""
Stage-3: Leave-one-virus-out (LOVO) receptor-ranking fine-tuning (LOVO56).

Pipeline:
  Stage-1 (general PPI, 39K pairs)
      -> Stage-2 (entry-domain fine-tune)
          -> Stage-3 (receptor-ranking LOVO, 84 gold-standard pairs)

Scientific goal:
  Given a viral surface protein, rank ALL 3,455 human membrane proteins
  (candidate pool) by probability of being the entry receptor.
  Evaluate using leave-one-virus-out cross-validation on 56 viruses with
  known receptors (84 gold-standard pairs).

Key differences from Stage-2:
  - Negative pool: human membrane protein candidate pool (3,455 proteins)
    → model must distinguish true receptors from other membrane proteins
  - neg_ratio: 50-100x  (hard ranking problem, vs 5x in Stage-2)
  - LOVO evaluation: for each held-out virus, score all 3,455 candidates
  - Heavy regularization: lr=1e-5, weight_decay=0.2, pos_weight=20
  - Initialize from Stage-2 best checkpoint
  - Optional: use expanded soft pairs (from expand_receptor_pairs_homolog.py)
    with fractional weights (<1.0) in BCE loss

LOVO protocol:
  For each unique viral protein V (56 with embeddings):
    1. Train on all gold pairs EXCEPT those involving V + negatives from receptor_pool
    2. Optionally include expanded soft pairs (also excluding V)
    3. Evaluate: score all receptor_pool candidates for V, record P@K / Recall@K / MRR
  Report mean ± std across all valid folds.

Usage:
    # Full LOVO (all folds)
    python train_stage3_receptor.py --config config/stage3_receptor_config.yaml

    # Single fold (parallelizable)
    python train_stage3_receptor.py --config config/stage3_receptor_config.yaml --fold_idx 3

    # Smoke test (3 folds, 5 epochs each)
    python train_stage3_receptor.py --config config/stage3_receptor_config.yaml \\
        --max_folds 3 --epochs 5 --out_dir outputs/stage3_smoke_test

    # Aggregate results after all folds finish
    python train_stage3_receptor.py --config config/stage3_receptor_config.yaml \\
        --aggregate_only
"""

import argparse
import csv
import json
import pickle
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from models.low_rank_model import LowRankInteractionModel


def extract_virus_family(virus_name: str) -> str:
    """
    Heuristic virus family extractor from a free-text virus label.

    Used for "taxon-level" leakage control in LOVO. When enabled, holding out a
    test virus also excludes *all* viruses from the same inferred family from
    training (gold + expanded pairs).

    Note: this is keyword-based and not a substitute for an explicit taxonomy map
    (NCBI TaxID / UniProt lineage). It's still useful to stress-test leakage.
    """
    if not virus_name:
        return "unknown"

    family_keywords = {
        "papillomavirus": "Papillomaviridae",
        "herpesvirus": "Herpesviridae",
        "influenza": "Orthomyxoviridae",
        "coronavirus": "Coronaviridae",
        "sars": "Coronaviridae",
        "mers": "Coronaviridae",
        "hiv": "Retroviridae",
        "hepatitis b": "Hepadnaviridae",
        "hepatitis c": "Flaviviridae",
        "dengue": "Flaviviridae",
        "zika": "Flaviviridae",
        "ebola": "Filoviridae",
        "measles": "Paramyxoviridae",
        "mumps": "Paramyxoviridae",
        "rsv": "Pneumoviridae",
        "respiratory syncytial": "Pneumoviridae",
        "rabies": "Rhabdoviridae",
        "polio": "Picornaviridae",
        "rhinovirus": "Picornaviridae",
        "adenovirus": "Adenoviridae",
        "parvovirus": "Parvoviridae",
        "rota": "Reoviridae",
        "norovirus": "Caliciviridae",
        "chikungunya": "Togaviridae",
        "sindbis": "Togaviridae",
    }

    s = str(virus_name).lower()
    for k, fam in family_keywords.items():
        if k in s:
            return fam
    return virus_name.split()[0] if virus_name else "unknown"

def load_viral_taxonomy_map(path: str) -> Dict[str, Dict]:
    """
    Load an explicit viral taxonomy map (JSON):
      viral_uniprot -> {taxid, family, genus, ...}

    Prefer this over keyword heuristics: deterministic + auditable + editable
    offline (taxonomy REST lookups are often unavailable on lab machines).
    """
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("viral taxonomy map must be a JSON object mapping viral_uniprot -> record")

    out: Dict[str, Dict] = {}
    for k, v in raw.items():
        vid = str(k).strip()
        if not vid:
            continue
        if not isinstance(v, dict):
            raise ValueError(f"viral taxonomy map record for {vid!r} must be a JSON object")
        out[vid] = v
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WeightedPPIDataset(Dataset):
    """Dataset supporting fractional sample weights for soft positives."""

    def __init__(self, pairs: List[Dict], embeddings: Dict[str, np.ndarray]):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        viral_emb = torch.from_numpy(self.embeddings[p["viral_id"]]).float()
        host_emb = torch.from_numpy(self.embeddings[p["host_id"]]).float()
        label = torch.tensor(float(p["label"]), dtype=torch.float32)
        weight = torch.tensor(float(p.get("weight", 1.0)), dtype=torch.float32)
        return viral_emb, host_emb, label, weight


class ListwiseReceptorDataset(Dataset):
    """
    Per-virus listwise ranking dataset.

    Each sample is: (viral_id, positive_receptor) + K negatives sampled from the
    receptor candidate pool (optionally biased toward hard negatives).
    """

    def __init__(
        self,
        pos_examples: List[Dict],
        embeddings: Dict[str, np.ndarray],
        receptor_pool: List[str],
        pos_by_virus: Dict[str, Set[str]],
        hard_pool_by_virus: Optional[Dict[str, List[str]]],
        k: int,
        hard_frac: float,
        seed: int,
    ):
        self.pos_examples = pos_examples
        self.embeddings = embeddings
        self.receptor_pool = receptor_pool
        self.pos_by_virus = pos_by_virus
        self.hard_pool_by_virus = hard_pool_by_virus or {}
        self.k = int(k)
        self.hard_frac = float(hard_frac)
        self.seed = int(seed)
        self.epoch = 0
        self.rng = np.random.default_rng(self.seed)
        self.receptor_index: Dict[str, int] = {rid: i for i, rid in enumerate(self.receptor_pool)}

        self.neg_pool_by_virus: Dict[str, List[str]] = {}
        for p in pos_examples:
            v = p["viral_id"]
            if v in self.neg_pool_by_virus:
                continue
            pos_set = self.pos_by_virus.get(v, set())
            self.neg_pool_by_virus[v] = [r for r in self.receptor_pool if r not in pos_set]

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.rng = np.random.default_rng(self.seed + 100000 * self.epoch)

    def set_hard_pool_by_virus(self, hard_pool_by_virus: Dict[str, List[str]]):
        self.hard_pool_by_virus = hard_pool_by_virus or {}

    def __len__(self):
        return len(self.pos_examples)

    def _sample_negs(self, viral_id: str, pos_receptor_id: str) -> List[str]:
        k = self.k
        n_hard = int(round(k * self.hard_frac))
        n_hard = max(0, min(k, n_hard))

        hard_pool = [x for x in self.hard_pool_by_virus.get(viral_id, []) if x != pos_receptor_id]
        neg_pool = [x for x in self.neg_pool_by_virus.get(viral_id, []) if x != pos_receptor_id]

        negs: List[str] = []
        if n_hard > 0 and hard_pool:
            replace = len(hard_pool) < n_hard
            pick = self.rng.choice(hard_pool, size=n_hard, replace=replace).tolist()
            negs.extend(pick)

        n_left = k - len(negs)
        if n_left > 0:
            if not neg_pool:
                raise RuntimeError(f"Empty negative pool for viral_id={viral_id}")
            replace = len(neg_pool) < n_left
            pick = self.rng.choice(neg_pool, size=n_left, replace=replace).tolist()
            negs.extend(pick)

        return negs

    def __getitem__(self, idx):
        p = self.pos_examples[idx]
        v = p["viral_id"]
        r_pos = p["host_id"]
        w = float(p.get("weight", 1.0))

        viral_emb = torch.from_numpy(self.embeddings[v]).float()
        pos_emb = torch.from_numpy(self.embeddings[r_pos]).float()

        neg_ids = self._sample_negs(v, r_pos)
        neg_embs = [torch.from_numpy(self.embeddings[r]).float() for r in neg_ids]
        host_embs = torch.stack([pos_emb] + neg_embs, dim=0)  # [1+K, D]
        host_indices = torch.tensor(
            [self.receptor_index[r] for r in [r_pos] + neg_ids],
            dtype=torch.long,
        )

        weight = torch.tensor(w, dtype=torch.float32)
        return viral_emb, host_embs, weight, str(v), host_indices


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(logits: torch.Tensor, labels: torch.Tensor,
                          margin: float = 0.5, max_pairs: int = 4096) -> torch.Tensor:
    """
    For each (positive, negative) pair, penalise if positive logit is not
    at least `margin` above negative logit.
    """
    logits = logits.view(-1)
    labels = labels.view(-1)
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.sum() * 0.0
    P, N = pos.numel(), neg.numel()
    if P * N > max_pairs:
        k = max_pairs
        i = torch.randint(0, P, (k,), device=logits.device)
        j = torch.randint(0, N, (k,), device=logits.device)
        diff = pos[i] - neg[j]
    else:
        diff = pos[:, None] - neg[None, :]
    return F.softplus(margin - diff).mean()


def listwise_softmax_loss(
    logits: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 1.0,
    pos_margin: float = 0.0,
) -> torch.Tensor:
    """
    Listwise loss per sample: assume logits shape [B, 1+K] with positive at index 0.
    Returns weighted mean cross-entropy.
    """
    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError(f"logits must be [B, >=2], got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = logits / float(temperature)
    if pos_margin != 0.0:
        # Subtracting margin from the positive logit makes the task harder,
        # pushing the model to separate positives from negatives more strongly.
        logits = logits.clone()
        logits[:, 0] = logits[:, 0] - float(pos_margin)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    per = F.cross_entropy(logits, targets, reduction="none")
    return (per * weights).mean()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "embeddings" in data:
        return data["embeddings"]
    return data


def load_candidate_pool(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_gold_pairs(path: str) -> List[Dict]:
    """Load gold-standard virus-receptor pairs from CSV."""
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get("viral_uniprot", "").strip()
            r = row.get("receptor_uniprot", "").strip()
            if not v or not r or v == "-" or r == "-":
                continue
            if " " in r or "MULTIPLE" in r.upper():
                continue
            pairs.append({
                "viral_uniprot": v,
                "receptor_uniprot": r,
                "virus": row.get("virus", ""),
                "viral_protein": row.get("viral_protein", ""),
                "receptor": row.get("receptor", ""),
                "mechanism": row.get("mechanism", ""),
                "weight": 1.0,
                "source": "gold_standard",
            })
    return pairs


def load_expanded_pairs(path: Optional[str]) -> List[Dict]:
    if not path or not Path(path).exists():
        return []
    with open(path) as f:
        data = json.load(f)
    # Normalise field names to match gold_pairs schema
    normalised = []
    for p in data:
        normalised.append({
            "viral_uniprot": p.get("viral_uniprot", ""),
            "receptor_uniprot": p.get("receptor_uniprot", ""),
            "virus": p.get("virus", ""),
            "viral_protein": p.get("viral_protein", ""),
            "receptor": p.get("receptor", ""),
            "mechanism": p.get("mechanism", ""),
            "weight": float(p.get("weight", 0.5)),
            "source": p.get("source", "expanded"),
        })
    dedup: Dict[Tuple[str, str], Dict] = {}
    for p in normalised:
        v = str(p.get("viral_uniprot", "")).strip()
        r = str(p.get("receptor_uniprot", "")).strip()
        if not v or not r:
            continue
        key = (v, r)
        prev = dedup.get(key)
        if prev is None or float(p.get("weight", 0.0)) > float(prev.get("weight", 0.0)):
            dedup[key] = p
    return list(dedup.values())


def cap_expanded_pairs_per_receptor(
    expanded_pairs: List[Dict],
    max_per_receptor: int,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    Cap expanded soft positives per receptor to reduce head-receptor dominance.
    """
    k = int(max_per_receptor)
    if k <= 0:
        return expanded_pairs, {
            "enabled": 0.0,
            "cap_k": float(k),
            "n_pairs_before": float(len(expanded_pairs)),
            "n_pairs_after": float(len(expanded_pairs)),
            "n_receptors": 0.0,
            "n_capped_receptors": 0.0,
            "dropped_pairs": 0.0,
        }

    by_receptor: Dict[str, List[Dict]] = defaultdict(list)
    for p in expanded_pairs:
        r = str(p.get("receptor_uniprot", "")).strip()
        if not r:
            continue
        by_receptor[r].append(p)

    kept: List[Dict] = []
    n_capped = 0
    dropped = 0
    for r, rows in by_receptor.items():
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                -float(x.get("weight", 0.0)),
                str(x.get("viral_uniprot", "")),
                str(x.get("source", "")),
            ),
        )
        keep_rows = rows_sorted[:k]
        keep_n = len(keep_rows)
        if len(rows_sorted) > k:
            n_capped += 1
        dropped += max(0, len(rows_sorted) - keep_n)
        kept.extend(keep_rows)

    stats = {
        "enabled": 1.0,
        "cap_k": float(k),
        "n_pairs_before": float(len(expanded_pairs)),
        "n_pairs_after": float(len(kept)),
        "n_receptors": float(len(by_receptor)),
        "n_capped_receptors": float(n_capped),
        "dropped_pairs": float(dropped),
    }
    return kept, stats


# ---------------------------------------------------------------------------
# Stage-3 helpers: build positives + hard negative pools
# ---------------------------------------------------------------------------

def build_fold_pos_examples(
    gold_pairs: List[Dict],
    expanded_pairs: List[Dict],
    heldout_viral_ids: Set[str],
    embeddings: Dict[str, np.ndarray],
) -> Tuple[List[Dict], Dict[str, Set[str]]]:
    pos_by_virus: Dict[str, Set[str]] = defaultdict(set)
    examples: List[Dict] = []

    for p in gold_pairs:
        v, r = p["viral_uniprot"], p["receptor_uniprot"]
        if v in heldout_viral_ids:
            continue
        if v not in embeddings or r not in embeddings:
            continue
        pos_by_virus[v].add(r)
        examples.append({"viral_id": v, "host_id": r, "label": 1, "weight": float(p.get("weight", 1.0))})

    for p in expanded_pairs:
        v, r = p["viral_uniprot"], p["receptor_uniprot"]
        if v in heldout_viral_ids:
            continue
        if v not in embeddings or r not in embeddings:
            continue
        pos_by_virus[v].add(r)
        examples.append({"viral_id": v, "host_id": r, "label": 1, "weight": float(p.get("weight", 0.5))})

    return examples, pos_by_virus


def _compute_reweight_factors(
    freq: Dict[str, float],
    mode: str,
    alpha: float,
    w_min: float,
    w_max: float,
) -> Dict[str, float]:
    mode = str(mode).strip().lower()
    if mode in ("", "none", "off", "false"):
        return {}
    if mode not in ("inv_sqrt", "inv"):
        raise ValueError(f"Unsupported reweight mode: {mode}")
    if not freq:
        return {}

    factors: Dict[str, float] = {}
    for key, f in freq.items():
        ff = max(float(f), 1e-8)
        if mode == "inv_sqrt":
            raw = 1.0 / (np.sqrt(ff) ** float(alpha))
        else:
            raw = 1.0 / (ff ** float(alpha))
        factors[key] = float(raw)

    mean_factor = float(np.mean(np.asarray(list(factors.values()), dtype=np.float64)))
    mean_factor = max(mean_factor, 1e-8)
    for key in list(factors.keys()):
        v = factors[key] / mean_factor
        factors[key] = float(np.clip(v, float(w_min), float(w_max)))
    return factors


def apply_tail_reweight_to_pos_examples(
    pos_examples: List[Dict],
    mode: str = "none",
    alpha: float = 0.5,
    w_min: float = 0.5,
    w_max: float = 2.0,
) -> Dict[str, float]:
    """
    Reweight listwise positive groups by receptor frequency to upweight tail receptors.

    Reweight factor is computed from fold-internal positive frequency per receptor:
      - inv_sqrt: factor ~ 1 / sqrt(freq)^alpha
      - inv:      factor ~ 1 / freq^alpha
    The factor is mean-normalized to preserve overall loss scale, then clipped.
    """
    freq: Dict[str, float] = defaultdict(float)
    for p in pos_examples:
        r = str(p["host_id"])
        freq[r] += float(p.get("weight", 1.0))
    factors = _compute_reweight_factors(
        freq=freq,
        mode=mode,
        alpha=alpha,
        w_min=w_min,
        w_max=w_max,
    )
    if not factors:
        return {"enabled": 0.0, "n_pos": float(len(pos_examples)), "n_receptors": float(len(freq)), "factor_mean": 1.0, "factor_min": 1.0, "factor_max": 1.0}

    factor_vals = np.asarray(list(factors.values()), dtype=np.float64)
    for p in pos_examples:
        r = str(p["host_id"])
        p["weight"] = float(p.get("weight", 1.0)) * float(factors.get(r, 1.0))

    return {
        "enabled": 1.0,
        "n_pos": float(len(pos_examples)),
        "n_receptors": float(len(freq)),
        "factor_mean": float(np.mean(factor_vals)),
        "factor_min": float(np.min(factor_vals)),
        "factor_max": float(np.max(factor_vals)),
    }


def apply_dual_axis_reweight_to_pos_examples(
    pos_examples: List[Dict],
    receptor_mode: str = "none",
    receptor_alpha: float = 0.5,
    receptor_w_min: float = 0.5,
    receptor_w_max: float = 2.0,
    virus_mode: str = "none",
    virus_alpha: float = 0.5,
    virus_w_min: float = 0.5,
    virus_w_max: float = 2.0,
    combine_mode: str = "mul",
) -> Dict[str, float]:
    """
    Dual-axis positive reweighting:
      - receptor-axis: upweight tail receptors
      - virus-axis:    upweight tail viruses
    """
    if not pos_examples:
        return {
            "enabled": 0.0,
            "n_pos": 0.0,
            "n_receptors": 0.0,
            "n_viruses": 0.0,
            "combine_mode": str(combine_mode),
            "receptor_enabled": 0.0,
            "virus_enabled": 0.0,
            "receptor_factor_min": 1.0,
            "receptor_factor_max": 1.0,
            "virus_factor_min": 1.0,
            "virus_factor_max": 1.0,
            "sample_factor_min": 1.0,
            "sample_factor_max": 1.0,
            "sample_factor_mean": 1.0,
        }

    receptor_freq: Dict[str, float] = defaultdict(float)
    virus_freq: Dict[str, float] = defaultdict(float)
    for p in pos_examples:
        w = float(p.get("weight", 1.0))
        receptor_freq[str(p["host_id"])] += w
        virus_freq[str(p["viral_id"])] += w

    receptor_factors = _compute_reweight_factors(
        freq=receptor_freq,
        mode=receptor_mode,
        alpha=receptor_alpha,
        w_min=receptor_w_min,
        w_max=receptor_w_max,
    )
    virus_factors = _compute_reweight_factors(
        freq=virus_freq,
        mode=virus_mode,
        alpha=virus_alpha,
        w_min=virus_w_min,
        w_max=virus_w_max,
    )

    combine = str(combine_mode).strip().lower()
    if combine not in ("mul", "mean", "max"):
        raise ValueError(f"Unsupported dual_pos_reweight_combine: {combine_mode}")

    sample_factors: List[float] = []
    for p in pos_examples:
        rf = float(receptor_factors.get(str(p["host_id"]), 1.0))
        vf = float(virus_factors.get(str(p["viral_id"]), 1.0))
        if combine == "mul":
            sf = rf * vf
        elif combine == "mean":
            sf = 0.5 * (rf + vf)
        else:
            sf = max(rf, vf)
        p["weight"] = float(p.get("weight", 1.0)) * float(sf)
        sample_factors.append(float(sf))

    sf_arr = np.asarray(sample_factors, dtype=np.float64) if sample_factors else np.asarray([1.0], dtype=np.float64)
    r_arr = np.asarray(list(receptor_factors.values()), dtype=np.float64) if receptor_factors else np.asarray([1.0], dtype=np.float64)
    v_arr = np.asarray(list(virus_factors.values()), dtype=np.float64) if virus_factors else np.asarray([1.0], dtype=np.float64)
    return {
        "enabled": 1.0 if (receptor_factors or virus_factors) else 0.0,
        "n_pos": float(len(pos_examples)),
        "n_receptors": float(len(receptor_freq)),
        "n_viruses": float(len(virus_freq)),
        "combine_mode": combine,
        "receptor_enabled": 1.0 if receptor_factors else 0.0,
        "virus_enabled": 1.0 if virus_factors else 0.0,
        "receptor_factor_min": float(np.min(r_arr)),
        "receptor_factor_max": float(np.max(r_arr)),
        "virus_factor_min": float(np.min(v_arr)),
        "virus_factor_max": float(np.max(v_arr)),
        "sample_factor_min": float(np.min(sf_arr)),
        "sample_factor_max": float(np.max(sf_arr)),
        "sample_factor_mean": float(np.mean(sf_arr)),
    }


def build_virus_conditional_prior_by_virus(
    pos_examples: List[Dict],
    receptor_pool: List[str],
    alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Build training-time virus-conditional prior vectors on receptor pool.

    prior(v, r) = log p(r|v) - log p(r|bg), with Laplace smoothing + per-virus z-score.
    """
    receptor_pool = [str(x) for x in receptor_pool]
    if not receptor_pool:
        return {}
    alpha = max(float(alpha), 1e-6)
    n = len(receptor_pool)

    bg_counts: Dict[str, float] = defaultdict(float)
    by_virus_counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in pos_examples:
        v = str(p.get("viral_id", "")).strip()
        r = str(p.get("host_id", "")).strip()
        if not v or not r:
            continue
        w = float(p.get("weight", 1.0))
        bg_counts[r] += w
        by_virus_counts[v][r] += w

    z_bg = float(sum(bg_counts.values())) + alpha * float(n)
    if z_bg <= 0:
        return {}

    prior_by_virus: Dict[str, np.ndarray] = {}
    for v, pos_counts in by_virus_counts.items():
        z_pos = float(sum(pos_counts.values())) + alpha * float(n)
        if z_pos <= 0:
            prior_by_virus[v] = np.zeros((n,), dtype=np.float32)
            continue

        arr = np.empty((n,), dtype=np.float32)
        for i, rid in enumerate(receptor_pool):
            p = (float(pos_counts.get(rid, 0.0)) + alpha) / z_pos
            q = (float(bg_counts.get(rid, 0.0)) + alpha) / z_bg
            arr[i] = float(np.log(max(p, 1e-12)) - np.log(max(q, 1e-12)))

        mu = float(arr.mean())
        sd = float(arr.std(ddof=0))
        sd = sd if sd > 1e-8 else 1.0
        arr = ((arr - mu) / sd).astype(np.float32, copy=False)
        prior_by_virus[v] = arr
    return prior_by_virus


@torch.no_grad()
def build_hard_negative_pools(
    model: nn.Module,
    viruses: List[str],
    receptor_pool: List[str],
    embeddings: Dict[str, np.ndarray],
    pos_by_virus: Dict[str, Set[str]],
    device: str,
    hard_pool_size: int,
    score_batch_size: int = 512,
    receptor_family_neighbors: Optional[Dict[str, List[str]]] = None,
    hard_family_frac: float = 0.0,
) -> Dict[str, List[str]]:
    model.eval()
    hard_pool_by_virus: Dict[str, List[str]] = {}

    receptor_pool = [r for r in receptor_pool if r in embeddings]
    if not receptor_pool:
        return hard_pool_by_virus

    host_mat = torch.from_numpy(np.stack([embeddings[r] for r in receptor_pool], axis=0)).float().to(device)

    use_family_proxy = bool(receptor_family_neighbors) and float(hard_family_frac) > 0.0

    def _inject_family_proxy(
        ranked: List[str],
        pos_set: Set[str],
        fam_neighbors: Dict[str, List[str]],
        pool_size: int,
        fam_frac: float,
    ) -> List[str]:
        """
        Keep baseline top-hard pool, then minimally inject family-proxy negatives.

        This avoids destructive replacement of high-confidence hard negatives while
        still guaranteeing a controllable fraction of near-family negatives.
        """
        hard = ranked[: int(pool_size)]
        if not hard or fam_frac <= 0.0:
            return hard

        fam_cands: Set[str] = set()
        for r in pos_set:
            fam_cands.update(fam_neighbors.get(r, []))
        fam_cands.difference_update(pos_set)
        if not fam_cands:
            return hard

        target_n = int(round(float(pool_size) * float(fam_frac)))
        target_n = max(0, min(int(pool_size), target_n))
        cur_n = sum(1 for rid in hard if rid in fam_cands)
        need = target_n - cur_n
        if need <= 0:
            return hard

        fam_extra = [rid for rid in ranked if rid in fam_cands and rid not in hard]
        if not fam_extra:
            return hard

        # Replace from tail, preferring non-family entries, to preserve the hardest head.
        replace_idx = [i for i in range(len(hard) - 1, -1, -1) if hard[i] not in fam_cands]
        if not replace_idx:
            return hard

        take = min(int(need), len(fam_extra), len(replace_idx))
        for i in range(take):
            hard[replace_idx[i]] = fam_extra[i]
        return hard

    for v in viruses:
        if v not in embeddings:
            continue
        pos_set = pos_by_virus.get(v, set())
        viral_vec = torch.from_numpy(embeddings[v]).float().to(device)
        scores: List[float] = []
        for i in range(0, host_mat.size(0), score_batch_size):
            h = host_mat[i : i + score_batch_size]
            vv = viral_vec.unsqueeze(0).expand(h.size(0), -1)
            logits = model(vv, h)
            scores.extend(logits.detach().float().cpu().tolist())

        idx_sorted = np.argsort(np.array(scores))[::-1]
        ranked: List[str] = []
        for idx in idx_sorted:
            rid = receptor_pool[int(idx)]
            if rid in pos_set:
                continue
            ranked.append(rid)

        if use_family_proxy:
            hard = _inject_family_proxy(
                ranked=ranked,
                pos_set=pos_set,
                fam_neighbors=receptor_family_neighbors,
                pool_size=int(hard_pool_size),
                fam_frac=float(hard_family_frac),
            )
        else:
            hard = ranked[: int(hard_pool_size)]
        hard_pool_by_virus[v] = hard

    return hard_pool_by_virus


def _load_model_state_compatible(model: nn.Module, state: dict) -> Tuple[int, int]:
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return len(compatible), len(state)


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    """Return 1..N ranks where rank=1 is highest score."""
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.int32)
    return ranks


@torch.no_grad()
def build_hard_negative_pools_mined(
    mining_models: List[nn.Module],
    method: str,
    viruses: List[str],
    receptor_pool: List[str],
    embeddings: Dict[str, np.ndarray],
    pos_by_virus: Dict[str, Set[str]],
    device: str,
    hard_pool_size: int,
    score_batch_size: int = 512,
    receptor_family_neighbors: Optional[Dict[str, List[str]]] = None,
    hard_family_frac: float = 0.0,
) -> Dict[str, List[str]]:
    """
    Build hard-negative pools using a fixed set of mining models.

    This is useful when you want "consensus hard negatives" (high-scoring
    negatives across multiple models) rather than hard negatives mined from
    the current (possibly unstable) model.
    """
    if not mining_models:
        raise ValueError("mining_models is empty")
    method = str(method).strip().lower()
    if method not in ("rank_mean", "mean_logit", "max_logit"):
        raise ValueError(f"Unsupported hard_pool_mining_method: {method}")

    for m in mining_models:
        m.eval()

    hard_pool_by_virus: Dict[str, List[str]] = {}
    receptor_pool = [r for r in receptor_pool if r in embeddings]
    if not receptor_pool:
        return hard_pool_by_virus

    host_mat = torch.from_numpy(np.stack([embeddings[r] for r in receptor_pool], axis=0)).float().to(device)

    use_family_proxy = bool(receptor_family_neighbors) and float(hard_family_frac) > 0.0

    def _inject_family_proxy(
        ranked: List[str],
        pos_set: Set[str],
        fam_neighbors: Dict[str, List[str]],
        pool_size: int,
        fam_frac: float,
    ) -> List[str]:
        hard = ranked[: int(pool_size)]
        if not hard or fam_frac <= 0.0:
            return hard

        fam_cands: Set[str] = set()
        for r in pos_set:
            fam_cands.update(fam_neighbors.get(r, []))
        fam_cands.difference_update(pos_set)
        if not fam_cands:
            return hard

        target_n = int(round(float(pool_size) * float(fam_frac)))
        target_n = max(0, min(int(pool_size), target_n))
        cur_n = sum(1 for rid in hard if rid in fam_cands)
        need = target_n - cur_n
        if need <= 0:
            return hard

        fam_extra = [rid for rid in ranked if rid in fam_cands and rid not in hard]
        if not fam_extra:
            return hard

        replace_idx = [i for i in range(len(hard) - 1, -1, -1) if hard[i] not in fam_cands]
        if not replace_idx:
            return hard

        take = min(int(need), len(fam_extra), len(replace_idx))
        for i in range(take):
            hard[replace_idx[i]] = fam_extra[i]
        return hard

    for v in viruses:
        if v not in embeddings:
            continue
        pos_set = pos_by_virus.get(v, set())
        viral_vec = torch.from_numpy(embeddings[v]).float().to(device)

        scores_runs: List[np.ndarray] = []
        for model in mining_models:
            scores: List[float] = []
            for i in range(0, host_mat.size(0), score_batch_size):
                h = host_mat[i : i + score_batch_size]
                vv = viral_vec.unsqueeze(0).expand(h.size(0), -1)
                logits = model(vv, h)
                scores.extend(logits.detach().float().cpu().tolist())
            scores_runs.append(np.asarray(scores, dtype=np.float32))

        if method == "rank_mean":
            ranks = np.stack([_ranks_desc(s) for s in scores_runs], axis=0).astype(np.float32)
            fused = ranks.mean(axis=0)  # lower = harder
            idx_sorted = np.argsort(fused)  # ascending
        elif method == "mean_logit":
            fused = np.mean(np.stack(scores_runs, axis=0), axis=0)
            idx_sorted = np.argsort(fused)[::-1]
        else:  # max_logit
            fused = np.max(np.stack(scores_runs, axis=0), axis=0)
            idx_sorted = np.argsort(fused)[::-1]

        ranked: List[str] = []
        for idx in idx_sorted:
            rid = receptor_pool[int(idx)]
            if rid in pos_set:
                continue
            ranked.append(rid)

        if use_family_proxy:
            hard = _inject_family_proxy(
                ranked=ranked,
                pos_set=pos_set,
                fam_neighbors=receptor_family_neighbors,
                pool_size=int(hard_pool_size),
                fam_frac=float(hard_family_frac),
            )
        else:
            hard = ranked[: int(hard_pool_size)]
        hard_pool_by_virus[v] = hard

    return hard_pool_by_virus


def build_receptor_similarity_neighbors(
    receptor_pool: List[str],
    embeddings: Dict[str, np.ndarray],
    topk: int = 64,
) -> Dict[str, List[str]]:
    """
    Build a receptor \"family proxy\" graph from embedding cosine similarity.

    This provides a data-driven fallback when curated receptor-family labels are
    unavailable. Each receptor gets its top-k nearest receptor neighbors.
    """
    receptor_pool = [r for r in receptor_pool if r in embeddings]
    if not receptor_pool:
        return {}
    if len(receptor_pool) == 1:
        return {receptor_pool[0]: []}

    mat = np.stack([embeddings[r] for r in receptor_pool], axis=0).astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    mat = mat / norms

    sim = mat @ mat.T
    np.fill_diagonal(sim, -np.inf)

    k = int(max(1, min(int(topk), len(receptor_pool) - 1)))
    idx_part = np.argpartition(sim, -k, axis=1)[:, -k:]
    neighbors: Dict[str, List[str]] = {}
    for i, rid in enumerate(receptor_pool):
        row_idx = idx_part[i]
        row_scores = sim[i, row_idx]
        order = row_idx[np.argsort(row_scores)[::-1]]
        neighbors[rid] = [receptor_pool[int(j)] for j in order if int(j) != i][:k]
    return neighbors


# ---------------------------------------------------------------------------
# Build training pairs for one LOVO fold
# ---------------------------------------------------------------------------

def build_fold_train_pairs(
    gold_pairs: List[Dict],
    expanded_pairs: List[Dict],
    heldout_viral_ids: Set[str],
    receptor_pool: List[str],
    embeddings: Dict[str, np.ndarray],
    neg_ratio: float,
    seed: int,
) -> List[Dict]:
    """
    Build training dataset for one LOVO fold:
      Positives: all gold pairs where viral_uniprot NOT IN heldout_viral_ids AND
                 both proteins have embeddings.
      Soft positives: expanded pairs with same filter and fractional weight.
      Negatives: sample from receptor_pool, excluding known positives for
                 each training virus. Ratio = neg_ratio per positive.
    """
    rng = np.random.default_rng(seed)

    # Collect all positives (gold + expanded), excluding test virus
    pos_by_virus: Dict[str, Set[str]] = defaultdict(set)
    all_pos_rows: List[Dict] = []

    for p in gold_pairs:
        v, r = p["viral_uniprot"], p["receptor_uniprot"]
        if v in heldout_viral_ids:
            continue
        if v not in embeddings or r not in embeddings:
            continue
        pos_by_virus[v].add(r)
        all_pos_rows.append({
            "viral_id": v, "host_id": r, "label": 1,
            "weight": p.get("weight", 1.0),
        })

    for p in expanded_pairs:
        v, r = p["viral_uniprot"], p["receptor_uniprot"]
        if v in heldout_viral_ids:
            continue
        if v not in embeddings or r not in embeddings:
            continue
        pos_by_virus[v].add(r)   # track as known positive to avoid as negative
        all_pos_rows.append({
            "viral_id": v, "host_id": r, "label": 1,
            "weight": p.get("weight", 0.5),
        })

    if not all_pos_rows:
        return []

    # Sample negatives per virus from receptor_pool
    receptor_arr = [c for c in receptor_pool if c in embeddings]
    all_neg_rows: List[Dict] = []

    for v, pos_set in pos_by_virus.items():
        valid_neg = [c for c in receptor_arr if c not in pos_set]
        n_pos = len(pos_set)
        n_neg = min(len(valid_neg), int(round(n_pos * neg_ratio)))
        if n_neg == 0:
            continue
        chosen = rng.choice(valid_neg, size=n_neg, replace=False).tolist()
        for r in chosen:
            all_neg_rows.append({
                "viral_id": v, "host_id": r, "label": 0, "weight": 1.0,
            })

    result = all_pos_rows + all_neg_rows
    random.shuffle(result)

    n_pos = sum(1 for p in result if p["label"] == 1)
    n_neg = sum(1 for p in result if p["label"] == 0)
    print(f"    Train pairs: {n_pos} pos (gold+expanded) + {n_neg} neg")
    return result


# ---------------------------------------------------------------------------
# Evaluation: rank all receptor candidates for one held-out viral protein
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_fold(
    model: nn.Module,
    test_viral_id: str,
    known_receptor_ids: List[str],
    receptor_pool: List[str],
    embeddings: Dict[str, np.ndarray],
    device: str,
    ks: List[int],
    batch_size: int = 512,
) -> Dict:
    """Score all receptor_pool candidates and compute ranking metrics."""
    model.eval()

    if test_viral_id not in embeddings:
        return {"viral_id": test_viral_id, "error": "no embedding"}

    valid_candidates = [c for c in receptor_pool if c in embeddings]
    known_set = set(known_receptor_ids)
    # Only count receptors that are actually in the candidate pool
    evaluable_receptors = [r for r in known_receptor_ids if r in set(valid_candidates)]

    if not evaluable_receptors:
        return {
            "viral_id": test_viral_id,
            "known_receptors": known_receptor_ids,
            "pos_count": 0,
            "candidate_pool_size": len(valid_candidates),
            "note": "Known receptors not in candidate pool",
        }

    # Score all candidates in batches
    viral_emb = torch.from_numpy(embeddings[test_viral_id]).float()
    scores_list = []
    for start in range(0, len(valid_candidates), batch_size):
        end = min(start + batch_size, len(valid_candidates))
        chunk = valid_candidates[start:end]
        h_batch = torch.from_numpy(
            np.stack([embeddings[h] for h in chunk])
        ).float().to(device)
        v_batch = viral_emb.unsqueeze(0).expand(len(chunk), -1).to(device)
        logits = model(v_batch, h_batch)
        scores_list.append(torch.sigmoid(logits).cpu().numpy())

    scores = np.concatenate(scores_list)
    labels = np.array([1 if c in known_set else 0 for c in valid_candidates])

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_ids = [valid_candidates[i] for i in order]

    pos_count = int(labels.sum())
    pos_ranks = np.where(sorted_labels == 1)[0] + 1  # 1-based

    metrics: Dict = {
        "viral_id": test_viral_id,
        "known_receptors": known_receptor_ids,
        "evaluable_receptors": evaluable_receptors,
        "pos_count": pos_count,
        "candidate_pool_size": len(valid_candidates),
        "best_pos_rank": int(pos_ranks[0]),
        "mrr": float(1.0 / pos_ranks[0]),
        "all_pos_ranks": pos_ranks.tolist(),
        "top10_ids": sorted_ids[:10],
    }
    for k in ks:
        top_k = sorted_labels[:k]
        metrics[f"has_pos_at_{k}"] = int(top_k.sum() > 0)
        metrics[f"p_at_{k}"] = float(top_k.mean())

    return metrics


# ---------------------------------------------------------------------------
# Model initialisation (partial loading to support different esm_dim)
# ---------------------------------------------------------------------------

def init_model_from_checkpoint(cfg: Dict, device: str) -> LowRankInteractionModel:
    model_cfg = cfg["model"]
    model = LowRankInteractionModel(
        esm_dim=model_cfg["esm_dim"],
        proj_dim=model_cfg["proj_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
    )
    ckpt_path = model_cfg.get("checkpoint", "")
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model_state = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        incompatible = [k for k in state if k not in compatible]
        model_state.update(compatible)
        model.load_state_dict(model_state)
        print(f"    Checkpoint: {ckpt_path} ({len(compatible)}/{len(state)} layers loaded)")
        if incompatible:
            print(f"    Skipped (shape mismatch): {incompatible}")
    else:
        print(f"    No checkpoint at '{ckpt_path}'; random init.")
    return model.to(device)


# ---------------------------------------------------------------------------
# Train one LOVO fold
# ---------------------------------------------------------------------------

def train_fold(
    fold_idx: int,
    test_viral_id: str,
    known_receptor_ids: List[str],
    gold_pairs: List[Dict],
    expanded_pairs: List[Dict],
    receptor_pool: List[str],
    embeddings: Dict[str, np.ndarray],
    cfg: Dict,
    device: str,
    seed: int,
    out_dir: Path,
    ks: List[int],
    family_by_viral_id: Optional[Dict[str, str]] = None,
    genus_by_viral_id: Optional[Dict[str, str]] = None,
    lovo_holdout_scope: str = "viral_id",
    receptor_family_neighbors: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Train and evaluate one LOVO fold. Returns best evaluation metrics."""
    train_cfg = cfg["training"]

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx} | Test virus: {test_viral_id}")
    print(f"  Known receptors: {known_receptor_ids}")

    heldout_viral_ids: Set[str] = {test_viral_id}
    heldout_family: Optional[str] = None
    heldout_genus: Optional[str] = None
    if str(lovo_holdout_scope) == "virus_family":
        if family_by_viral_id is None:
            print("  WARNING: lovo_holdout_scope=virus_family but family map missing; using viral_id-only holdout.")
        else:
            heldout_family = family_by_viral_id.get(test_viral_id, None)
            if heldout_family and heldout_family != "unknown":
                heldout_viral_ids = {v for v, fam in family_by_viral_id.items() if fam == heldout_family}
                heldout_viral_ids.add(test_viral_id)
    elif str(lovo_holdout_scope) == "virus_genus":
        if genus_by_viral_id is None:
            print("  WARNING: lovo_holdout_scope=virus_genus but genus map missing; using viral_id-only holdout.")
        else:
            heldout_genus = genus_by_viral_id.get(test_viral_id, None)
            if heldout_genus and heldout_genus != "unknown":
                heldout_viral_ids = {v for v, gen in genus_by_viral_id.items() if gen == heldout_genus}
                heldout_viral_ids.add(test_viral_id)

    loss_mode = str(train_cfg.get("loss", "bce_pairwise")).lower().strip()

    # --- Model (fresh copy from checkpoint per fold) ---
    torch.manual_seed(seed + fold_idx)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = init_model_from_checkpoint(cfg, device)

    # --- Build training data (loss-specific) ---
    if loss_mode == "listwise":
        fold_dir = out_dir / f"fold_{fold_idx:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        pos_examples, pos_by_virus = build_fold_pos_examples(
            gold_pairs=gold_pairs,
            expanded_pairs=expanded_pairs,
            heldout_viral_ids=heldout_viral_ids,
            embeddings=embeddings,
        )
        if not pos_examples:
            print(f"  SKIP: no training positives for fold {fold_idx}")
            return {"fold_idx": fold_idx, "viral_id": test_viral_id, "skipped": True,
                    "known_receptors": known_receptor_ids}

        listwise_k = int(train_cfg.get("listwise_k", 50))
        hard_pool_size = int(train_cfg.get("hard_pool_size", 400))
        hard_frac = float(train_cfg.get("hard_neg_frac", 0.7))
        hard_family_frac = float(train_cfg.get("hard_neg_family_frac", 0.0))
        tail_mode = str(train_cfg.get("tail_pos_reweight_mode", "none"))
        tail_alpha = float(train_cfg.get("tail_pos_reweight_alpha", 0.5))
        tail_w_min = float(train_cfg.get("tail_pos_reweight_min", 0.5))
        tail_w_max = float(train_cfg.get("tail_pos_reweight_max", 2.0))
        receptor_reweight_mode = str(train_cfg.get("receptor_pos_reweight_mode", tail_mode))
        receptor_reweight_alpha = float(train_cfg.get("receptor_pos_reweight_alpha", tail_alpha))
        receptor_reweight_min = float(train_cfg.get("receptor_pos_reweight_min", tail_w_min))
        receptor_reweight_max = float(train_cfg.get("receptor_pos_reweight_max", tail_w_max))
        virus_reweight_mode = str(train_cfg.get("virus_pos_reweight_mode", "none"))
        virus_reweight_alpha = float(train_cfg.get("virus_pos_reweight_alpha", 0.5))
        virus_reweight_min = float(train_cfg.get("virus_pos_reweight_min", 0.5))
        virus_reweight_max = float(train_cfg.get("virus_pos_reweight_max", 2.0))
        dual_reweight_combine = str(train_cfg.get("dual_pos_reweight_combine", "mul"))
        score_bs = int(train_cfg.get("hard_score_batch_size", 512))
        hard_refresh_every = int(train_cfg.get("hard_refresh_every", 0))
        train_virus_prior_weight = float(train_cfg.get("train_virus_prior_weight", 0.0))
        train_virus_prior_alpha = float(train_cfg.get("train_virus_prior_alpha", 1.0))
        family_dropout_p = float(train_cfg.get("family_dropout_p", 0.0))
        family_dropout_p = min(max(family_dropout_p, 0.0), 1.0)

        tail_stats = apply_dual_axis_reweight_to_pos_examples(
            pos_examples=pos_examples,
            receptor_mode=receptor_reweight_mode,
            receptor_alpha=receptor_reweight_alpha,
            receptor_w_min=receptor_reweight_min,
            receptor_w_max=receptor_reweight_max,
            virus_mode=virus_reweight_mode,
            virus_alpha=virus_reweight_alpha,
            virus_w_min=virus_reweight_min,
            virus_w_max=virus_reweight_max,
            combine_mode=dual_reweight_combine,
        )
        if tail_stats.get("enabled", 0.0) > 0:
            print(
                "  Positive reweight enabled: "
                f"receptor=({receptor_reweight_mode}, alpha={receptor_reweight_alpha:.3f}, "
                f"clip=[{receptor_reweight_min:.2f},{receptor_reweight_max:.2f}]), "
                f"virus=({virus_reweight_mode}, alpha={virus_reweight_alpha:.3f}, "
                f"clip=[{virus_reweight_min:.2f},{virus_reweight_max:.2f}]), "
                f"combine={tail_stats['combine_mode']}, "
                f"sample_factor(mean/min/max)="
                f"{tail_stats['sample_factor_mean']:.3f}/"
                f"{tail_stats['sample_factor_min']:.3f}/"
                f"{tail_stats['sample_factor_max']:.3f}"
            )
        if family_dropout_p > 0.0 and train_virus_prior_weight == 0.0:
            print("  WARNING: family_dropout_p > 0 but train_virus_prior_weight == 0 (dropout will be no-op).")

        virus_prior_by_virus: Dict[str, np.ndarray] = {}
        if train_virus_prior_weight != 0.0:
            virus_prior_by_virus = build_virus_conditional_prior_by_virus(
                pos_examples=pos_examples,
                receptor_pool=receptor_pool,
                alpha=train_virus_prior_alpha,
            )
            print(
                "  Train virus-prior enabled: "
                f"wv={train_virus_prior_weight:.3f}, alpha={train_virus_prior_alpha:.3f}, "
                f"family_dropout_p={family_dropout_p:.2f}, "
                f"viruses_with_prior={len(virus_prior_by_virus)}"
            )

        viruses = sorted(pos_by_virus.keys())
        mining_run_dirs = train_cfg.get("hard_pool_mining_run_dirs", None)
        mining_method = str(train_cfg.get("hard_pool_mining_method", "rank_mean"))
        mining_union_with_self = bool(train_cfg.get("hard_pool_mining_union_with_self", False))
        hard_pool_by_virus: Dict[str, List[str]]
        if isinstance(mining_run_dirs, list) and mining_run_dirs:
            print(f"  Hard pool mining: {len(mining_run_dirs)} runs, method={mining_method}")
            mining_models: List[nn.Module] = []
            model_cfg = cfg["model"]
            for rd in mining_run_dirs:
                ckpt_path = Path(str(rd)) / f"fold_{fold_idx:03d}" / "model_best.pth"
                if not ckpt_path.exists():
                    raise FileNotFoundError(str(ckpt_path))
                m = LowRankInteractionModel(
                    esm_dim=model_cfg["esm_dim"],
                    proj_dim=model_cfg["proj_dim"],
                    hidden_dim=model_cfg["hidden_dim"],
                    dropout=model_cfg["dropout"],
                ).to(device)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                state = ckpt.get("model_state_dict", ckpt)
                loaded, total = _load_model_state_compatible(m, state)
                print(f"    Miner ckpt: {ckpt_path} ({loaded}/{total} tensors loaded)")
                m.eval()
                mining_models.append(m)

            mined_pool_by_virus = build_hard_negative_pools_mined(
                mining_models=mining_models,
                method=mining_method,
                viruses=viruses,
                receptor_pool=receptor_pool,
                embeddings=embeddings,
                pos_by_virus=pos_by_virus,
                device=device,
                hard_pool_size=hard_pool_size,
                score_batch_size=score_bs,
                receptor_family_neighbors=receptor_family_neighbors,
                hard_family_frac=hard_family_frac,
            )
            hard_pool_by_virus = mined_pool_by_virus
            if mining_union_with_self:
                print("  Hard pool mining mode: UNION(mined, self)")
                self_pool_by_virus = build_hard_negative_pools(
                    model=model,
                    viruses=viruses,
                    receptor_pool=receptor_pool,
                    embeddings=embeddings,
                    pos_by_virus=pos_by_virus,
                    device=device,
                    hard_pool_size=hard_pool_size,
                    score_batch_size=score_bs,
                    receptor_family_neighbors=receptor_family_neighbors,
                    hard_family_frac=hard_family_frac,
                )
                merged: Dict[str, List[str]] = {}
                for v in viruses:
                    seen = set()
                    out = []
                    for rid in mined_pool_by_virus.get(v, []) + self_pool_by_virus.get(v, []):
                        if rid in seen:
                            continue
                        seen.add(rid)
                        out.append(rid)
                        if len(out) >= hard_pool_size:
                            break
                    merged[v] = out
                hard_pool_by_virus = merged
            # Avoid overriding mined pools via refresh.
            if hard_refresh_every > 0:
                print(f"  NOTE: hard_refresh_every={hard_refresh_every} ignored because mined hard pools are enabled.")
                hard_refresh_every = 0
            with open(fold_dir / "hard_pool_mined.json", "w", encoding="utf-8") as f:
                json.dump(hard_pool_by_virus, f, indent=2, ensure_ascii=False)
            if mining_union_with_self:
                with open(fold_dir / "hard_pool_mined_only.json", "w", encoding="utf-8") as f:
                    json.dump(mined_pool_by_virus, f, indent=2, ensure_ascii=False)
        else:
            hard_pool_by_virus = build_hard_negative_pools(
                model=model,
                viruses=viruses,
                receptor_pool=receptor_pool,
                embeddings=embeddings,
                pos_by_virus=pos_by_virus,
                device=device,
                hard_pool_size=hard_pool_size,
                score_batch_size=score_bs,
                receptor_family_neighbors=receptor_family_neighbors,
                hard_family_frac=hard_family_frac,
            )

        dataset: Dataset = ListwiseReceptorDataset(
            pos_examples=pos_examples,
            embeddings=embeddings,
            receptor_pool=receptor_pool,
            pos_by_virus=pos_by_virus,
            hard_pool_by_virus=hard_pool_by_virus,
            k=listwise_k,
            hard_frac=hard_frac,
            seed=seed + fold_idx * 1000,
        )
        batch_size = min(int(train_cfg.get("batch_size", 16)), len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        n_pos = len(pos_examples)
        n_neg = n_pos * listwise_k
        print(f"  Train groups: {n_pos} pos (gold+expanded) + {n_neg} implicit neg (K={listwise_k})")
    else:
        train_pairs = build_fold_train_pairs(
            gold_pairs=gold_pairs,
            expanded_pairs=expanded_pairs,
            heldout_viral_ids=heldout_viral_ids,
            receptor_pool=receptor_pool,
            embeddings=embeddings,
            neg_ratio=float(train_cfg["neg_ratio"]),
            seed=seed + fold_idx * 1000,
        )
        n_pos = sum(1 for p in train_pairs if p["label"] == 1)
        if n_pos == 0:
            print(f"  SKIP: no training positives for fold {fold_idx}")
            return {"fold_idx": fold_idx, "viral_id": test_viral_id, "skipped": True,
                    "known_receptors": known_receptor_ids}

        dataset = WeightedPPIDataset(train_pairs, embeddings)
        batch_size = min(int(train_cfg.get("batch_size", 64)), len(train_pairs))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["epochs"]),
        eta_min=float(train_cfg["learning_rate"]) * 0.01,
    )

    pos_weight_val = float(train_cfg.get("pos_weight", 20.0))
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    ranking_weight = float(train_cfg.get("ranking_loss_weight", 1.0))
    ranking_margin = float(train_cfg.get("ranking_margin", 0.5))
    listwise_temp = float(train_cfg.get("listwise_temperature", 1.0))
    listwise_pos_margin = float(train_cfg.get("listwise_pos_margin", 0.0))
    gradient_clip = float(train_cfg.get("gradient_clip", 1.0))
    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg.get("early_stopping_patience", 10))
    selection_metric = str(train_cfg.get("selection_metric", "mrr")).strip()

    # --- Training loop ---
    best_mrr = -1.0
    best_metrics: Dict = {}
    patience_counter = 0
    best_state: Optional[Dict] = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if loss_mode == "listwise":
            assert isinstance(dataset, ListwiseReceptorDataset)
            if hard_refresh_every > 0 and epoch > 1 and (epoch % hard_refresh_every == 0):
                print(f"    Refresh hard negatives (epoch {epoch})")
                new_hard = build_hard_negative_pools(
                    model=model,
                    viruses=viruses,
                    receptor_pool=receptor_pool,
                    embeddings=embeddings,
                    pos_by_virus=pos_by_virus,
                    device=device,
                    hard_pool_size=hard_pool_size,
                    score_batch_size=score_bs,
                    receptor_family_neighbors=receptor_family_neighbors,
                    hard_family_frac=hard_family_frac,
                )
                dataset.set_hard_pool_by_virus(new_hard)
            dataset.set_epoch(epoch)

        for batch in loader:
            optimizer.zero_grad()

            if loss_mode == "listwise":
                viral_emb, host_embs, weights, viral_ids, host_indices = batch
                viral_emb = viral_emb.to(device)          # [B, D]
                host_embs = host_embs.to(device)          # [B, 1+K, D]
                weights = weights.to(device)              # [B]
                host_indices = host_indices.to(device)    # [B, 1+K]

                B, Kp1, D = host_embs.shape
                viral_flat = viral_emb.unsqueeze(1).expand(B, Kp1, D).reshape(B * Kp1, D)
                host_flat = host_embs.reshape(B * Kp1, D)
                logits_base = model(viral_flat, host_flat).view(B, Kp1)
                logits = logits_base
                if train_virus_prior_weight != 0.0 and virus_prior_by_virus:
                    prior_vals = torch.zeros((B, Kp1), device=device, dtype=logits_base.dtype)
                    host_idx_np = host_indices.detach().cpu().numpy()
                    for i, vid in enumerate(viral_ids):
                        pv = virus_prior_by_virus.get(str(vid))
                        if pv is None:
                            continue
                        prior_vals[i] = torch.from_numpy(pv[host_idx_np[i]]).to(
                            device=device,
                            dtype=logits_base.dtype,
                        )
                    if family_dropout_p > 0.0:
                        keep = (torch.rand((B, 1), device=device) >= family_dropout_p).to(logits_base.dtype)
                        prior_vals = prior_vals * keep
                    logits = logits_base + float(train_virus_prior_weight) * prior_vals
                loss = listwise_softmax_loss(
                    logits,
                    weights=weights,
                    temperature=listwise_temp,
                    pos_margin=listwise_pos_margin,
                )
            else:
                viral_emb, host_emb, labels, weights = batch
                viral_emb = viral_emb.to(device)
                host_emb = host_emb.to(device)
                labels = labels.to(device)
                weights = weights.to(device)

                logits = model(viral_emb, host_emb)

                bce_per_sample = criterion(logits, labels)
                bce = (bce_per_sample * weights).mean()

                rloss = pairwise_ranking_loss(logits, labels,
                                              margin=ranking_margin, max_pairs=4096)
                loss = bce + ranking_weight * rloss

            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on held-out virus
        metrics = evaluate_fold(
            model, test_viral_id, known_receptor_ids,
            receptor_pool, embeddings, device, ks,
        )
        mrr = metrics.get("mrr", 0.0)
        rank = metrics.get("best_pos_rank", len(receptor_pool) + 1)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            has10 = metrics.get("has_pos_at_10", 0)
            has50 = metrics.get("has_pos_at_50", 0)
            print(f"    Ep {epoch:3d} | loss={avg_loss:.4f} | "
                  f"rank={rank:4d} MRR={mrr:.4f} R@10={has10} R@50={has50}")

        if selection_metric == "has_pos_at_1":
            sel = float(metrics.get("has_pos_at_1", 0.0))
            best_sel = float(best_metrics.get("has_pos_at_1", -1.0))
            improved = (sel > best_sel) or (sel == best_sel and mrr > best_mrr)
        else:
            improved = mrr > best_mrr

        if improved:
            best_mrr = mrr
            best_metrics = dict(metrics)
            best_metrics["fold_idx"] = fold_idx
            best_metrics["best_epoch"] = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    # --- Save fold outputs ---
    fold_dir = out_dir / f"fold_{fold_idx:03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Attach holdout metadata for reproducibility/debugging
    best_metrics["lovo_holdout_scope"] = str(lovo_holdout_scope)
    best_metrics["heldout_family"] = heldout_family
    best_metrics["heldout_genus"] = heldout_genus
    best_metrics["heldout_viral_ids"] = sorted(list(heldout_viral_ids))

    if best_state is not None:
        torch.save(
            {"model_state_dict": best_state,
             "fold_idx": fold_idx,
             "test_viral_id": test_viral_id,
             "best_mrr": best_mrr},
            fold_dir / "model_best.pth",
        )

    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2, ensure_ascii=False)

    print(f"  RESULT: rank={best_metrics.get('best_pos_rank', 'N/A')} "
          f"MRR={best_mrr:.4f} "
          f"R@10={best_metrics.get('has_pos_at_10', 0)} "
          f"R@50={best_metrics.get('has_pos_at_50', 0)}")

    return best_metrics


# ---------------------------------------------------------------------------
# Aggregate fold results
# ---------------------------------------------------------------------------

def aggregate_fold_results(fold_results: List[Dict], ks: List[int]) -> Dict:
    valid = [r for r in fold_results
             if not r.get("skipped") and not r.get("error") and "mrr" in r]

    if not valid:
        return {"n_folds": 0, "error": "no valid folds"}

    def mean_std(key: str) -> Dict:
        vals = [r[key] for r in valid if key in r]
        if not vals:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    agg: Dict = {
        "n_folds_total": len(fold_results),
        "n_folds_valid": len(valid),
        "mrr": mean_std("mrr"),
        "best_pos_rank": mean_std("best_pos_rank"),
    }
    for k in ks:
        agg[f"recall_at_{k}"] = mean_std(f"has_pos_at_{k}")
        agg[f"p_at_{k}"] = mean_std(f"p_at_{k}")

    # Per-virus detail
    agg["per_virus"] = [
        {"viral_id": r.get("viral_id"), "mrr": r.get("mrr", 0),
         "best_pos_rank": r.get("best_pos_rank"),
         "known_receptors": r.get("known_receptors", [])}
        for r in valid
    ]

    print(f"\n{'='*60}")
    print(f"=== LOVO Aggregate Results ({len(valid)}/{len(fold_results)} valid folds) ===")
    print(f"  MRR:            {agg['mrr']['mean']:.4f} ± {agg['mrr']['std']:.4f}")
    print(f"  Mean best rank: {agg['best_pos_rank']['mean']:.1f} / {valid[0].get('candidate_pool_size', '?')}")
    for k in ks:
        r = agg[f"recall_at_{k}"]
        print(f"  Recall@{k:<4d}: {r['mean']:.4f} ± {r['std']:.4f}")

    return agg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage-3 LOVO receptor-ranking fine-tuning")
    p.add_argument("--config", required=True, help="Path to stage3_receptor_config.yaml")
    p.add_argument("--out_dir", default=None, help="Override output directory from config")
    p.add_argument("--checkpoint", default=None, help="Override model.checkpoint in config")
    p.add_argument("--gold_pairs", default=None, help="Override data.gold_pairs in config")
    p.add_argument("--expanded_pairs", default=None, help="Override data.expanded_pairs in config")
    p.add_argument("--embeddings_path", default=None, help="Override data.embeddings_path in config")
    p.add_argument("--receptor_candidate_pool", default=None, help="Override data.receptor_candidate_pool in config")
    p.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (auto-detect if not set)")
    p.add_argument("--fold_idx", type=int, default=None,
                   help="Run only this fold index (0-indexed); None = run all folds")
    p.add_argument(
        "--fold_indices",
        default=None,
        help="Comma-separated fold indices to run (overrides --fold_idx/--max_folds). Example: 0,3,6,9",
    )
    p.add_argument("--max_folds", type=int, default=None,
                   help="Limit number of folds run (smoke test). E.g. --max_folds 3")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip training; aggregate existing fold_{NNN}/metrics.json files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.checkpoint:
        cfg.setdefault("model", {})["checkpoint"] = args.checkpoint
    if args.gold_pairs:
        cfg.setdefault("data", {})["gold_pairs"] = args.gold_pairs
    if args.expanded_pairs:
        cfg.setdefault("data", {})["expanded_pairs"] = args.expanded_pairs
    if args.embeddings_path:
        cfg.setdefault("data", {})["embeddings_path"] = args.embeddings_path
    if args.receptor_candidate_pool:
        cfg.setdefault("data", {})["receptor_candidate_pool"] = args.receptor_candidate_pool
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = cfg.get("evaluation", {}).get("ks", [1, 5, 10, 50, 100, 500])
    data_cfg = cfg["data"]

    # --- Aggregate-only mode ---
    if args.aggregate_only:
        fold_results = []
        for fold_dir in sorted(out_dir.glob("fold_*")):
            m_path = fold_dir / "metrics.json"
            if m_path.exists():
                with open(m_path) as f:
                    fold_results.append(json.load(f))
        print(f"Found {len(fold_results)} fold result files.")
        agg = aggregate_fold_results(fold_results, ks)
        with open(out_dir / "aggregate_metrics.json", "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"Saved to {out_dir}/aggregate_metrics.json")
        return

    # --- Load data ---
    print("Loading embeddings...")
    embeddings = load_embeddings(data_cfg["embeddings_path"])
    print(f"  {len(embeddings)} proteins in cache")

    print("Loading receptor candidate pool...")
    receptor_pool = load_candidate_pool(data_cfg["receptor_candidate_pool"])
    receptor_pool = [c for c in receptor_pool if c in embeddings]
    print(f"  {len(receptor_pool)} candidates with embeddings")

    # Optional: receptor \"family proxy\" (embedding-neighbor graph) for
    # family-aware hard negative sampling in listwise training.
    train_cfg = cfg.get("training", {})
    loss_mode_global = str(train_cfg.get("loss", "bce_pairwise")).lower().strip()
    hard_family_frac = float(train_cfg.get("hard_neg_family_frac", 0.0))
    hard_family_topk = int(train_cfg.get("hard_neg_family_topk", 64))
    receptor_family_neighbors: Optional[Dict[str, List[str]]] = None
    if loss_mode_global == "listwise" and hard_family_frac > 0.0:
        print(
            f"Building receptor family-proxy neighbors "
            f"(topk={hard_family_topk}, frac={hard_family_frac:.2f})..."
        )
        receptor_family_neighbors = build_receptor_similarity_neighbors(
            receptor_pool=receptor_pool,
            embeddings=embeddings,
            topk=hard_family_topk,
        )
        print(f"  Built neighbors for {len(receptor_family_neighbors)} receptors")

    print("Loading gold-standard receptor pairs...")
    gold_pairs = load_gold_pairs(data_cfg["gold_pairs"])
    gold_pairs = [p for p in gold_pairs
                  if p["viral_uniprot"] in embeddings and p["receptor_uniprot"] in embeddings]
    print(f"  {len(gold_pairs)} gold pairs with embeddings")

    expanded_path = data_cfg.get("expanded_pairs")
    expanded_pairs = load_expanded_pairs(expanded_path)
    if expanded_pairs:
        expanded_pairs = [p for p in expanded_pairs
                          if p["viral_uniprot"] in embeddings
                          and p["receptor_uniprot"] in embeddings]
        expanded_cap_per_receptor = int(train_cfg.get("expanded_cap_per_receptor", 0))
        cap_stats: Dict[str, float] = {}
        if expanded_cap_per_receptor > 0:
            expanded_pairs, cap_stats = cap_expanded_pairs_per_receptor(
                expanded_pairs=expanded_pairs,
                max_per_receptor=expanded_cap_per_receptor,
            )
        print(f"  {len(expanded_pairs)} expanded soft pairs with embeddings")
        if cap_stats.get("enabled", 0.0) > 0:
            print(
                "  Expanded cap enabled: "
                f"per_receptor<={int(cap_stats['cap_k'])}, "
                f"receptors_capped={int(cap_stats['n_capped_receptors'])}, "
                f"dropped={int(cap_stats['dropped_pairs'])}, "
                f"pairs_before/after={int(cap_stats['n_pairs_before'])}/{int(cap_stats['n_pairs_after'])}"
            )
    else:
        print("  No expanded pairs (run scripts/expand_receptor_pairs_homolog.py to generate)")

    # --- Build LOVO fold structure ---
    receptors_by_virus: Dict[str, List[str]] = defaultdict(list)
    for p in gold_pairs:
        r = p["receptor_uniprot"]
        if r not in receptors_by_virus[p["viral_uniprot"]]:
            receptors_by_virus[p["viral_uniprot"]].append(r)

    fold_viruses = sorted(receptors_by_virus.keys())
    print(f"\n  LOVO: {len(fold_viruses)} unique viral proteins (folds)")

    # --- Optional: "taxon-level" leakage control (explicit map preferred; heuristic fallback) ---
    lovo_holdout_scope = str(data_cfg.get("lovo_holdout_scope", "viral_id")).strip()
    family_by_viral_id: Dict[str, str] = {}
    genus_by_viral_id: Dict[str, str] = {}

    taxonomy_map_path = data_cfg.get("viral_taxonomy_map", None)
    taxonomy_map: Dict[str, Dict] = {}
    if taxonomy_map_path:
        print(f"Loading viral taxonomy map: {taxonomy_map_path}")
        taxonomy_map = load_viral_taxonomy_map(str(taxonomy_map_path))
        print(f"  {len(taxonomy_map)} viral IDs in map")

    if lovo_holdout_scope in ("virus_family", "virus_genus"):
        if taxonomy_map:
            for v, rec in taxonomy_map.items():
                family_by_viral_id[v] = str(rec.get("family", "")).strip() or "unknown"
                genus_by_viral_id[v] = str(rec.get("genus", "")).strip() or "unknown"
        else:
            virus_label_by_viral: Dict[str, str] = {}
            for p in gold_pairs + expanded_pairs:
                v = str(p.get("viral_uniprot", "")).strip()
                lab = str(p.get("virus", "")).strip()
                if v and lab and v not in virus_label_by_viral:
                    virus_label_by_viral[v] = lab
            for v, lab in virus_label_by_viral.items():
                family_by_viral_id[v] = extract_virus_family(lab)

        if lovo_holdout_scope == "virus_family":
            missing = [v for v in fold_viruses if v not in family_by_viral_id]
            if missing:
                print(
                    f"  WARNING: family map missing for {len(missing)} fold viruses; "
                    f"they will fall back to viral_id holdout."
                )

            fam_counts: Dict[str, int] = defaultdict(int)
            for v in fold_viruses:
                fam_counts[family_by_viral_id.get(v, "unknown")] += 1
            top = sorted(fam_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
            print("  Holdout scope: virus_family. Fold virus family counts (top):")
            for fam, n in top:
                print(f"    - {fam}: {n}")
        else:
            missing = [v for v in fold_viruses if v not in genus_by_viral_id]
            if missing:
                print(
                    f"  WARNING: genus map missing for {len(missing)} fold viruses; "
                    f"they will fall back to viral_id holdout."
                )

            gen_counts: Dict[str, int] = defaultdict(int)
            for v in fold_viruses:
                gen_counts[genus_by_viral_id.get(v, "unknown")] += 1
            top = sorted(gen_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
            print("  Holdout scope: virus_genus. Fold virus genus counts (top):")
            for gen, n in top:
                print(f"    - {gen}: {n}")
    else:
        if lovo_holdout_scope != "viral_id":
            print(f"  WARNING: unknown lovo_holdout_scope={lovo_holdout_scope!r}; using viral_id.")
        lovo_holdout_scope = "viral_id"

    # Apply fold selection
    if args.fold_indices:
        raw = [x.strip() for x in str(args.fold_indices).split(",") if x.strip()]
        try:
            idxs = [int(x) for x in raw]
        except ValueError:
            print(f"Error: invalid --fold_indices '{args.fold_indices}' (must be comma-separated ints)")
            sys.exit(1)
        bad = [i for i in idxs if i < 0 or i >= len(fold_viruses)]
        if bad:
            print(f"Error: fold indices out of range: {bad} (max {len(fold_viruses)-1})")
            sys.exit(1)
        fold_indices = idxs
        fold_viruses_run = [fold_viruses[i] for i in fold_indices]
    elif args.fold_idx is not None:
        if args.fold_idx >= len(fold_viruses):
            print(f"Error: fold_idx {args.fold_idx} out of range (max {len(fold_viruses)-1})")
            sys.exit(1)
        fold_indices = [args.fold_idx]
        fold_viruses_run = [fold_viruses[args.fold_idx]]
    else:
        fold_indices = list(range(len(fold_viruses)))
        fold_viruses_run = fold_viruses
        if args.max_folds is not None:
            fold_indices = fold_indices[:args.max_folds]
            fold_viruses_run = fold_viruses_run[:args.max_folds]

    # Save config snapshot
    with open(out_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # Snapshot taxonomy map for reproducibility (if used)
    if taxonomy_map_path:
        try:
            src = Path(str(taxonomy_map_path))
            if src.exists():
                shutil.copyfile(src, out_dir / "viral_taxonomy_map_snapshot.json")
        except Exception as e:
            print(f"  WARNING: failed to snapshot viral_taxonomy_map ({taxonomy_map_path}): {e}")

    # Save LOVO fold map
    fold_map = {str(i): {"viral_id": v, "known_receptors": receptors_by_virus[v]}
                for i, v in enumerate(fold_viruses)}
    with open(out_dir / "lovo_fold_map.json", "w") as f:
        json.dump(fold_map, f, indent=2, ensure_ascii=False)

    # Save LOVO settings (metadata) separately to keep lovo_fold_map.json backward-compatible
    with open(out_dir / "lovo_settings.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "lovo_holdout_scope": lovo_holdout_scope,
                "viral_taxonomy_map": str(taxonomy_map_path) if taxonomy_map_path else None,
                "family_by_viral_id": family_by_viral_id if lovo_holdout_scope == "virus_family" else None,
                "genus_by_viral_id": genus_by_viral_id if lovo_holdout_scope == "virus_genus" else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --- Run folds ---
    fold_results: List[Dict] = []

    for fold_idx, test_viral_id in zip(fold_indices, fold_viruses_run):
        known_receptors = receptors_by_virus[test_viral_id]
        result = train_fold(
            fold_idx=fold_idx,
            test_viral_id=test_viral_id,
            known_receptor_ids=known_receptors,
            gold_pairs=gold_pairs,
            expanded_pairs=expanded_pairs,
            receptor_pool=receptor_pool,
            embeddings=embeddings,
            cfg=cfg,
            device=device,
            seed=seed,
            out_dir=out_dir,
            ks=ks,
            family_by_viral_id=family_by_viral_id if family_by_viral_id else None,
            genus_by_viral_id=genus_by_viral_id if genus_by_viral_id else None,
            lovo_holdout_scope=lovo_holdout_scope,
            receptor_family_neighbors=receptor_family_neighbors,
        )
        fold_results.append(result)

    # --- Save all fold results ---
    with open(out_dir / "fold_results.json", "w") as f:
        json.dump(fold_results, f, indent=2, ensure_ascii=False)

    # --- Aggregate (if >=3 folds completed) ---
    if len(fold_results) >= 3:
        agg = aggregate_fold_results(fold_results, ks)
        with open(out_dir / "aggregate_metrics.json", "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
