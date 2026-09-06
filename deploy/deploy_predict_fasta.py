#!/usr/bin/env python3
"""DeepEntry deployment: rank the fixed 3,455-protein candidate pool for any
viral protein given as FASTA (single sequence).

Scoring is identical to the verified Rubella E1 example (deploy_predict.py):
frozen ESM-2 3B dual-end embedding -> 168-checkpoint ensemble (56 folds x 3
seeds) -> per-checkpoint sigmoid -> pool-level z-score -> per-seed mean ->
ensemble mean -> ranking.

Usage:
    python deploy_predict_fasta.py --fasta query.fasta --out results_dir
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_ROOT = Path(os.environ.get("DEEPENTRY_ROOT",
                    str(Path(__file__).resolve().parent.parent / "release")))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from low_rank_model import LowRankInteractionModel
import esm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POOL_IDS = MODEL_ROOT / "data/candidate_pool_3455.ids.txt"
ANNOTATION = MODEL_ROOT / "data/candidate_pool_3455.annotation.tsv"
EMB_CACHE = MODEL_ROOT / "data/embeddings_esm2_3b_dualends.pkl"
RUN_DIRS = {s: MODEL_ROOT / f"checkpoints/stage3_seed{s}"
            for s in (42, 43, 44)}


def load_ids(p):
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def read_fasta(p):
    lines = p.read_text().strip().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">"))


def mean_pool(esm_model, batch_converter, s):
    _, _, toks = batch_converter([("q", s)])
    with torch.no_grad():
        out = esm_model(toks.to(DEVICE), repr_layers=[36], return_contacts=False)
    rep = out["representations"][36][0, 1:len(s) + 1].float()
    return rep.mean(0).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = read_fasta(Path(args.fasta))
    print(f"Device: {DEVICE}")
    print(f"Query length: {len(seq)} aa")

    ids = load_ids(POOL_IDS)
    print(f"Pool: {len(ids)}")

    with open(EMB_CACHE, "rb") as f:
        emb_all = pickle.load(f)["embeddings"]
    host_mat = np.stack([emb_all[r] for r in ids], axis=0).astype(np.float32)

    annot = {}
    with open(ANNOTATION) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        col = {c: i for i, c in enumerate(hdr)}
        for line in f:
            row = line.rstrip("\n").split("\t")
            annot[row[col["uniprot"]]] = {"gene": row[col["primary_gene_symbol"]],
                                          "entry": row[col["uniprot_entry_name"]],
                                          "names": row[col["protein_names"]]}

    print("Embedding query (ESM-2 3B, layer 36, dual-end)...", flush=True)
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    esm_model = esm_model.to(DEVICE).half().eval()
    batch_converter = alphabet.get_batch_converter()
    if len(seq) <= 1024:
        e = mean_pool(esm_model, batch_converter, seq)
        emb = np.concatenate([e, e])
    else:
        emb = np.concatenate([mean_pool(esm_model, batch_converter, seq[:1024]),
                              mean_pool(esm_model, batch_converter, seq[-1024:])])
    print(f"Embedding: {emb.shape}", flush=True)
    del esm_model
    torch.cuda.empty_cache()

    v = torch.from_numpy(emb).to(DEVICE)
    h = torch.from_numpy(host_mat).to(DEVICE)

    seed_z = {}
    for seed, run_dir in RUN_DIRS.items():
        ckpts = sorted([d / "model_best.pth" for d in sorted(run_dir.glob("fold_*"))
                        if (d / "model_best.pth").exists()])
        print(f"Seed {seed}: {len(ckpts)} ckpts", flush=True)
        zsum = np.zeros(len(ids), dtype=np.float64)
        for ck in ckpts:
            ckpt = torch.load(ck, map_location="cpu", weights_only=True)
            sd = ckpt["model_state_dict"]
            esm_dim = int(sd["viral_norm.weight"].numel())
            proj_dim = int(sd["viral_proj.weight"].shape[0])
            hidden_dim = int(sd["classifier.0.weight"].shape[0])
            model = LowRankInteractionModel(esm_dim=esm_dim, proj_dim=proj_dim, hidden_dim=hidden_dim)
            model.load_state_dict(sd, strict=True)
            model.to(DEVICE).eval()
            logits = []
            with torch.no_grad():
                for i in range(0, len(ids), 512):
                    hh = h[i:i + 512]
                    vv = v.unsqueeze(0).expand(hh.size(0), -1)
                    logits.append(model(vv, hh).cpu().numpy().reshape(-1))
            logits = np.concatenate(logits).astype(np.float64)
            p = 1.0 / (1.0 + np.exp(-logits))
            z = (p - p.mean()) / (p.std() + 1e-12)
            zsum += z
            del model
        seed_z[seed] = zsum / len(ckpts)

    mean_z = sum(seed_z.values()) / len(seed_z)
    order = np.argsort(-mean_z)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full = out / "full_ranking.tsv"
    with open(full, "w") as f:
        f.write("rank\tuniprot\tgene\tentry\tnames\tmean_z\tz42\tz43\tz44\n")
        for rank, idx in enumerate(order, 1):
            a = annot.get(ids[idx], {"gene": "", "entry": "", "names": ""})
            f.write(f"{rank}\t{ids[idx]}\t{a['gene']}\t{a['entry']}\t{a['names']}\t"
                    f"{mean_z[idx]:.6f}\t{seed_z[42][idx]:.6f}\t{seed_z[43][idx]:.6f}\t{seed_z[44][idx]:.6f}\n")
    print(f"Full ranking: {full}")

    print(f"\nTop 20 of {len(ids)}:")
    for rank, idx in enumerate(order[:20], 1):
        a = annot.get(ids[idx], {"gene": "", "names": ""})
        print(f"  {rank:>3}  {ids[idx]:<10} {a['gene']:<10} {a['names'][:50]}  z={mean_z[idx]:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
