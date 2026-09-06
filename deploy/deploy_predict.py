#!/usr/bin/env python3
"""Predict CD36 rank for Rubella E1 (P08563, zero-shot)."""
import os
import pickle, sys
from pathlib import Path
import numpy as np
import torch

MODEL_ROOT = Path(os.environ.get("DEEPENTRY_ROOT",
                    str(Path(__file__).resolve().parent.parent / "release")))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from low_rank_model import LowRankInteractionModel
import esm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Rubella E1 ectodomain (P08563, 583-1063)
E1_SEQ = ("EEAFTYLCTAPGCATQTPVPVRLAGVRFESKIVDGGCFAPWDLEATGACICEIPTDVSCEGLGAWVP"
          "TAPCARIWNGTQRACTFWAVNAYSSGGYAQLASYFNPGGSYYKQYHPTACEVEPAFGHSDAACWGFP"
          "TDTVMSVFALASYVQHPHKTVRVKFHTETRTVWQLSVAGVSCNVTTEHPFCNTPHGQLEVQVPPDPG"
          "DLVEYIMNYTGNQQSRWGLGSPNCHGPDWASPVCQRHSPDCSRLVGATPERPRLRLVDADDPLLRTA"
          "PGPGEVWVTPVIGSQARKCGLHIRAGPYGHATVEMPEWIHAHTTSDPWHPPGPLGLKFKTVRPVALP"
          "RALAPPRNVRVTGCYQCGTPALVEGLAPGGGNCHLTVNGEDVGAFPPGKFVTAALLNTPPPYQVSCG"
          "GESDRASARVIDPAAQSFTGVVYGTHTTAVSETRQTWAEWAAAHWWQLTLGAICALLLAGLLACCAK"
          "CLYYLRGAIAPR")

POOL_IDS = MODEL_ROOT / "data/candidate_pool_3455.ids.txt"
ANNOTATION = MODEL_ROOT / "data/candidate_pool_3455.annotation.tsv"
EMB_CACHE = MODEL_ROOT / "data/embeddings_esm2_3b_dualends.pkl"

RUN_DIRS = {s: MODEL_ROOT / f"checkpoints/stage3_seed{s}" for s in (42,43,44)}

OUT = Path(__file__).resolve().parent / "example_output"
OUT.mkdir(parents=True, exist_ok=True)

def load_pickle(p): return pickle.loads(p.read_bytes())
def load_ids(p): return [l.strip() for l in p.read_text().splitlines() if l.strip()]

ids = load_ids(POOL_IDS)
print(f"Pool: {len(ids)}")

emb_all = load_pickle(EMB_CACHE)["embeddings"]
host_mat = np.stack([emb_all[r] for r in ids], axis=0).astype(np.float32)

# Annotation
annot = {}
with open(ANNOTATION) as f:
    hdr = f.readline().rstrip("\n").split("\t")
    col = {c: i for i, c in enumerate(hdr)}
    for line in f:
        row = line.rstrip("\n").split("\t")
        annot[row[col["uniprot"]]] = {"gene": row[col["primary_gene_symbol"]],
                                       "entry": row[col["uniprot_entry_name"]],
                                       "names": row[col["protein_names"]]}

# Embed E1
print(f"Embedding Rubella E1 ({len(E1_SEQ)} aa)...", flush=True)
esm_model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(DEVICE).half().eval()

def mean_pool(s):
    _, _, toks = batch_converter([("q", s)])
    with torch.no_grad():
        out = esm_model(toks.to(DEVICE), repr_layers=[36], return_contacts=False)
    rep = out["representations"][36][0, 1:len(s)+1].float()
    return rep.mean(0).cpu().numpy().astype(np.float32)

if len(E1_SEQ) <= 1024:
    e = mean_pool(E1_SEQ)
    emb = np.concatenate([e, e])
else:
    emb = np.concatenate([mean_pool(E1_SEQ[:1024]), mean_pool(E1_SEQ[-1024:])])
print(f"Embedding: {emb.shape}", flush=True)
del esm_model; torch.cuda.empty_cache()

v = torch.from_numpy(emb).to(DEVICE)
h = torch.from_numpy(host_mat).to(DEVICE)

# Score with 168 checkpoints (56 folds × 3 seeds)
seed_z = {}
for seed, run_dir in RUN_DIRS.items():
    ckpts = sorted([d / "model_best.pth" for d in sorted(run_dir.glob("fold_*")) if (d / "model_best.pth").exists()])
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
                hh = h[i:i+512]
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

# Find CD36
for rank, idx in enumerate(order, 1):
    if annot.get(ids[idx], {}).get("gene") == "CD36":
        rank42 = int(np.argsort(-seed_z[42])[idx]) + 1
        rank43 = int(np.argsort(-seed_z[43])[idx]) + 1
        rank44 = int(np.argsort(-seed_z[44])[idx]) + 1
        print(f"\n{'='*60}")
        print(f"Rubella E1 → CD36 RANK: {rank} / 3455")
        print(f"  Per-seed: {rank42} / {rank43} / {rank44}")
        print(f"  mean_zscore: {mean_z[idx]:.6f}")
        print(f"  z42={seed_z[42][idx]:.6f} z43={seed_z[43][idx]:.6f} z44={seed_z[44][idx]:.6f}")
        print(f"{'='*60}")
        break

# Save top 100
top100 = OUT / "RUBV_E1_top100.tsv"
with open(top100, "w") as f:
    f.write("rank\tuniprot\tgene\tentry\tnames\tmean_z\tz42\tz43\tz44\trank42\trank43\trank44\n")
    for rank, idx in enumerate(order[:100], 1):
        a = annot.get(ids[idx], {"gene":"","entry":"","names":""})
        f.write(f"{rank}\t{ids[idx]}\t{a['gene']}\t{a['entry']}\t{a['names']}\t"
                f"{mean_z[idx]:.6f}\t{seed_z[42][idx]:.6f}\t{seed_z[43][idx]:.6f}\t{seed_z[44][idx]:.6f}\t"
                f"{int(np.argsort(-seed_z[42])[idx])+1}\t{int(np.argsort(-seed_z[43])[idx])+1}\t{int(np.argsort(-seed_z[44])[idx])+1}\n")
print(f"Top100: {top100}")
print("DONE")
