# DeepEntry Reviewer Package — User Manual (draft v0.1)

This manual documents how to run the released DeepEntry deployment pipeline: given one
viral protein sequence, rank the fixed 3,455-protein human candidate pool and obtain the
receptor/entry-factor shortlist. All paths below were verified against the release assets
on 2026-08-25; every command comes from the actual code, not from documentation.

Repository: https://github.com/adistman/DeepEntry (MIT license, tag v1.1.0)

## 1. Environment

The deployment pipeline was executed with:

- Python 3.12, PyTorch 2.9.1+cu128, NumPy 1.26.4, CUDA GPU (NVIDIA RTX 4090 D, 24 GB)
- The `esm` package (facebookresearch/esm) for ESM-2 3B (`esm2_t36_3B_UR50D`)
- sklearn for metrics in the evaluation scripts

## 2. Input assets (paths relative to the release-asset root; set the environment variable `DEEPENTRY_ROOT` to that root, or use the local layout described in the release)

| Asset | Path | Notes |
|---|---|---|
| Candidate pool (3,455 UniProt IDs) | `data/candidate_pool_3455.ids.txt` | one ID per line |
| Pool annotation | `data/candidate_pool_3455.annotation.tsv` | gene / entry name / protein names |
| Host embeddings | `data/embeddings_esm2_3b_dualends.pkl` | `{"embeddings": {uniprot: float32[5120]}}` |
| Deployment checkpoints | `checkpoints/stage3_seed{42,43,44}/fold_*/model_best.pth` | 56 fold checkpoints per seed, 168 total |
| Model definition | `LowRankInteractionModel` | repository `models/` directory |

The pool has four layers (Table S1d): HumanCore_r2 (2,853) + Tier-0 priority entry factors
(96) + Tier-1 UniProt-reviewed GO cell-surface (502) + gold supplement (4) = 3,455.

## 3. Embedding a query protein (dual-end procedure)

The viral protein is embedded with frozen ESM-2 3B, representation layer 36, mean pooling,
dual-end construction:

- sequence length ≤ 1,024: the full-sequence mean vector is used for both ends
- sequence length > 1,024: the N-terminal 1,024-residue window and the C-terminal
  1,024-residue window are mean-pooled separately and concatenated

Result: one 5,120-dimensional float32 vector per protein (2 × 2,560).

```python
def mean_pool(s):
    _, _, toks = batch_converter([("q", s)])
    with torch.no_grad():
        out = esm_model(toks.to(DEVICE), repr_layers=[36], return_contacts=False)
    rep = out["representations"][36][0, 1:len(s)+1].float()
    return rep.mean(0).cpu().numpy().astype(np.float32)

emb = np.concatenate([mean_pool(seq[:1024]), mean_pool(seq[-1024:])])   # if len > 1024
emb = np.concatenate([mean_pool(seq), mean_pool(seq)])                 # if len <= 1024
```

## 4. Scoring against the candidate pool (168-checkpoint ensemble)

For each of the 168 checkpoints:

1. Load the state dict; infer architecture dimensions from the state dict itself:
   `esm_dim = viral_norm.weight.numel()`, `proj_dim = viral_proj.weight.shape[0]`,
   `hidden_dim = classifier.0.weight.shape[0]`.
2. Instantiate `LowRankInteractionModel(esm_dim, proj_dim, hidden_dim)`, load strictly,
   move to GPU, `eval()`.
3. Forward-pass the query embedding against all 3,455 host embeddings in batches of 512.
4. Transform logits: `p = sigmoid(logits)`; per-checkpoint z-score across the pool:
   `z = (p - p.mean()) / (p.std() + 1e-12)`.
5. Average z over the 56 fold checkpoints of each seed → `z42`, `z43`, `z44`.
6. Ensemble score = mean of the three per-seed z-scores. Rank the pool descending.

The per-checkpoint pool-level z-scoring makes the 56-fold × 3-seed ensemble directly
comparable across folds that use different ranker heads.

## 5. Worked example 1 — Rubella virus E1 (P08563, residues 583–1063)

Sequence: the E1 ectodomain used in the post-freeze deployment check (in
`deploy_predict.py` as `E1_SEQ`).

Run:

```bash
python deploy_predict.py   # use a Python 3.12 environment with torch, numpy and esm
```

Expected output (frozen reference, shipped as `examples/example_output/RUBV_E1_top100.tsv`):

- NECTIN4 (Q96NY8) ranked 12th of 3,455 (ensemble mean_z = 6.930868)
- CD36 (P16671) ranked 65th of 3,455
- Output columns: `rank, uniprot, gene, entry, names, mean_z, z42, z43, z44, rank42, rank43, rank44`

[Verification status: end-to-end reproduction run completed 2026-08-25 with
`deploy_predict.py`. Output is byte-identical to the frozen reference:
NECTIN4 (Q96NY8) rank 12, ensemble mean_z 6.930868 (z42 6.569529 / z43 7.222405 /
z44 7.000671); CD36 (P16671) rank 65. GPU wall time ≈ 12 min on one RTX 4090 D.]

## 6. Worked example 2 — SARS-CoV-2 Spike (P0DTC2)

Identical procedure; replace the query sequence (use `deploy_predict_fasta.py`).
ACE2 (Q9BYF1) is the expected top-ranked canonical receptor.

[Verification status: end-to-end run completed 2026-08-25 with
`deploy_predict_fasta.py` on the full 1,273-aa spike (UniProt P0DTC2). ACE2
ranked 1st of 3,455 (ensemble z = 34.4032, far above rank 2 ANPEP at z = 15.09);
TMPRSS2 (O15393), the established spike co-factor, ranked 5th (z = 8.67).
GPU wall time ≈ 12 min on one RTX 4090 D.]

## 7. Reproducing the LOVO56 nested leave-one-viral-protein-out evaluation

The full benchmark evaluation is heavier than deployment inference: it requires the
Stage-2 checkpoint (ranking pre-training, LOVO56 evaluation viruses excluded), the
accession-confirmed fold map and gold-pair table, and 56-fold Stage-3 fine-tuning under
nested validation with per-fold inner validation split.

Entry points (repository `scripts/` directory):

- `train_baseline.py` — Stage-1/2/3 training and evaluation harness (the archived
  Stage-1 held-out metrics were reproduced exactly from this path on 2026-08-25)
- `cross_validate.py` — cross-validation driver
- Data: release archive `data/` (positive pairs, sampled negatives and the
  split-by-virus fold map)
- Supplementary Tables S2/S2b/S3 contain the per-fold and per-virus reference outputs
  against which a reproduction can be checked.

## 8. Availability

- Code: https://github.com/adistman/DeepEntry (MIT, tag v1.1.0; training, evaluation,
  release-verification and figure-generation scripts)
- Data, checkpoints and benchmark outputs: Zenodo deposit (per the Data availability
  statement of the manuscript)
- Reference outputs: release archive `results/` (per-virus ranks, holdout summaries,
  external deployment checks)
