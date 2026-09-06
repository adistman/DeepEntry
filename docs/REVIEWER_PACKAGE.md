# DeepEntry Reviewer Package (draft)

Nature Methods requires that code, and ideally a user manual and example data for
testing the code, be made available to reviewers. This directory is that package,
prepared 2026-08-25. The main manual is `USER_MANUAL.md`.

## Files

| File | Purpose | Verification |
|---|---|---|
| `USER_MANUAL.md` | Full user manual (environment, assets, embedding, scoring, examples, LOVO56 reproduction entry points) | Written from actual code paths; every command traced to a real file |
| `deploy_predict.py` | Worked example: Rubella E1 (P08563, 583–1063) → 3,455-protein pool ranking | **Run end-to-end 2026-08-25; output byte-identical to the frozen reference** (NECTIN4 rank 12, CD36 rank 65) |
| `deploy_predict_fasta.py` | General entry point: any viral protein (single-sequence FASTA) → full pool ranking | **Run end-to-end 2026-08-25 on SARS-CoV-2 Spike P0DTC2** (ACE2 rank 1, TMPRSS2 rank 5) |
| `example_fasta/P0DTC2_spike.fasta` | SARS-CoV-2 Spike (UniProt P0DTC2, 1,273 aa) | Sequence from the project UniProt records cache |
| `example_fasta/RUBV_E1_ectodomain.fasta` | Rubella E1 ectodomain (P08563, 583–1063, 481 aa) | Same sequence embedded in the verified run |
| `example_output/RUBV_E1_top100.tsv` | Example 1 output | Reproduced exactly |
| `example2_output_P0DTC2/full_ranking.tsv` | Example 2 output (3,455 ranks) | Produced 2026-08-25 |

## Verified facts in this package

- Deployment ensemble: `checkpoints/stage3_seed{42,43,44}`, 56 `fold_*/model_best.pth` per seed = **168 checkpoints**, matching the manuscript's Methods description.
- Candidate pool: 3,455 IDs (Table S1d layers), fixed, accession-confirmed.
- Embedding: frozen ESM-2 3B (`esm2_t36_3B_UR50D`), layer 36, mean pooling, dual-end 2×1,024 construction → 5,120-dim.
- Scoring: per-checkpoint sigmoid → pool-level z-score → per-seed mean (56 folds) → ensemble mean of seeds 42/43/44.
- Both examples reproduce the numbers cited in the manuscript (NECTIN4 12th, CD36 60–74 range across Sapovirus queries elsewhere; ACE2 top-ranked).

## Known limits

- Full LOVO56 nested-validation reproduction (Stage-2 + 56-fold Stage-3 retraining)
  is described in `USER_MANUAL.md` §7 but not included in this light package; the
  archived per-fold and per-virus reference outputs (Supplementary Tables S2/S2b/S3)
  allow a reproduction check.
- Zenodo data deposit and the final GitHub user-facing README updates are pending
  (see Data availability statement of the manuscript).
