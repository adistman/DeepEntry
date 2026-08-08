# DeepEntry

DeepEntry is a leakage-aware framework for prioritizing host receptor candidates for viral entry proteins, evaluated on the LOVO56 benchmark: a 56-fold leave-one-viral-protein-out (LOVO) retrieval benchmark (56 viral proteins from 52 virus names) over a 3,455-candidate human receptor pool. This repository provides verification utilities, public-facing result tables, and documentation for using the accompanying v1.1.0 data release and model checkpoints.

This repository accompanies a manuscript under consideration. It is not a clinical diagnostic tool. CRISPR/siRNA host-factor screens are used as host-factor context and do not constitute direct receptor validation.

Repository URL: https://github.com/adistman/DeepEntry

## Releases

### v1.1.0 - LOVO56 (56-fold benchmark)

The current data release provides the 56-fold leave-one-viral-protein-out
(LOVO) benchmark with a 3,455-candidate receptor pool:

- **Split units**: each fold holds out one viral entry protein (viral_id);
  the 56 folds correspond to 52 unique virus names, because four viruses
  contribute two entry proteins each: EBV (P03200/P03231), CMV
  (P06473/P12824), HSV-1 (P06477/Q69091) and Mammalian orthoreovirus
  (P03527/P03528). Metrics are aggregated over the 56 protein units, so
  these four viruses are weighted twice.

- **GitHub Release**: https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Package: `deepentry-dataset-v1.1.0.tar.gz`
- Contents: curated gold receptor pairs, ESM2-3B embeddings, trained model checkpoint and configuration, full-rank predictions, 11-method benchmark comparison, prior-knowledge ablation, CRISPR/siRNA validation data, main and supplementary figure PDFs, and Supplementary Tables S1-S9.

- **Zenodo archive**: https://doi.org/10.5281/zenodo.20049088

## Benchmark results (v1.1.0 LOVO56)

The manuscript reports the **nested-validation protocol** as the primary
result; the 11-method comparison is evaluated under the
test-time-selection protocol (see `docs/REPRODUCE.md`, "Model selection
protocol") so that DeepEntry and the published baselines are scored under
identical conditions.

### Primary result (nested-validation protocol)

Metrics of the reference model (the `baseline_no_prior` variant, i.e. the
same model as "DeepEntry" in the 11-method table below), per seed and
averaged over seeds 42/43/44, from
`results/ablation/lovo56_prior_ablation_metrics_by_seed.tsv`:

| Seed | MRR | R@1 | R@5 | R@10 | R@50 | R@100 | Mean best rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.2922 | 0.1786 | 0.4286 | 0.5000 | 0.6429 | 0.6786 | 210.7 |
| 43 | 0.3149 | 0.2321 | 0.4107 | 0.4821 | 0.6250 | 0.7143 | 216.9 |
| 44 | 0.3096 | 0.2143 | 0.4107 | 0.4821 | 0.6250 | 0.6964 | 195.5 |

Mean over seeds 42/43/44: **MRR 0.3056**, R@1 0.2083, R@5 0.4167, R@10
0.4881, R@50 0.6310, R@100 0.6964, mean best rank 207.7. Recall@20 was
not reported under this protocol.

### 11-method comparison (test-time-selection protocol)

Aggregate metrics over the 56 benchmark units (viral proteins; 52 unique
virus names), mean of per-seed scores for seeds 42/43/44. Under this
protocol each fold's checkpoint is selected on the held-out unit itself,
which imparts an optimistic bias relative to the nested-validation
numbers above (see `docs/REPRODUCE.md`); the table is directly comparable
only within itself, where all methods share identical folds, candidate
pool and selection protocol. Method naming follows the release file
`results/benchmark/lovo56_benchmark_11_methods.tsv`:

| Method | MRR | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| DeepEntry | 0.4927 | 0.6488 | 0.6845 |
| DeepEntry (weighted, no family prior) | 0.4965 | 0.6488 | 0.6845 |
| DeepEntry (family-hard) | 0.4772 | 0.6488 | 0.6845 |
| Prior (host frequency) | 0.1347 | 0.1429 | 0.3214 |
| DeepViral (RF) | 0.1343 | 0.2917 | 0.3869 |
| ESM2 cosine | 0.0035 | 0.0179 | 0.0179 |
| Random | 0.0031 | 0.0060 | 0.0119 |
| BLAST transfer | 0.0031 | 0.0000 | 0.0000 |
| DSCRIPT (human v1) | 0.0016 | 0.0000 | 0.0000 |
| Prior (family) | 0.0011 | 0.0000 | 0.0000 |
| Prior (virus label) | 0.0008 | 0.0000 | 0.0000 |

Full per-virus ranks and per-seed scores are included in the v1.1.0
release package. The per-method baseline prediction files behind this
table (193,480 scored pairs per file, one file per method and seed,
sha256-checked) are included in the package under
`results/baseline_predictions/`; method definitions, parameters and
source references are documented in `README_baselines.md`.

## Metric definition and aggregation

Reported metrics are computed per seed and per benchmark virus. For each
held-out virus, only its best-known receptor counts: among the gold pairs of
that virus, the smallest rank is taken. Per-virus MRR is the reciprocal of
that rank (viruses without a known receptor contribute 0); per-virus
Recall@10 and Recall@20 indicate whether the best-known receptor ranks within
the top 10 or top 20. Per-seed scores are the mean over the 56 benchmark
viruses; the headline numbers are the mean of the three per-seed scores
(seeds 42, 43 and 44). `scripts/aggregate_lovo56_metrics.py` recomputes the
headline metrics from `results/lovo56_fullrank_3seed_zscore_mean.tsv` and
checks them against `results/lovo56_benchmark_11_methods.tsv`.

## Repository contents

- `models/`: public model definition (`low_rank_model.py`, LowRankInteractionModel) used by the released checkpoints.
- `train_stage3_receptor.py`: leave-one-viral-protein-out (LOVO) fine-tuning entry point for the LOVO56 benchmark.
- `scripts/verify_release.py`: validates a companion data bundle and reproduces headline table checks.
- `scripts/check_checkpoint.py`: verifies that a released checkpoint can be loaded with the public model definition.
- `configs/deepentry_lovo56_public.yaml`: LOVO56 configuration template with package-relative paths.
- `results/`: manuscript-facing benchmark and diagnostic tables.
- `tests/`: unit tests for the public model definition.
- `docs/`: installation, data and reproducibility notes.

Large data files, complete fold-checkpoint sets and figure PDFs are distributed in the companion archives listed under Releases.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Verification utilities

`scripts/verify_release.py` validates the structure of a companion data bundle: required files, row counts (84 gold pairs, 3,455-candidate pool, 193,480 full-rank predictions, 11 benchmark methods) and the LOVO56 headline metrics of the reference model. `scripts/check_checkpoint.py` loads a released checkpoint with the public model definition, infers the architecture dimensions from the state dict, and reports parameter counts and validation metrics.

```bash
python scripts/verify_release.py --zenodo-root /path/to/unpacked/deepentry-dataset-v1.1.0
python scripts/check_checkpoint.py --checkpoint /path/to/unpacked/deepentry-dataset-v1.1.0/models/lovo56/model_best.pth
```

See `docs/REPRODUCE.md` for full reproducibility instructions, including the LOVO56 training entry point.

## Citation

Please cite the associated manuscript and the archived companion dataset.

- Code repository: https://github.com/adistman/DeepEntry
- Data release (v1.1.0, LOVO56): https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Zenodo archive: https://doi.org/10.5281/zenodo.20049088

## License

Code is released under the MIT License. Data and figure files in the companion archives are intended for CC BY 4.0 release unless superseded by third-party source restrictions.
