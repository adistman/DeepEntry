# DeepEntry

DeepEntry is a leakage-aware framework for prioritizing host receptor candidates for viral entry proteins, evaluated on the LOVO56 benchmark: a 56-fold leave-one-viral-protein-out (LOVO) retrieval benchmark (56 viral proteins from 52 virus names) over a 3,455-candidate human receptor pool. This repository provides verification utilities, public-facing result tables, and documentation for using the accompanying v1.1.0 data release and model checkpoints.

This repository accompanies a manuscript under consideration. It is not a clinical diagnostic tool. CRISPR/siRNA host-factor screens are used as host-factor context and do not constitute direct receptor validation.

Repository URL: https://github.com/adistman/DeepEntry

## Releases

### v1.1.0 - LOVO56 (56-fold benchmark, current)

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

### v1.0.0 - 38-virus benchmark (historical)

The original 38-virus release is archived at:

- Data and model archive: https://doi.org/10.5281/zenodo.20049088

## Benchmark results (v1.1.0 LOVO56)

Aggregate metrics over the 56 benchmark units (viral proteins; 52 unique virus names; 3-seed z-score mean ensemble); method naming follows the release file `results/benchmark/lovo56_benchmark_11_methods.tsv`:

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

Full per-virus ranks and per-seed scores are included in the v1.1.0 release package.

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
- Current data release (v1.1.0, LOVO56): https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Historical data archive (v1.0.0): https://doi.org/10.5281/zenodo.20049088

## License

Code is released under the MIT License. Data and figure files in the companion archives are intended for CC BY 4.0 release unless superseded by third-party source restrictions.
