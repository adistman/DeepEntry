# DeepEntry

DeepEntry is a leakage-aware framework for prioritizing host receptor candidates for viral entry proteins, evaluated on the LOVO56 benchmark: a 56-virus leave-one-virus-out (LOVO) retrieval benchmark over a 3,455-candidate human receptor pool. This repository provides verification utilities, public-facing result tables, and documentation for using the accompanying v1.1.0 data release and model checkpoints.

This repository accompanies a manuscript under consideration. It is not a clinical diagnostic tool. CRISPR/siRNA host-factor screens are used as host-factor context and do not constitute direct receptor validation.

Repository URL: https://github.com/adistman/DeepEntry

## Releases

### v1.1.0 — LOVO56 (56-virus benchmark, current)

The current data release provides the 56-virus leave-one-virus-out (LOVO) benchmark with a 3,455-candidate receptor pool:

- **GitHub Release**: https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Package: `deepentry-dataset-v1.1.0.tar.gz`
- Contents: curated gold receptor pairs, ESM2-3B embeddings, trained model checkpoint and configuration, full-rank predictions, 11-method benchmark comparison, prior-knowledge ablation, CRISPR/siRNA validation data, main and supplementary figure PDFs, and Supplementary Tables S1-S9.

### v1.0.0 — 38-virus benchmark (historical)

The original 38-virus release is archived at:

- Data and model archive: https://doi.org/10.5281/zenodo.20049088

## Repository contents

- `src/deepentry/`: small utilities for loading release tables and computing ranking metrics.
- `scripts/verify_release.py`: validates the companion data bundle and reproduces headline table checks.
- `scripts/check_checkpoint.py`: verifies that a released checkpoint can be loaded with the public model definition.
- `configs/deepentry_ranker_public.yaml`: public configuration template with relative paths.
- `results/`: manuscript-facing benchmark and diagnostic tables.
- `docs/`: installation, data and reproducibility notes.

Large data files, complete fold-checkpoint sets and figure PDFs are distributed in the companion archives listed under Releases.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

## Quick verification

From the repository root, run:

```bash
python scripts/verify_release.py --zenodo-root ../deepentry-zenodo-release-complete
python scripts/check_checkpoint.py --checkpoint ../deepentry-zenodo-release-complete/models/receptor_ranker_38fold/leave_one_virus/replicate_01/fold_001/model_best.pth
```

Expected headline metrics for the v1.0.0 38-virus leave-one-virus-out benchmark:

| Model | MRR | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| DeepEntry | 0.4011 | 0.6491 | 0.7368 |
| DeepViral | 0.0597 | 0.1930 | 0.2456 |
| PIPR-style RCNN | 0.0237 | 0.0614 | 0.0789 |
| STEP-style Siamese | 0.0138 | 0.0088 | 0.0175 |

## Citation

Please cite the associated manuscript and the archived companion dataset.

- Code repository: https://github.com/adistman/DeepEntry
- Current data release (v1.1.0, LOVO56): https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Historical data archive (v1.0.0): https://doi.org/10.5281/zenodo.20049088

## License

Code is released under the MIT License. Data and figure files in the companion archives are intended for CC BY 4.0 release unless superseded by third-party source restrictions.
