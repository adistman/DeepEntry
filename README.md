# DeepEntry

DeepEntry is a leakage-aware framework for prioritizing host receptor candidates for viral entry proteins. The release provides lightweight verification utilities, public-facing result tables, and documentation for using the accompanying data and selected model checkpoints.

This repository accompanies a manuscript under consideration. It is not a clinical diagnostic tool. CRISPR/siRNA host-factor screens are used as plausibility support and do not constitute direct receptor validation.

## Repository contents

- `src/deepentry/`: small utilities for loading release tables and computing ranking metrics.
- `scripts/verify_release.py`: validates the Zenodo-style data bundle and reproduces headline table checks.
- `configs/deepentry_ranker_public.yaml`: public configuration template with relative paths.
- `results/`: manuscript-facing benchmark and diagnostic tables.
- `docs/`: installation, data and reproducibility notes.

Large data files, selected model checkpoints and figure PDFs are distributed in the companion Zenodo-style archive. After formal deposition, cite the Zenodo DOI in the manuscript data-availability statement.

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
python scripts/verify_release.py --zenodo-root ../deepentry-zenodo-release
```

Expected headline metrics for the 38-virus leave-one-virus-out benchmark:

| Model | MRR | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| DeepEntry | 0.4011 | 0.6491 | 0.7368 |
| DeepViral | 0.0597 | 0.1930 | 0.2456 |
| PIPR-style RCNN | 0.0237 | 0.0614 | 0.0789 |
| STEP-style Siamese | 0.0138 | 0.0088 | 0.0175 |

## Citation

Please cite the associated manuscript. A final DOI and formatted citation should be added after journal and data-archive deposition.

## License

Code is released under the MIT License. Data and figure files in the companion archive are intended for CC BY 4.0 release unless superseded by third-party source restrictions.
