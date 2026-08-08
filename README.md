# DeepEntry

DeepEntry is a leakage-aware framework for prioritizing host receptor candidates for viral entry proteins, evaluated on the LOVO56 benchmark: a 56-virus leave-one-virus-out (LOVO) retrieval benchmark over a 3,455-candidate human receptor pool. This repository provides verification utilities, public-facing result tables, and documentation for using the accompanying v1.1.0 data release and model checkpoints.

This repository accompanies a manuscript under consideration. It is not a clinical diagnostic tool. CRISPR/siRNA host-factor screens are used as host-factor context and do not constitute direct receptor validation.

Repository URL: https://github.com/adistman/DeepEntry

## Releases

### v1.1.0 - LOVO56 (56-virus benchmark, current)

The current data release provides the 56-virus leave-one-virus-out (LOVO) benchmark with a 3,455-candidate receptor pool:

- **GitHub Release**: https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Package: `deepentry-dataset-v1.1.0.tar.gz`
- Contents: curated gold receptor pairs, ESM2-3B embeddings, trained model checkpoint and configuration, full-rank predictions, 11-method benchmark comparison, prior-knowledge ablation, CRISPR/siRNA validation data, main and supplementary figure PDFs, and Supplementary Tables S1-S9.

### v1.0.0 - 38-virus benchmark (historical)

The original 38-virus release is archived at:

- Data and model archive: https://doi.org/10.5281/zenodo.20049088

## Benchmark results (v1.1.0 LOVO56)

Aggregate metrics over the 56 benchmark viruses (3-seed z-score mean ensemble); method naming follows the release file `results/benchmark/lovo56_benchmark_11_methods.tsv`:

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

## Repository contents

- `src/deepentry/`: small utilities for loading release tables and computing ranking metrics.
- `scripts/verify_release.py`: validates a companion data bundle and reproduces headline table checks.
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

## Verification utilities

`scripts/verify_release.py` and `scripts/check_checkpoint.py` validate the structure of a companion data bundle and verify that released checkpoints load with the public model definition. These utilities correspond to the v1.0.0 code snapshot; the v1.1.0 data package is self-described by its manifest (`MANIFEST.txt`) and can be inspected through the tables under `results/` in the release package.

## Citation

Please cite the associated manuscript and the archived companion dataset.

- Code repository: https://github.com/adistman/DeepEntry
- Current data release (v1.1.0, LOVO56): https://github.com/adistman/DeepEntry/releases/tag/v1.1.0
- Historical data archive (v1.0.0): https://doi.org/10.5281/zenodo.20049088

## License

Code is released under the MIT License. Data and figure files in the companion archives are intended for CC BY 4.0 release unless superseded by third-party source restrictions.
