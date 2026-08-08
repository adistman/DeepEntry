# Data Archive

The LOVO56 companion archive (`deepentry-dataset-v1.1.0.tar.gz`, v1.1.0) is
distributed through the GitHub Release assets of this repository
(https://github.com/adistman/DeepEntry/releases/tag/v1.1.0).

The companion archive contains:

- `data/training/` — gold receptor pairs (84 pairs, accession-confirmed),
  expanded soft pairs, and the 3,455-protein candidate pool;
- `data/validation/` — CRISPR/siRNA host-factor overlap tables;
- `data/embeddings/` — ESM-2 3B protein embeddings of the candidate pool;
- `data/interactions/by_virus/` — curated virus-receptor interactions per virus;
- `models/lovo56/` — released model checkpoint and LOVO56 configuration;
- `results/benchmark/` — 11-method benchmark table and full-rank predictions;
- `results/ablation/` — prior-knowledge ablation results;
- `results/biological_credibility/` — manually inspected boundary cases;
- `figures/` — manuscript main and supplementary figure PDFs;
- `supplementary_tables/` — Supplementary Tables S1–S9 (zipped XLSX).

Paths in `models/lovo56/config_seed42.yaml` are relative to the archive root,
so unpack the archive and run the training/reproducibility scripts from that
directory. Use the release scripts in this repository
(`scripts/verify_release.py`, `scripts/check_checkpoint.py`) to validate file
presence, row counts, and headline metrics.

The earlier 38-virus release (v1.0.0) remains archived at
https://doi.org/10.5281/zenodo.20049088.
