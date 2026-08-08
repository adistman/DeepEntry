# Data Archive

The LOVO56 companion archive (`deepentry-dataset-v1.1.0.tar.gz`, v1.1.0) is
distributed through the GitHub Release assets of this repository
(https://github.com/adistman/DeepEntry/releases/tag/v1.1.0).

The companion archive contains:

- `data/training/` — gold receptor pairs (84 pairs, accession-confirmed),
  expanded soft pairs, and the 3,455-protein candidate pool;
- `data/validation/` — CRISPR/siRNA host-factor overlap tables;
- `data/embeddings/` — ESM-2 3B protein embeddings; the file covers the
  3,455-protein candidate pool plus additional proteins used during training
  (10,160 entries in total; see the archive README.txt);
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

## Data provenance and third-party sources

- Protein embeddings were generated with ESM-2 (3B-parameter model;
  Lin et al., Science 379, 1123-1130, 2023;
  https://github.com/facebookresearch/esm).
- Gold interaction pairs carry PubMed evidence identifiers (pmid column);
  see the manuscript for the curation procedure.
- CRISPR/siRNA host-factor screens are used as host-factor context and are
  sourced from the published screens cited in the accompanying manuscript.
- `results/benchmark/lovo56_family_map.json` covers all viruses in the
  curation pipeline, not only the 56 benchmark viruses; a small number of
  non-standard or host-species labels may appear for entries outside the
  benchmark set.

## Metric aggregation

Reported metrics follow the convention documented in the README
("Metric definition and aggregation"): per-seed, best-known receptor per
virus, mean over the 56 benchmark viruses, then mean over seeds 42/43/44.
`scripts/aggregate_lovo56_metrics.py` recomputes the headline metrics from
`results/benchmark/lovo56_fullrank_3seed_zscore_mean.tsv` and checks them
against `results/benchmark/lovo56_benchmark_11_methods.tsv`.
