# Reproducibility

## Verify the release bundle

```bash
python scripts/verify_release.py --zenodo-root ../deepentry-zenodo-release
```

## Recreate headline table checks

The verifier reads `results/main_benchmark_38virus.tsv` from the repository and checks that the DeepEntry row matches the manuscript headline values.

## Recreate figures

Figure-generating scripts are maintained in the analysis repository. The companion archive includes the final PDF figures used for submission. If the analysis repository is available locally, regenerate figures from:

```bash
python paper/figures/generate_all_figures.py
```

Use the regenerated PDFs in `paper/figures/output/figures of manuscript/` for submission and archive synchronization.
