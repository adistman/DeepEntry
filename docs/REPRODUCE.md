# Reproducibility

## Verify the release bundle

```bash
python scripts/verify_release.py --zenodo-root ../deepentry-zenodo-release-complete
```

## Recreate headline table checks

The verifier reads `results/main_benchmark_38virus.tsv` from the repository and checks that the DeepEntry row matches the manuscript headline values.

## Verify model checkpoint compatibility

```bash
python scripts/check_checkpoint.py --checkpoint ../deepentry-zenodo-release-complete/models/receptor_ranker_38fold/leave_one_virus/replicate_01/fold_001/model_best.pth
```

## Recreate figures

The companion archive includes the final PDF figures used for submission. Figure-generation scripts and full training reruns require the complete analysis environment and the external ESM-2 model resources described in the manuscript.
