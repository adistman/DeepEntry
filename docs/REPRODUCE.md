# Reproducibility

## Verify the release bundle

```bash
python scripts/verify_release.py --zenodo-root /path/to/unpacked/deepentry-dataset-v1.1.0
```

The verifier checks the in-repo benchmark table (`results/lovo56_benchmark_11_methods.tsv`)
and the companion archive: required files, row counts (84 gold pairs, 3,455
candidate pool, 193,480 full-rank predictions, 11 benchmark methods), and the
LOVO56 headline metrics of the reference model.

## Verify model checkpoint compatibility

```bash
python scripts/check_checkpoint.py --checkpoint /path/to/unpacked/deepentry-dataset-v1.1.0/models/lovo56/model_best.pth
```

The script loads the checkpoint with the public model definition
(`models/low_rank_model.py`), infers the architecture dimensions from the
state dict, and reports the number of parameters and validation metrics.

## Reproduce the LOVO56 benchmark

Unpack the companion archive and run the training script from the archive
root, so that the config's relative paths resolve against the archive:

```bash
cd /path/to/unpacked/deepentry-dataset-v1.1.0
python /path/to/DeepEntry/train_stage3_receptor.py --config models/lovo56/config_seed42.yaml
```

This runs the leave-one-virus-out fine-tuning over all 56 folds and writes
per-fold results, which aggregate into the published benchmark table.
Single-fold runs and smoke tests are supported:

```bash
python /path/to/DeepEntry/train_stage3_receptor.py --config models/lovo56/config_seed42.yaml --fold_idx 3
python /path/to/DeepEntry/train_stage3_receptor.py --config models/lovo56/config_seed42.yaml --max_folds 3 --epochs 5
```

The training script requires the candidate pool embeddings from
`data/embeddings/` and the released Stage-2 checkpoint from `models/lovo56/`
as initialization.

## Recreate figures

The companion archive includes the final PDF figures used for submission.
Figure-generation scripts and full training reruns require the complete
analysis environment and the external ESM-2 model resources described in the
manuscript.
