#!/usr/bin/env python3
"""Validate the DeepEntry v1.1 (LOVO56) public release bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# LOVO56 headline metrics of the reference model (ALL56, 3-seed mean)
EXPECTED = {"MRR": 0.4927, "R@10": 0.6488, "R@20": 0.6845}


def check_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path}")


def check_count(label: str, observed: int, expected: int, errors: list[str]) -> None:
    if observed != expected:
        errors.append(f"{label} count mismatch: observed {observed}, expected {expected}")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenodo-root", type=Path, required=True, help="Unpacked companion archive root")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    errors: list[str] = []

    # ---- repository benchmark table (in-repo copy) ----
    benchmark = repo / "results" / "lovo56_benchmark_11_methods.tsv"
    check_file(benchmark, errors)
    check_count("benchmark table rows", count_lines(benchmark), 12, errors)
    if benchmark.exists():
        df = pd.read_csv(benchmark, sep="\t")
        row = df[df["model"] == "DeepEntry_current"]
        if row.empty:
            errors.append("DeepEntry_current row missing from benchmark table")
        else:
            row = row.iloc[0]
            for key, expected in EXPECTED.items():
                value = float(row[key])
                if abs(value - expected) > 5e-4:
                    errors.append(f"{key} mismatch: observed {value}, expected {expected}")

    # ---- companion archive (Zenodo) ----
    required = [
        args.zenodo_root / "README.txt",
        args.zenodo_root / "MANIFEST.txt",
        args.zenodo_root / "data/training/gold_receptor_pairs_accession_confirmed.csv",
        args.zenodo_root / "data/training/candidate_pool_3455_ids.txt",
        args.zenodo_root / "data/training/candidate_pool_3455_annotation.tsv",
        args.zenodo_root / "data/training/expanded_receptor_pairs.json",
        args.zenodo_root / "data/validation/crispr_sirna_overlap_forest.tsv",
        args.zenodo_root / "data/validation/crispr_sirna_overlap_meta.tsv",
        args.zenodo_root / "data/validation/crispr_sirna_overlap_summary.tsv",
        args.zenodo_root / "data/validation/crispr_sirna_raw_vs_leakage_compare.tsv",
        args.zenodo_root / "models/lovo56/config_seed42.yaml",
        args.zenodo_root / "models/lovo56/model_best.pth",
        args.zenodo_root / "results/benchmark/lovo56_benchmark_11_methods.tsv",
        args.zenodo_root / "results/benchmark/lovo56_fullrank_3seed_zscore_mean.tsv",
        args.zenodo_root / "results/benchmark/lovo56_split_defs.json",
        args.zenodo_root / "results/ablation/lovo56_prior_ablation_metrics_by_seed.tsv",
        args.zenodo_root / "results/biological_credibility/lovo56_watch_case_summary_20260622.tsv",
        args.zenodo_root / "supplementary_tables/Supplementary_Tables_NatComm_20260722.zip",
    ]
    for path in required:
        check_file(path, errors)

    check_count(
        "gold receptor pairs",
        count_lines(args.zenodo_root / "data/training/gold_receptor_pairs_accession_confirmed.csv"),
        85,
        errors,
    )
    check_count(
        "candidate pool",
        count_lines(args.zenodo_root / "data/training/candidate_pool_3455_ids.txt"),
        3455,
        errors,
    )
    check_count(
        "fullrank predictions",
        count_lines(args.zenodo_root / "results/benchmark/lovo56_fullrank_3seed_zscore_mean.tsv"),
        193481,
        errors,
    )
    check_count(
        "benchmark table rows",
        count_lines(args.zenodo_root / "results/benchmark/lovo56_benchmark_11_methods.tsv"),
        12,
        errors,
    )

    figures_main = [p.name for p in (args.zenodo_root / "figures/main").glob("Figure*_LOVO56_*.pdf")]
    check_count("main figures", len(figures_main), 6, errors)
    figures_supp = [p.name for p in (args.zenodo_root / "figures/supplementary").glob("FigureS*.pdf")]
    check_count("supplementary figures", len(figures_supp), 4, errors)

    if errors:
        print("Release verification failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Release verification passed.")
    print("DeepEntry headline metrics (LOVO56):")
    for key, value in EXPECTED.items():
        print(f"- {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
