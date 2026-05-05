#!/usr/bin/env python3
"""Validate the DeepEntry public release bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

EXPECTED = {"MRR": 0.4011, "Recall@10": 0.6491, "Recall@20": 0.7368}


def check_file(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path}")


def check_count(label: str, observed: int, expected: int, errors: list[str]) -> None:
    if observed != expected:
        errors.append(f"{label} count mismatch: observed {observed}, expected {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenodo-root", type=Path, required=True, help="Unpacked companion archive root")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    errors: list[str] = []

    benchmark = repo / "results" / "main_benchmark_38virus.tsv"
    check_file(benchmark, errors)
    if benchmark.exists():
        df = pd.read_csv(benchmark, sep="\t")
        row = df[df["model"] == "DeepEntry"]
        if row.empty:
            errors.append("DeepEntry row missing from main benchmark table")
        else:
            row = row.iloc[0]
            for key, expected in EXPECTED.items():
                value = float(row[key])
                if abs(value - expected) > 5e-4:
                    errors.append(f"{key} mismatch: observed {value}, expected {expected}")

    required = [
        args.zenodo_root / "README.txt",
        args.zenodo_root / "MANIFEST.txt",
        args.zenodo_root / "metadata/benchmark_run_manifest.json",
        args.zenodo_root / "data/training/integrated_dataset_clean.tsv",
        args.zenodo_root / "data/validation/crispr_recall_top200_20260308.tsv",
        args.zenodo_root / "figures/main/Figure1_workflow_updated.pdf",
        args.zenodo_root / "figures/main/Figure2_benchmark.pdf",
        args.zenodo_root / "figures/supplementary/FigureS1_bio_credibility.pdf",
        args.zenodo_root / "results/main_benchmark_38virus.tsv",
        args.zenodo_root / "results/full_per_fold/leave_one_virus/replicate_01/fold_results.json",
        args.zenodo_root / "results/full_per_fold/controlled_dynamic_prior/replicate_01/fold_results.json",
    ]
    for path in required:
        check_file(path, errors)

    model_root = args.zenodo_root / "models/receptor_ranker_38fold"
    if model_root.exists():
        checkpoint_files = list(model_root.glob("*/*/fold_*/model_best.pth"))
        check_count("complete benchmark checkpoint", len(checkpoint_files), 342, errors)
        for protocol in ["leave_one_virus", "leave_family_out", "leave_genus_out"]:
            protocol_count = len(list((model_root / protocol).glob("replicate_*/fold_*/model_best.pth")))
            check_count(f"{protocol} checkpoint", protocol_count, 114, errors)
    else:
        errors.append(f"missing: {model_root}")

    per_fold_root = args.zenodo_root / "results/full_per_fold"
    if per_fold_root.exists():
        for protocol, expected in {
            "leave_one_virus": 3,
            "controlled_dynamic_prior": 3,
            "leave_family_out": 3,
            "leave_genus_out": 3,
        }.items():
            count = len([p for p in (per_fold_root / protocol).glob("replicate_*/fold_results.json")])
            check_count(f"{protocol} per-fold result", count, expected, errors)
    else:
        errors.append(f"missing: {per_fold_root}")

    if errors:
        print("Release verification failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Release verification passed.")
    print("DeepEntry headline metrics:")
    for key, value in EXPECTED.items():
        print(f"- {key}: {value:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
