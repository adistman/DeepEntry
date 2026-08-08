#!/usr/bin/env python3
"""Recompute the LOVO56 headline metrics from the released full-rank predictions.

Aggregation convention (as documented in README.md and the manuscript):

  - For each held-out virus, only its best-known receptor counts: among the
    rows with is_known_receptor=1 for that virus, take the smallest rank.
  - Per-virus MRR = 1 / best_rank (viruses without a known receptor
    contribute 0), R@10 = (best_rank <= 10), R@20 = (best_rank <= 20).
  - Per-seed scores are the mean over the 56 benchmark viruses.
  - Headline numbers are the mean of the three per-seed scores
    (seeds 42, 43 and 44).

The full-rank file stores both a merged 'rank' column and per-seed columns
(rank_seed42/43/44). This script uses the per-seed columns, which is the
convention behind the reported benchmark table: the per-seed ranks are
averaged as metrics, not merged into a single ensemble ranking.

Usage:
    python scripts/aggregate_lovo56_metrics.py \
        --fullrank <archive>/results/benchmark/lovo56_fullrank_3seed_zscore_mean.tsv \
        --benchmark <repo>/results/lovo56_benchmark_11_methods.tsv

Exit code 0 on success (metrics match the benchmark table within tolerance),
1 otherwise.
"""

import argparse
import csv
import sys

TOL = 5e-4  # same tolerance as scripts/verify_release.py


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        sys.exit(f"error: no rows in {path}")
    return rows


def per_seed_metrics(rows, seed_col):
    """Best-known-receptor rank per virus -> per-seed MRR / R@10 / R@20."""
    best = {}  # viral_id -> best rank among known receptors
    for r in rows:
        if r["is_known_receptor"] == "1":
            vid = r["viral_id"]
            rk = int(r[seed_col])
            best[vid] = rk if vid not in best else min(rk, best[vid])
    n_viruses = len({r["viral_id"] for r in rows})
    mrr = sum(1.0 / rk for rk in best.values()) / n_viruses
    r10 = sum(1 for rk in best.values() if rk <= 10) / n_viruses
    r20 = sum(1 for rk in best.values() if rk <= 20) / n_viruses
    return mrr, r10, r20, len(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fullrank", required=True,
                    help="path to lovo56_fullrank_3seed_zscore_mean.tsv")
    ap.add_argument("--benchmark", required=True,
                    help="path to lovo56_benchmark_11_methods.tsv")
    args = ap.parse_args()

    rows = load_rows(args.fullrank)
    seed_cols = [c for c in ("rank_seed42", "rank_seed43", "rank_seed44")
                 if c in rows[0]]
    if len(seed_cols) != 3:
        sys.exit("error: expected rank_seed42/43/44 columns in the full-rank file")

    print(f"Per-seed metrics over {len({r['viral_id'] for r in rows})} viruses:")
    per_seed = []
    for col in seed_cols:
        mrr, r10, r20, n_known = per_seed_metrics(rows, col)
        per_seed.append((mrr, r10, r20))
        print(f"  {col}: MRR={mrr:.6f} R@10={r10:.6f} R@20={r20:.6f}"
              f"  ({n_known} viruses with a known receptor)")

    h_mrr = sum(x[0] for x in per_seed) / 3
    h_r10 = sum(x[1] for x in per_seed) / 3
    h_r20 = sum(x[2] for x in per_seed) / 3
    print(f"Headline (mean of seeds): MRR={h_mrr:.6f} R@10={h_r10:.6f} R@20={h_r20:.6f}")

    bench = load_rows(args.benchmark)
    ref = next((r for r in bench if r.get("model") == "DeepEntry_current"), None)
    if ref is None:
        sys.exit("error: no DeepEntry_current row in the benchmark table")
    b_mrr, b_r10, b_r20 = float(ref["MRR"]), float(ref["R@10"]), float(ref["R@20"])
    print(f"Benchmark table (DeepEntry_current): MRR={b_mrr:.6f}"
          f" R@10={b_r10:.6f} R@20={b_r20:.6f}")

    diffs = [("MRR", abs(h_mrr - b_mrr)),
             ("R@10", abs(h_r10 - b_r10)),
             ("R@20", abs(h_r20 - b_r20))]
    ok = all(d < TOL for _, d in diffs)
    for name, d in diffs:
        status = "OK" if d < TOL else "FAIL"
        print(f"  {name}: |diff|={d:.6f}  {status}")
    print("MATCH: recomputed headline metrics agree with the benchmark table"
          if ok else "MISMATCH: recomputed metrics differ from the benchmark table")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
