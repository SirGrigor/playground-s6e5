"""v22.400 — average multi-seed v22 TabPFN runs.

After running notebooks/29_v22_tabpfn_own.py multiple times with --seed N,
each seed writes:
  probs/v22_tabpfn_epochs6_seedN/holdout.npy
  probs/v22_tabpfn_epochs6_seedN/test.npy
  submissions/v22.300_seedN.csv

This script reads all matching seed runs and averages them in rank space,
producing:
  probs/v22_tabpfn_epochs6_avg/holdout.npy
  probs/v22_tabpfn_epochs6_avg/test.npy
  submissions/v22.400.csv

Plus a Step-4 analysis vs safar1_95449 (same as the single-run notebook).

Usage:
  python notebooks/30_v22_seed_average.py
  python notebooks/30_v22_seed_average.py --seeds 42 7 13
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import PROBS, SUBMISSIONS, TARGET, ID
from src.data import load_holdout, load_test
from src.evaluate import auc
from src.blend_math import predict_blend_lb, rank_normalize


def main(seeds: list[int]):
    print("=" * 70)
    print(f"v22.400 — averaging {len(seeds)} v22 TabPFN seed runs: {seeds}")
    print("=" * 70)

    holdout_preds, test_preds = [], []
    found_seeds = []
    for s in seeds:
        prob_dir = PROBS / f"v22_tabpfn_epochs6_seed{s}"
        h_path = prob_dir / "holdout.npy"
        t_path = prob_dir / "test.npy"
        if not (h_path.exists() and t_path.exists()):
            print(f"  seed {s}: MISSING ({prob_dir})")
            continue
        h = np.load(h_path)
        t = np.load(t_path)
        holdout_preds.append(rank_normalize(h))
        test_preds.append(rank_normalize(t))
        h_auc = auc(load_holdout()[TARGET].astype(int).to_numpy(), h)
        print(f"  seed {s}: solo holdout AUC = {h_auc:.5f}")
        found_seeds.append(s)

    if len(found_seeds) < 2:
        print(f"\nNeed ≥ 2 seeds with both holdout.npy and test.npy. Found: {found_seeds}")
        sys.exit(1)

    print(f"\nAveraging {len(found_seeds)} seeds (rank-uniform mean)...")
    holdout_avg = np.mean(holdout_preds, axis=0)
    test_avg = np.mean(test_preds, axis=0)

    y_holdout = load_holdout()[TARGET].astype(int).to_numpy()
    avg_holdout_auc = auc(y_holdout, holdout_avg)
    best_solo = max(auc(y_holdout, np.load(PROBS / f"v22_tabpfn_epochs6_seed{s}/holdout.npy"))
                    for s in found_seeds)
    print(f"\n  Avg holdout AUC: {avg_holdout_auc:.5f}")
    print(f"  Best solo:        {best_solo:.5f}  (Δ from avg: {best_solo - avg_holdout_auc:+.5f})")

    # Save
    out_dir = PROBS / "v22_tabpfn_epochs6_avg"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "holdout.npy", holdout_avg)
    np.save(out_dir / "test.npy", test_avg)

    test_ids = load_test()[ID].to_numpy()
    sub = pd.DataFrame({ID: test_ids, TARGET: test_avg})
    sub_path = SUBMISSIONS / "v22.400.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved avg submission to {sub_path}")

    # Step 4: math-driven blend analysis vs safar1_95449
    print("\n" + "=" * 70)
    print("Step 4 — Math-driven blend analysis vs safar1_95449")
    print("=" * 70)
    safar_path = Path("harvest/v21/safar1_lb-score-0-95449/submission.csv")
    if not safar_path.exists():
        print("  Try: kaggle kernels output safar1/lb-score-0-95449 -p harvest/v21/safar1_lb-score-0-95449")
        return
    safar = pd.read_csv(safar_path).set_index(ID).loc[test_ids][TARGET].to_numpy()
    rho = float(np.corrcoef(rankdata(test_avg), rankdata(safar))[0, 1])
    print(f"  ρ(v22.400_avg, safar1_95449) = {rho:.5f}")
    safar_lb = 0.95449
    proxy_lb = avg_holdout_auc
    print(f"  v22.400 holdout AUC: {proxy_lb:.5f}  (proxy for solo LB)")
    print(f"  safar1_95449 LB:     {safar_lb}")
    print()
    for w in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]:
        pred = predict_blend_lb(safar_lb, proxy_lb, rho, w)
        marker = "  ★" if pred > safar_lb else "   "
        print(f"    w_v22.400={w:.2f}: predicted LB={pred:.5f}{marker}")
    best_w = max(np.linspace(0, 1, 101),
                 key=lambda w: predict_blend_lb(safar_lb, proxy_lb, rho, w))
    best_pred = predict_blend_lb(safar_lb, proxy_lb, rho, best_w)
    print(f"\n  → MATH RECOMMENDS: w={best_w:.3f}, predicted LB={best_pred:.5f}")
    if best_pred > safar_lb + 0.00003:
        blend = (1 - best_w) * rank_normalize(safar) + best_w * rank_normalize(test_avg)
        mix_path = SUBMISSIONS / "v22.401.csv"
        pd.DataFrame({ID: test_ids, TARGET: blend}).to_csv(mix_path, index=False)
        print(f"  → saved math-optimal blend to {mix_path}")
    else:
        print(f"  → predicted lift < +0.00003 (noise floor); don't submit blend.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13],
                   help="Seeds to average (default: 42 7 13)")
    args = p.parse_args()
    main(args.seeds)
