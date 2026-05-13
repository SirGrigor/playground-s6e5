"""v17 — Final blend candidate selection (CPU only, runs on Colab OR local).

Runs AFTER v14b/v15/v16 have produced probs. Compares all available
candidates by holdout AUC and produces blend variants. Outputs a leaderboard
of candidates with predicted LB → user picks the single best to submit.

Designed to be infra-agnostic:
  - On Colab: probs/ live in /content from the same session; auto-fetches
    yekenot's submission via kaggle CLI if not already present.
  - On local: same, plus picks up v14 (only on local, never run on Colab).

Logic:
  1. Load all available OOFs + holdouts + tests from probs/
  2. Load yekenot test predictions (no aligned OOF available)
  3. Compute pairwise ρ on test predictions (diversity audit)
  4. Three blend strategies:
       a. Nelder-Mead on holdout AUC across our internal models
       b. 50/50 rank-blend of (best internal) + yekenot
       c. 30/70 rank-blend of (best internal) + yekenot (more weight to yekenot)
  5. Report holdout AUC for each candidate
  6. Save final candidate submissions to submissions/v17_*.csv

Usage:
  python notebooks/21_v17_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.config import PROBS, SUBMISSIONS, TARGET, ID, ROOT
from src.data import load_train_pool, load_holdout, load_test


def load_probs(version: str) -> dict | None:
    """Load probs/{version}/{oof,holdout,test}.npy if all present."""
    d = PROBS / version
    if not all((d / f"{x}.npy").exists() for x in ("oof", "holdout", "test")):
        return None
    return {
        "oof": np.load(d / "oof.npy"),
        "holdout": np.load(d / "holdout.npy"),
        "test": np.load(d / "test.npy"),
    }


def rank_norm(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def nelder_mead_weights(oof_matrix: np.ndarray, y_pool: np.ndarray) -> np.ndarray:
    """Optimize weights to maximize AUC(rank_avg(weighted_oof), y_pool)."""
    n_models = oof_matrix.shape[1]
    ranks = np.column_stack([rank_norm(oof_matrix[:, i]) for i in range(n_models)])

    def neg_auc(w):
        w = np.abs(w)
        w = w / w.sum()
        return -roc_auc_score(y_pool, ranks @ w)

    best_auc = -np.inf
    best_w = None
    for _ in range(5):
        x0 = np.random.default_rng().dirichlet(np.ones(n_models))
        res = minimize(neg_auc, x0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6})
        if -res.fun > best_auc:
            best_auc = -res.fun
            best_w = np.abs(res.x) / np.abs(res.x).sum()
    return best_w, best_auc


def main() -> int:
    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()
    print(f"pool: {len(pool):,} rows   holdout: {len(holdout):,} rows   test: {len(test):,} rows\n")

    # Candidates to consider
    candidates = ["v9_realmlp_big", "v10_tabm", "v11_blend_4way", "v14_yekenot_repro",
                  "v14b_external", "v15_stats", "v16_factory_yekenot"]
    available = {}
    print("=== Loading internal candidates ===")
    for v in candidates:
        p = load_probs(v)
        if p:
            available[v] = p
            ho_auc = roc_auc_score(y_holdout, p["holdout"])
            print(f"  ✓ {v:<28}  holdout AUC = {ho_auc:.5f}")
        else:
            print(f"  ✗ {v:<28}  (probs not found, skipped)")
    if not available:
        print("\nNo internal candidates found. Run v14/v14b/v15/v16 first.")
        return 1
    print()

    # Yekenot (test-only — no aligned OOF for our pool)
    # Auto-fetch via kaggle CLI if not present (Colab-friendly).
    yek_sub_path = ROOT / "harvest/v13/yekenot_ensemble/submission.csv"
    yekenot_test = None
    if not yek_sub_path.exists():
        print(f"yekenot submission not found at {yek_sub_path} — attempting kaggle CLI fetch...")
        import subprocess
        yek_sub_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["kaggle", "kernels", "output", "yekenot/ps-s6-e5-realmlp-pytabkit",
                 "-p", str(yek_sub_path.parent)],
                check=True, capture_output=True, text=True, timeout=120,
            )
            print(f"  fetched to {yek_sub_path.parent}")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"  ✗ fetch failed ({type(exc).__name__}) — proceeding without yekenot mix")

    if yek_sub_path.exists():
        yekenot_test = pd.read_csv(yek_sub_path).set_index("id").loc[test[ID]]["PitNextLap"].to_numpy()
        print(f"yekenot test predictions loaded ({len(yekenot_test):,} rows)\n")

    # Individual candidate scores
    print("=== Individual candidate holdout AUC ===")
    scores: dict[str, float] = {}
    for v, p in available.items():
        scores[v] = roc_auc_score(y_holdout, p["holdout"])
    for v, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        marker = " ← best internal" if v == max(scores, key=scores.get) else ""
        print(f"  {v:<28}  {s:.5f}{marker}")
    best_internal = max(scores, key=scores.get)
    print(f"\nBest internal: {best_internal} ({scores[best_internal]:.5f} holdout)")
    print()

    # Nelder-Mead blend on internal OOFs
    if len(available) >= 2:
        print("=== Nelder-Mead blend (optimized on pool OOF AUC) ===")
        names = list(available.keys())
        oof_matrix = np.column_stack([available[n]["oof"] for n in names])
        holdout_matrix = np.column_stack([available[n]["holdout"] for n in names])
        test_matrix = np.column_stack([available[n]["test"] for n in names])

        weights, oof_blend_auc = nelder_mead_weights(oof_matrix, y_pool)
        print(f"Weights:")
        for n, w in sorted(zip(names, weights), key=lambda kv: kv[1], reverse=True):
            if w > 0.001:
                print(f"  {n:<28}  {w:.4f}")
        print(f"OOF blend AUC:     {oof_blend_auc:.5f}")

        # Apply same weights to holdout (rank-normalize each column first)
        holdout_ranks = np.column_stack([rank_norm(holdout_matrix[:, i]) for i in range(len(names))])
        blend_holdout = holdout_ranks @ weights
        scores["v17_NM_blend"] = roc_auc_score(y_holdout, blend_holdout)
        print(f"Holdout blend AUC: {scores['v17_NM_blend']:.5f}")
        print()

        # Save blended test predictions
        test_ranks = np.column_stack([rank_norm(test_matrix[:, i]) for i in range(len(names))])
        blend_test = test_ranks @ weights
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: test[ID].astype("int64"), TARGET: blend_test}).to_csv(
            SUBMISSIONS / "v17_NM_blend.csv", index=False
        )

    # Yekenot mixes (test-only — holdout AUC unmeasurable, just for diversity)
    if yekenot_test is not None and best_internal in available:
        print("=== Yekenot mix candidates (holdout AUC estimated from best-internal alone) ===")
        best_p = available[best_internal]
        best_test_rank = rank_norm(best_p["test"])
        yek_test_rank = rank_norm(yekenot_test)

        for alpha, name in [(0.5, "v17_5050_internal_yekenot"), (0.3, "v17_3070_internal_yekenot"),
                            (0.7, "v17_7030_internal_yekenot")]:
            blend = (1 - alpha) * best_test_rank + alpha * yek_test_rank
            pd.DataFrame({ID: test[ID].astype("int64"), TARGET: blend}).to_csv(
                SUBMISSIONS / f"{name}.csv", index=False
            )
            print(f"  → submissions/{name}.csv  (alpha={alpha} to yekenot)")
        print()

    # Final leaderboard
    print("=== FINAL CANDIDATE LEADERBOARD (by holdout AUC, measurable rows only) ===")
    for v, s in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {v:<28}  {s:.5f}")
    print()
    print("Yekenot mixes cannot be holdout-scored (yekenot has no holdout predictions).")
    print("Expected LB for yekenot mixes: between best_internal LB and yekenot LB (0.95356).")
    print()

    best_holdout_candidate = max(scores, key=scores.get)
    print(f"=== RECOMMENDED SUBMISSION ===")
    print(f"By holdout: {best_holdout_candidate} ({scores[best_holdout_candidate]:.5f})")
    print(f"Expected LB: ~{scores[best_holdout_candidate] - 0.0007:.5f} (holdout-LB gap ~0.0007 from v14)")
    print(f"\nAlternative (gambling on diversity): v17_5050_internal_yekenot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
