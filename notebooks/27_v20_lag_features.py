"""v20 — Yekenot FE + lag features computed across pool ∪ holdout ∪ test.

Pre-experiment checklist:
  Hypothesis: The dataset is a lap-interleaved random split of full F1 stints
              (gap=1 dominates in train+test combined). Within (Race,Driver,Year),
              ~60% of test rows have their previous lap somewhere in the union.
              v14b/yekenot/kospintr/all top public solutions treat each row as
              independent — they do NOT consume previous-lap state. Adding lag1
              features (LapTime, LapTime_Delta, Cumulative_Degradation, Position,
              Position_Change, TyreLife, PitStop, Stint) plus current-minus-lag1
              deltas should be an orthogonal signal to the public ecosystem.
  Parent: v14_yekenot_repro (holdout 0.95194, LB 0.95125)
  Predicted Δ holdout: +0.0015 to +0.0030
  Confidence: medium — informed by 2026-05-17 data audit (combined gap=1 = 260K rows)
  Risk: if synthetic-data generator broke temporal coherence at row level
        (suspicious Stint non-monotonicity for ALB on Canadian GP 2022), lag
        features may be noise. Sanity check on 5K subsample first.
  Validation plan: 5-fold StratifiedKFold(42) + sacred 20% holdout
  Abort signal: sanity OOF AUC < 0.85, fold collapse > 0.05, or holdout Δ < -0.002
  Output: probs/v20_lag_features/{oof,holdout,test}.npy + submissions/v20.001.csv

Usage:
  python notebooks/27_v20_lag_features.py --subsample 5000   # local CPU sanity (~2 min)
  python notebooks/27_v20_lag_features.py --gpu              # full Colab/Kaggle (~12 min)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROBS, SUBMISSIONS, TARGET, ID, MODEL_SEED
from src.data import load_train_pool, load_holdout, load_test
from src.features import (
    yekenot_fe_fit,
    yekenot_fe_transform,
    yekenot_feature_lists,
    build_lag_features,
)
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v20 — Yekenot FE + lag features (computed across pool ∪ holdout ∪ test)")
    print("=" * 70)

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}")

    # Lag features FIRST — computed on the full union so each row sees its
    # actual previous lap. Target-free (uses only physical race-state columns).
    print("\n--- Building lag features across pool ∪ holdout ∪ test ---")
    pool, holdout, test, lag_cols = build_lag_features(pool, holdout, test, lags=(1,))
    print(f"  added {len(lag_cols)} lag/delta columns")
    coverage = (pool["is_first_in_RDY"] == 0).mean()
    print(f"  pool coverage (rows with valid lag): {coverage:.3f}")
    coverage_test = (test["is_first_in_RDY"] == 0).mean()
    print(f"  test coverage (rows with valid lag): {coverage_test:.3f}")

    if subsample is not None:
        rng = np.random.default_rng(MODEL_SEED)
        idx = rng.choice(len(pool), size=min(subsample, len(pool)), replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)
        print(f"  -- SANITY MODE -- subsampled pool to {len(pool):,} rows")

    print("\n--- Yekenot FE ---")
    pool_fe, fe_state = yekenot_fe_fit(pool)
    holdout_fe = yekenot_fe_transform(holdout, fe_state)
    test_fe = yekenot_fe_transform(test, fe_state)

    numeric_feats, cat_feats = yekenot_feature_lists(fe_state)
    # Append lag features as numeric (RealMLP normalizes; tree models split natively).
    numeric_feats = numeric_feats + lag_cols
    feature_cols = numeric_feats + cat_feats
    print(f"feature_cols ({len(feature_cols)}): {len(numeric_feats)} numeric + {len(cat_feats)} categorical")
    print(f"  +lag/delta numeric: {lag_cols}")

    te_pairs = [("Race", "Compound"), ("Race", "Year")]
    print(f"\nte_pairs (fold-safe TE applied by train_variant): {te_pairs}")

    y_pool = pool_fe[TARGET].astype(int).to_numpy()
    y_holdout = holdout_fe[TARGET].astype(int).to_numpy()

    if subsample:
        params_override = {"n_ens": 4, "n_epochs": 2, "verbosity": 0}
    else:
        params_override = {"n_ens": 24, "n_epochs": 6}

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v20_lag_features",
            parent="v14_yekenot_repro",
            hypothesis=(
                "Dataset is lap-interleaved random split (gap=1 dominates in train+test "
                "combined, 260K rows). Top public solutions and our v14b treat each row "
                "as independent. Adding past-lag features (LapTime, LapTime_Delta, "
                "Cumulative_Degradation, Position, Position_Change, TyreLife, PitStop, "
                "Stint) computed across pool ∪ holdout ∪ test plus current-minus-lag1 "
                "deltas should be orthogonal signal to the converged public ecosystem "
                "(ρ ≈ 1.0 at top tier). Predicted Δ holdout vs v14 (0.95194): +0.0015 to +0.0030."
            ),
            predicted_delta=0.0022,
            confidence="medium",
            feature_changes=[
                "+ 8 lag1 features (LAG_BASE_COLS × lag=1)",
                "+ 8 current-minus-lag1 delta features",
                "+ is_first_in_RDY flag",
            ],
            config_changes={"n_ens": 24, "n_epochs": 6},
            pipeline_changes=[
                "+ build_lag_features in src/features.py (operates on pool ∪ holdout ∪ test)",
            ],
            cloud_or_local="local" if not gpu else "cloud",
        )

    print(f"\n--- Training RealMLP (n_ens={params_override.get('n_ens', 24)}, n_epochs={params_override.get('n_epochs', 6)}) ---")
    result = train_variant(
        algo="realmlp",
        X_pool=pool_fe,
        y_pool=y_pool,
        X_holdout=holdout_fe,
        X_test=test_fe,
        feature_cols=feature_cols,
        categorical_cols=cat_feats,
        params=params_override,
        use_gpu=gpu,
        te_pairs=te_pairs,
    )

    oof_auc_mean = float(np.mean(result["fold_aucs"]))
    holdout_auc = auc(y_holdout, result["holdout_pred"])
    print()
    print(f"OOF AUC (mean of folds): {oof_auc_mean:.5f}")
    print(f"OOF AUC (per fold):       {[f'{x:.5f}' for x in result['fold_aucs']]}")
    print(f"Holdout AUC:              {holdout_auc:.5f}")
    print(f"v14 holdout (parent):     0.95194")
    print(f"Δ vs v14:                 {holdout_auc - 0.95194:+.5f}")
    print(f"Runtime:                  {result['runtime_sec']:.1f}s")

    if is_sanity:
        if oof_auc_mean < 0.85:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.85 (pipeline broken).")
            sys.exit(2)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly.")
        return

    out_dir = PROBS / "v20_lag_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    sub = pd.DataFrame({ID: test_fe[ID].to_numpy(), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v20.001.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    if exp is not None:
        exp.record(
            oof_auc_mean=oof_auc_mean,
            oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
            holdout_auc=float(holdout_auc),
            runtime_sec=float(result["runtime_sec"]),
            extra={
                "n_features": len(feature_cols),
                "n_lag_features": len(lag_cols),
                "categorical_cols": cat_feats[:3],
                "model_family": "pytabkit.RealMLP_TD_Classifier",
                "lag_coverage_pool": float(coverage),
                "lag_coverage_test": float(coverage_test),
            },
        )
        exp.commit()
        print("\nExperiment recorded to experiments.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(args.subsample, args.gpu)
