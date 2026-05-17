"""v20.002 — Yekenot FE + lag features with MEDIAN-FILL encoding.

Pre-experiment checklist:
  Hypothesis: v20.001 regressed -0.00075 vs v14 because lag/delta features used
              -1 sentinel for first-in-RDY rows (6.7% of data). RealMLP's
              continuous normalization treated -1 as a wildly out-of-range
              value, distorting PLR embeddings for the affected columns. Worse,
              `delta = current - (-1) = current + 1` produced fake values that
              looked real to the normalizer.
              Fix (in src/features.py::build_lag_features):
                - lag1 cols: fill NaN with median of valid values (looks normal)
                - delta1 cols: fill NaN with 0 (no-change default)
                - is_first_in_RDY indicator preserved → network can gate lag
                  features on this flag if needed.
              Local LGB 50K re-sanity with new encoding: +0.00054 OOF / +0.00016
              holdout (vs +0.00079/+0.00009 with the -1 sentinel). Slight LGB
              drop because LGB had implicitly exploited the -1 as a missing
              indicator; now is_first_in_RDY carries that info explicitly.
              Predicted Δ holdout vs v14 (0.95194): +0.0008 to +0.0020 (lower
              than v20.001's prediction because the regression evidence narrows
              what we can claim).
  Parent: v14_yekenot_repro (holdout 0.95194, LB 0.95125)
  Predicted Δ holdout: +0.0008 to +0.0020
  Confidence: medium — encoding fix is principled; LGB-vs-RealMLP gap may persist
  Risk: even with clean encoding, lag signal at RealMLP scale may be marginal;
        the real value is ρ(v20.002, v19.019) for blend diversity.
  Validation plan: 5-fold StratifiedKFold(42) + sacred 20% holdout
  Abort signal: holdout Δ < -0.001 (worse than v20.001 → drop direction)
  Output: probs/v20_002_lag_median/{oof,holdout,test}.npy + submissions/v20.002.csv

Usage:
  python notebooks/28_v20_002_lag_median.py --subsample 5000   # local CPU sanity (~2 min)
  python notebooks/28_v20_002_lag_median.py --gpu              # full Colab/Kaggle (~10 min)
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
    print("v20.002 — Yekenot FE + lag features (median-fill encoding)")
    print("=" * 70)

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}")

    print("\n--- Building lag features across pool ∪ holdout ∪ test ---")
    pool, holdout, test, lag_cols = build_lag_features(pool, holdout, test, lags=(1,))
    print(f"  added {len(lag_cols)} lag/delta columns")
    coverage = (pool["is_first_in_RDY"] == 0).mean()
    coverage_test = (test["is_first_in_RDY"] == 0).mean()
    print(f"  pool coverage (rows with valid lag): {coverage:.3f}")
    print(f"  test coverage (rows with valid lag): {coverage_test:.3f}")
    # Encoding spot-check: first-in-RDY row should have lag = median, delta = 0
    sample_row = pool[pool["is_first_in_RDY"] == 1].iloc[0]
    print(f"  first-in-RDY spot-check:")
    print(f"    LapTime (s)_lag1   = {sample_row['LapTime (s)_lag1']:.3f}  (should be near column median)")
    print(f"    LapTime (s)_delta1 = {sample_row['LapTime (s)_delta1']:.3f}  (should be 0)")

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
    numeric_feats = numeric_feats + lag_cols
    feature_cols = numeric_feats + cat_feats
    print(f"feature_cols ({len(feature_cols)}): {len(numeric_feats)} numeric + {len(cat_feats)} categorical")

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
            version="v20_002_lag_median",
            parent="v14_yekenot_repro",
            hypothesis=(
                "v20.001 regressed -0.00075 vs v14 because -1 sentinel for first-in-RDY "
                "rows (6.7%) poisoned RealMLP's continuous normalization; deltas became "
                "current + 1 (nonsense). Fix: median-fill for lag1 cols, 0-fill for delta1 "
                "cols, is_first_in_RDY indicator preserved. Local LGB 50K re-sanity: "
                "+0.00054 OOF / +0.00016 holdout with new encoding (vs +0.00079/+0.00009 "
                "with -1 sentinel — LGB lost implicit missing signal but is_first_in_RDY "
                "now carries it explicitly). Predicted Δ holdout vs v14 (0.95194): "
                "+0.0008 to +0.0020. Even if standalone is modest, the real value is "
                "ρ(v20.002, v19.019) for blend decorrelation."
            ),
            predicted_delta=0.0014,
            confidence="medium",
            feature_changes=[
                "+ 8 lag1 features (LAG_BASE_COLS × lag=1), median-fill for first-in-RDY",
                "+ 8 current-minus-lag1 delta features, 0-fill for first-in-RDY",
                "+ is_first_in_RDY indicator flag",
            ],
            config_changes={"n_ens": 24, "n_epochs": 6},
            pipeline_changes=[
                "+ build_lag_features encoding fix in src/features.py (NaN→median/0 instead of -1)",
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
    print(f"v20.001 holdout (broken): 0.95119")
    print(f"Δ vs v14:                 {holdout_auc - 0.95194:+.5f}")
    print(f"Δ vs v20.001:             {holdout_auc - 0.95119:+.5f}")
    print(f"Runtime:                  {result['runtime_sec']:.1f}s")

    # ρ vs v19.019 — the actual decision metric for blend candidacy.
    # v19.019 = the LB-plateau-tied submission; if rho < 0.99, blend could lift LB.
    try:
        v19_path = SUBMISSIONS / "v19.019.csv"
        if v19_path.exists():
            v19 = pd.read_csv(v19_path)
            v19_aligned = v19.set_index(ID).loc[test_fe[ID]][TARGET].to_numpy()
            rho = float(np.corrcoef(result["test_pred"], v19_aligned)[0, 1])
            print(f"\nρ(v20.002_test, v19.019_test) = {rho:.5f}")
            if rho < 0.95:
                print(f"  → ρ < 0.95: GENUINE diversity. Blend test is high priority.")
            elif rho < 0.99:
                print(f"  → ρ in [0.95, 0.99): mild diversity. Blend test worth trying.")
            else:
                print(f"  → ρ ≥ 0.99: no usable diversity. Standalone weakness dominates.")
    except Exception as e:
        print(f"\n(ρ vs v19.019 skipped: {e})")

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

    out_dir = PROBS / "v20_002_lag_median"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    sub = pd.DataFrame({ID: test_fe[ID].to_numpy(), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v20.002.csv"
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
                "model_family": "pytabkit.RealMLP_TD_Classifier",
                "lag_coverage_pool": float(coverage),
                "lag_coverage_test": float(coverage_test),
                "encoding": "median-fill lag1, 0-fill delta1, is_first_in_RDY indicator",
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
