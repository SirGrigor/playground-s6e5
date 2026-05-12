"""v4 — XGBoost on same Block 1 features (algo-family diversity test).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: v2 + v3 confirmed LGB plateaued on Block 1 (any added feature
              crowds productive splits and regresses). Time to change MODEL,
              not features. XGBoost uses level-wise growth + ordinal categorical
              encoding (vs LGB's leaf-wise + Fisher splits). Different decision
              boundaries → genuine diversity, sets up v6 blend.
  Parent: v1_lgb (same Block 1, different model family)
  Predicted Δ holdout: +0.0010 single-model
                        (XGB tends to slightly outperform LGB on tabular at
                         default settings; even similar = useful for blending)
  Most relevant pitfall: #9 (algorithm-family IS valid diversity when models
                              see same features but use different splitting)
  Validation plan: same 5-fold StratifiedKFold(seed=42) + sacred holdout
  Abort signal: subsample-mode pipeline integrity (AUC > 0.85, no fold collapse)
  Output: probs/v4_xgb/{oof,holdout,test}.npy + submissions/v4_xgb.csv

Even if XGB matches or slightly underperforms LGB solo (e.g., 0.942 vs 0.943),
the OOF correlation ρ < 0.99 is what we're after — sets up v6 multi-model blend
where decorrelated errors compound to +0.002-0.004.

Usage:
  python notebooks/06_v4_xgb_block1.py --subsample 10000
  python notebooks/06_v4_xgb_block1.py
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
from src.features import build_block1
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v4 — XGBoost on Block 1 (algo-family diversity test)")
    print("=" * 70)

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}")

    if subsample is not None:
        rng = np.random.default_rng(MODEL_SEED)
        idx = rng.choice(len(pool), size=min(subsample, len(pool)), replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)
        print(f"  -- SANITY MODE -- subsampled pool to {len(pool):,} rows")

    pool = build_block1(pool)
    holdout = build_block1(holdout)
    test = build_block1(test)

    # Same feature space as v1 — that's the experiment (different model, same features)
    categorical_cols = ["Driver", "Compound", "Race"]
    drop_cols = {ID, TARGET, "race_lap_mean", "race_lap_std"}
    feature_cols = [c for c in pool.columns if c not in drop_cols]

    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"categorical_cols (XGB native via enable_categorical=True): {categorical_cols}")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v4_xgb",
            parent="v1_lgb",  # comparing to LGB on same features — clean algo-family delta
            hypothesis=(
                "v2 and v3 proved LGB on Block 1 is plateaued. Switching to XGBoost "
                "(level-wise growth + ordinal cat encoding vs LGB leaf-wise + Fisher). "
                "Same features, different model family. Predicted Δ holdout vs v1 LGB: "
                "+0.0010 single-model; sets up v6 blend where decorrelated errors compound."
            ),
            predicted_delta=0.0010,
            confidence="medium",
            feature_changes=[],
            config_changes={"algo": "xgb", "n_estimators": 5000, "max_depth": 6, "learning_rate": 0.05},
            pipeline_changes=["+ xgboost (replaces lightgbm as base model)"],
            cloud_or_local="local" if subsample else "cloud",
        )

    print("\n--- Training XGBoost ---")
    result = train_variant(
        algo="xgb",
        X_pool=pool,
        y_pool=y_pool,
        X_holdout=holdout,
        X_test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        use_gpu=gpu,
    )

    oof_auc_mean = float(np.mean(result["fold_aucs"]))
    holdout_auc = auc(y_holdout, result["holdout_pred"])
    print()
    print(f"OOF AUC (mean of folds): {oof_auc_mean:.5f}")
    print(f"OOF AUC (per fold):       {[f'{x:.5f}' for x in result['fold_aucs']]}")
    print(f"Holdout AUC:              {holdout_auc:.5f}")
    print(f"Gap (holdout - oof):      {holdout_auc - oof_auc_mean:+.5f}")
    print(f"Runtime:                  {result['runtime_sec']:.1f}s")

    if is_sanity:
        if oof_auc_mean < 0.85:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.85 (pipeline broken).")
            sys.exit(2)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse detected (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly.")
        return

    out_dir = PROBS / "v4_xgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v4_xgb.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features": len(feature_cols),
            "categorical_cols": categorical_cols,
            "model_family": "xgboost",
        },
    )
    exp.commit()
    print(f"\nExperiment v4_xgb committed to experiments.jsonl.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
