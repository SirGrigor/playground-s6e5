"""v1 — LightGBM baseline on raw + Block 1 (target-free physical features).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: Block 1 captures pit-window phenomena per compound; LGB with
              default-ish params should land OOF AUC in [0.91, 0.93].
  Predicted Δ vs parent: 0 (no parent — this is the floor we measure against).
  Most relevant pitfall: #10 (versions trap) — keep this lean, one algo only.
  Validation plan: 5-fold StratifiedKFold(seed=42) on train_pool + sacred
                   holdout evaluation at end.
  Abort signal: if subsample-mode CV mean < 0.85, stop and investigate before
                spending compute on full run.
  Output: probs/v1_lgb/{oof,holdout,test}.npy + submissions/v1_lgb.csv

Usage:
  python notebooks/03_v1_lgb_baseline.py --subsample 10000     # sanity
  python notebooks/03_v1_lgb_baseline.py                       # full data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROBS, SUBMISSIONS, TARGET, ID, CV_SEED, MODEL_SEED
from src.data import load_train_pool, load_holdout, load_test
from src.features import build_block1
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, force_local: bool, gpu: bool):
    print("=" * 70)
    print("v1 — LightGBM baseline (raw + Block 1)")
    print("=" * 70)

    # ---- load via the sacred holdout protocol
    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}")

    if subsample is not None:
        rng = np.random.default_rng(MODEL_SEED)
        idx = rng.choice(len(pool), size=min(subsample, len(pool)), replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)
        print(f"  -- SANITY MODE -- subsampled pool to {len(pool):,} rows")

    # ---- features (Block 1 only — target-free)
    pool = build_block1(pool)
    holdout = build_block1(holdout)
    test = build_block1(test)

    # ---- column set
    categorical_cols = ["Driver", "Compound", "Race"]
    drop_cols = {ID, TARGET, "race_lap_mean", "race_lap_std"}  # intermediates we don't feed to model
    feature_cols = [c for c in pool.columns if c not in drop_cols]
    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"categorical_cols: {categorical_cols}")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    # ---- experiment registration (only commit on the FULL run, not sanity)
    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v1_lgb",
            parent=None,
            hypothesis=(
                "Plain LGB on raw + Block 1 (compound features + pit-window + "
                "tyre-life ratios + race-pace z-score). Expect OOF AUC in [0.91, 0.93]."
            ),
            predicted_delta=0.0,
            confidence="high",
            feature_changes=["+ block1 (compound, pit_window, tyre_life_ratio, race_pace_z)"],
            config_changes={"n_estimators": 5000, "early_stopping_rounds": 100, "learning_rate": 0.05},
            pipeline_changes=["+ sacred_holdout_v1", "+ stratified_kfold_5"],
            cloud_or_local="local",
        )

    # ---- train
    print("\n--- Training LGB ---")
    result = train_variant(
        algo="lgb",
        X_pool=pool,
        y_pool=y_pool,
        X_holdout=holdout,
        X_test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        use_gpu=gpu,
    )

    # ---- evaluate
    oof_auc_mean = float(np.mean(result["fold_aucs"]))
    holdout_auc = auc(y_holdout, result["holdout_pred"])
    print()
    print(f"OOF AUC (mean of folds): {oof_auc_mean:.5f}")
    print(f"OOF AUC (per fold):       {[f'{x:.5f}' for x in result['fold_aucs']]}")
    print(f"Holdout AUC:              {holdout_auc:.5f}")
    print(f"Gap (holdout - oof):      {holdout_auc - oof_auc_mean:+.5f}")
    print(f"Runtime:                  {result['runtime_sec']:.1f}s")

    # ---- abort if sanity fails
    if is_sanity:
        if oof_auc_mean < 0.85:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.85 threshold.")
            print("Investigate before running full data.")
            sys.exit(2)
        print("\n✓ Sanity pass — pipeline works end-to-end. Re-run without --subsample for full.")
        return

    # ---- save artifacts (full run only)
    out_dir = PROBS / "v1_lgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v1_lgb.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    # ---- record experiment
    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "categorical_cols": categorical_cols,
        },
    )
    exp.commit()
    print(f"\nExperiment v1_lgb committed to experiments.jsonl.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None, help="row subsample size (sanity mode)")
    p.add_argument("--force-local", action="store_true", help="acknowledge local-run regardless of size")
    p.add_argument("--gpu", action="store_true", help="use LGB GPU")
    args = p.parse_args()
    main(subsample=args.subsample, force_local=args.force_local, gpu=args.gpu)
