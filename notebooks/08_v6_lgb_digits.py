"""v6 — LGB + Block 4 (digit-extraction features, yunsuxiaozi pattern).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: Synthetic data often has decimal-quantization patterns that
              LGB cannot discover via threshold splits (e.g. "value ending
              in digit 7"). yunsuxiaozi's pattern adds digit-at-position(k)
              features as int8 to expose those patterns.
              If S6E5 synthetic generator quantizes values to specific
              precisions, these features capture orthogonal information.
              Uses np.round-after-scale (S6E4 L10 lesson) for float safety.
  Parent: v1_lgb (best LGB-only — our solid baseline)
  Predicted Δ holdout: +0.001 to +0.002 (single-model, ORTHOGONAL info vs
                       v2/v3 which were redundant cat encodings)
  Most relevant pitfall: #10 (one change per version), L10 (digit float bug)
  Validation plan: 5-fold StratifiedKFold(42) + sacred holdout
  Abort signal: subsample-mode pipeline integrity (AUC > 0.85, no fold collapse)
  Output: probs/v6_lgb_digits/{oof,holdout,test}.npy + submissions/v6_lgb_digits.csv

Single change vs v1: ADD 21 digit-extraction int8 columns from:
  TyreLife, LapNumber, LapTime (s), LapTime_Delta, Cumulative_Degradation,
  RaceProgress, Position_Change — at meaningful k positions per feature range.

IMPORTANT — different from v2/v3:
  v2 (TE) and v3 (cat interactions) FAILED because the new features were
  redundant with raw cats LGB already partitions. v6 features are orthogonal
  modular-arithmetic info LGB literally cannot build via splits — so
  the redundancy issue does NOT apply here.

Usage:
  python notebooks/08_v6_lgb_digits.py --subsample 10000
  python notebooks/08_v6_lgb_digits.py
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
from src.features import build_block1, add_digit_features, DIGIT_POSITIONS
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v6 — LGB + Block 4 (digit-extraction features)")
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

    # Block 1 (same as v1) + Block 4 digit features
    pool = build_block1(pool)
    holdout = build_block1(holdout)
    test = build_block1(test)

    pool = add_digit_features(pool)
    holdout = add_digit_features(holdout)
    test = add_digit_features(test)

    n_digit_features = sum(len(ks) for ks in DIGIT_POSITIONS.values())
    print(f"Block 4: added {n_digit_features} digit features across "
          f"{len(DIGIT_POSITIONS)} numeric source columns")

    categorical_cols = ["Driver", "Compound", "Race"]
    drop_cols = {ID, TARGET, "race_lap_mean", "race_lap_std"}
    feature_cols = [c for c in pool.columns if c not in drop_cols]

    digit_cols = [c for c in feature_cols if "_digit" in c]
    print(f"feature_cols total: {len(feature_cols)}  ({len(digit_cols)} digit features)")
    print(f"sample digit cols: {digit_cols[:5]} ...")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v6_lgb_digits",
            parent="v1_lgb",
            hypothesis=(
                f"Block 4: {n_digit_features} digit-extraction int8 features "
                f"(yunsuxiaozi pattern) across 7 numeric source columns. These expose "
                f"decimal-quantization patterns LGB cannot build via threshold splits "
                f"(modular arithmetic). Orthogonal info — different from v2/v3 which "
                f"were redundant cat encodings. Predicted Δ holdout vs v1: +0.0015."
            ),
            predicted_delta=0.0015,
            confidence="medium",
            feature_changes=[f"+ {n_digit_features} digit features (Block 4 yunsuxiaozi)"],
            config_changes={},
            pipeline_changes=[],
            cloud_or_local="local" if subsample else "cloud",
        )

    print("\n--- Training LGB + Block 4 (digit features) ---")
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
        print(f"  21 digit features added; LGB will tell us if any encode useful patterns.")
        return

    out_dir = PROBS / "v6_lgb_digits"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v6_lgb_digits.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features_total": len(feature_cols),
            "n_digit_features": n_digit_features,
            "digit_positions": DIGIT_POSITIONS,
        },
    )
    exp.commit()
    print(f"\nExperiment v6_lgb_digits committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
