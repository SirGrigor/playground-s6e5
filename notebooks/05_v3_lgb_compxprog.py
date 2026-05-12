"""v3 — LGB + Block 2: Compound x RaceProgress_bin (single interaction).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: The strongest signal interaction per EDA is Compound × RaceProgress
              (different compounds have completely different pit windows: SOFT
              0.05-0.225, HARD 0.50-0.70, MEDIUM 0.55-0.725). A single new
              categorical feature combining compound + progress_bin(40) lets LGB
              partition the joint pit-window space directly via native cat splits.
  Parent: v1_lgb (OOF 0.94236, holdout 0.94379, LB 0.94285)
  Predicted Δ holdout: +0.0015 (mid of [0.001, 0.002] — single feature; full Block 2 of 4 features could be +0.002-0.003)
  Most relevant pitfall: #10 (avoid version explosion — keep ONE change per version)
  Validation plan: same 5-fold StratifiedKFold(seed=42) + sacred holdout
  Abort signal: subsample-mode pipeline integrity check (no fold collapse, AUC > 0.85)
  Output: probs/v3_lgb_compxprog/{oof,holdout,test}.npy + submissions/v3_lgb_compxprog.csv

Single change from v1: ADD 1 categorical feature
  - compound_x_progress_bin: e.g. "HARD_b25" (compound + 40-bin RaceProgress index)

NOTE: This is fundamentally different from v2's TargetEncoder approach.
- v2 TE: collapsed each (Driver,Compound) to a single numeric pit-rate.
- v3 native cat: tells LGB "here are 200 new categories, find best splits".
LGB has MORE freedom in v3, less freedom in v2. Tests whether the lever is
mechanism (TE vs native cat) or content (these specific categoricals).

Usage:
  python notebooks/05_v3_lgb_compxprog.py --subsample 10000
  python notebooks/05_v3_lgb_compxprog.py
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
from src.features import build_block1, add_compound_x_progress_bin
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v3 — LGB + Block 2 (Compound × RaceProgress_bin native cat)")
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

    # Block 1 (same as v1) + Block 2 single interaction
    pool = build_block1(pool)
    holdout = build_block1(holdout)
    test = build_block1(test)

    pool = add_compound_x_progress_bin(pool)
    holdout = add_compound_x_progress_bin(holdout)
    test = add_compound_x_progress_bin(test)

    # Original categoricals + the new interaction
    categorical_cols = ["Driver", "Compound", "Race", "compound_x_progress_bin"]
    drop_cols = {ID, TARGET, "race_lap_mean", "race_lap_std"}
    feature_cols = [c for c in pool.columns if c not in drop_cols]

    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")
    print(f"categorical_cols (includes new interaction): {categorical_cols}")
    print(f"compound_x_progress_bin unique values: {pool['compound_x_progress_bin'].nunique()}")
    print(f"  (expected up to 200; depends on which combos actually occur in train)")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v3_lgb_compxprog",
            parent="v1_lgb",   # NOT v2 — we threw v2 away
            hypothesis=(
                "Compound × RaceProgress is the strongest EDA-confirmed interaction "
                "(different pit windows per compound). Adding it as a single native "
                "categorical (5 × 40 = up to 200 combos) lets LGB partition the joint "
                "pit-window space directly. Predicted Δ holdout vs v1: +0.0015."
            ),
            predicted_delta=0.0015,
            confidence="medium",
            feature_changes=["+ compound_x_progress_bin (native cat)"],
            config_changes={},
            pipeline_changes=[],
            cloud_or_local="local" if subsample else "cloud",
        )

    print("\n--- Training LGB + Block 2 (compound_x_progress_bin) ---")
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
        print(f"  Per-fold spread is tight, no collapse. Full Colab run will tell us")
        print(f"  if compound_x_progress_bin delivers the predicted +0.0015 on 350K data.")
        return

    out_dir = PROBS / "v3_lgb_compxprog"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v3_lgb_compxprog.csv"
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
            "new_feature": "compound_x_progress_bin",
            "new_feature_cardinality": int(pool["compound_x_progress_bin"].nunique()),
        },
    )
    exp.commit()
    print(f"\nExperiment v3_lgb_compxprog committed to experiments.jsonl.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
