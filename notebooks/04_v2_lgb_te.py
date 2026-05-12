"""v2 — LGB + Block 3 (sklearn TargetEncoder, fold-safe).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: Driver is weak alone (MI=0.009) but (Driver, Compound) and
              (Race, Compound) encode strategic personality / race-pace.
              sklearn TargetEncoder(cv=5) does fold-safe encoding — NEVER manual TE
              per S6E4 Phase 12 fold-collapse lesson.
  Parent: v1_lgb (OOF 0.94236, holdout 0.94379, LB 0.94285)
  Predicted Δ holdout: +0.004 (mid of [0.003, 0.006] from audit roadmap)
  Most relevant pitfall: #5 — manual TE causes fold collapse. Use sklearn only.
  Validation plan: same 5-fold StratifiedKFold(seed=42) + sacred holdout
  Abort signal: subsample-mode OOF < 0.93 (would indicate TE broke the pipeline)
  Output: probs/v2_lgb_te/{oof,holdout,test}.npy + submissions/v2_lgb_te.csv

Single change from v1: ADD 4 TargetEncoder columns
  - TE_Driver
  - TE_Race
  - TE_Driver_x_Compound (pair)
  - TE_Race_x_Compound (pair)

Usage:
  python notebooks/04_v2_lgb_te.py --subsample 10000
  python notebooks/04_v2_lgb_te.py
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
    print("v2 — LGB + Block 3 (fold-safe sklearn TargetEncoder)")
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

    categorical_cols = ["Driver", "Compound", "Race"]
    drop_cols = {ID, TARGET, "race_lap_mean", "race_lap_std"}
    feature_cols = [c for c in pool.columns if c not in drop_cols]

    te_cols = ["Driver", "Race"]
    te_pairs = [("Driver", "Compound"), ("Race", "Compound")]

    print(f"feature_cols (block1 + raw, {len(feature_cols)}): {feature_cols}")
    print(f"categorical_cols: {categorical_cols}")
    print(f"target-encoded cols: {te_cols}")
    print(f"target-encoded pairs: {te_pairs}")
    print(f"NEW features will be: TE_Driver, TE_Race, TE_Driver_x_Compound, TE_Race_x_Compound (+4 cols)")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v2_lgb_te",
            parent="v1_lgb",
            hypothesis=(
                "Driver weak alone (MI=0.009) but (Driver,Compound) and (Race,Compound) "
                "encode strategic personality. sklearn TargetEncoder(cv=5) adds 4 fold-safe "
                "numeric features. Predicted Δ holdout vs v1: +0.004."
            ),
            predicted_delta=0.004,
            confidence="medium",
            feature_changes=["+ TE_Driver, TE_Race, TE_Driver_x_Compound, TE_Race_x_Compound"],
            config_changes={},
            pipeline_changes=["+ sklearn_TargetEncoder(cv=5)"],
            cloud_or_local="local" if subsample else "cloud",
        )

    print("\n--- Training LGB + TE ---")
    result = train_variant(
        algo="lgb",
        X_pool=pool,
        y_pool=y_pool,
        X_holdout=holdout,
        X_test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        te_cols=te_cols,
        te_pairs=te_pairs,
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
        # NOTE: TE is sample-size sensitive. On 10K rows each (Driver,Compound) combo
        # has ~2 observations → smoothing pushes values to global mean, adding noise
        # without signal. On full 350K data each combo gets ~80 observations → signal
        # emerges. Therefore the sanity threshold is set to "pipeline runs cleanly"
        # (>= 0.85), NOT "TE delivers the predicted +0.004" (which only happens at full scale).
        if oof_auc_mean < 0.85:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.85 threshold.")
            print("Pipeline likely broken (not a TE-vs-no-TE comparison issue).")
            sys.exit(2)
        # Check pipeline integrity: per-fold AUC sane (no collapse)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse detected (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly.")
        print(f"  Note: TE may underperform v1 at 10K scale due to ~2 obs/category. ")
        print(f"  Full-data run on Colab will tell us if TE actually delivers the predicted +0.004.")
        return

    out_dir = PROBS / "v2_lgb_te"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v2_lgb_te.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features_base": len(feature_cols),
            "n_features_added": 4,
            "te_cols": te_cols,
            "te_pairs": [list(p) for p in te_pairs],
        },
    )
    exp.commit()
    print(f"\nExperiment v2_lgb_te committed to experiments.jsonl.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
