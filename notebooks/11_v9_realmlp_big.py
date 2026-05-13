"""v9 — RealMLP_TD with n_ens=24 (Path A — bigger ensemble for variance reduction).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: v7 used n_ens=8 because of cost concerns. Yekenot's top-voted
              public S6E5 notebook uses n_ens=24. Going from 8→24 should reduce
              within-model variance by ~25-30% via 1/sqrt(N) scaling. Each
              ensemble member has independent initialization so the averaging
              extracts more stable predictions. Expected Δ holdout vs v7: +0.001.
  Parent: v7_realmlp (our current best solo, n_ens=8 holdout 0.94792)
  Predicted Δ holdout: +0.0008 to +0.0015
  Most relevant pitfall: #10 (one change), L17 (pytabkit baseline)
  Validation plan: 5-fold StratifiedKFold(42) + sacred holdout
  Abort signal: sanity OOF AUC drops vs v7 (would mean regression — unexpected)
  Output: probs/v9_realmlp_big/{oof,holdout,test}.npy + submissions/v9_realmlp_big.csv

Single change vs v7: n_ens 8 → 24. Same architecture, transforms, lr schedule.
Cost: ~7 min on T4 (v7 with n_ens=8 took 140s; n_ens=24 = ~3× = 420s).

Usage:
  python notebooks/11_v9_realmlp_big.py --subsample 10000  # local sanity
  python notebooks/11_v9_realmlp_big.py                    # full data on Colab
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
    print("v9 — RealMLP_TD with n_ens=24 (bigger ensemble)")
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

    print(f"feature_cols ({len(feature_cols)}): same as v7 (Block 1)")
    print(f"categorical_cols: {categorical_cols}")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    # Sanity: light config so it finishes fast
    # Full: n_ens=24 (yekenot's config)
    if subsample:
        params_override = {"n_ens": 4, "n_epochs": 2, "verbosity": 0}
    else:
        params_override = {"n_ens": 24}  # everything else from _realmlp_defaults()

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v9_realmlp_big",
            parent="v7_realmlp",
            hypothesis=(
                "Single change: n_ens 8 → 24. Same architecture, transforms, "
                "lr schedule. Bigger ensemble extracts more stable predictions via "
                "1/sqrt(N) variance reduction. Predicted Δ holdout vs v7: +0.001."
            ),
            predicted_delta=0.001,
            confidence="medium",
            feature_changes=[],
            config_changes={"n_ens": 24, "_prev_n_ens": 8},
            pipeline_changes=[],
            cloud_or_local="local" if subsample else "cloud",
        )

    print(f"\n--- Training RealMLP_TD (n_ens={params_override.get('n_ens', 24)}) ---")
    result = train_variant(
        algo="realmlp",
        X_pool=pool,
        y_pool=y_pool,
        X_holdout=holdout,
        X_test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        params=params_override,
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
        if oof_auc_mean < 0.80:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.80 (pipeline broken).")
            sys.exit(2)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse detected (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly.")
        print(f"  Sanity used n_ens=4. Full Colab run uses 24.")
        return

    out_dir = PROBS / "v9_realmlp_big"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v9_realmlp_big.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features": len(feature_cols),
            "n_ens": 24,
            "model_family": "pytabkit.RealMLP_TD_Classifier",
        },
    )
    exp.commit()
    print(f"\nExperiment v9_realmlp_big committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
