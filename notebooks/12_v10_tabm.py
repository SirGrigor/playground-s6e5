"""v10 — TabM_D (pytabkit) on Block 1 features (Path D — architectural diversity).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: v9 RealMLP n_ens=24 is the strongest single model we've trained
              (LB 0.94757). The blend with trees plateaued at the noise floor
              (v8 only +0.00007 over v7) because ρ(NN, tree) = 0.97.
              TabM_D is a DIFFERENT neural architecture in pytabkit (different
              paper, different network design — token-based / mixer-style).
              Its predictions should have lower correlation with RealMLP than
              RealMLP-variants of itself. Expected solo Δ vs v9: 0 to +0.001
              (TabM_D competitive but not necessarily better solo).
              REAL VALUE: blend(v9 RealMLP, v10 TabM_D, ± trees) should lift
              meaningfully over v9 solo because ρ(RealMLP, TabM_D) should
              be 0.90-0.95 (vs RealMLP self-correlation ~0.99).
  Parent: v9_realmlp_big
  Predicted Δ holdout vs v9: -0.001 to +0.001 (solo); blend payoff in v11.
  Most relevant pitfall: #10 (one change), L17 (pytabkit baseline)
  Validation plan: 5-fold StratifiedKFold(42) + sacred holdout
  Abort signal: subsample-mode pipeline integrity (AUC > 0.80, no fold collapse)
  Output: probs/v10_tabm/{oof,holdout,test}.npy + submissions/v10_tabm.csv

Single change vs v9: pytabkit RealMLP_TD → TabM_D.
Same features, same fold structure, different model class within pytabkit.

Usage:
  python notebooks/12_v10_tabm.py --subsample 10000
  python notebooks/12_v10_tabm.py
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
    print("v10 — TabM_D on Block 1 (architectural diversity within neural family)")
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

    print(f"feature_cols ({len(feature_cols)}): same as v7/v9 (Block 1)")
    print(f"categorical_cols: {categorical_cols}")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    # For sanity: lighter config to verify pipeline runs fast
    sanity_params = {"n_epochs": 1, "verbosity": 0} if subsample else {}

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v10_tabm",
            parent="v9_realmlp_big",
            hypothesis=(
                "TabM_D (pytabkit) — different neural architecture than RealMLP. "
                "Same Block 1 features, different network design (token-based / mixer-style). "
                "Predicted Δ vs v9 solo: -0.001 to +0.001 (architectural neutrality). "
                "Real value: ρ(RealMLP, TabM_D) should be 0.90-0.95 → blend payoff in v11."
            ),
            predicted_delta=0.0,  # neutral solo prediction
            confidence="low",
            feature_changes=[],
            config_changes={"algo": "tabm", "tabm_k": 32, "n_epochs": 3, "d_block": 256, "n_blocks": 3},
            pipeline_changes=["+ pytabkit.TabM_D_Classifier (different neural architecture)"],
            cloud_or_local="local" if subsample else "cloud",
        )

    print("\n--- Training TabM_D ---")
    result = train_variant(
        algo="tabm",
        X_pool=pool,
        y_pool=y_pool,
        X_holdout=holdout,
        X_test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        params=sanity_params,
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
        if oof_auc_mean < 0.75:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.75 (pipeline broken).")
            sys.exit(2)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse detected (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly. Full Colab run will tell us")
        print(f"  if TabM_D delivers meaningful diversity vs RealMLP.")
        return

    out_dir = PROBS / "v10_tabm"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v10_tabm.csv"
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
            "model_family": "pytabkit.TabM_D_Classifier",
        },
    )
    exp.commit()
    print(f"\nExperiment v10_tabm committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
