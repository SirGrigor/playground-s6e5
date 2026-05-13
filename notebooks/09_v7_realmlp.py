"""v7 — RealMLP_TD (pytabkit) on Block 1 features.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: We empirically proved that "more LGB features" and "different
              tree algorithm" both plateau at ~0.943. RealMLP is a genuinely
              different model class — neural net with PLR (Periodic Linear ReLU)
              embeddings + learned categorical embeddings. Its decision
              boundaries differ structurally from tree splits. Expected blend
              correlation with LGB/XGB: 0.85-0.93 (much lower than tree-tree 0.99).
  Parent: v1_lgb (our best solo)
  Predicted Δ holdout vs v1: 0 to +0.001 (RealMLP is competitive but not
                              expected to beat trees solo on this data)
  Real payoff: blend(v1, v4, v7) should compound +0.001-0.003 over v5 because
               ρ(v1, v7) and ρ(v4, v7) will be much lower than ρ(v1, v4)=0.989
  Most relevant pitfall: #10 (one change), L17 (pytabkit baseline for tabular)
  Validation plan: 5-fold StratifiedKFold(42) + sacred holdout
  Abort signal: subsample-mode pipeline integrity (AUC > 0.85, no fold collapse)
  Output: probs/v7_realmlp/{oof,holdout,test}.npy + submissions/v7_realmlp.csv

Config: yekenot's S6E5 top-voted notebook params, with n_ens=8 (vs his 24)
for cost/quality balance.

Usage:
  python notebooks/09_v7_realmlp.py --subsample 10000  # local sanity
  python notebooks/09_v7_realmlp.py                    # full data on Colab
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
    print("v7 — RealMLP_TD on Block 1 (genuine model-class diversity)")
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

    # Same feature space as v1/v4 — experiment is the model class, not features
    categorical_cols = ["Driver", "Compound", "Race"]
    drop_cols = {ID, TARGET, "race_lap_mean", "race_lap_std"}
    feature_cols = [c for c in pool.columns if c not in drop_cols]

    print(f"feature_cols ({len(feature_cols)}): same as v1/v4 (Block 1)")
    print(f"categorical_cols (pytabkit auto-embeds): {categorical_cols}")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    # For sanity: use lighter config (fast verify pipeline works)
    sanity_params = {
        "n_ens": 2,
        "n_epochs": 2,
        "verbosity": 0,
    } if subsample else {}

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v7_realmlp",
            parent="v1_lgb",
            hypothesis=(
                "RealMLP_TD (pytabkit) on Block 1 — genuine model-class diversity. "
                "Empirically, more LGB features and different tree algo both plateaued "
                "at ~0.943. Neural net w/ PLR embeddings has structurally different "
                "decision boundaries than trees. Expected solo Δ vs v1: 0 to +0.001. "
                "Real value: ρ(v7, v1) and ρ(v7, v4) should be 0.85-0.93 (vs 0.99 for "
                "tree-tree) → big blend payoff in v8 = blend(v1, v4, v7)."
            ),
            predicted_delta=0.0005,  # midpoint of [0, 0.001]
            confidence="low",  # honest: RealMLP solo might match or slightly underperform LGB
            feature_changes=[],
            config_changes={"algo": "realmlp", "n_ens": 8, "n_epochs": 5, "hidden_sizes": [512, 256, 128]},
            pipeline_changes=["+ pytabkit.RealMLP_TD_Classifier (neural net)"],
            cloud_or_local="local" if subsample else "cloud",
        )

    print("\n--- Training RealMLP_TD ---")
    result = train_variant(
        algo="realmlp",
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
        if oof_auc_mean < 0.80:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.80 (pipeline broken).")
            sys.exit(2)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse detected (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly.")
        print(f"  RealMLP sanity uses n_ens=2, n_epochs=2. Full Colab run uses 8/5.")
        return

    out_dir = PROBS / "v7_realmlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v7_realmlp.csv"
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
            "model_family": "pytabkit.RealMLP_TD_Classifier",
        },
    )
    exp.commit()
    print(f"\nExperiment v7_realmlp committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
