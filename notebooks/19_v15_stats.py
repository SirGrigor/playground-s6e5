"""v15 — v14b + group target-mean (yekenot notebook #2 statistical features).

Pre-experiment checklist:
  Hypothesis: Adding fold-safe CV-target-encoding on each categorical and each
              KBins-binned numerical (yekenot notebook #2 pattern) gives the
              model per-group target-rate signal that pair-TE alone doesn't
              capture. The marginal target_mean of e.g. RaceProgress_bin
              encodes "historical pit rate at this race-progress slice"
              independent of compound/race interactions.
  Parent: v14b_external (predicted holdout ~0.953-0.954)
  Predicted Δ holdout: +0.0005 to +0.0010
  Confidence: medium-low — most signal may be redundant with pair-TE + embeddings
  Most relevant pitfall: #5 (always sklearn TargetEncoder, never manual TE)
  Output: probs/v15_stats/{oof,holdout,test}.npy + submissions/v15_stats.csv

Difference from v14b:
  Only the te_cols list grows. Same external merge, same architecture, same FE.
  Cleanest possible test of "more marginal TE columns" lever in isolation.

Usage:
  python notebooks/19_v15_stats.py --subsample 5000   # local CPU sanity
  python notebooks/19_v15_stats.py --gpu              # full run (~12 min)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROBS, SUBMISSIONS, TARGET, ID, MODEL_SEED
from src.data import load_train_pool, load_holdout, load_test, load_original
from src.features import yekenot_fe_fit, yekenot_fe_transform, yekenot_feature_lists
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v15 — yekenot FE + external merge + extended marginal TE")
    print("=" * 70)

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    orig = load_original()
    if "Normalized_TyreLife" in orig.columns:
        orig = orig.drop(columns=["Normalized_TyreLife"])
    orig = orig.dropna()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}   orig: {orig.shape}")

    if subsample is not None:
        rng = np.random.default_rng(MODEL_SEED)
        idx = rng.choice(len(pool), size=min(subsample, len(pool)), replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)
        orig_sub = max(int(subsample * 0.3), 100)
        orig = orig.iloc[rng.choice(len(orig), size=min(orig_sub, len(orig)), replace=False)].reset_index(drop=True)
        print(f"  -- SANITY MODE -- pool→{len(pool):,}, orig→{len(orig):,}")

    print("\n--- Yekenot FE ---")
    pool_fe, fe_state = yekenot_fe_fit(pool)
    holdout_fe = yekenot_fe_transform(holdout, fe_state)
    test_fe = yekenot_fe_transform(test, fe_state)
    orig_X = orig.drop(columns=[TARGET])
    orig_y = orig[TARGET].astype(int).to_numpy()
    orig_X_fe = yekenot_fe_transform(orig_X, fe_state)

    numeric_feats, cat_feats = yekenot_feature_lists(fe_state)
    feature_cols = numeric_feats + cat_feats

    # v15 lever: extended marginal TE (yekenot notebook #2 statistical features).
    # Each becomes a fold-safe target_mean column via sklearn TargetEncoder(cv=5).
    te_cols = [
        # Original categoricals
        "Compound", "Year", "PitStop",
        # Floor-factorized numericals (yekenot's _cat_ codes)
        "Stint_cat_", "Position_cat_", "LapNumber_cat_", "TyreLife_cat_",
        "LapTime (s)_cat_", "RaceProgress_cat_",
        # KBins discretized
        "RaceProgress_200_quantile_bin_", "LapTime (s)_7_quantile_bin_",
    ]
    te_pairs = [("Race", "Compound"), ("Race", "Year")]
    print(f"\nfeature_cols ({len(feature_cols)}): {len(numeric_feats)} numeric + {len(cat_feats)} categorical")
    print(f"te_cols (marginal target mean per col, NEW vs v14b): {te_cols}")
    print(f"te_pairs (joint TE, same as v14b): {te_pairs}")
    print(f"external rows added per fold: {len(orig_X_fe):,}")

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
            version="v15_stats",
            parent="v14b_external",
            hypothesis=(
                "v14b + extended marginal target encoding (CV-safe sklearn TargetEncoder) "
                "on 11 single-column groups: 3 raw cats + 6 floor-factorized + 2 KBins. "
                "Mirrors yekenot notebook #2's statistical-features pattern but minimal "
                "(only mean, no std/skew/min/max). Tests whether marginal target signal "
                "adds anything beyond pair-TE + raw embeddings."
            ),
            predicted_delta=0.0008,
            confidence="low",
            feature_changes=[
                f"+ extended te_cols ({len(te_cols)} marginal TE columns)",
            ],
            config_changes={},
            pipeline_changes=[],
            cloud_or_local="local",
        )

    print(f"\n--- Training RealMLP (n_ens={params_override.get('n_ens', 24)}) ---")
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
        te_cols=te_cols,
        te_pairs=te_pairs,
        external_X=orig_X_fe,
        external_y=orig_y,
    )

    oof_auc_mean = float(np.mean(result["fold_aucs"]))
    holdout_auc = auc(y_holdout, result["holdout_pred"])
    print()
    print(f"OOF AUC (mean of folds): {oof_auc_mean:.5f}")
    print(f"Holdout AUC:              {holdout_auc:.5f}")
    print(f"v14 holdout (grandparent): 0.95194")
    print(f"v14b predicted target:    ~0.953-0.954")
    print(f"v15 predicted target:     ~0.9535-0.955")
    print(f"Runtime:                  {result['runtime_sec']:.1f}s")

    if is_sanity:
        if oof_auc_mean < 0.80:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.80")
            sys.exit(2)
        print(f"\n✓ Sanity pass.")
        return

    out_dir = PROBS / "v15_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test_fe[ID].astype("int64"), TARGET: result["test_pred"]})
    sub.to_csv(SUBMISSIONS / "v15_stats.csv", index=False)

    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features": len(feature_cols),
            "te_cols": te_cols,
            "te_pairs": [list(p) for p in te_pairs],
            "external_data_used": True,
            "external_rows_per_fold": len(orig_X_fe),
        },
    )
    exp.commit()
    print(f"\nExperiment v15_stats committed. Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
