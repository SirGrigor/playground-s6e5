"""v12 — Internal Diversity Factory.

Uses ml-variant-factory (v0.1.0) to sample N variants from a configuration
space and blend them. This replaces hand-built single models with a
methodology that mirrors what public Kagglers do collectively but solo.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: Internal architectural diversity (v8/v11) plateaued at the noise
              floor because ρ > 0.97 — same features, splits, seeds. The factory
              samples across 5 axes simultaneously (feature subset, CV seed,
              model seed, hyperparams, model class) → variants are MORE
              decorrelated than any single hand-built config can produce.
              Each variant is one sample from the cross-product space.
  Parent: v11_blend_4way (current best LB 0.94774)
  Predicted Δ holdout: +0.001 to +0.003 (more decorrelation than v11)
  Most relevant pitfall: #10 (one change), L13 (real diversity is multi-axis)
  Output: probs/v12_factory/{oof,holdout,test}.npy + submissions/v12_factory.csv

Configuration:
  10 variants total — 6 RealMLP, 2 TabM_D, 1 LGB, 1 XGB
  Each samples feature_fraction ∈ [0.6, 1.0], random model_seed and cv_seed,
  random hyperparameters from the search space.

Usage:
  python notebooks/14_v12_factory.py --subsample 5000   # sanity (smaller)
  python notebooks/14_v12_factory.py                    # full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROBS, SUBMISSIONS, TARGET, ID, MODEL_SEED
from src.data import load_train_pool, load_holdout, load_test
from src.features import build_block1
from src.evaluate import auc
from src.observer import Experiment

from ml_variant_factory import sample_variants, run_factory, aggregate
from ml_variant_factory.search import (
    realmlp_default,
    tabm_default,
    lgb_default,
    xgb_default,
)


N_VARIANTS = 10
ALGO_WEIGHTS = {
    "realmlp": 6.0,  # 60% — strongest single model on this comp
    "tabm":    2.0,  # 20%
    "lgb":     1.0,  # 10%
    "xgb":     1.0,  # 10%
}


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print(f"v12 — Internal Diversity Factory (n={N_VARIANTS} variants)")
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

    print(f"feature_cols ({len(feature_cols)}): same as v9/v11 (Block 1)")
    print(f"categorical_cols: {categorical_cols}")

    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    # --- Define search space (per-algo) ---
    search_space = {
        "realmlp": realmlp_default(),
        "tabm":    tabm_default(),
        "lgb":     lgb_default(),
        "xgb":     xgb_default(),
    }
    if subsample:
        # In sanity mode, smaller / faster configs to verify pipeline
        search_space["realmlp"] = {**search_space["realmlp"], "n_ens": [2], "n_epochs": [2]}
        search_space["tabm"] = {**search_space["tabm"], "n_epochs": [1]}
        search_space["lgb"] = {**search_space["lgb"], "n_estimators": [200]}
        search_space["xgb"] = {**search_space["xgb"], "n_estimators": [200]}

    # --- Sample variants ---
    n = 3 if subsample else N_VARIANTS
    variants = sample_variants(
        n=n,
        search_space=search_space,
        seed=42,
        algo_weights=ALGO_WEIGHTS,
        name_prefix="v12",
    )
    print(f"\n--- Sampled {len(variants)} variants ---")
    for v in variants:
        print(f"  {v.name:30s}  algo={v.algo:8s} ff={v.feature_fraction:.2f} "
              f"model_seed={v.model_seed} cv_seed={v.cv_seed}")

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v12_factory",
            parent="v11_blend_4way",
            hypothesis=(
                f"Factory of {N_VARIANTS} variants sampled across 5 diversity axes "
                f"(feature subset, CV seed, model seed, hyperparams, model class). "
                f"Tests whether multi-axis sampling produces lower ρ than v11's "
                f"single-axis architectural diversity. Predicted Δ holdout vs v11: "
                f"+0.001 to +0.003."
            ),
            predicted_delta=0.0015,
            confidence="medium",
            feature_changes=[],
            config_changes={"n_variants": N_VARIANTS, "algo_weights": ALGO_WEIGHTS,
                            "search_space": "ml-variant-factory v0.1.0 defaults"},
            pipeline_changes=["+ ml-variant-factory @ v0.1.0"],
            cloud_or_local="local" if subsample else "cloud",
        )

    # --- Run factory ---
    print("\n--- Training factory ---")
    results = run_factory(
        variants,
        X_pool=pool,
        y_pool=y_pool,
        X_holdout=holdout,
        X_test=test,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        n_folds=5,
        verbose=True,
    )

    # --- Per-variant report ---
    print("\n--- Per-variant results ---")
    print(f"{'name':30s}  {'algo':8s}  {'OOF AUC':>10s}  {'holdout':>10s}  {'runtime':>8s}")
    for r in results:
        ho_auc = auc(y_holdout, r["holdout_pred"])
        print(f"{r['variant'].name:30s}  {r['variant'].algo:8s}  "
              f"{r['oof_auc']:>10.5f}  {ho_auc:>10.5f}  {r['runtime_sec']:>8.1f}s")

    # --- Aggregation ---
    blended = aggregate(
        results,
        y_pool=y_pool,
        y_holdout=y_holdout,
        method="auto",
        quality_min_auc=None,   # no filter for now; let Nelder-Mead drop weak ones
        n_restarts=8,           # extra restarts for many-variant blends
    )

    print(f"\n--- Aggregation chose: {blended['method']} ---")
    print(f"Final OOF AUC:     {blended['oof_auc']:.5f}")
    print(f"Final holdout AUC: {blended['holdout_auc']:.5f}")
    print(f"vs v11 holdout (0.94833):  Δ = {blended['holdout_auc'] - 0.94833:+.5f}")
    print(f"vs v9 holdout  (0.94808):  Δ = {blended['holdout_auc'] - 0.94808:+.5f}")
    print()
    print("Weight distribution:")
    for name, w in zip(blended["contributor_names"], blended["weights"], strict=True):
        print(f"  {name:30s}  {w:.4f}")

    if is_sanity:
        print("\n✓ Sanity pass — factory runs end-to-end. Re-run without --subsample for full.")
        return

    # --- Save artifacts ---
    out_dir = PROBS / "v12_factory"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", blended["oof"])
    np.save(out_dir / "holdout.npy", blended["holdout"])
    np.save(out_dir / "test.npy", blended["test"])

    # Save per-variant audit trail
    audit = {
        "variants": [v.to_dict() for v in variants],
        "results_summary": [
            {
                "name": r["variant"].name,
                "algo": r["variant"].algo,
                "feature_fraction": r["variant"].feature_fraction,
                "model_seed": r["variant"].model_seed,
                "cv_seed": r["variant"].cv_seed,
                "oof_auc": r["oof_auc"],
                "holdout_auc": float(auc(y_holdout, r["holdout_pred"])),
                "runtime_sec": r["runtime_sec"],
                "n_features_used": r["n_features_used"],
                "algo_params": r["variant"].algo_params,
            }
            for r in results
        ],
        "blend": {
            "method": blended["method"],
            "weights": blended["weights"],
            "contributor_names": blended["contributor_names"],
            "oof_auc": blended["oof_auc"],
            "holdout_auc": blended["holdout_auc"],
        },
    }
    with open(out_dir / "audit.json", "w") as fp:
        json.dump(audit, fp, indent=2, default=str)
    print(f"\nSaved probs + audit to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: blended["test"]})
    sub_path = SUBMISSIONS / "v12_factory.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=float(blended["oof_auc"]),
        oof_auc_per_fold=[float(blended["oof_auc"])] * 5,
        holdout_auc=float(blended["holdout_auc"]),
        runtime_sec=float(sum(r["runtime_sec"] for r in results)),
        extra={
            "n_variants": len(variants),
            "blend_method": blended["method"],
            "weights": blended["weights"],
            "contributor_names": blended["contributor_names"],
            "per_variant_oof_auc": {r["variant"].name: r["oof_auc"] for r in results},
            "ml_variant_factory_version": "0.1.0",
        },
    )
    exp.commit()
    print(f"\nExperiment v12_factory committed. Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
