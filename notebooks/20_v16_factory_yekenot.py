"""v16 — ml-variant-factory on yekenot FE space (the two-stage hypothesis test).

Pre-experiment checklist:
  Hypothesis: The v12 factory plateaued at holdout 0.94998 because Block 1
              features set a hard ceiling. With yekenot FE (38 features, TE
              on combos) the same factory methodology should reach ~0.952
              and produce diverse variants that v17 can blend with v14/v14b.
              This is the empirical test of the "sequential factories"
              architecture discussed in v14 retrospective.
  Parent: v14_yekenot_repro (holdout 0.95194)
  Predicted Δ holdout (factory blend): -0.001 to +0.001 vs v14 solo (factory
              tradeoff: weaker individual variants + diversity gain ≈ wash)
  Confidence: low — exploratory, real value is in v17 blend not v16 solo
  Most relevant pitfall: NOTE — no external data merge here (factory doesn't
                         support it without scope creep into ml-variant-factory).
  Output: probs/v16_factory_yekenot/{oof,holdout,test}.npy
          + submissions/v16_factory_yekenot.csv

Usage:
  python notebooks/20_v16_factory_yekenot.py --subsample 5000   # CPU sanity
  python notebooks/20_v16_factory_yekenot.py --gpu              # full run (~45 min)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROBS, SUBMISSIONS, TARGET, ID, MODEL_SEED, CV_SEED, N_FOLDS
from src.data import load_train_pool, load_holdout, load_test
from src.features import yekenot_fe_fit, yekenot_fe_transform, yekenot_feature_lists
from src.evaluate import auc
from src.observer import Experiment

from ml_variant_factory import sample_variants, run_factory, aggregate
from ml_variant_factory.search import realmlp_default, tabm_default, lgb_default


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v16 — ml-variant-factory on yekenot FE")
    print("=" * 70)

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}")

    if subsample is not None:
        rng = np.random.default_rng(MODEL_SEED)
        idx = rng.choice(len(pool), size=min(subsample, len(pool)), replace=False)
        pool = pool.iloc[idx].reset_index(drop=True)
        print(f"  -- SANITY MODE -- pool→{len(pool):,}")

    print("\n--- Yekenot FE ---")
    pool_fe, fe_state = yekenot_fe_fit(pool)
    holdout_fe = yekenot_fe_transform(holdout, fe_state)
    test_fe = yekenot_fe_transform(test, fe_state)
    numeric_feats, cat_feats = yekenot_feature_lists(fe_state)
    feature_cols = numeric_feats + cat_feats
    print(f"feature_cols ({len(feature_cols)}): {len(numeric_feats)} numeric + {len(cat_feats)} categorical")

    # Skew weights toward RealMLP (proven strong on this comp) + LGB (cheap)
    # Skip TabM (weak on Block 1 in v12, no reason to expect different here)
    n_variants = 8 if not subsample else 3
    print(f"\n--- Sampling {n_variants} variants ---")
    search_specs = {"realmlp": realmlp_default(), "lgb": lgb_default()}
    algo_weights = {"realmlp": 5.0, "lgb": 3.0}

    variants = sample_variants(
        n=n_variants,
        search_space=search_specs,
        algo_weights=algo_weights,
        seed=MODEL_SEED,
    )
    for v in variants:
        print(f"  {v.name:<32} algo={v.algo:<8} ff={v.feature_fraction:.2f}")

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v16_factory_yekenot",
            parent="v14_yekenot_repro",
            hypothesis=(
                "ml-variant-factory on yekenot FE space (no external data) "
                "tests whether the v12 factory methodology delivers diversity "
                "gain on a richer feature space. Expected solo holdout ~0.951 "
                "(below v14b/v15) but diverse OOFs valuable for v17 blend."
            ),
            predicted_delta=-0.001,
            confidence="low",
            feature_changes=[],
            config_changes={"n_variants": n_variants, "algo_weights": algo_weights},
            pipeline_changes=["+ run_factory on yekenot FE state"],
            cloud_or_local="cloud",
        )

    print(f"\n--- Running factory ({n_variants} variants) ---")
    # cv_seed is per-variant (set by sample_variants); use_gpu is auto-detected by pytabkit
    results = run_factory(
        variants=variants,
        X_pool=pool_fe,
        y_pool=pool_fe[TARGET].astype(int).to_numpy(),
        X_holdout=holdout_fe,
        X_test=test_fe,
        feature_cols=feature_cols,
        categorical_cols=cat_feats,
        n_folds=N_FOLDS,
    )

    print("\n--- Per-variant (holdout AUC computed locally — run_variant returns predictions, not holdout AUC) ---")
    from sklearn.metrics import roc_auc_score
    y_holdout = holdout_fe[TARGET].astype(int).to_numpy()
    per_variant = []
    for r in results:
        ho_auc = roc_auc_score(y_holdout, r["holdout_pred"])
        per_variant.append({"name": r["variant"].name, "oof_auc": r["oof_auc"], "holdout_auc": ho_auc, "runtime_sec": r["runtime_sec"]})
        print(f"  {r['variant'].name:<32}  OOF={r['oof_auc']:.5f}  holdout={ho_auc:.5f}  rt={r['runtime_sec']:.1f}s")

    # Aggregate via Nelder-Mead — aggregate() returns its own oof_auc/holdout_auc
    print("\n--- Aggregating ---")
    agg = aggregate(
        results=results,
        y_pool=pool_fe[TARGET].astype(int).to_numpy(),
        y_holdout=y_holdout,
        method="auto",
    )
    blend_oof = agg["oof_auc"]
    blend_holdout = agg["holdout_auc"]
    print(f"\nFINAL BLEND OOF:     {blend_oof:.5f}")
    print(f"FINAL BLEND Holdout: {blend_holdout:.5f}")
    print(f"v14 holdout (parent): 0.95194")
    print(f"v15 holdout:         0.95372 (current best internal)")
    print(f"v12 factory hold:    0.94998 (Block 1)")

    if is_sanity:
        print(f"\n✓ Sanity pass.")
        return

    out_dir = PROBS / "v16_factory_yekenot"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", agg["oof"])
    np.save(out_dir / "holdout.npy", agg["holdout"])
    np.save(out_dir / "test.npy", agg["test"])

    audit = {
        "variants_summary": [{
            "name": v["name"], "oof_auc": float(v["oof_auc"]),
            "holdout_auc": float(v["holdout_auc"]), "runtime_sec": float(v["runtime_sec"]),
        } for v in per_variant],
        "blend_weights": list(agg.get("weights", [])),
        "blend_method": agg.get("method", "unknown"),
        "blend_oof_auc": float(blend_oof),
        "blend_holdout_auc": float(blend_holdout),
    }
    (out_dir / "audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Saved probs + audit to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test_fe[ID].astype("int64"), TARGET: agg["test"]})
    sub.to_csv(SUBMISSIONS / "v16_factory_yekenot.csv", index=False)

    exp.record(
        oof_auc_mean=float(blend_oof),
        oof_auc_per_fold=[float(r["oof_auc"]) for r in results],
        holdout_auc=float(blend_holdout),
        runtime_sec=float(sum(r["runtime_sec"] for r in results)),
        extra={
            "n_variants": n_variants,
            "blend_weights": list(agg.get("weights", [])),
            "fe_state": "yekenot_only_no_external",
            "n_features": len(feature_cols),
        },
    )
    exp.commit()
    print(f"\nExperiment v16_factory_yekenot committed. Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
