"""v14 — Yekenot FE reproduction (no external data, A path from triage).

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: yekenot's OOF 0.9537 comes mostly from feature engineering, not
              from external data merging. With our sacred holdout protocol
              (80% pool / 20% locked holdout) and yekenot's FE recipe
              (arithmetic interactions + floor-factorize + count encoding +
               KBins(RaceProgress×200, LapTime×7) + TE on Race×Compound and
               Race×Year), but WITHOUT merging the F1 strategy external dataset,
              we should still get +0.003 to +0.005 over v9 (0.9479 holdout).
              If FE alone closes most of the gap, the docs/pitfalls.md:44 rule
              banning external data for training is vindicated. If FE alone
              gets only +0.001, the external merge IS the lever — file v14b
              to test that path with explicit override.
  Parent: v9_realmlp_big (current solo RealMLP at holdout 0.94808)
  Predicted Δ holdout: +0.003 to +0.005
  Confidence: medium-high — direct OOF measurement on yekenot's file confirmed 0.9537
  Most relevant pitfall: #10 (one change rule violated — multi-FE bundled. Mitigation:
                         this is a reproduction not an iteration; the bundle IS the test.)
  Validation plan: 5-fold StratifiedKFold(42) + sacred 20% holdout
  Abort signal: sanity OOF AUC < 0.90 → pipeline broken; fold collapse > 0.05 → leakage
  Output: probs/v14_yekenot_repro/{oof,holdout,test}.npy + submissions/v14_yekenot_repro.csv

Usage:
  python notebooks/17_v14_yekenot_repro.py --subsample 5000   # local CPU sanity (~2 min)
  python notebooks/17_v14_yekenot_repro.py --gpu              # full Colab/Kaggle (~12 min)
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
from src.features import yekenot_fe_fit, yekenot_fe_transform, yekenot_feature_lists
from src.train import train_variant
from src.evaluate import auc
from src.observer import Experiment


def main(subsample: int | None, gpu: bool):
    print("=" * 70)
    print("v14 — Yekenot FE reproduction (no external data)")
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

    print("\n--- Yekenot FE ---")
    pool_fe, fe_state = yekenot_fe_fit(pool)
    holdout_fe = yekenot_fe_transform(holdout, fe_state)
    test_fe = yekenot_fe_transform(test, fe_state)

    numeric_feats, cat_feats = yekenot_feature_lists(fe_state)
    feature_cols = numeric_feats + cat_feats
    print(f"feature_cols ({len(feature_cols)}): {len(numeric_feats)} numeric + {len(cat_feats)} categorical")
    print(f"  numeric:  {numeric_feats}")
    print(f"  categorical (passed to RealMLP embeddings): {cat_feats}")

    # TE on the two combo categoricals — same as yekenot's important_combos.
    # train_variant applies fold-safe sklearn TargetEncoder(cv=5).
    te_pairs = [("Race", "Compound"), ("Race", "Year")]
    print(f"\nte_pairs (fold-safe TE applied by train_variant): {te_pairs}")

    y_pool = pool_fe[TARGET].astype(int).to_numpy()
    y_holdout = holdout_fe[TARGET].astype(int).to_numpy()

    # Hyperparams: yekenot's RealMLP config (n_ens=24 same as our v9; n_epochs=6 vs our default).
    if subsample:
        params_override = {"n_ens": 4, "n_epochs": 2, "verbosity": 0}
    else:
        params_override = {"n_ens": 24, "n_epochs": 6}

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v14_yekenot_repro",
            parent="v9_realmlp_big",
            hypothesis=(
                "Yekenot's OOF 0.9537 comes mostly from FE, not external data merge. "
                "With our sacred holdout protocol + yekenot's FE (arith interactions, "
                "floor-factorize, count encoding, KBins, TE on Race×Compound and Race×Year) "
                "but WITHOUT external merge (docs/pitfalls.md:44 rule honored), we should "
                "still gain +0.003 to +0.005 over v9 (0.9479)."
            ),
            predicted_delta=0.004,
            confidence="medium",
            feature_changes=[
                "+ arith interactions (LapNumber/RaceProgress, TyreLife/LapNumber)",
                "+ floor-factorize numericals as categorical codes (13 new cat cols)",
                "+ count encoding on cat cols + Year_cat/PitStop_cat",
                "+ KBins (RaceProgress×200, LapTime×7)",
                "+ TE on Race×Compound and Race×Year pair combos",
            ],
            config_changes={"n_ens": 24, "n_epochs": 6, "_prev_n_epochs": 5},
            pipeline_changes=[
                "+ yekenot_fe_fit / yekenot_fe_transform in src/features.py",
                "+ TE wiring through _train_realmlp (mirrors LGB pattern)",
            ],
            cloud_or_local="local",  # local RTX 4070, parallel to v12 on Colab
        )

    print(f"\n--- Training RealMLP (n_ens={params_override.get('n_ens', 24)}, n_epochs={params_override.get('n_epochs', 6)}) ---")
    # Pre-yekenot pool has 'Driver','Compound','Race' as raw strings; train_variant
    # handles them via _align_category_codes + pytabkit embeddings.
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
        te_pairs=te_pairs,
    )

    oof_auc_mean = float(np.mean(result["fold_aucs"]))
    holdout_auc = auc(y_holdout, result["holdout_pred"])
    print()
    print(f"OOF AUC (mean of folds): {oof_auc_mean:.5f}")
    print(f"OOF AUC (per fold):       {[f'{x:.5f}' for x in result['fold_aucs']]}")
    print(f"Holdout AUC:              {holdout_auc:.5f}")
    print(f"Yekenot OOF (reference):  0.95368")
    print(f"v9 holdout (parent):      0.94808")
    print(f"Δ vs v9:                  {holdout_auc - 0.94808:+.5f}")
    print(f"Runtime:                  {result['runtime_sec']:.1f}s")

    if is_sanity:
        if oof_auc_mean < 0.80:
            print(f"\n⚠ SANITY ABORT — OOF mean {oof_auc_mean:.5f} below 0.80 (pipeline broken).")
            sys.exit(2)
        fold_min = min(result["fold_aucs"])
        if fold_min < oof_auc_mean - 0.05:
            print(f"\n⚠ SANITY ABORT — fold collapse (fold_min={fold_min:.5f})")
            sys.exit(2)
        print(f"\n✓ Sanity pass — pipeline runs cleanly.")
        return

    out_dir = PROBS / "v14_yekenot_repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", result["oof_pred"])
    np.save(out_dir / "holdout.npy", result["holdout_pred"])
    np.save(out_dir / "test.npy", result["test_pred"])
    print(f"\nSaved probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test_fe[ID].astype("int64"), TARGET: result["test_pred"]})
    sub_path = SUBMISSIONS / "v14_yekenot_repro.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=oof_auc_mean,
        oof_auc_per_fold=[float(x) for x in result["fold_aucs"]],
        holdout_auc=float(holdout_auc),
        runtime_sec=float(result["runtime_sec"]),
        extra={
            "n_features": len(feature_cols),
            "n_numeric": len(numeric_feats),
            "n_categorical": len(cat_feats),
            "te_pairs": [list(p) for p in te_pairs],
            "n_ens": 24,
            "model_family": "pytabkit.RealMLP_TD_Classifier",
            "yekenot_reference_oof": 0.95368,
            "external_data_used": False,
        },
    )
    exp.commit()
    print(f"\nExperiment v14_yekenot_repro committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    main(subsample=args.subsample, gpu=args.gpu)
