"""v22 — Own-trained FinetunedTabPFN on yekenot FE + sacred holdout protocol.

Pre-experiment checklist:
  Hypothesis: karltonkxb's vanilla TabPFN-3 + domain FE got solo LB 0.94922
              (49 min on Kaggle GPU). With yekenot FE (which gave RealMLP
              +0.0037 lift, v9→v14) layered on the same TabPFN training
              backbone, we should hit ≥0.952 solo. At ≥0.952 solo and
              ρ≈0.972 with safar1_95449 (currently best at 0.95449),
              math says blend at w≈0.30 lifts to ~0.9547 (+0.00021 LB).
              At ≥0.953 solo, blend lifts to ~0.95516 (+0.00067 LB).
  Parent: karltonkxb/tabpfn-3-s6e5-predicting-f1-pit-stops (0.94922) + v14 FE
  Predicted Δ holdout vs karltonkxb baseline: +0.002 to +0.004
  Predicted blend lift vs safar1_95449: +0.00021 to +0.00067 (depending on solo strength)
  Confidence: medium-high — yekenot FE proven on RealMLP; TabPFN family responds well to good FE
  Risk: TabPFN's prior may be less aligned with synthetic Playground data than
        RealMLP. If solo lands at 0.949-0.951, blend won't help — pivot to
        karltonkxb-FE-style fine-tuning or more ensembles.
  Validation plan: sacred 20% holdout + ρ-check + math-optimal blend weight
  Output: probs/v22_tabpfn_own/{holdout,test}.npy + submissions/v22.xxx.csv
          PLUS math-recommended next submission via src.blend_math

PREREQUISITE — TabPFN token (one-time setup):
  1. Register/login at https://ux.priorlabs.ai
  2. Accept license on Licenses tab
  3. Copy API key from https://ux.priorlabs.ai/account
  4. Add it ONCE — choose whichever environment you use:
     • Colab: click the 🔑 key icon in left sidebar → Add new secret
              Name: TABPFN_TOKEN, Value: <your-key>, toggle "Notebook access" ON
     • Kaggle notebook: Add-ons → Secrets → Add new secret named TABPFN_TOKEN
     • Local shell: add `export TABPFN_TOKEN=<your-key>` to ~/.bashrc, then `source ~/.bashrc`
  The notebook auto-loads from Colab/Kaggle Secrets, then env var, then local file.

Usage:
  python notebooks/29_v22_tabpfn_own.py --subsample 5000   # local sanity if pytabkit installed
  python notebooks/29_v22_tabpfn_own.py --gpu              # full Colab/Kaggle (~60 min)
  python notebooks/29_v22_tabpfn_own.py --gpu --with-original  # add external dataset
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import PROBS, SUBMISSIONS, TARGET, ID, MODEL_SEED
from src.data import load_train_pool, load_holdout, load_test, load_original
from src.features import yekenot_fe_fit, yekenot_fe_transform, yekenot_feature_lists
from src.evaluate import auc
from src.observer import Experiment
from src.blend_math import (
    fit_quadratic_lb, recommend_next_weight, predict_blend_lb, rank_normalize,
)


def predict_proba_in_batches(model, X_data, batch_size: int = 50_000) -> np.ndarray:
    """TabPFN API has row-count limits — chunk the inference."""
    all_probas = []
    n = len(X_data)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        print(f"  predicting rows {start:,} → {end:,} of {n:,}")
        batch = X_data.iloc[start:end] if hasattr(X_data, "iloc") else X_data[start:end]
        proba = model.predict_proba(batch)[:, 1]
        all_probas.append(proba)
    return np.concatenate(all_probas)


def load_tabpfn_token() -> str:
    """Load TABPFN_TOKEN from (in order): env, Colab Secrets, Kaggle Secrets, local file.

    Mirrors the same UX as KAGGLE_API_TOKEN — set it once in your environment,
    then every notebook/script picks it up automatically.
    """
    tok = os.environ.get("TABPFN_TOKEN", "")
    if tok:
        return tok
    # Colab Secrets (sidebar key icon)
    try:
        from google.colab import userdata  # type: ignore
        tok = userdata.get("TABPFN_TOKEN")
        if tok:
            os.environ["TABPFN_TOKEN"] = tok
            print("  TABPFN_TOKEN loaded from Colab Secrets.")
            return tok
    except Exception:
        pass
    # Kaggle Secrets (Add-ons → Secrets)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        tok = UserSecretsClient().get_secret("TABPFN_TOKEN")
        if tok:
            os.environ["TABPFN_TOKEN"] = tok
            print("  TABPFN_TOKEN loaded from Kaggle Secrets.")
            return tok
    except Exception:
        pass
    # Local file fallback (gitignored)
    for candidate in [Path("tabpfn_token.txt"), Path.home() / ".tabpfn_token"]:
        if candidate.exists():
            tok = candidate.read_text().strip()
            os.environ["TABPFN_TOKEN"] = tok
            print(f"  TABPFN_TOKEN loaded from {candidate}.")
            return tok
    return ""


def main(subsample: int | None, gpu: bool, with_original: bool):
    print("=" * 70)
    print("v22 — Own FinetunedTabPFN on yekenot FE + sacred holdout")
    print("=" * 70)

    # Step 0: require TabPFN token (cloud fine-tuning) unless local sanity
    if not subsample:
        if not load_tabpfn_token():
            print("\nERROR: TABPFN_TOKEN not found in any source.")
            print("  Tried: env var, Colab Secrets, Kaggle Secrets, ~/.tabpfn_token, ./tabpfn_token.txt")
            print("  See header docstring for one-time setup steps.")
            sys.exit(1)

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    print(f"pool: {pool.shape}   holdout: {holdout.shape}   test: {test.shape}")

    if with_original:
        try:
            orig = load_original()
            if TARGET in orig.columns:
                # Original has target; align columns to pool's schema (no holdout split applies)
                common_cols = [c for c in pool.columns if c in orig.columns]
                orig_aligned = orig[common_cols].copy()
                # Synthesize an id column so the merge logic works
                if "id" not in orig_aligned.columns:
                    orig_aligned["id"] = -np.arange(1, len(orig_aligned) + 1)  # negative ids → external
                pool = pd.concat([pool, orig_aligned], axis=0, ignore_index=True)
                print(f"  + original dataset merged → pool: {pool.shape}")
        except Exception as e:
            print(f"  (--with-original requested but skipped: {e})")

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

    y_pool = pool_fe[TARGET].astype(int).to_numpy()
    y_holdout = holdout_fe[TARGET].astype(int).to_numpy()
    X_pool = pool_fe[feature_cols]
    X_holdout = holdout_fe[feature_cols]
    X_test = test_fe[feature_cols]

    categorical_indices = [X_pool.columns.get_loc(c) for c in cat_feats if c in X_pool.columns]

    is_sanity = subsample is not None
    exp = None
    if not is_sanity:
        exp = Experiment.start(
            version="v22_tabpfn_own",
            parent="v14_yekenot_repro",
            hypothesis=(
                "karltonkxb's vanilla TabPFN-3 + simple domain FE got 0.94922 LB. "
                "With yekenot FE (the +0.0037 lever proven on RealMLP v9→v14) on "
                "the same FinetunedTabPFN backbone (epochs=10, lr=2e-3), we should "
                "reach solo LB ≥0.952. At ρ≈0.972 with safar1_95449 (LB 0.95449), "
                "math says blend at w≈0.30 lifts to ~0.9547 (+0.00021). At solo "
                "≥0.953, blend lifts to ~0.95516 (+0.00067)."
            ),
            predicted_delta=0.003,
            confidence="medium-high",
            feature_changes=[
                "+ yekenot FE (vs karltonkxb's domain FE)",
                "+ sacred 20% holdout for validation",
            ],
            config_changes={
                "model": "FinetunedTabPFNClassifier",
                "epochs": 10,
                "learning_rate": 2e-3,
                "n_estimators": 1,
                "tune_decision_thresholds": True,
            },
            pipeline_changes=[
                "+ TabPFN API integration via TABPFN_TOKEN",
                "+ src/blend_math.py — Step 4 quadratic-LB fit + NM holdout optimizer",
            ],
            cloud_or_local="cloud",
        )

    # --- Train FinetunedTabPFN ---
    print(f"\n--- FinetunedTabPFN training ({len(X_pool):,} rows × {len(feature_cols)} feats) ---")
    if is_sanity:
        print("  [SANITY] skipping TabPFN fit — pipeline integrity only")
        return

    from tabpfn import TabPFNClassifier  # noqa: F401 (informational import)
    from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier

    classifier = FinetunedTabPFNClassifier(
        device="cuda" if gpu else "cpu",
        epochs=10,
        learning_rate=2e-3,
        n_estimators_finetune=1,
        n_estimators_validation=1,
        n_estimators_final_inference=1,
        random_state=MODEL_SEED,
        eval_metric="log_loss",
        extra_classifier_kwargs={
            "categorical_features_indices": categorical_indices,
            "ignore_pretraining_limits": True,
            "tuning_config": {"tune_decision_thresholds": True},
        },
    )

    t0 = time.time()
    print("  fitting...")
    classifier.fit(X_pool, y_pool)
    fit_time = time.time() - t0
    print(f"  fit completed in {fit_time:.1f}s")

    # --- Predict holdout + test ---
    t0 = time.time()
    print("  predicting holdout...")
    holdout_pred = predict_proba_in_batches(classifier, X_holdout, batch_size=50_000)
    print(f"  predicting test...")
    test_pred = predict_proba_in_batches(classifier, X_test, batch_size=50_000)
    pred_time = time.time() - t0
    print(f"  inference completed in {pred_time:.1f}s")

    holdout_auc = auc(y_holdout, holdout_pred)
    print()
    print(f"Holdout AUC:                  {holdout_auc:.5f}")
    print(f"v14 holdout (parent):         0.95194")
    print(f"karltonkxb LB (no yekenot):   0.94922")
    print(f"Δ vs v14:                     {holdout_auc - 0.95194:+.5f}")
    print(f"Δ vs karltonkxb:              {holdout_auc - 0.94922:+.5f}")
    print(f"Total runtime:                {fit_time + pred_time:.1f}s")

    # --- Save artifacts ---
    out_dir = PROBS / "v22_tabpfn_own"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "holdout.npy", holdout_pred)
    np.save(out_dir / "test.npy", test_pred)
    print(f"\nSaved holdout/test probs to {out_dir}/")

    sub = pd.DataFrame({ID: test_fe[ID].to_numpy(), TARGET: test_pred})
    sub_path = SUBMISSIONS / "v22.100.csv"  # .100 marks first own-TabPFN; mixes will be .1XX
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    # --- Step 4: ρ check + math-optimal blend recommendation ---
    print("\n" + "=" * 70)
    print("Step 4 — Math-driven blend analysis vs current best (safar1_95449)")
    print("=" * 70)
    safar_path = SUBMISSIONS / "v22.001.csv"
    if not safar_path.exists():
        safar_path = Path("harvest/v21/safar1_lb-score-0-95449/submission.csv")
    if safar_path.exists():
        test_ids = test_fe[ID].to_numpy()
        safar = pd.read_csv(safar_path).set_index(ID).loc[test_ids][TARGET].to_numpy()
        rho = float(np.corrcoef(rankdata(test_pred), rankdata(safar))[0, 1])
        print(f"  ρ(v22_tabpfn_own, safar1_95449) = {rho:.5f}")

        # Predict LB at several weights (using bonus_coef calibrated on yesterday's TabPFN data)
        safar_lb = 0.95449
        # Use measured holdout as proxy for solo LB
        proxy_lb = holdout_auc  # rough — actual LB tends to be slightly lower
        print(f"  v22 holdout AUC: {proxy_lb:.5f}  (proxy for solo LB)")
        print(f"  safar1_95449 confirmed LB: {safar_lb:.5f}")
        print()
        print("  Predicted blend LB (math):")
        for w in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70]:
            pred = predict_blend_lb(safar_lb, proxy_lb, rho, w)
            marker = "  ★" if pred > safar_lb else "   "
            print(f"    w_v22={w:.2f}: predicted LB={pred:.5f}{marker}")

        best_w = max(np.linspace(0, 1, 101),
                     key=lambda w: predict_blend_lb(safar_lb, proxy_lb, rho, w))
        best_pred = predict_blend_lb(safar_lb, proxy_lb, rho, best_w)
        print(f"\n  → MATH RECOMMENDS: w_v22={best_w:.3f}, predicted LB={best_pred:.5f}")
        if best_pred > safar_lb + 0.00003:
            blend = (1 - best_w) * rank_normalize(safar) + best_w * rank_normalize(test_pred)
            mix_path = SUBMISSIONS / "v22.101.csv"
            pd.DataFrame({ID: test_ids, TARGET: blend}).to_csv(mix_path, index=False)
            print(f"  → saved math-optimal blend to {mix_path}")
        else:
            print(f"  → predicted lift < +0.00003 (noise floor); don't submit blend.")

    if exp is not None:
        exp.record(
            oof_auc_mean=0.0,  # no CV in TabPFN flow
            oof_auc_per_fold=[],
            holdout_auc=float(holdout_auc),
            runtime_sec=float(fit_time + pred_time),
            extra={
                "n_features": len(feature_cols),
                "n_pool_rows": len(X_pool),
                "model_family": "FinetunedTabPFNClassifier",
                "epochs": 10,
                "n_estimators": 1,
            },
        )
        exp.commit()
        print("\nExperiment recorded to experiments.jsonl")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--with-original", action="store_true",
                   help="Merge external dataset into training pool (matches karltonkxb)")
    args = p.parse_args()
    main(args.subsample, args.gpu, args.with_original)
