"""v22 — Own-trained FinetunedTabPFN on simplified yekenot FE + sacred holdout.

Pre-experiment checklist (UPDATED 2026-05-18 after v22.100 attempt failed):

  v22.100 attempt: full yekenot FE (33 features, 18 categorical) → holdout
  AUC 0.50009. Loss plateaued at 0.499 = log-loss at base rate. Classic
  embedding-collapse: TabPFN couldn't handle 13 floor-factorize cats with
  cardinality up to thousands (LapTime_cat_ ≈ 4000 unique values), and
  treating them as numeric is worse (factorize indices are arbitrary).

  v22.200 (this attempt): simplified yekenot FE (15 numeric + 5 cat).
              Drops the 13 floor-factorize cats + 2 KBins entirely. Keeps
              the high-value Yekenot signals: arith ratios, count encodings,
              base 5 low-card cats. This matches karltonkxb's working scheme
              + yekenot's count-encoding extras. Expected: solo LB 0.949-0.952.
  Parent: karltonkxb/tabpfn-3-s6e5-predicting-f1-pit-stops (0.94922) + simplified yekenot
  Predicted Δ holdout vs karltonkxb (0.94922): +0.000 to +0.003
  Predicted blend lift vs safar1_95449: 0 to +0.0003 (only if solo ≥ 0.952)
  Confidence: medium — karltonkxb already showed simple-FE TabPFN works at 0.949
              on this data. Count encodings + Year_cat_/PitStop_cat_ may add a
              little signal beyond his scheme.
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
from src.features import (
    yekenot_fe_fit, yekenot_fe_transform,
    yekenot_feature_lists, yekenot_feature_lists_for_tabpfn,
)
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


def quick_diagnose(gpu: bool):
    """10-min diagnostic: run two short fits on 20K subsample to find the cause
    of v22.100/v22.200's prior-collapse failure (loss flat at 0.499).

    Test A: lr=2e-4 (10x smaller than karltonkxb default), keep all 5 cats.
            If loss DROPS → LR was too aggressive for our data shape.
    Test B: default lr=2e-3, drop Driver (cardinality 871) from cats.
            If loss DROPS → Driver cardinality was still the issue.

    Each test: 3 epochs only (~3-4 min). Total ~10 min.
    Auto-verdict: COLLAPSED if all 3 epochs ≥ 0.5005, else LEARNING.
    """
    from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier

    print("=" * 70)
    print("QUICK DIAGNOSE — find cause of v22.100/v22.200 prior-collapse")
    print("=" * 70)
    pool = load_train_pool()
    rng = np.random.default_rng(MODEL_SEED)
    idx = rng.choice(len(pool), size=20000, replace=False)
    pool_sub = pool.iloc[idx].reset_index(drop=True)

    pool_fe, fe_state = yekenot_fe_fit(pool_sub)
    numeric_feats, cat_feats_full = yekenot_feature_lists_for_tabpfn(fe_state)
    y = pool_fe[TARGET].astype(int).to_numpy()

    results = {}
    for label, (lr, cats) in [
        ("A: lr=2e-4, all 5 cats",   (2e-4, cat_feats_full)),
        ("B: lr=2e-3, NO Driver",    (2e-3, [c for c in cat_feats_full if c != "Driver"])),
    ]:
        print(f"\n--- {label} ---")
        cols = numeric_feats + cats
        # If Driver dropped from cats, add it to numerics (raw codes) so info isn't lost
        if "Driver" not in cats and "Driver" not in numeric_feats:
            # Driver is a string; can't go into numeric block raw. Use _Driver_count instead
            # (already in numerics via yekenot count features). Just skip Driver entirely.
            pass
        X = pool_fe[cols].copy()
        cat_idx = [X.columns.get_loc(c) for c in cats if c in X.columns]
        print(f"  features: {len(cols)} ({len(numeric_feats)} num + {len(cats)} cat)")
        print(f"  lr={lr}, cat_indices={cat_idx}")

        clf = FinetunedTabPFNClassifier(
            device="cuda" if gpu else "cpu",
            epochs=3,
            learning_rate=lr,
            n_estimators_finetune=1,
            n_estimators_validation=1,
            n_estimators_final_inference=1,
            random_state=MODEL_SEED,
            eval_metric="log_loss",
            extra_classifier_kwargs={
                "categorical_features_indices": cat_idx,
                "ignore_pretraining_limits": True,
                "tuning_config": {"tune_decision_thresholds": True},
            },
        )
        t0 = time.time()
        clf.fit(X, y)
        dt = time.time() - t0
        print(f"  fit in {dt:.0f}s")
        # We can't easily grab per-epoch losses from clf — but we can do quick AUC on subset
        # Use the last 2K of the subsample as a mini-validation
        from sklearn.metrics import roc_auc_score
        val_X = X.iloc[-2000:]; val_y = y[-2000:]
        proba = clf.predict_proba(val_X)[:, 1]
        auc = roc_auc_score(val_y, proba)
        unique_preds = len(np.unique(np.round(proba, 4)))
        verdict = "★ LEARNING" if auc > 0.55 else "✗ COLLAPSED"
        print(f"  mini-val AUC: {auc:.4f}  unique preds (rounded 4dp): {unique_preds}  → {verdict}")
        results[label] = {"auc": float(auc), "unique_preds": int(unique_preds), "verdict": verdict}

    print("\n" + "=" * 70)
    print("VERDICT TABLE")
    print("=" * 70)
    for label, r in results.items():
        print(f"  {label:<25}  AUC={r['auc']:.4f}  uniq={r['unique_preds']:>5}  {r['verdict']}")
    print()
    a_ok = results["A: lr=2e-4, all 5 cats"]["auc"] > 0.55
    b_ok = results["B: lr=2e-3, NO Driver"]["auc"] > 0.55
    if a_ok and b_ok:
        print("→ BOTH learn → safest: lr=2e-4 + Driver-as-cat. Full run is ~50 min.")
    elif a_ok and not b_ok:
        print("→ Only A learns → LR is the issue. Use lr=2e-4 for full run.")
    elif b_ok and not a_ok:
        print("→ Only B learns → Driver cardinality is the issue. Drop Driver from cats.")
    else:
        print("→ Neither learns → deeper issue (TabPFN-data incompatibility or hyperparam).")
        print("  Pivot recommendation: try plain TabPFNClassifier (no finetuning) with subsample bagging.")


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

    # IMPORTANT: TabPFN-friendly feature split. The default yekenot_feature_lists
    # produces 18 categoricals (incl. floor-factorize cats with cardinality up to
    # thousands), which TabPFN cannot embed → model collapses to predicting the
    # marginal prior (observed v22.100 attempt: holdout AUC 0.50009).
    # _for_tabpfn keeps only Driver/Compound/Race/Year_cat_/PitStop_cat_ as cats
    # (cardinalities 887/5/26/4/2) and promotes high-card factorize codes to numeric.
    numeric_feats, cat_feats = yekenot_feature_lists_for_tabpfn(fe_state)
    feature_cols = numeric_feats + cat_feats
    print(f"feature_cols ({len(feature_cols)}): {len(numeric_feats)} numeric + {len(cat_feats)} categorical")
    print(f"  cats: {cat_feats}  (cardinalities should all be ≤ ~30 except Driver)")
    for c in cat_feats:
        print(f"    {c}: {pool_fe[c].nunique()} unique")

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
            version="v22_tabpfn_simplified",
            parent="v14_yekenot_repro",
            hypothesis=(
                "v22.100 attempt with full yekenot FE collapsed to AUC 0.50 — 13 "
                "floor-factorize cats (cardinality up to ~4000) broke TabPFN's "
                "embedding tables. Simplified FE: 15 numerics (8 raw + 2 yekenot "
                "arith + 5 count encodings) + 5 low-card cats (Driver/Compound/"
                "Race/Year_cat_/PitStop_cat_). Matches karltonkxb's working scheme "
                "(LB 0.94922) plus yekenot's count-encoding extras. Predicted solo: "
                "0.949-0.952. Blend lift vs safar1_95449 only realistic if solo ≥ 0.952."
            ),
            predicted_delta=0.0,
            confidence="medium",
            feature_changes=[
                "+ simplified yekenot FE (dropped 13 floor-factorize cats + 2 KBins)",
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
    out_dir = PROBS / "v22_tabpfn_simplified"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "holdout.npy", holdout_pred)
    np.save(out_dir / "test.npy", test_pred)
    print(f"\nSaved holdout/test probs to {out_dir}/")

    sub = pd.DataFrame({ID: test_fe[ID].to_numpy(), TARGET: test_pred})
    sub_path = SUBMISSIONS / "v22.200.csv"  # .200 = simplified-FE retry; .2XX for mixes
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
            mix_path = SUBMISSIONS / "v22.201.csv"
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
    p.add_argument("--quick-diagnose", action="store_true",
                   help="10-min diagnostic: two short fits to find cause of prior-collapse")
    args = p.parse_args()
    if args.quick_diagnose:
        # Token still required for model download
        if not load_tabpfn_token():
            print("ERROR: TABPFN_TOKEN not found. See header for setup."); sys.exit(1)
        quick_diagnose(args.gpu)
    else:
        main(args.subsample, args.gpu, args.with_original)
