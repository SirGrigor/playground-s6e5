"""v8 — Weighted 3-way blend (v1 LGB + v4 XGB + v7 RealMLP).

NO new training. Reads contributor probs, computes pairwise correlations,
runs Nelder-Mead optimization on holdout to find optimal weights, AND
compares against simple average. Picks the better of the two and saves
that submission.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: v7 RealMLP solo holdout AUC = 0.9479 (best). v1 LGB = 0.9438,
              v4 XGB = 0.9437. Tree-tree blend (v5) lifted only +0.0003
              because ρ(v1, v4) = 0.989. NN vs tree should give ρ in the
              0.85-0.92 range — meaningful diversity. Optimal weights
              will likely favor v7 heavily but trees should still contribute
              non-zero weight if their errors are decorrelated.
  Parent: v7_realmlp (current best solo)
  Predicted Δ holdout: +0.001 to +0.003
  Reports: pairwise ρ, simple-average AUC, Nelder-Mead-optimized AUC,
           plus the optimal weight vector
  Output: probs/v8_blend_3way/{oof,holdout,test}.npy + submissions/v8_blend_3way.csv

Reuses sync logic from v5 — if probs/<contributor>/ missing, pulls from
Drive (Colab) or errors helpfully (local).

Usage:
  python notebooks/10_v8_blend_3way.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import PROBS, SUBMISSIONS, TARGET, ID
from src.data import load_train_pool, load_holdout, load_test
from src.evaluate import auc
from src.observer import Experiment

CONTRIBUTORS = ["v1_lgb", "v4_xgb", "v7_realmlp"]
DRIVE_PROBS_DEFAULT = Path("/content/drive/MyDrive/Colab Notebooks/kaggle/s6e5/probs")


def ensure_probs_present():
    missing = []
    for name in CONTRIBUTORS:
        local_dir = PROBS / name
        if not all((local_dir / f).exists() for f in ["oof.npy", "holdout.npy", "test.npy"]):
            missing.append(name)
    if not missing:
        print(f"All contributor probs present locally: {CONTRIBUTORS}")
        return
    if DRIVE_PROBS_DEFAULT.exists():
        print(f"Syncing missing probs from Drive: {missing}")
        for name in missing:
            src = DRIVE_PROBS_DEFAULT / name
            dst = PROBS / name
            if not src.exists():
                print(f"  MISSING on Drive: {src}")
                sys.exit(2)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  copied {name}")
    else:
        print(f"\nERROR: missing locally and Drive not mounted: {missing}")
        print(f"  Run on Colab with Drive mounted at {DRIVE_PROBS_DEFAULT}")
        sys.exit(2)


def load_contributor(name: str) -> dict:
    d = PROBS / name
    return {
        "oof": np.load(d / "oof.npy"),
        "holdout": np.load(d / "holdout.npy"),
        "test": np.load(d / "test.npy"),
    }


def main():
    print("=" * 70)
    print("v8 — Weighted 3-way blend (v1 LGB + v4 XGB + v7 RealMLP)")
    print("=" * 70)

    ensure_probs_present()

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    contribs = {name: load_contributor(name) for name in CONTRIBUTORS}

    # ---- Per-model AUC reconfirmation
    print("\n--- Per-model AUC ---")
    for name, c in contribs.items():
        print(f"{name:14s}  OOF: {auc(y_pool, c['oof']):.5f}   holdout: {auc(y_holdout, c['holdout']):.5f}")

    # ---- Pairwise correlations
    print("\n--- Pairwise correlations (OOF) ---")
    names = list(contribs.keys())
    for i, a in enumerate(names):
        for b in names[i+1:]:
            r = float(np.corrcoef(contribs[a]["oof"], contribs[b]["oof"])[0, 1])
            print(f"  ρ({a}, {b}) = {r:.5f}")

    # ---- Simple 3-way average
    avg_oof = np.mean([c["oof"] for c in contribs.values()], axis=0)
    avg_hold = np.mean([c["holdout"] for c in contribs.values()], axis=0)
    avg_test = np.mean([c["test"] for c in contribs.values()], axis=0)
    avg_oof_auc = auc(y_pool, avg_oof)
    avg_hold_auc = auc(y_holdout, avg_hold)
    print(f"\n--- Simple 3-way average (weights = [1/3, 1/3, 1/3]) ---")
    print(f"  OOF AUC:     {avg_oof_auc:.5f}")
    print(f"  Holdout AUC: {avg_hold_auc:.5f}")

    # ---- Nelder-Mead weighted blend optimization (on OOF, validated on holdout)
    oof_stack = np.column_stack([c["oof"] for c in contribs.values()])
    hold_stack = np.column_stack([c["holdout"] for c in contribs.values()])
    test_stack = np.column_stack([c["test"] for c in contribs.values()])

    def neg_oof_auc(w):
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 0
        w = w / w.sum()
        blend = oof_stack @ w
        return -auc(y_pool, blend)

    # Multiple restarts for robustness
    best_result = None
    best_oof_auc = -np.inf
    for seed_start in [
        np.array([1/3, 1/3, 1/3]),
        np.array([0.5, 0.2, 0.3]),
        np.array([0.1, 0.1, 0.8]),  # favors v7
    ]:
        res = minimize(neg_oof_auc, seed_start, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 500})
        oof_score = -res.fun
        if oof_score > best_oof_auc:
            best_oof_auc = oof_score
            best_result = res

    w_optimal = np.clip(best_result.x, 0, None)
    w_optimal = w_optimal / w_optimal.sum()
    print(f"\n--- Nelder-Mead optimal weights ---")
    for name, w in zip(names, w_optimal):
        print(f"  {name:14s}  {w:.4f}")
    print(f"  sum: {w_optimal.sum():.4f}  (should be 1.0)")

    opt_oof = oof_stack @ w_optimal
    opt_hold = hold_stack @ w_optimal
    opt_test = test_stack @ w_optimal
    opt_oof_auc = auc(y_pool, opt_oof)
    opt_hold_auc = auc(y_holdout, opt_hold)
    print(f"  OOF AUC:     {opt_oof_auc:.5f}  (vs simple avg {avg_oof_auc:.5f})")
    print(f"  Holdout AUC: {opt_hold_auc:.5f}  (vs simple avg {avg_hold_auc:.5f})")

    # ---- Pick the better blend (by holdout AUC, since we tuned weights on OOF)
    if opt_hold_auc > avg_hold_auc:
        choice = "weighted"
        chosen_oof, chosen_hold, chosen_test = opt_oof, opt_hold, opt_test
        chosen_oof_auc, chosen_hold_auc = opt_oof_auc, opt_hold_auc
        weights_used = list(map(float, w_optimal))
    else:
        choice = "simple_average"
        chosen_oof, chosen_hold, chosen_test = avg_oof, avg_hold, avg_test
        chosen_oof_auc, chosen_hold_auc = avg_oof_auc, avg_hold_auc
        weights_used = [1/3, 1/3, 1/3]

    print(f"\n--- CHOSEN: {choice} blend ---")
    print(f"  Holdout AUC: {chosen_hold_auc:.5f}")
    print(f"  vs v7 solo holdout (0.94792): Δ = {chosen_hold_auc - 0.94792:+.5f}")
    print(f"  vs v5 prev best holdout (0.94412): Δ = {chosen_hold_auc - 0.94412:+.5f}")

    # ---- Observer
    exp = Experiment.start(
        version="v8_blend_3way",
        parent="v7_realmlp",
        hypothesis=(
            f"Weighted 3-way blend of v1 LGB + v4 XGB + v7 RealMLP. v7 solo wins "
            f"(holdout 0.9479) but trees may add decorrelated info. Nelder-Mead on "
            f"OOF finds optimal weights; compare with simple 1/3 average. "
            f"Predicted Δ holdout vs v7 solo: +0.001 to +0.003."
        ),
        predicted_delta=0.0015,
        confidence="medium",
        feature_changes=[],
        config_changes={"blend_method": choice, "weights": str(weights_used)},
        pipeline_changes=["+ 3-way blend with Nelder-Mead weight tuning"],
        cloud_or_local="local" if not DRIVE_PROBS_DEFAULT.exists() else "cloud",
    )

    out_dir = PROBS / "v8_blend_3way"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", chosen_oof)
    np.save(out_dir / "holdout.npy", chosen_hold)
    np.save(out_dir / "test.npy", chosen_test)
    print(f"\nSaved blend probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: chosen_test})
    sub_path = SUBMISSIONS / "v8_blend_3way.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    pairwise_rho = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pairwise_rho[f"{a}__{b}"] = float(np.corrcoef(contribs[a]["oof"], contribs[b]["oof"])[0, 1])

    exp.record(
        oof_auc_mean=float(chosen_oof_auc),
        oof_auc_per_fold=[float(chosen_oof_auc)] * 5,
        holdout_auc=float(chosen_hold_auc),
        runtime_sec=0.0,
        extra={
            "contributors": CONTRIBUTORS,
            "blend_method": choice,
            "weights": weights_used,
            "simple_avg_oof_auc": float(avg_oof_auc),
            "simple_avg_hold_auc": float(avg_hold_auc),
            "weighted_oof_auc": float(opt_oof_auc),
            "weighted_hold_auc": float(opt_hold_auc),
            "pairwise_rho_oof": pairwise_rho,
        },
    )
    exp.commit()
    print(f"\nExperiment v8_blend_3way committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    main()
