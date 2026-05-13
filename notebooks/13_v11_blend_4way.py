"""v11 — Weighted 4-way blend (v9 RealMLP big + v10 TabM_D + v1 LGB + v4 XGB).

NO new training. Tests if architectural diversity within pytabkit (RealMLP vs
TabM_D) gives enough decorrelation for a meaningful blend lift over v9 solo.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: v9 is current best solo (LB 0.94757). v10 TabM_D underperformed
              solo by 0.001 but is a fundamentally different neural architecture.
              KEY DIAGNOSTIC: ρ(v9, v10) — if < 0.95, decorrelated errors
              compound; if > 0.97, redundant. Trees (v1, v4) also added in case
              they bring marginal diversity.
  Parent: v9_realmlp_big (current best solo)
  Predicted Δ holdout: +0.0005 to +0.002 (depends on ρ(v9, v10))
  Reports: ALL pairwise correlations + simple-avg + Nelder-Mead optimal
  Output: probs/v11_blend_4way/{oof,holdout,test}.npy + submissions/v11_blend_4way.csv

Usage:
  python notebooks/13_v11_blend_4way.py
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

CONTRIBUTORS = ["v9_realmlp_big", "v10_tabm", "v1_lgb", "v4_xgb"]
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
    print("v11 — Weighted 4-way blend (v9 + v10 + v1 + v4)")
    print("=" * 70)

    ensure_probs_present()

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    contribs = {name: load_contributor(name) for name in CONTRIBUTORS}

    print("\n--- Per-model AUC ---")
    for name, c in contribs.items():
        print(f"{name:18s}  OOF: {auc(y_pool, c['oof']):.5f}   holdout: {auc(y_holdout, c['holdout']):.5f}")

    print("\n--- Pairwise correlations (OOF) ---")
    names = list(contribs.keys())
    pairwise_rho = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            r = float(np.corrcoef(contribs[a]["oof"], contribs[b]["oof"])[0, 1])
            pairwise_rho[f"{a}__{b}"] = r
            note = " ← KEY (neural-neural)" if (a, b) == ("v9_realmlp_big", "v10_tabm") or (b, a) == ("v9_realmlp_big", "v10_tabm") else ""
            print(f"  ρ({a:18s}, {b:18s}) = {r:.5f}{note}")

    # Simple 4-way average
    avg_oof = np.mean([c["oof"] for c in contribs.values()], axis=0)
    avg_hold = np.mean([c["holdout"] for c in contribs.values()], axis=0)
    avg_test = np.mean([c["test"] for c in contribs.values()], axis=0)
    avg_oof_auc = auc(y_pool, avg_oof)
    avg_hold_auc = auc(y_holdout, avg_hold)
    print(f"\n--- Simple 4-way average (weights = [0.25, 0.25, 0.25, 0.25]) ---")
    print(f"  OOF AUC:     {avg_oof_auc:.5f}")
    print(f"  Holdout AUC: {avg_hold_auc:.5f}")

    # Nelder-Mead optimization
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

    best_result = None
    best_oof_auc = -np.inf
    for seed_start in [
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.5, 0.3, 0.1, 0.1]),  # favor v9
        np.array([0.7, 0.1, 0.1, 0.1]),  # heavily favor v9
        np.array([0.4, 0.4, 0.1, 0.1]),  # favor neural
        np.array([0.1, 0.1, 0.4, 0.4]),  # favor trees
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
        print(f"  {name:18s}  {w:.4f}")

    opt_oof = oof_stack @ w_optimal
    opt_hold = hold_stack @ w_optimal
    opt_test = test_stack @ w_optimal
    opt_oof_auc = auc(y_pool, opt_oof)
    opt_hold_auc = auc(y_holdout, opt_hold)
    print(f"  OOF AUC:     {opt_oof_auc:.5f}")
    print(f"  Holdout AUC: {opt_hold_auc:.5f}")

    # Pick better blend by holdout
    if opt_hold_auc > avg_hold_auc:
        choice = "weighted"
        chosen_oof, chosen_hold, chosen_test = opt_oof, opt_hold, opt_test
        chosen_oof_auc, chosen_hold_auc = opt_oof_auc, opt_hold_auc
        weights_used = list(map(float, w_optimal))
    else:
        choice = "simple_average"
        chosen_oof, chosen_hold, chosen_test = avg_oof, avg_hold, avg_test
        chosen_oof_auc, chosen_hold_auc = avg_oof_auc, avg_hold_auc
        weights_used = [0.25] * 4

    print(f"\n--- CHOSEN: {choice} blend ---")
    print(f"  Holdout AUC: {chosen_hold_auc:.5f}")
    print(f"  vs v9 solo holdout (0.94808): Δ = {chosen_hold_auc - 0.94808:+.5f}")
    print(f"  vs v8 prev blend (0.94803):   Δ = {chosen_hold_auc - 0.94803:+.5f}")

    exp = Experiment.start(
        version="v11_blend_4way",
        parent="v9_realmlp_big",
        hypothesis=(
            "4-way blend of v9 RealMLP big + v10 TabM_D + v1 LGB + v4 XGB. "
            "Tests architectural diversity within pytabkit. Predicted Δ vs v9 solo: "
            "+0.0005 to +0.002 depending on ρ(v9, v10)."
        ),
        predicted_delta=0.0010,
        confidence="medium",
        feature_changes=[],
        config_changes={"blend_method": choice, "weights": str(weights_used)},
        pipeline_changes=["+ 4-way Nelder-Mead blend"],
        cloud_or_local="local" if not DRIVE_PROBS_DEFAULT.exists() else "cloud",
    )

    out_dir = PROBS / "v11_blend_4way"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", chosen_oof)
    np.save(out_dir / "holdout.npy", chosen_hold)
    np.save(out_dir / "test.npy", chosen_test)
    print(f"\nSaved blend probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: chosen_test})
    sub_path = SUBMISSIONS / "v11_blend_4way.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

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
    print(f"\nExperiment v11_blend_4way committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    main()
