"""v5 — Simple-average blend of v1 LGB + v4 XGB.

NO new training. Reads probs/v1_lgb/{oof,holdout,test}.npy and
probs/v4_xgb/{oof,holdout,test}.npy, simple-averages them, saves blended
submission.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: v1 and v4 are TIED solo (all metrics within 0.0003) but use
              fundamentally different decision algorithms (LGB Fisher splits
              vs XGB ordinal cat). If their OOF correlation < 0.97, simple
              averaging will compound decorrelated errors and lift AUC.
              If ρ > 0.99, blend won't help (L13 lesson from S6E4 audit).
  Parent: v1_lgb (the marginally-better solo on LB)
  Predicted Δ holdout: +0.0008 (mid of [0.0005, 0.0015])
  Most relevant pitfall: #9 (model-family diversity IS valid; #10 (no version
                              explosion — keep solo models cached)
  Validation plan: report OOF correlation FIRST, then blend AUC on OOF/holdout
  Abort signal: if ρ > 0.99 between v1 and v4 OOF, blend won't help meaningfully

If running on Colab: pulls probs from Drive (s6e5/probs/) into local probs/.
If running locally: requires probs/v1_lgb/ and probs/v4_xgb/ to already exist.

Usage:
  python notebooks/07_v5_blend_lgb_xgb.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import PROBS, SUBMISSIONS, TARGET, ID
from src.data import load_train_pool, load_holdout, load_test
from src.evaluate import auc
from src.observer import Experiment

CONTRIBUTORS = ["v1_lgb", "v4_xgb"]
DRIVE_PROBS_DEFAULT = Path("/content/drive/MyDrive/Colab Notebooks/kaggle/s6e5/probs")


def ensure_probs_present():
    """If running on Colab and probs/<contributor>/ missing, sync from Drive."""
    missing = []
    for name in CONTRIBUTORS:
        local_dir = PROBS / name
        needed_files = ["oof.npy", "holdout.npy", "test.npy"]
        if not all((local_dir / f).exists() for f in needed_files):
            missing.append(name)

    if not missing:
        print(f"All contributor probs present locally: {CONTRIBUTORS}")
        return

    if DRIVE_PROBS_DEFAULT.exists():
        print(f"Syncing missing probs from Drive: {missing}")
        for name in missing:
            src = DRIVE_PROBS_DEFAULT / name
            dst = PROBS / name
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"  copied {src} → {dst}")
            else:
                print(f"  MISSING on Drive: {src}")
                sys.exit(2)
    else:
        print(f"\nERROR: probs missing locally and Drive not mounted.")
        print(f"  Missing: {missing}")
        print(f"  Expected at: probs/<contributor>/<oof|holdout|test>.npy")
        print(f"  Or run on Colab with Drive mounted at {DRIVE_PROBS_DEFAULT}")
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
    print("v5 — Simple-average blend (v1 LGB + v4 XGB)")
    print("=" * 70)

    ensure_probs_present()

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    v1 = load_contributor("v1_lgb")
    v4 = load_contributor("v4_xgb")

    # ---- Diagnostics: correlation + per-model AUC reconfirmation
    print("\n--- Per-model AUC reconfirmation ---")
    print(f"v1_lgb  OOF AUC: {auc(y_pool, v1['oof']):.5f}   holdout AUC: {auc(y_holdout, v1['holdout']):.5f}")
    print(f"v4_xgb  OOF AUC: {auc(y_pool, v4['oof']):.5f}   holdout AUC: {auc(y_holdout, v4['holdout']):.5f}")

    rho_oof = float(np.corrcoef(v1["oof"], v4["oof"])[0, 1])
    rho_test = float(np.corrcoef(v1["test"], v4["test"])[0, 1])
    print(f"\n--- OOF correlation ρ(v1, v4) ---")
    print(f"  on OOF predictions:  {rho_oof:.5f}")
    print(f"  on test predictions: {rho_test:.5f}")

    if rho_oof > 0.99:
        print(f"\n⚠ WARNING: ρ > 0.99 — predictions nearly identical, blend won't help (L13 lesson)")
    elif rho_oof > 0.97:
        print(f"\n⚠ ρ > 0.97 — minimal blend gain expected")
    else:
        print(f"\n✓ ρ < 0.97 — meaningful diversity, blend should lift AUC")

    # ---- Simple average blend
    blend_oof = (v1["oof"] + v4["oof"]) / 2
    blend_hold = (v1["holdout"] + v4["holdout"]) / 2
    blend_test = (v1["test"] + v4["test"]) / 2

    blend_oof_auc = auc(y_pool, blend_oof)
    blend_hold_auc = auc(y_holdout, blend_hold)

    print(f"\n--- Blend results ---")
    print(f"Blend OOF AUC:     {blend_oof_auc:.5f}")
    print(f"Blend holdout AUC: {blend_hold_auc:.5f}")
    print()
    print(f"vs v1 holdout 0.94379:  Δ = {blend_hold_auc - 0.94379:+.5f}")
    print(f"vs v4 holdout 0.94375:  Δ = {blend_hold_auc - 0.94375:+.5f}")
    print(f"vs best single solo:    Δ = {blend_hold_auc - max(0.94379, 0.94375):+.5f}")

    # ---- Observer
    exp = Experiment.start(
        version="v5_blend_lgb_xgb",
        parent="v1_lgb",   # marginally better solo on LB
        hypothesis=(
            "Simple average of v1 LGB + v4 XGB. Both tied solo (all metrics within "
            "0.0003). Different decision algorithms (LGB Fisher vs XGB ordinal) → "
            "if OOF correlation < 0.97, decorrelated errors compound. Predicted "
            "Δ holdout vs best single solo: +0.0008 (mid of [0.0005, 0.0015])."
        ),
        predicted_delta=0.0008,
        confidence="medium",
        feature_changes=[],
        config_changes={"blend_method": "simple_average", "weights": "0.5/0.5"},
        pipeline_changes=["+ multi-model blend (v1_lgb + v4_xgb)"],
        cloud_or_local="local" if not DRIVE_PROBS_DEFAULT.exists() else "cloud",
    )

    # ---- Save artifacts
    out_dir = PROBS / "v5_blend_lgb_xgb"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", blend_oof)
    np.save(out_dir / "holdout.npy", blend_hold)
    np.save(out_dir / "test.npy", blend_test)
    print(f"\nSaved blend probs to {out_dir}/")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID].astype("int64"), TARGET: blend_test})
    sub_path = SUBMISSIONS / "v5_blend_lgb_xgb.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

    exp.record(
        oof_auc_mean=float(blend_oof_auc),
        oof_auc_per_fold=[float(blend_oof_auc)] * 5,  # blend is post-fold, no per-fold meaning
        holdout_auc=float(blend_hold_auc),
        runtime_sec=0.0,
        extra={
            "contributors": CONTRIBUTORS,
            "rho_oof": rho_oof,
            "rho_test": rho_test,
            "blend_method": "simple_average",
            "weights": [0.5, 0.5],
        },
    )
    exp.commit()
    print(f"\nExperiment v5_blend_lgb_xgb committed.")
    print(f"Flags: {exp.flags or '(none)'}")


if __name__ == "__main__":
    main()
