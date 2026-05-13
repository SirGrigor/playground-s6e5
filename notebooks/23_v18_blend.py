"""v18 — Blend with bulk-harvested public OOFs + submissions.

Consumes harvest/v18/manifest.json from 22_harvest_v2.py. Auto-loads:
  - All INCLUDE-OOF entries: aligned OOFs → Nelder-Mead matrix
  - All INCLUDE-TEST entries: test predictions for rank-blend mixing
  - Our internal probs (v14b, v15) from probs/

Generates blend candidates and picks the best by holdout AUC.

Designed to run on Colab OR local. Auto-restores from Drive if needed.

Usage:
  python notebooks/23_v18_blend.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ID, PROBS, ROOT, SUBMISSIONS, TARGET
from src.data import load_holdout, load_test, load_train_pool


COLAB_DRIVE_BASE = Path("/content/drive/MyDrive/Colab Notebooks/kaggle/s6e5")


def restore_from_drive_if_needed() -> None:
    """Pull probs/ and harvest/v18/ from Drive if local is empty."""
    drive_probs = COLAB_DRIVE_BASE / "probs"
    if drive_probs.exists():
        PROBS.mkdir(parents=True, exist_ok=True)
        for src_dir in drive_probs.iterdir():
            if not src_dir.is_dir():
                continue
            dst_dir = PROBS / src_dir.name
            if not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)
                print(f"  restored probs/{src_dir.name} from Drive")

    drive_v18 = COLAB_DRIVE_BASE / "harvest" / "v18"
    local_v18 = ROOT / "harvest" / "v18"
    if drive_v18.exists() and not local_v18.exists():
        local_v18.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive_v18, local_v18)
        print(f"  restored harvest/v18 from Drive")


def rank_norm(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def nelder_mead_weights(oof_matrix: np.ndarray, y_pool: np.ndarray, n_restarts: int = 8):
    """Optimize weights maximizing AUC(weighted rank-avg, y). Returns (weights, auc)."""
    n_models = oof_matrix.shape[1]
    ranks = np.column_stack([rank_norm(oof_matrix[:, i]) for i in range(n_models)])

    def neg_auc(w):
        w = np.abs(w)
        w = w / w.sum()
        return -roc_auc_score(y_pool, ranks @ w)

    best_auc, best_w = -np.inf, None
    rng = np.random.default_rng(42)
    for _ in range(n_restarts):
        x0 = rng.dirichlet(np.ones(n_models))
        res = minimize(neg_auc, x0, method="Nelder-Mead",
                       options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
        if -res.fun > best_auc:
            best_auc, best_w = -res.fun, np.abs(res.x) / np.abs(res.x).sum()
    return best_w, best_auc


def load_internal_probs(version: str) -> dict | None:
    """Load probs/{version}/oof+holdout+test.npy if all present."""
    d = PROBS / version
    if not all((d / f"{x}.npy").exists() for x in ("oof", "holdout", "test")):
        return None
    return {
        "oof": np.load(d / "oof.npy"),
        "holdout": np.load(d / "holdout.npy"),
        "test": np.load(d / "test.npy"),
    }


def load_v18_oof(entry: dict, pool_ids: np.ndarray) -> np.ndarray | None:
    """Load a harvested OOF, align to our pool ids, return predictions array.

    Returns None if not loadable / not aligned.
    """
    oof_path_rel = entry.get("files_found", {}).get("oof")
    if oof_path_rel is None:
        return None
    oof_path = ROOT / oof_path_rel
    if not oof_path.exists():
        return None
    try:
        df = pd.read_csv(oof_path)
    except Exception:
        return None
    pred_cols = [c for c in df.columns if c != ID]
    if len(pred_cols) != 1:
        return None
    pool_id_set = set(int(x) for x in pool_ids)
    df_pool = df[df[ID].isin(pool_id_set)].set_index(ID).sort_index()
    pool_ids_sorted = np.sort(pool_ids)
    if not np.array_equal(df_pool.index.to_numpy(), pool_ids_sorted):
        # Not full coverage — author trained on subset of our pool. Skip.
        return None
    # Re-order from sorted-by-id back to pool's actual ordering
    df_orig_order = df_pool.loc[pool_ids]
    return df_orig_order[pred_cols[0]].to_numpy()


def load_v18_test(entry: dict, test_ids: np.ndarray) -> np.ndarray | None:
    """Load a harvested test submission aligned to our test ids."""
    sub_path_rel = entry.get("files_found", {}).get("submission")
    if sub_path_rel is None:
        return None
    sub_path = ROOT / sub_path_rel
    if not sub_path.exists():
        return None
    try:
        df = pd.read_csv(sub_path)
    except Exception:
        return None
    pred_cols = [c for c in df.columns if c != ID]
    if len(pred_cols) != 1:
        return None
    df_aligned = df.set_index(ID).loc[test_ids]
    return df_aligned[pred_cols[0]].to_numpy()


def main() -> int:
    print("=== Drive recovery (if on Colab) ===")
    restore_from_drive_if_needed()
    print()

    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    pool_ids = pool[ID].to_numpy()
    test_ids = test[ID].to_numpy()
    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()
    print(f"pool: {len(pool):,}  holdout: {len(holdout):,}  test: {len(test):,}\n")

    # Load internal probs (v14b, v15 — anything with full probs/ tuple)
    internal_versions = ["v14b_external", "v15_stats", "v14_yekenot_repro",
                         "v9_realmlp_big", "v10_tabm", "v11_blend_4way"]
    internal: dict[str, dict] = {}
    print("=== Internal candidates ===")
    for v in internal_versions:
        p = load_internal_probs(v)
        if p is not None:
            internal[v] = p
            print(f"  ✓ {v:<28}  holdout={roc_auc_score(y_holdout, p['holdout']):.5f}")
        else:
            print(f"  ✗ {v} (missing)")
    print()

    # Load harvest/v18 manifest
    manifest_path = ROOT / "harvest" / "v18" / "manifest.json"
    if not manifest_path.exists():
        print(f"No harvest/v18/manifest.json — run 22_harvest_v2.py first.")
        return 1
    manifest = json.loads(manifest_path.read_text())
    print(f"=== Harvest v18: {len(manifest)} entries ===")

    # Build OOF stack: internal + harvested INCLUDE-OOF
    oof_models: dict[str, np.ndarray] = {}
    holdout_preds: dict[str, np.ndarray] = {}
    test_preds: dict[str, np.ndarray] = {}

    for name, p in internal.items():
        oof_models[name] = p["oof"]
        holdout_preds[name] = p["holdout"]
        test_preds[name] = p["test"]

    # Harvested OOFs aligned to our pool
    harvested_oof_count = 0
    for e in manifest:
        if e.get("verdict") != "INCLUDE-OOF":
            continue
        oof_arr = load_v18_oof(e, pool_ids)
        if oof_arr is None:
            continue
        # OOF available — but we don't have aligned holdout/test for it,
        # so we use it ONLY for NM weight learning, then apply to test via rank-norm
        test_arr = load_v18_test(e, test_ids)
        if test_arr is None:
            continue
        tag = "v18_" + e["tag"]
        oof_models[tag] = oof_arr
        test_preds[tag] = test_arr
        # No aligned holdout — will skip in holdout blend
        holdout_preds[tag] = None
        harvested_oof_count += 1
        print(f"  + {tag} (OOF, votes={e['votes']}, ρ={e.get('rho_with_anchor', 0):.4f})")
    print(f"\nTotal OOF stack: {len(oof_models)} models ({harvested_oof_count} from harvest/v18, {len(internal)} internal)\n")

    # Harvested test-only submissions (for rank-blend mixing later)
    test_only: dict[str, np.ndarray] = {}
    for e in manifest:
        if e.get("verdict") != "INCLUDE-TEST":
            continue
        # Skip if we already loaded it as INCLUDE-OOF
        tag = "v18_" + e["tag"]
        if tag in oof_models:
            continue
        test_arr = load_v18_test(e, test_ids)
        if test_arr is None:
            continue
        test_only[tag] = test_arr
        print(f"  + {tag} (test-only, votes={e['votes']}, ρ={e.get('rho_with_anchor', 0):.4f})")
    print(f"Test-only candidates: {len(test_only)}\n")

    # ---- Nelder-Mead on OOF stack -------------------------------------
    print("=== Nelder-Mead blend on OOF stack ===")
    names = list(oof_models.keys())
    oof_matrix = np.column_stack([oof_models[n] for n in names])
    weights, blend_oof_auc = nelder_mead_weights(oof_matrix, y_pool, n_restarts=10)
    print(f"Blend OOF AUC: {blend_oof_auc:.5f}")
    print("Weights:")
    for n, w in sorted(zip(names, weights, strict=True), key=lambda kv: kv[1], reverse=True):
        if w > 0.001:
            print(f"  {n:<40}  {w:.4f}")

    # Apply same weights to test (rank-normalize each column first)
    n_test = len(test_ids)
    test_ranks = np.column_stack([rank_norm(test_preds[n]) for n in names])
    blend_test = test_ranks @ weights

    # Holdout AUC: only computable on rows where we have all holdout preds
    holdout_avail = [n for n in names if holdout_preds[n] is not None]
    if holdout_avail:
        # Rescale weights to only internal models for holdout estimate
        w_idx = [names.index(n) for n in holdout_avail]
        w_internal = weights[w_idx]
        w_internal = w_internal / w_internal.sum()
        holdout_ranks = np.column_stack([rank_norm(holdout_preds[n]) for n in holdout_avail])
        blend_holdout = holdout_ranks @ w_internal
        blend_holdout_auc = roc_auc_score(y_holdout, blend_holdout)
        print(f"Blend Holdout AUC (internal-only): {blend_holdout_auc:.5f}")
    print()

    # Save the v18 NM blend
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS / "v18_NM_blend.csv"
    pd.DataFrame({ID: test_ids, TARGET: blend_test}).to_csv(sub_path, index=False)
    print(f"  → {sub_path}")

    # ---- Test-only mix candidates -------------------------------------
    if test_only:
        # Mix v18_NM_blend with weighted average of test_only candidates
        all_test_only = np.column_stack([rank_norm(v) for v in test_only.values()])
        test_only_mean = all_test_only.mean(axis=1)
        blend_test_rank = rank_norm(blend_test)
        for alpha, name in [(0.10, "v18_blend_90_pubmix_10"),
                             (0.20, "v18_blend_80_pubmix_20"),
                             (0.30, "v18_blend_70_pubmix_30")]:
            mix = (1 - alpha) * blend_test_rank + alpha * test_only_mean
            pd.DataFrame({ID: test_ids, TARGET: mix}).to_csv(
                SUBMISSIONS / f"{name}.csv", index=False)
            print(f"  → submissions/{name}.csv  (alpha={alpha} to public-test mix of {len(test_only)})")

    # ---- Final report -------------------------------------------------
    print("\n=== Summary ===")
    print(f"OOF stack: {len(names)} models")
    print(f"OOF blend AUC: {blend_oof_auc:.5f}")
    if holdout_avail:
        print(f"Holdout blend AUC (internal-only): {blend_holdout_auc:.5f}")
    print(f"v17 NM blend (prev best): holdout 0.95388, LB 0.95353")
    print(f"\nExpected LB for v18_NM_blend: ~{blend_oof_auc - 0.0007:.5f} to ~{blend_oof_auc - 0.0003:.5f}")
    print("\nCandidates produced:")
    for p in sorted(SUBMISSIONS.glob("v18_*.csv")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
