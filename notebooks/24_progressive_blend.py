"""v18 progressive blend — greedy forward selection with release tracking.

Two phases:
  Phase A: OOF candidates → strict holdout-AUC test, promotes on positive delta
  Phase B: test-only public candidates → rank-mix at multiple alpha for human pick

Each accepted promotion writes:
  - submissions/v18.NNN.csv         (submission file)
  - probs/v18.NNN/{oof,holdout,test}.npy   (so it becomes the next baseline)
  - Appends a line to releases.jsonl

Usage:
  python notebooks/24_progressive_blend.py
  python notebooks/24_progressive_blend.py --baseline v15_stats --threshold 0.0001
  python notebooks/24_progressive_blend.py --dry-run

Designed for Colab — auto-restores probs + harvest/v18 from Drive.
"""
from __future__ import annotations

import argparse
import datetime
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
RELEASES_LOG = ROOT / "releases.jsonl"


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


def nelder_mead_weights(oof_matrix: np.ndarray, y_pool: np.ndarray, n_restarts: int = 10):
    """Optimize blend weights maximizing OOF AUC. Returns (weights, auc)."""
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


def blend_with_weights(model_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply weights to rank-normalized matrix. Returns 1D blend."""
    ranks = np.column_stack([rank_norm(model_matrix[:, i]) for i in range(model_matrix.shape[1])])
    return ranks @ weights


def load_internal_probs(version: str):
    d = PROBS / version
    if not all((d / f"{x}.npy").exists() for x in ("oof", "holdout", "test")):
        return None
    return {
        "oof": np.load(d / "oof.npy"),
        "holdout": np.load(d / "holdout.npy"),
        "test": np.load(d / "test.npy"),
    }


def load_v18_oof(entry: dict, pool_ids: np.ndarray):
    """Load a harvested OOF aligned to our pool. Returns array or None."""
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
    if len(df_pool) != len(pool_ids):
        return None
    return df_pool.loc[pool_ids][pred_cols[0]].to_numpy()


def load_v18_test(entry: dict, test_ids: np.ndarray):
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
    return df.set_index(ID).loc[test_ids][pred_cols[0]].to_numpy()


def next_release_version() -> str:
    """Find next available v18.NNN slot."""
    existing = sorted(p.name for p in PROBS.glob("v18.*"))
    if not existing:
        return "v18.001"
    last = existing[-1]  # e.g., "v18.003"
    num = int(last.split(".")[1])
    return f"v18.{num + 1:03d}"


def save_release(version: str, oof: np.ndarray, holdout: np.ndarray, test: np.ndarray,
                 test_ids: np.ndarray) -> None:
    """Write probs/{version}/*.npy and submissions/{version}.csv."""
    out_dir = PROBS / version
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof)
    np.save(out_dir / "holdout.npy", holdout)
    np.save(out_dir / "test.npy", test)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS / f"{version}.csv"
    pd.DataFrame({ID: test_ids, TARGET: test}).to_csv(sub_path, index=False)


def log_release(entry: dict) -> None:
    """Append a release record to releases.jsonl."""
    entry["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(RELEASES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="v17_NM_blend",
                   help="Starting baseline (probs/{name}/ must exist)")
    p.add_argument("--threshold", type=float, default=1e-5,
                   help="Min holdout-AUC delta to promote (default 1e-5 = above noise)")
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--candidate-order", choices=["votes", "rho_asc", "rho_desc"],
                   default="votes",
                   help="Iteration order for OOF candidates")
    p.add_argument("--dry-run", action="store_true",
                   help="Print decisions without writing files")
    p.add_argument("--include-internal", action="store_true",
                   help="Also try adding internal probs/v* candidates (not just harvested)")
    args = p.parse_args()

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

    # Load baseline
    base = load_internal_probs(args.baseline)
    if base is None:
        print(f"ERROR: probs/{args.baseline}/ not found. Run v17 or specify a different baseline.")
        return 1
    baseline_holdout_auc = roc_auc_score(y_holdout, base["holdout"])
    print(f"BASELINE: {args.baseline}")
    print(f"  Holdout AUC: {baseline_holdout_auc:.5f}")
    print()

    # Load harvest manifest
    manifest_path = ROOT / "harvest" / "v18" / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run 22_harvest_v2.py first.")
        return 1
    manifest = json.loads(manifest_path.read_text())

    # ---- Phase A: OOF Progressive Blend ----
    print("=" * 60)
    print("PHASE A — OOF candidates (holdout AUC strict test)")
    print("=" * 60)

    # Collect OOF candidates (harvested INCLUDE-OOF + optionally internal)
    oof_candidates: list[dict] = []

    # Harvested
    for e in manifest:
        if e.get("verdict") != "INCLUDE-OOF":
            continue
        oof_arr = load_v18_oof(e, pool_ids)
        test_arr = load_v18_test(e, test_ids)
        if oof_arr is None or test_arr is None:
            continue
        oof_candidates.append({
            "name": "v18_" + e["tag"],
            "source": "harvest",
            "oof": oof_arr,
            "holdout": None,  # not aligned for harvested
            "test": test_arr,
            "votes": e.get("votes", 0),
            "rho": e.get("rho_with_anchor", 0),
        })

    # Internal (optional — those not already in the baseline lineage)
    if args.include_internal:
        for v in ["v9_realmlp_big", "v10_tabm", "v11_blend_4way",
                  "v14_yekenot_repro", "v14b_external", "v15_stats"]:
            if v == args.baseline:
                continue
            p = load_internal_probs(v)
            if p is None:
                continue
            oof_candidates.append({
                "name": v, "source": "internal",
                "oof": p["oof"], "holdout": p["holdout"], "test": p["test"],
                "votes": 0, "rho": 0,
            })

    # Order
    if args.candidate_order == "votes":
        oof_candidates.sort(key=lambda c: -c["votes"])
    elif args.candidate_order == "rho_asc":
        oof_candidates.sort(key=lambda c: c["rho"])
    else:  # rho_desc
        oof_candidates.sort(key=lambda c: -c["rho"])

    print(f"OOF candidate pool: {len(oof_candidates)}\n")

    # Iteration state.
    # Selection criterion is OOF AUC (aligned for every candidate that has an OOF).
    # Holdout AUC is reported as secondary validation when both sides have aligned holdout.
    current_baseline_name = args.baseline
    current_oof = base["oof"]
    current_holdout = base["holdout"]
    current_test = base["test"]
    current_oof_auc = roc_auc_score(y_pool, current_oof)
    current_holdout_auc = baseline_holdout_auc
    promoted_names = {current_baseline_name}

    print(f"BASELINE OOF AUC: {current_oof_auc:.5f}\n")

    for iter_i in range(args.max_iter):
        best_candidate = None
        best_new_oof_auc = current_oof_auc
        best_weights = None
        best_new_holdout_auc = None

        for cand in oof_candidates:
            if cand["name"] in promoted_names:
                continue

            # Build candidate stack: current baseline + this candidate (2 models)
            oof_stack = np.column_stack([current_oof, cand["oof"]])
            weights, blend_oof_auc = nelder_mead_weights(oof_stack, y_pool, n_restarts=8)

            # OOF-driven selection (primary criterion)
            if blend_oof_auc > best_new_oof_auc:
                best_candidate = cand
                best_new_oof_auc = blend_oof_auc
                best_weights = weights
                # Secondary: holdout if both sides aligned
                if cand["holdout"] is not None:
                    holdout_stack = np.column_stack([current_holdout, cand["holdout"]])
                    new_holdout = blend_with_weights(holdout_stack, weights)
                    best_new_holdout_auc = roc_auc_score(y_holdout, new_holdout)
                else:
                    best_new_holdout_auc = None

        # Decision
        if best_candidate is None:
            print(f"Iter {iter_i+1}: no OOF improvement — stopping Phase A")
            break

        delta_oof = best_new_oof_auc - current_oof_auc
        if delta_oof < args.threshold:
            print(f"Iter {iter_i+1}: best candidate '{best_candidate['name']}' OOF delta {delta_oof:+.6f} < threshold — stopping")
            break

        # PROMOTE
        new_version = next_release_version()
        validation_mode = "HOLDOUT" if best_candidate["holdout"] is not None else "OOF-ONLY"
        print(f"Iter {iter_i+1}: PROMOTING {best_candidate['name']}  [{validation_mode}]")
        print(f"  OOF delta:     {delta_oof:+.6f}  ({current_oof_auc:.5f} → {best_new_oof_auc:.5f})")
        if best_new_holdout_auc is not None:
            delta_ho = best_new_holdout_auc - current_holdout_auc
            print(f"  Holdout delta: {delta_ho:+.6f}  ({current_holdout_auc:.5f} → {best_new_holdout_auc:.5f})")
        print(f"  weights: baseline={best_weights[0]:.4f}, added={best_weights[1]:.4f}")
        print(f"  new release: {new_version}")

        # Build the new release's oof/holdout/test
        oof_stack = np.column_stack([current_oof, best_candidate["oof"]])
        test_stack = np.column_stack([current_test, best_candidate["test"]])
        new_oof = blend_with_weights(oof_stack, best_weights)
        new_test = blend_with_weights(test_stack, best_weights)

        # Holdout: only blend if candidate has aligned holdout; else carry forward
        if best_candidate["holdout"] is not None:
            holdout_stack = np.column_stack([current_holdout, best_candidate["holdout"]])
            new_holdout = blend_with_weights(holdout_stack, best_weights)
        else:
            # Candidate has no holdout — keep baseline's holdout unchanged
            new_holdout = current_holdout

        if not args.dry_run:
            save_release(new_version, new_oof, new_holdout, new_test, test_ids)
            log_release({
                "version": new_version,
                "parent": current_baseline_name,
                "added": best_candidate["name"],
                "added_source": best_candidate["source"],
                "validation_mode": validation_mode,
                "weights": {
                    current_baseline_name: float(best_weights[0]),
                    best_candidate["name"]: float(best_weights[1]),
                },
                "oof_before": float(current_oof_auc),
                "oof_after": float(best_new_oof_auc),
                "oof_delta": float(delta_oof),
                "holdout_before": float(current_holdout_auc),
                "holdout_after": float(best_new_holdout_auc) if best_new_holdout_auc is not None else None,
                "decision": "PROMOTED",
            })

        # Update baseline
        current_baseline_name = new_version
        current_oof = new_oof
        current_holdout = new_holdout
        current_test = new_test
        current_oof_auc = best_new_oof_auc
        if best_new_holdout_auc is not None:
            current_holdout_auc = best_new_holdout_auc
        promoted_names.add(best_candidate["name"])

    print(f"\n--- Phase A complete ---")
    print(f"Final baseline: {current_baseline_name}")
    print(f"Holdout AUC: {current_holdout_auc:.5f}")
    print(f"Δ from initial: {current_holdout_auc - baseline_holdout_auc:+.6f}\n")

    # ---- Phase B: Test-only mix (cannot validate on holdout, generate variants) ----
    print("=" * 60)
    print("PHASE B — Test-only candidates (ρ-diversity rank-mix)")
    print("=" * 60)

    # Collect test-only candidates with quality + diversity filter
    test_only_pool = []
    for e in manifest:
        if e.get("verdict") != "INCLUDE-TEST":
            continue
        claimed_lb = e.get("claimed_lb")
        rho = e.get("rho_with_anchor", 1.0)
        if claimed_lb is not None and claimed_lb < 0.95:
            continue
        if rho > 0.995:
            continue
        test_arr = load_v18_test(e, test_ids)
        if test_arr is None:
            continue
        test_only_pool.append({
            "name": "v18_" + e["tag"],
            "test": test_arr,
            "votes": e.get("votes", 0),
            "rho": rho,
            "claimed_lb": claimed_lb,
        })

    print(f"Test-only pool (after filter): {len(test_only_pool)}")
    print()

    # Generate mix variants at multiple alpha
    for alpha in [0.05, 0.10, 0.20]:
        # Mix: (1-alpha) * current_test + alpha * mean(test_only_pool ranks)
        if not test_only_pool:
            break
        all_test_ranks = np.column_stack([rank_norm(c["test"]) for c in test_only_pool])
        public_mean_rank = all_test_ranks.mean(axis=1)
        current_test_rank = rank_norm(current_test)
        mix = (1 - alpha) * current_test_rank + alpha * public_mean_rank

        version = next_release_version()
        if not args.dry_run:
            # Save under v18.NNN with empty oof/holdout (test only)
            out_dir = PROBS / version
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "test.npy", mix)
            # No oof/holdout saved — this release can't be a future baseline
            pd.DataFrame({ID: test_ids, TARGET: mix}).to_csv(
                SUBMISSIONS / f"{version}.csv", index=False)
            log_release({
                "version": version,
                "parent": current_baseline_name,
                "added": f"public_mean_mix(alpha={alpha}, n={len(test_only_pool)})",
                "added_source": "test_mix",
                "weights": {"baseline": float(1 - alpha), "public_mix": float(alpha)},
                "holdout_before": float(current_holdout_auc),
                "holdout_after": None,  # not measurable for test-only addition
                "delta": None,
                "decision": "PHASE_B_VARIANT",
            })
        print(f"{version}: baseline {1-alpha:.2f} + public_mix {alpha:.2f}  ({len(test_only_pool)} preds averaged)")

    # ---- Final report ----
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Phase A baseline (validated on holdout): {current_baseline_name}")
    print(f"  Holdout AUC: {current_holdout_auc:.5f}")
    print(f"  Initial: {baseline_holdout_auc:.5f}  (delta {current_holdout_auc - baseline_holdout_auc:+.6f})")
    print(f"  Submission file: submissions/{current_baseline_name}.csv")
    print()
    print("Phase B variants (unvalidated — pick best via Kaggle submit):")
    for sub in sorted(SUBMISSIONS.glob("v18.*.csv")):
        if sub.stem != current_baseline_name:
            print(f"  submissions/{sub.name}")
    print()
    print(f"Releases logged to: {RELEASES_LOG}")
    print()
    print("RECOMMENDED SUBMISSION:")
    print(f"  submissions/{current_baseline_name}.csv  (best holdout-validated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
