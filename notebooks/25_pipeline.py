"""v18 master pipeline — pre-flight validation + progressive blend in one script.

Philosophy: validate ALL data prerequisites BEFORE any heavy work. If anything
is missing, abort early with a clear actionable error. No mid-run discoveries.

Pipeline stages:
  0. Pre-flight checks (fast, ~5 sec):
       - Drive mounted
       - probs/ on Drive: at least one valid baseline candidate
       - harvest/v18/ on Drive: manifest.json + expected kernel subdirs
       - For each INCLUDE-OOF entry: oof_preds.csv physically present
       - For each INCLUDE-TEST entry: submission.csv physically present
       ABORT if any check fails. Prints exactly what's missing.

  1. Restore data Drive → /content (only if pre-flight passed)

  2. Auto-detect baseline (highest-holdout among available probs/)

  3. Progressive blend Phase A (greedy forward selection)
       For each iter: try every candidate, promote best, stop when no improvement

  4. Progressive blend Phase B (test-only public mix variants)
       3 variants at α=0.05/0.10/0.20

  5. Sync back to Drive (probs, submissions, releases.jsonl)

  6. Final report — winner, all releases, kaggle submit command

Usage:
  python notebooks/25_pipeline.py
  python notebooks/25_pipeline.py --baseline v18.002   # override auto-detect
  python notebooks/25_pipeline.py --skip-phase-b       # OOF only, no public mix
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

# Internal candidates eligible for the blend (in addition to harvested INCLUDE-OOF)
INTERNAL_CANDIDATES = [
    "v9_realmlp_big", "v10_tabm", "v11_blend_4way",
    "v14_yekenot_repro", "v14b_external", "v15_stats",
    # Also include any v18.NNN releases from a previous run
]


# =========================================================================
# Pre-flight validation
# =========================================================================

class PreflightError(Exception):
    """Raised when a pre-flight check fails. Aborts before any heavy work."""


def preflight_drive_mounted() -> dict:
    """Check Drive is mounted and the s6e5 folder exists."""
    if not COLAB_DRIVE_BASE.exists():
        raise PreflightError(
            f"Drive folder not found: {COLAB_DRIVE_BASE}\n"
            f"  Fix: Run cell 1 (drive.mount). If already mounted, check that the\n"
            f"  s6e5 folder name in DRIVE_S6E5 matches your Drive layout."
        )
    return {"drive_root": str(COLAB_DRIVE_BASE)}


def preflight_probs(min_baselines: int = 1) -> dict:
    """Check there's at least one usable baseline on Drive."""
    drive_probs = COLAB_DRIVE_BASE / "probs"
    if not drive_probs.exists():
        raise PreflightError(
            f"Drive probs/ folder missing: {drive_probs}\n"
            f"  Fix: Make sure earlier runs synced probs to Drive (cell 7)."
        )

    candidates = []
    for d in sorted(drive_probs.iterdir()):
        if not d.is_dir():
            continue
        if all((d / f"{x}.npy").exists() for x in ("oof", "holdout", "test")):
            candidates.append(d.name)

    if len(candidates) < min_baselines:
        raise PreflightError(
            f"Drive probs/ has {len(candidates)} valid baseline(s), need ≥ {min_baselines}.\n"
            f"  Found: {candidates}\n"
            f"  Fix: Re-run an earlier model script (e.g. v14b_external) and cell 7."
        )

    return {"baseline_candidates": candidates}


def preflight_harvest_v18() -> dict:
    """Check harvest/v18 on Drive has manifest + all referenced kernel subdirs."""
    drive_v18 = COLAB_DRIVE_BASE / "harvest" / "v18"
    if not drive_v18.exists():
        raise PreflightError(
            f"Drive harvest/v18 missing: {drive_v18}\n"
            f"  Fix: Upload harvest/v18/ folder from local machine\n"
            f"  (source: /home/ilgrig/IdeaProjects/kaggle/playground-s6e5/harvest/v18/)"
        )

    manifest_path = drive_v18 / "manifest.json"
    if not manifest_path.exists():
        raise PreflightError(
            f"harvest/v18/manifest.json missing on Drive.\n"
            f"  Fix: Re-upload manifest.json to {drive_v18}/"
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        raise PreflightError(f"harvest/v18/manifest.json is corrupt: {e}") from e

    # Verify referenced kernel subdirs exist on Drive
    missing_subdirs = []
    missing_oof = []
    missing_sub = []
    for entry in manifest:
        verdict = entry.get("verdict")
        if verdict in ("EDA", "DOWNLOAD-FAIL", "EXCEPTION", "WEAK", "UNCLASSIFIED"):
            continue
        tag = entry.get("tag")
        if not tag:
            continue
        subdir = drive_v18 / tag
        if not subdir.exists():
            missing_subdirs.append(tag)
            continue
        # Check expected files based on verdict
        files = entry.get("files_found", {})
        if files.get("oof") and not (drive_v18.parent.parent / files["oof"]).name in [f.name for f in subdir.iterdir()]:
            # The path in manifest is harvest/v18/{tag}/oof_preds.csv — check by basename
            oof_name = Path(files["oof"]).name
            if not (subdir / oof_name).exists():
                missing_oof.append(f"{tag}/{oof_name}")
        if files.get("submission"):
            sub_name = Path(files["submission"]).name
            if not (subdir / sub_name).exists():
                missing_sub.append(f"{tag}/{sub_name}")

    if missing_subdirs:
        raise PreflightError(
            f"harvest/v18: {len(missing_subdirs)} kernel subdirectories missing on Drive.\n"
            f"  Examples: {missing_subdirs[:5]}\n"
            f"  Fix: Re-upload the missing folders to {drive_v18}/\n"
            f"  Original location: /home/ilgrig/IdeaProjects/kaggle/playground-s6e5/harvest/v18/"
        )

    warnings = []
    if missing_oof:
        warnings.append(f"  {len(missing_oof)} OOF files missing (will reduce Phase A pool)")
    if missing_sub:
        warnings.append(f"  {len(missing_sub)} submission files missing (will reduce Phase B pool)")

    return {
        "manifest_entries": len(manifest),
        "kernel_subdirs_present": sum(1 for e in manifest if (drive_v18 / e.get("tag", "_")).exists()),
        "include_oof_count": sum(1 for e in manifest if e.get("verdict") == "INCLUDE-OOF"),
        "include_test_count": sum(1 for e in manifest if e.get("verdict") == "INCLUDE-TEST"),
        "warnings": warnings,
    }


def preflight_data_files() -> dict:
    """Check data/raw/test.csv and train.csv exist (needed by the script)."""
    raw_dir = ROOT / "data" / "raw"
    missing = []
    for fn in ("train.csv", "test.csv"):
        if not (raw_dir / fn).exists():
            # Try Drive
            drive_src = COLAB_DRIVE_BASE / "data" / "raw" / fn
            if drive_src.exists():
                raw_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(drive_src, raw_dir / fn)
                continue
            missing.append(fn)
    if missing:
        raise PreflightError(
            f"data/raw/ missing files: {missing}\n"
            f"  Fix: Run cell 5 (sync data Drive → /content) OR check Drive has them."
        )
    return {"data_raw": "OK"}


def run_preflight() -> dict:
    """Run all pre-flight checks. Returns report dict. Raises PreflightError on failure."""
    print("=" * 70)
    print("PRE-FLIGHT VALIDATION")
    print("=" * 70)

    checks = [
        ("Drive mounted", preflight_drive_mounted),
        ("Data files (train.csv, test.csv)", preflight_data_files),
        ("probs/ on Drive (baseline candidates)", preflight_probs),
        ("harvest/v18/ structure on Drive", preflight_harvest_v18),
    ]

    report = {}
    for name, check_fn in checks:
        try:
            result = check_fn()
            report[name] = result
            print(f"  ✓ {name}")
            for k, v in result.items():
                if isinstance(v, list):
                    if not v:
                        continue
                    if len(v) <= 5:
                        print(f"      {k}: {v}")
                    else:
                        print(f"      {k}: {v[:3]} ... ({len(v)} total)")
                else:
                    print(f"      {k}: {v}")
        except PreflightError as exc:
            print(f"\n  ✗ {name}")
            print(f"\n{'=' * 70}")
            print(f"PRE-FLIGHT ABORTED — {name}")
            print(f"{'=' * 70}")
            print(f"\n{exc}")
            sys.exit(1)
    print()
    return report


# =========================================================================
# Data restore (only runs if pre-flight passed)
# =========================================================================

def restore_from_drive() -> None:
    """Copy probs/ and harvest/v18 from Drive to /content."""
    drive_probs = COLAB_DRIVE_BASE / "probs"
    if drive_probs.exists():
        PROBS.mkdir(parents=True, exist_ok=True)
        for src_dir in drive_probs.iterdir():
            if not src_dir.is_dir():
                continue
            dst_dir = PROBS / src_dir.name
            if not dst_dir.exists():
                shutil.copytree(src_dir, dst_dir)

    drive_v18 = COLAB_DRIVE_BASE / "harvest" / "v18"
    local_v18 = ROOT / "harvest" / "v18"
    if drive_v18.exists() and not local_v18.exists():
        local_v18.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive_v18, local_v18)


# =========================================================================
# Baseline auto-detection
# =========================================================================

def autodetect_baseline(y_holdout: np.ndarray) -> str:
    """Pick the highest-holdout-AUC release from probs/ as the starting baseline."""
    candidates = []
    for d in sorted(PROBS.iterdir()):
        if not d.is_dir():
            continue
        if not all((d / f"{x}.npy").exists() for x in ("oof", "holdout", "test")):
            continue
        try:
            holdout_pred = np.load(d / "holdout.npy")
            if len(holdout_pred) != len(y_holdout):
                continue
            auc = roc_auc_score(y_holdout, holdout_pred)
            candidates.append((d.name, auc))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No valid baseline found in probs/")
    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"Auto-detected baseline candidates by holdout AUC:")
    for name, auc in candidates[:5]:
        print(f"  {name:<28}  {auc:.5f}")
    return candidates[0][0]


# =========================================================================
# Blend math (shared with 24_progressive_blend.py)
# =========================================================================

def rank_norm(x: np.ndarray) -> np.ndarray:
    return (rankdata(x, method="average") - 1) / (len(x) - 1)


def nelder_mead_weights(oof_matrix: np.ndarray, y_pool: np.ndarray, n_restarts: int = 8):
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
    ranks = np.column_stack([rank_norm(model_matrix[:, i]) for i in range(model_matrix.shape[1])])
    return ranks @ weights


# =========================================================================
# Candidate loaders
# =========================================================================

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
    existing = sorted(p.name for p in PROBS.glob("v18.*"))
    if not existing:
        return "v18.001"
    last = existing[-1]
    num = int(last.split(".")[1])
    return f"v18.{num + 1:03d}"


def save_release(version, oof, holdout, test, test_ids):
    out_dir = PROBS / version
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof.npy", oof)
    np.save(out_dir / "holdout.npy", holdout)
    np.save(out_dir / "test.npy", test)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test_ids, TARGET: test}).to_csv(SUBMISSIONS / f"{version}.csv", index=False)


def log_release(entry):
    entry["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(RELEASES_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =========================================================================
# Sync back to Drive
# =========================================================================

def sync_back_to_drive():
    """Copy probs/v18.*/, submissions/v18.*.csv, releases.jsonl to Drive."""
    drive_probs = COLAB_DRIVE_BASE / "probs"
    drive_subs = COLAB_DRIVE_BASE / "submissions"
    drive_probs.mkdir(parents=True, exist_ok=True)
    drive_subs.mkdir(parents=True, exist_ok=True)

    # probs/v18.*/ (new releases only — don't re-sync existing)
    for src_dir in PROBS.glob("v18.*"):
        if not src_dir.is_dir():
            continue
        dst_dir = drive_probs / src_dir.name
        if dst_dir.exists():
            continue
        shutil.copytree(src_dir, dst_dir)
        print(f"  → Drive probs/{src_dir.name}")

    # submissions/v18.*.csv
    for sub in SUBMISSIONS.glob("v18.*.csv"):
        dst = drive_subs / sub.name
        if not dst.exists() or dst.stat().st_size != sub.stat().st_size:
            shutil.copyfile(sub, dst)
            print(f"  → Drive submissions/{sub.name}")

    # releases.jsonl
    if RELEASES_LOG.exists():
        shutil.copyfile(RELEASES_LOG, COLAB_DRIVE_BASE / "releases.jsonl")
        print(f"  → Drive releases.jsonl")


# =========================================================================
# Main pipeline
# =========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=None,
                   help="Override auto-detection (e.g. 'v18.002')")
    p.add_argument("--threshold", type=float, default=1e-5)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--skip-phase-b", action="store_true",
                   help="Skip the test-only public mix variants")
    args = p.parse_args()

    # ---- Stage 0: Pre-flight ----
    report = run_preflight()

    # ---- Stage 1: Restore data ----
    print("=" * 70)
    print("STAGE 1 — Restore data from Drive")
    print("=" * 70)
    restore_from_drive()
    print(f"  probs/ entries: {len(list(PROBS.iterdir()))}")
    print(f"  harvest/v18/ entries: {len(list((ROOT / 'harvest' / 'v18').iterdir()))}")
    print()

    # Load training/holdout/test ids and labels
    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    pool_ids = pool[ID].to_numpy()
    test_ids = test[ID].to_numpy()
    y_pool = pool[TARGET].astype(int).to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()

    # ---- Stage 2: Auto-detect baseline ----
    print("=" * 70)
    print("STAGE 2 — Auto-detect baseline")
    print("=" * 70)
    baseline_name = args.baseline if args.baseline else autodetect_baseline(y_holdout)
    base = load_internal_probs(baseline_name)
    if base is None:
        raise RuntimeError(f"Baseline '{baseline_name}' has incomplete probs/")
    baseline_oof_auc = roc_auc_score(y_pool, base["oof"])
    baseline_holdout_auc = roc_auc_score(y_holdout, base["holdout"])
    print(f"\nBaseline selected: {baseline_name}")
    print(f"  OOF AUC: {baseline_oof_auc:.5f}")
    print(f"  Holdout AUC: {baseline_holdout_auc:.5f}\n")

    # ---- Stage 3: Build candidate pool ----
    print("=" * 70)
    print("STAGE 3 — Build candidate pool")
    print("=" * 70)

    candidates = []
    # Internal
    for v in INTERNAL_CANDIDATES:
        if v == baseline_name:
            continue
        c = load_internal_probs(v)
        if c is None:
            continue
        candidates.append({
            "name": v, "source": "internal",
            "oof": c["oof"], "holdout": c["holdout"], "test": c["test"],
            "votes": 0, "rho": 0,
        })

    # Also include any v18.NNN releases that aren't the baseline
    for v in sorted(PROBS.glob("v18.*")):
        if v.name == baseline_name:
            continue
        c = load_internal_probs(v.name)
        if c is None:
            continue
        candidates.append({
            "name": v.name, "source": "internal_v18",
            "oof": c["oof"], "holdout": c["holdout"], "test": c["test"],
            "votes": 0, "rho": 0,
        })

    # Harvested INCLUDE-OOF
    manifest = json.loads((ROOT / "harvest" / "v18" / "manifest.json").read_text())
    for e in manifest:
        if e.get("verdict") != "INCLUDE-OOF":
            continue
        oof_arr = load_v18_oof(e, pool_ids)
        test_arr = load_v18_test(e, test_ids)
        if oof_arr is None or test_arr is None:
            continue
        candidates.append({
            "name": "v18_" + e["tag"],
            "source": "harvest_oof",
            "oof": oof_arr, "holdout": None, "test": test_arr,
            "votes": e.get("votes", 0), "rho": e.get("rho_with_anchor", 0),
        })

    print(f"\nCandidate pool: {len(candidates)} models")
    src_counts = {}
    for c in candidates:
        src_counts[c["source"]] = src_counts.get(c["source"], 0) + 1
    for src, n in src_counts.items():
        print(f"  {src:<20}  {n}")
    print()

    # ---- Stage 4: Phase A — greedy forward selection ----
    print("=" * 70)
    print("STAGE 4 — Phase A (greedy forward selection)")
    print("=" * 70)

    current_name = baseline_name
    current_oof = base["oof"]
    current_holdout = base["holdout"]
    current_test = base["test"]
    current_oof_auc = baseline_oof_auc
    current_holdout_auc = baseline_holdout_auc
    promoted = {baseline_name}

    for iter_i in range(args.max_iter):
        best = None
        best_new_oof = current_oof_auc
        best_weights = None
        best_new_holdout = None

        for cand in candidates:
            if cand["name"] in promoted:
                continue
            stack = np.column_stack([current_oof, cand["oof"]])
            w, new_oof_auc = nelder_mead_weights(stack, y_pool, n_restarts=6)
            if new_oof_auc > best_new_oof:
                best = cand
                best_new_oof = new_oof_auc
                best_weights = w
                if cand["holdout"] is not None:
                    h_stack = np.column_stack([current_holdout, cand["holdout"]])
                    h_pred = blend_with_weights(h_stack, w)
                    best_new_holdout = roc_auc_score(y_holdout, h_pred)
                else:
                    best_new_holdout = None

        if best is None or (best_new_oof - current_oof_auc) < args.threshold:
            print(f"Iter {iter_i+1}: no candidate improves OOF AUC by ≥ {args.threshold} — stopping")
            break

        delta_oof = best_new_oof - current_oof_auc
        delta_ho = (best_new_holdout - current_holdout_auc) if best_new_holdout is not None else None
        new_ver = next_release_version()
        mode = "HOLDOUT" if best["holdout"] is not None else "OOF-ONLY"
        print(f"Iter {iter_i+1}: PROMOTING {best['name']}  [{mode}]")
        print(f"  OOF delta:     {delta_oof:+.6f}  ({current_oof_auc:.5f} → {best_new_oof:.5f})")
        if delta_ho is not None:
            print(f"  Holdout delta: {delta_ho:+.6f}  ({current_holdout_auc:.5f} → {best_new_holdout:.5f})")
        print(f"  weights: baseline={best_weights[0]:.4f}, added={best_weights[1]:.4f}")
        print(f"  release: {new_ver}\n")

        # Build new release
        oof_stack = np.column_stack([current_oof, best["oof"]])
        test_stack = np.column_stack([current_test, best["test"]])
        new_oof = blend_with_weights(oof_stack, best_weights)
        new_test = blend_with_weights(test_stack, best_weights)
        if best["holdout"] is not None:
            h_stack = np.column_stack([current_holdout, best["holdout"]])
            new_holdout = blend_with_weights(h_stack, best_weights)
        else:
            new_holdout = current_holdout

        save_release(new_ver, new_oof, new_holdout, new_test, test_ids)
        log_release({
            "version": new_ver, "parent": current_name, "added": best["name"],
            "added_source": best["source"], "validation_mode": mode,
            "weights": {current_name: float(best_weights[0]), best["name"]: float(best_weights[1])},
            "oof_before": float(current_oof_auc), "oof_after": float(best_new_oof),
            "oof_delta": float(delta_oof),
            "holdout_before": float(current_holdout_auc),
            "holdout_after": float(best_new_holdout) if best_new_holdout is not None else None,
            "decision": "PROMOTED",
        })

        current_name = new_ver
        current_oof = new_oof
        current_holdout = new_holdout
        current_test = new_test
        current_oof_auc = best_new_oof
        if best_new_holdout is not None:
            current_holdout_auc = best_new_holdout
        promoted.add(best["name"])

    print(f"--- Phase A complete ---")
    print(f"Final Phase A baseline: {current_name}")
    print(f"  OOF AUC: {current_oof_auc:.5f}  (initial: {baseline_oof_auc:.5f}, Δ {current_oof_auc - baseline_oof_auc:+.6f})")
    print(f"  Holdout AUC: {current_holdout_auc:.5f}  (initial: {baseline_holdout_auc:.5f}, Δ {current_holdout_auc - baseline_holdout_auc:+.6f})")
    print()

    phase_a_final = current_name

    # ---- Stage 5: Phase B — test-only public mix ----
    if not args.skip_phase_b:
        print("=" * 70)
        print("STAGE 5 — Phase B (public test-only mix variants)")
        print("=" * 70)

        test_only_pool = []
        for e in manifest:
            if e.get("verdict") != "INCLUDE-TEST":
                continue
            claimed_lb = e.get("claimed_lb")
            rho = e.get("rho_with_anchor", 0)
            if rho is None:
                rho = 0  # unknown — treat as diverse
            if claimed_lb is not None and claimed_lb < 0.95:
                continue
            if rho > 0.995:
                continue
            test_arr = load_v18_test(e, test_ids)
            if test_arr is None:
                continue
            test_only_pool.append(test_arr)

        print(f"\nTest-only pool: {len(test_only_pool)} candidates after quality+diversity filter")
        if test_only_pool:
            public_mean = np.column_stack([rank_norm(t) for t in test_only_pool]).mean(axis=1)
            current_test_rank = rank_norm(current_test)
            for alpha in [0.05, 0.10, 0.20]:
                mix = (1 - alpha) * current_test_rank + alpha * public_mean
                ver = next_release_version()
                out_dir = PROBS / ver
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "test.npy", mix)
                pd.DataFrame({ID: test_ids, TARGET: mix}).to_csv(SUBMISSIONS / f"{ver}.csv", index=False)
                log_release({
                    "version": ver, "parent": current_name,
                    "added": f"public_mean_mix(alpha={alpha}, n={len(test_only_pool)})",
                    "added_source": "test_mix",
                    "weights": {"baseline": float(1 - alpha), "public_mix": float(alpha)},
                    "oof_before": float(current_oof_auc), "oof_after": None,
                    "oof_delta": None,
                    "holdout_before": float(current_holdout_auc),
                    "holdout_after": None,
                    "decision": "PHASE_B_VARIANT",
                })
                print(f"  → {ver}: {1-alpha:.2f} baseline + {alpha:.2f} public_mix")
        print()

    # ---- Stage 6: Sync back ----
    print("=" * 70)
    print("STAGE 6 — Sync to Drive")
    print("=" * 70)
    sync_back_to_drive()
    print()

    # ---- Stage 7: Final report ----
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"Starting baseline:    {baseline_name}  (holdout {baseline_holdout_auc:.5f})")
    print(f"Phase A winner:       {phase_a_final}  (holdout {current_holdout_auc:.5f})")
    print(f"  Δ from baseline:    {current_holdout_auc - baseline_holdout_auc:+.6f}")
    print()
    print("All releases this run (probs/v18.*/, submissions/v18.*.csv):")
    for sub in sorted(SUBMISSIONS.glob("v18.*.csv")):
        print(f"  {sub.name}")
    print()
    print(f"RECOMMENDED SUBMISSION: submissions/{phase_a_final}.csv")
    print(f"\nLocal Kaggle command:")
    print(f"  kaggle competitions submit -c playground-series-s6e5 \\")
    print(f"      -f submissions/{phase_a_final}.csv \\")
    print(f"      -m '{phase_a_final} — progressive blend pipeline'")
    print()


if __name__ == "__main__":
    main()
