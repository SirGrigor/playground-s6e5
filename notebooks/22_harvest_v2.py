"""v18 / harvester_v2 — bulk-scan top N public notebooks for OOFs + submissions.

Designed to be run on Colab (Drive-friendly outputs) or local. Replaces the
hand-curated KERNELS list in 15_v13_harvest.py with auto-listing of top-voted
public notebooks via Kaggle CLI.

Per-kernel pipeline:
  1. Download all kernel outputs to harvest/v18/{slug_tag}/
  2. Find oof*.csv + submission*.csv + .log
  3. Extract claimed_lb from title (regex)
  4. Extract claimed_oof_auc from log (regex on stdout)
  5. Validate submission file (row count, id alignment, value range)
  6. Validate OOF file if present (aligned to OUR pool ids — drop holdout rows)
  7. Compute ρ vs our v17 anchor (if available)
  8. Categorize (INCLUDE-OOF / INCLUDE-TEST / TEST-ONLY / LEAKY / EDA / FAIL)

Outputs:
  harvest/v18/manifest.json  — full structured registry
  harvest/v18/audit.md       — human-readable report

Usage:
  python notebooks/22_harvest_v2.py                    # top 50 from S6E5
  python notebooks/22_harvest_v2.py --top 30           # top 30 only
  python notebooks/22_harvest_v2.py --comp other-slug  # different comp
  python notebooks/22_harvest_v2.py --skip-existing    # don't redownload
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ID, RAW, ROOT, TARGET
from src.data import load_train_pool


HARVEST_BASE = ROOT / "harvest" / "v18"
DEFAULT_COMP = "playground-series-s6e5"
DEFAULT_TOP_N = 50
SLEEP_BETWEEN_DOWNLOADS = 0.5  # be polite to Kaggle


# ---- Step 1: List top kernels via Kaggle CLI -----------------------------

def list_top_kernels(comp: str, n: int) -> list[dict[str, str]]:
    """Run `kaggle kernels list -v` and parse the CSV output."""
    print(f"Listing top {n} kernels for {comp}...")
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--competition", comp,
         "--sort-by", "voteCount", "--page-size", str(n), "-v"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed: {result.stderr}")
    # Skip the deprecation warning line if present
    lines = [l for l in result.stdout.splitlines() if not l.startswith("Warning")]
    reader = csv.DictReader(lines)
    return list(reader)


# ---- Step 2: Download a kernel's outputs ---------------------------------

def slug_to_tag(slug: str) -> str:
    """user/kernel-name → user_kernel-name (path-safe)."""
    return slug.replace("/", "_").replace(" ", "_")


def download_kernel(slug: str, out_dir: Path, timeout: int = 180) -> tuple[bool, str]:
    """Download kernel outputs. Returns (success, error_msg)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", str(out_dir)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            err = (result.stderr or "").strip().splitlines()
            return False, err[-1] if err else "unknown CLI error"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"


# ---- Step 3: Find relevant files inside the kernel output ---------------

def find_files(out_dir: Path) -> dict[str, Path | None]:
    """Locate oof, submission, and log files. Handles common name variants."""
    files = {"oof": None, "submission": None, "log": None}

    # OOF — anything with 'oof' in the name, prefer .csv
    oof_candidates = list(out_dir.rglob("*oof*.csv")) + list(out_dir.rglob("*OOF*.csv"))
    if oof_candidates:
        files["oof"] = max(oof_candidates, key=lambda p: p.stat().st_size)

    # Submission — anything matching submission*.csv (largest if multiple)
    sub_candidates = list(out_dir.rglob("submission*.csv")) + list(out_dir.rglob("Submission*.csv"))
    if not sub_candidates:
        # Some authors save the prediction in a differently-named csv
        # but with the right column structure — last-resort scan
        all_csvs = list(out_dir.rglob("*.csv"))
        # Skip ones that look like OOF or metadata
        sub_candidates = [c for c in all_csvs if "oof" not in c.name.lower() and c not in (files["oof"],)]
    if sub_candidates:
        files["submission"] = max(sub_candidates, key=lambda p: p.stat().st_size)

    # Log
    log_candidates = list(out_dir.rglob("*.log"))
    if log_candidates:
        files["log"] = log_candidates[0]

    return files


# ---- Step 4: Title and log parsing ---------------------------------------

LB_REGEX = re.compile(r"\b0\.\d{3,5}\b")


def extract_claimed_lb(title: str) -> float | None:
    """Find AUC-like number in kernel title. Prefer values in 0.90-0.99 range."""
    matches = [float(m) for m in LB_REGEX.findall(title)]
    plausible = [v for v in matches if 0.90 <= v <= 0.99]
    return max(plausible) if plausible else None


OOF_AUC_REGEXES = [
    re.compile(r"OOF AUC[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
    re.compile(r"oof_auc[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
    re.compile(r"Overall OOF[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
    re.compile(r"CV[^0-9]+score[^0-9]*(\d\.\d{4,5})", re.IGNORECASE),
]


def extract_claimed_oof_auc(log_path: Path | None) -> float | None:
    """Parse kernel log file for printed OOF AUC. Handles JSON-stream and plain text."""
    if log_path is None or not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return None
    # Kaggle logs are sometimes JSON-stream (one event per line) — concatenate stdout
    if text.lstrip().startswith("["):
        try:
            events = json.loads(text)
            text = "\n".join(e.get("data", "") for e in events if isinstance(e, dict))
        except Exception:
            pass  # fall through with original text
    candidates: list[float] = []
    for rgx in OOF_AUC_REGEXES:
        candidates += [float(m) for m in rgx.findall(text)]
    plausible = [v for v in candidates if 0.85 <= v <= 0.99]
    return max(plausible) if plausible else None


# ---- Step 5: Validate submission file -----------------------------------

def validate_submission(sub_path: Path, test_ids: np.ndarray) -> dict[str, Any]:
    """Check that the submission has correct shape, id alignment, plausible values."""
    diag: dict[str, Any] = {"valid": False, "issues": []}
    try:
        df = pd.read_csv(sub_path)
    except Exception as e:
        diag["issues"].append(f"read failed: {e}")
        return diag
    if ID not in df.columns:
        diag["issues"].append(f"missing '{ID}' col")
        return diag
    pred_cols = [c for c in df.columns if c != ID]
    if len(pred_cols) != 1:
        diag["issues"].append(f"expected 1 pred col, got {pred_cols}")
        return diag
    if len(df) != len(test_ids):
        diag["issues"].append(f"row count {len(df)} != {len(test_ids)}")
        return diag
    if set(df[ID].to_numpy()) != set(test_ids):
        diag["issues"].append("id set mismatch")
        return diag
    preds = df[pred_cols[0]].to_numpy()
    if np.isnan(preds).any() or not np.isfinite(preds).all():
        diag["issues"].append("nan or inf predictions")
        return diag
    diag.update({
        "valid": True, "pred_col": pred_cols[0],
        "n_rows": len(df), "mean": float(preds.mean()),
        "min": float(preds.min()), "max": float(preds.max()),
    })
    return diag


# ---- Step 6: Validate OOF file (aligned to our pool) ---------------------

def validate_oof(oof_path: Path, pool_ids: np.ndarray, pool_y: np.ndarray) -> dict[str, Any]:
    """Load OOF, filter to our pool ids (drop holdout rows), compute AUC."""
    diag: dict[str, Any] = {"valid": False, "issues": []}
    try:
        df = pd.read_csv(oof_path)
    except Exception as e:
        diag["issues"].append(f"read failed: {e}")
        return diag
    if ID not in df.columns:
        diag["issues"].append(f"missing '{ID}' col")
        return diag
    pred_cols = [c for c in df.columns if c != ID]
    if len(pred_cols) != 1:
        diag["issues"].append(f"expected 1 pred col, got {pred_cols}")
        return diag
    pool_id_set = set(int(x) for x in pool_ids)
    df_pool = df[df[ID].isin(pool_id_set)].sort_values(ID).reset_index(drop=True)
    if len(df_pool) < len(pool_ids) * 0.95:
        diag["issues"].append(f"only {len(df_pool)}/{len(pool_ids)} pool ids in OOF — author trained on subset")
        return diag
    # Build aligned arrays
    pool_id_to_y = dict(zip([int(x) for x in pool_ids], pool_y))
    aligned_y = np.array([pool_id_to_y[int(i)] for i in df_pool[ID]])
    aligned_pred = df_pool[pred_cols[0]].to_numpy()
    if np.isnan(aligned_pred).any():
        diag["issues"].append("nan predictions in OOF")
        return diag
    auc = float(roc_auc_score(aligned_y, aligned_pred))
    diag.update({
        "valid": True, "pred_col": pred_cols[0],
        "n_rows_in_pool": len(df_pool), "n_rows_total": len(df),
        "auc_on_our_pool": auc,
    })
    return diag


# ---- Step 7: ρ vs our anchor --------------------------------------------

def rho_to_anchor(pred: np.ndarray, anchor: np.ndarray) -> float:
    """Spearman ρ — fast via Pearson on ranks."""
    r1 = rankdata(pred, method="average")
    r2 = rankdata(anchor, method="average")
    return float(np.corrcoef(r1, r2)[0, 1])


# ---- Step 8: Categorize --------------------------------------------------

def categorize(entry: dict[str, Any]) -> str:
    """Heuristic verdict based on entry data."""
    if entry.get("download_failed"):
        return "DOWNLOAD-FAIL"

    sub_valid = entry.get("submission_diag", {}).get("valid", False)
    oof_valid = entry.get("oof_diag", {}).get("valid", False)

    if not sub_valid and not oof_valid:
        # No predictions at all — likely an EDA notebook
        return "EDA"

    claimed_oof = entry.get("claimed_oof_auc")
    claimed_lb = entry.get("claimed_lb")
    rho = entry.get("rho_with_anchor")

    # Leakage check: claimed OOF much higher than claimed LB
    if claimed_oof is not None and claimed_lb is not None:
        if claimed_oof - claimed_lb > 0.005:
            return "LEAKY"

    # Quality check: minimum LB level worth blending
    if claimed_lb is not None and claimed_lb < 0.945:
        return "WEAK"

    # Diversity check: ρ < 0.99 with our v17
    if rho is not None and rho > 0.995:
        return "REDUNDANT"

    if oof_valid:
        return "INCLUDE-OOF"
    if sub_valid:
        return "INCLUDE-TEST"

    return "UNCLASSIFIED"


# ---- Main pipeline -------------------------------------------------------

def harvest_one(slug: str, title: str, votes: int, test_ids: np.ndarray,
                pool_ids: np.ndarray, pool_y: np.ndarray,
                anchor_test: np.ndarray | None, skip_existing: bool) -> dict[str, Any]:
    """End-to-end harvest pipeline for one kernel."""
    tag = slug_to_tag(slug)
    out_dir = HARVEST_BASE / tag

    entry: dict[str, Any] = {
        "slug": slug, "title": title, "votes": int(votes), "tag": tag,
        "claimed_lb": extract_claimed_lb(title),
        "download_failed": False, "download_err": None,
        "verdict": "UNCLASSIFIED",  # defensive default; overwritten before return
    }

    # Download (skip if directory already populated and --skip-existing)
    if skip_existing and out_dir.exists() and any(out_dir.iterdir()):
        print(f"  · skip (existing): {slug}")
    else:
        ok, err = download_kernel(slug, out_dir)
        if not ok:
            entry["download_failed"] = True
            entry["download_err"] = err
            return entry
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    files = find_files(out_dir)
    entry["files_found"] = {k: (str(v.relative_to(ROOT)) if v else None) for k, v in files.items()}

    # Claimed OOF AUC from log
    entry["claimed_oof_auc"] = extract_claimed_oof_auc(files["log"])

    # Validate submission
    if files["submission"]:
        entry["submission_diag"] = validate_submission(files["submission"], test_ids)
    else:
        entry["submission_diag"] = {"valid": False, "issues": ["no submission file found"]}

    # Validate OOF
    if files["oof"]:
        entry["oof_diag"] = validate_oof(files["oof"], pool_ids, pool_y)
    else:
        entry["oof_diag"] = {"valid": False, "issues": ["no OOF file found"]}

    # ρ vs anchor (test-side)
    if anchor_test is not None and entry["submission_diag"]["valid"]:
        sub_df = pd.read_csv(files["submission"]).set_index(ID).loc[test_ids]
        sub_pred = sub_df[entry["submission_diag"]["pred_col"]].to_numpy()
        entry["rho_with_anchor"] = rho_to_anchor(sub_pred, anchor_test)
    else:
        entry["rho_with_anchor"] = None

    entry["verdict"] = categorize(entry)
    return entry


def write_audit_md(entries: list[dict[str, Any]], path: Path) -> None:
    """Generate human-readable markdown report."""
    lines = [
        "# Harvest v18 — Audit Report",
        "",
        f"Total kernels examined: {len(entries)}",
    ]
    # Verdict counts
    from collections import Counter
    counts = Counter(e["verdict"] for e in entries)
    lines.append("")
    lines.append("## Verdict counts")
    lines.append("")
    for verdict, cnt in counts.most_common():
        lines.append(f"- **{verdict}**: {cnt}")
    lines.append("")
    lines.append("## Per-kernel table")
    lines.append("")
    lines.append("| Verdict | Slug | Votes | Claimed LB | Claimed OOF | OOF AUC on our pool | ρ vs v17 | Sub | OOF |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in sorted(entries, key=lambda x: (x["verdict"], -x["votes"])):
        sub_ok = "✓" if e.get("submission_diag", {}).get("valid") else "—"
        oof_ok = "✓" if e.get("oof_diag", {}).get("valid") else "—"
        oof_auc = e.get("oof_diag", {}).get("auc_on_our_pool")
        oof_auc_str = f"{oof_auc:.5f}" if oof_auc else "—"
        cl = e.get("claimed_lb")
        cl_str = f"{cl:.5f}" if cl else "—"
        co = e.get("claimed_oof_auc")
        co_str = f"{co:.5f}" if co else "—"
        rho = e.get("rho_with_anchor")
        rho_str = f"{rho:.4f}" if rho is not None else "—"
        lines.append(
            f"| {e['verdict']} | `{e['slug']}` | {e['votes']} | {cl_str} | "
            f"{co_str} | {oof_auc_str} | {rho_str} | {sub_ok} | {oof_ok} |"
        )
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comp", default=DEFAULT_COMP)
    p.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip kernels whose output dir already exists")
    p.add_argument("--anchor-sub", default="submissions/v17_NM_blend.csv",
                   help="Submission to use as ρ anchor")
    args = p.parse_args()

    HARVEST_BASE.mkdir(parents=True, exist_ok=True)

    # Load test ids + our pool ids + labels (for OOF validation)
    print("Loading test/train ids and labels...")
    test_df = pd.read_csv(RAW / "test.csv", usecols=[ID])
    test_ids = test_df[ID].to_numpy()
    pool = load_train_pool()
    pool_ids = pool[ID].to_numpy()
    pool_y = pool[TARGET].astype(int).to_numpy()

    # Load anchor submission if available
    anchor_path = ROOT / args.anchor_sub
    anchor_test = None
    if anchor_path.exists():
        anchor_df = pd.read_csv(anchor_path).set_index(ID).loc[test_ids]
        anchor_test = anchor_df.iloc[:, 0].to_numpy()
        print(f"Anchor: {args.anchor_sub} (ρ baseline)\n")
    else:
        print(f"Anchor {args.anchor_sub} not found — skipping ρ computation\n")

    # List top-N kernels
    kernels = list_top_kernels(args.comp, args.top)
    print(f"\nFound {len(kernels)} kernels. Starting harvest...\n")

    entries = []
    for i, k in enumerate(kernels, 1):
        slug = k["ref"]
        title = k["title"]
        votes = k.get("totalVotes", "0")
        print(f"[{i}/{len(kernels)}] {slug}  ({votes} votes)")
        try:
            entry = harvest_one(slug, title, int(votes), test_ids,
                               pool_ids, pool_y, anchor_test, args.skip_existing)
            print(f"    → {entry['verdict']}")
            entries.append(entry)
        except Exception as exc:
            print(f"    ✗ exception: {exc}")
            entries.append({"slug": slug, "title": title, "votes": int(votes),
                           "verdict": "EXCEPTION", "error": str(exc)})

    # Write outputs
    manifest_path = HARVEST_BASE / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2, default=str))
    print(f"\nManifest written: {manifest_path}")

    audit_path = HARVEST_BASE / "audit.md"
    write_audit_md(entries, audit_path)
    print(f"Audit report: {audit_path}")

    # Summary print
    from collections import Counter
    counts = Counter(e["verdict"] for e in entries)
    print("\n=== Verdict summary ===")
    for verdict, cnt in counts.most_common():
        print(f"  {verdict:<20} {cnt}")


if __name__ == "__main__":
    main()
