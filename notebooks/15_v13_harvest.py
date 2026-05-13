"""v13a — Harvest public submission CSVs for blending.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: Public notebooks for S6E5 use model classes / CV regimes our
              stack lacks (TabPFN transformer, CatBoost ordered boosting,
              NN-residual). Their test predictions should rank-correlate
              weakly enough with our v11/v12 that adding them to the blend
              yields the diversity gain we cannot manufacture internally
              (L16 from S6E4 audit).
  Parent: v11_blend_4way (current best LB 0.94774)
  Predicted Δ holdout: N/A — public predictions have no labels for us.
                       Diversity proxy: pairwise rank-Spearman with v11
                       should be < 0.95 to be worth including.
  Most relevant pitfall: #6 (don't trust public LB blindly — author may have
                         overfit by feedback). Treat each candidate as one
                         additional signal, not as ground truth.
  Output: harvest/v13/{slug}/submission.csv  + harvest/v13/manifest.json

This is the I/O half of v13. Pure download + validate, no compute.
The blend itself lives in notebooks/16_v13_blend.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ID, RAW, ROOT, TARGET

HARVEST = ROOT / "harvest" / "v13"
HARVEST.mkdir(parents=True, exist_ok=True)

# Manually curated. Add/remove as scouting finds new strong public notebooks.
# Each entry: kaggle kernel slug + claimed public LB (from author's notebook).
KERNELS = [
    {
        "slug": "sidcodegg/tabpfn-is-all-you-need",
        "claimed_lb": None,  # to be filled from notebook header
        "tag": "tabpfn",
    },
    {
        "slug": "anthonytherrien/predicting-f1-pit-stops-nn-residual-network",
        "claimed_lb": None,
        "tag": "nn_residual",
    },
    {
        "slug": "rohit8527kmr7518/ps-s6e5-catboost-10-fold-cv",
        "claimed_lb": None,
        "tag": "catboost_10fold",
    },
]


def download_kernel_output(slug: str, out_dir: Path) -> Path | None:
    """Run `kaggle kernels output` for a public kernel. Returns the path
    where outputs landed, or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {slug}")
    try:
        subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", str(out_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        print(f"    ✗ kaggle CLI failed: {exc.stderr.strip().splitlines()[-1]}")
        return None
    except subprocess.TimeoutExpired:
        print("    ✗ timeout after 120s")
        return None
    return out_dir


def find_submission_csv(out_dir: Path) -> Path | None:
    """Find the submission CSV among the kernel's output files.
    Public Playground notebooks usually name it `submission.csv` but not always."""
    candidates = list(out_dir.glob("*.csv")) + list(out_dir.glob("*.csv.gz"))
    if not candidates:
        return None
    # Prefer files literally named submission.*
    preferred = [c for c in candidates if c.name.lower().startswith("submission")]
    if preferred:
        return preferred[0]
    # Otherwise return the largest CSV (often the prediction file is bigger than aux outputs)
    return max(candidates, key=lambda p: p.stat().st_size)


def validate_submission(sub: pd.DataFrame, test_ids: np.ndarray, slug: str) -> dict:
    """Hard checks. Returns diagnostics dict; sets 'valid' to True/False."""
    diag: dict = {"slug": slug, "valid": False, "issues": []}

    if ID not in sub.columns:
        diag["issues"].append(f"missing '{ID}' column (found: {list(sub.columns)})")
        return diag

    pred_cols = [c for c in sub.columns if c != ID]
    if len(pred_cols) != 1:
        diag["issues"].append(f"expected exactly 1 prediction column, found {pred_cols}")
        return diag
    pred_col = pred_cols[0]

    if len(sub) != len(test_ids):
        diag["issues"].append(f"row count {len(sub)} != test {len(test_ids)}")
        return diag

    sub_ids = sub[ID].to_numpy()
    if not np.array_equal(np.sort(sub_ids), np.sort(test_ids)):
        diag["issues"].append("id set does not match test set")
        return diag

    preds = sub[pred_col].to_numpy()
    if np.isnan(preds).any():
        diag["issues"].append(f"{np.isnan(preds).sum()} NaN predictions")
        return diag
    if not np.isfinite(preds).all():
        diag["issues"].append("non-finite predictions")
        return diag

    diag["valid"] = True
    diag["pred_col"] = pred_col
    diag["min"] = float(preds.min())
    diag["max"] = float(preds.max())
    diag["mean"] = float(preds.mean())
    return diag


def main() -> int:
    test = pd.read_csv(RAW / "test.csv", usecols=[ID])
    test_ids = test[ID].to_numpy()
    print(f"Test set: {len(test_ids)} rows")
    print(f"Harvest dir: {HARVEST}")
    print(f"Target column expected: {TARGET}")
    print()

    manifest: list[dict] = []
    for entry in KERNELS:
        slug = entry["slug"]
        tag = entry["tag"]
        kernel_dir = HARVEST / tag

        # Clean prior attempt if present
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)

        out = download_kernel_output(slug, kernel_dir)
        if out is None:
            manifest.append({**entry, "valid": False, "issues": ["download failed"]})
            continue

        sub_path = find_submission_csv(out)
        if sub_path is None:
            print(f"    ✗ {tag}: no CSV in kernel output (author didn't save one)")
            print()
            manifest.append({**entry, "valid": False, "issues": ["no CSV in kernel output"]})
            continue

        sub = pd.read_csv(sub_path)
        diag = validate_submission(sub, test_ids, slug)
        diag["downloaded_csv"] = str(sub_path.relative_to(ROOT))
        diag["tag"] = tag
        manifest.append({**entry, **diag})

        if diag["valid"]:
            print(f"    ✓ {tag}: pred_col={diag['pred_col']} "
                  f"range=[{diag['min']:.4f}, {diag['max']:.4f}] "
                  f"mean={diag['mean']:.4f}")
        else:
            print(f"    ✗ {tag}: {diag['issues']}")
        print()

    (HARVEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    valid = sum(1 for m in manifest if m.get("valid"))
    print(f"=== Harvest complete: {valid}/{len(manifest)} valid ===")
    print(f"Manifest: {HARVEST / 'manifest.json'}")
    return 0 if valid > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
