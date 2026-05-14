"""v19 curated-mix explorer — standalone tool for K × ratio grid sweep.

The S6E5 L19 lesson operationalized: filter public submissions by claimed_lb,
sweep top-K × ratio grid, output ranked recommendations.

Generates `submissions/{name}.csv` for each (K, ratio) combination tested.
Outputs `curated_audit.md` with a sorted table of candidates by predicted_lb.

Run AFTER 25_pipeline.py has produced a baseline (e.g., v18.007 or v19.006).

Usage:
  python notebooks/26_curated_explorer.py
  python notebooks/26_curated_explorer.py --baseline v18.007
  python notebooks/26_curated_explorer.py --baseline v18.007 --claimed-lb-min 0.954
  python notebooks/26_curated_explorer.py --k 3,4,5,6 --ratios 0.20,0.30,0.50,0.70,0.90,1.00
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ID, PROBS, ROOT, SUBMISSIONS, TARGET
from src.curated import (
    load_public_subset,
    ratio_sweep,
)
from src.data import load_holdout, load_test, load_train_pool


COLAB_DRIVE_BASE = Path("/content/drive/MyDrive/Colab Notebooks/kaggle/s6e5")


def restore_from_drive_if_needed() -> None:
    """Auto-restore probs/ and harvest/v18/ from Drive on Colab."""
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
    if drive_v18.exists():
        local_v18.mkdir(parents=True, exist_ok=True)
        for src in drive_v18.iterdir():
            dst = local_v18 / src.name
            if src.is_dir():
                if not dst.exists():
                    shutil.copytree(src, dst)
            elif not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copyfile(src, dst)


def load_baseline_test(version: str, test_ids: np.ndarray) -> np.ndarray:
    """Load baseline test predictions.

    Prefers probs/{version}/test.npy (the canonical form), falls back to
    submissions/{version}.csv (rank-normed back to raw probability ranks).
    """
    npy_path = PROBS / version / "test.npy"
    if npy_path.exists():
        return np.load(npy_path)
    csv_path = SUBMISSIONS / f"{version}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path).set_index(ID).loc[test_ids]
        return df.iloc[:, 0].to_numpy()
    raise FileNotFoundError(
        f"No probs/{version}/test.npy or submissions/{version}.csv found"
    )


def baseline_lb_estimate(version: str, y_holdout: np.ndarray) -> float:
    """Estimate the baseline's LB from its holdout AUC.

    Empirical S6E5 observation: holdout AUC predicts LB with gap ~0.0003-0.0007.
    Use holdout AUC - 0.0005 as a midpoint estimate.
    """
    holdout_path = PROBS / version / "holdout.npy"
    if holdout_path.exists():
        holdout_pred = np.load(holdout_path)
        holdout_auc = float(roc_auc_score(y_holdout, holdout_pred))
        return holdout_auc - 0.0005
    return 0.954  # fallback if no holdout data


def next_release_version() -> str:
    """Find next available v19.NNN slot (continues from existing)."""
    existing = sorted(p.name for p in SUBMISSIONS.glob("v19.*.csv"))
    if not existing:
        return "v19.001"
    last_num = max(
        int(p.stem.split(".")[1]) for p in SUBMISSIONS.glob("v19.*.csv")
        if p.stem.split(".")[1].isdigit()
    )
    return f"v19.{last_num + 1:03d}"


def write_audit_md(
    results: list,
    baseline: str,
    baseline_lb: float,
    out_path: Path,
    n_public_candidates: int,
    claimed_lb_threshold: float | None,
) -> None:
    lines = [
        "# v19 Curated Mix Explorer — Audit Report",
        "",
        f"**Baseline**: `{baseline}` (estimated LB: {baseline_lb:.5f})",
        f"**Public candidates after filter**: {n_public_candidates}",
        f"**claimed_lb threshold**: {claimed_lb_threshold or 'none'}",
        f"**Variants generated**: {len(results)}",
        "",
        "## Top candidates by predicted LB",
        "",
        "| Rank | Name | K | Ratio | Public avg LB | ρ vs baseline | Predicted LB | Submission |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results[:15], 1):
        lines.append(
            f"| {i} | `{r.name}` | {r.K} | {r.ratio:.2f} | "
            f"{r.public_avg_claimed_lb:.5f} | {r.rho_with_baseline:.4f} | "
            f"**{r.predicted_lb:.5f}** | `submissions/{r.name}.csv` |"
        )
    lines.append("")
    lines.append("## Composition of top variant")
    lines.append("")
    top = results[0]
    lines.append(f"`{top.name}` = {(1-top.ratio):.2f} × baseline + {top.ratio:.2f} × mean of top-{top.K} curated public:")
    lines.append("")
    for tag in top.selected_tags:
        lines.append(f"- `{tag}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"Submit top 3 by predicted_lb:")
    for r in results[:3]:
        lines.append(f"  ```bash")
        lines.append(f"  kaggle competitions submit -c playground-series-s6e5 \\")
        lines.append(f"      -f submissions/{r.name}.csv -m '{r.name} curated mix'")
        lines.append(f"  ```")
    out_path.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=None,
                   help="Baseline version (auto-detect highest-LB if omitted)")
    p.add_argument("--claimed-lb-min", type=float, default=None,
                   help="Filter public submissions to claimed_lb >= this. "
                        "Default: baseline_lb - 0.001")
    p.add_argument("--k", default="3,4,5,6",
                   help="Comma-separated K values for top-K curated mean")
    p.add_argument("--ratios", default="0.20,0.30,0.50,0.70,0.90,1.00",
                   help="Comma-separated ratios for public weight")
    p.add_argument("--name-prefix", default="v19_curated",
                   help="Prefix for variant names")
    p.add_argument("--dry-run", action="store_true",
                   help="Don't write submission CSVs, just audit")
    args = p.parse_args()

    # ---- Drive recovery (Colab) ----
    print("=" * 70)
    print("DRIVE RECOVERY (if on Colab)")
    print("=" * 70)
    restore_from_drive_if_needed()
    print()

    # ---- Load data ----
    pool = load_train_pool()
    holdout = load_holdout()
    test = load_test()
    test_ids = test[ID].to_numpy()
    y_holdout = holdout[TARGET].astype(int).to_numpy()
    print(f"pool: {len(pool):,}  holdout: {len(holdout):,}  test: {len(test):,}")
    print()

    # ---- Baseline ----
    if args.baseline is None:
        # Auto-detect highest holdout AUC
        candidates = []
        for d in sorted(PROBS.iterdir()):
            if not d.is_dir():
                continue
            ho = d / "holdout.npy"
            if ho.exists():
                try:
                    auc = float(roc_auc_score(y_holdout, np.load(ho)))
                    candidates.append((d.name, auc))
                except Exception:
                    pass
        if not candidates:
            raise SystemExit("No baseline with probs/{name}/holdout.npy found")
        candidates.sort(key=lambda x: -x[1])
        args.baseline = candidates[0][0]
        print(f"Auto-detected baseline: {args.baseline} (holdout AUC {candidates[0][1]:.5f})")
    else:
        print(f"Using baseline: {args.baseline}")

    baseline_test = load_baseline_test(args.baseline, test_ids)
    baseline_lb = baseline_lb_estimate(args.baseline, y_holdout)
    print(f"Baseline LB estimate: {baseline_lb:.5f}")
    print()

    # ---- Load + filter public manifest ----
    manifest_path = ROOT / "harvest" / "v18" / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path}. Run 22_harvest_v2.py first.")
    manifest = json.loads(manifest_path.read_text())

    claimed_lb_threshold = args.claimed_lb_min
    if claimed_lb_threshold is None:
        # Auto: baseline - 0.001 (just below current best, to include strong-only)
        claimed_lb_threshold = baseline_lb - 0.001

    public_subs = load_public_subset(
        manifest,
        claimed_lb_threshold=claimed_lb_threshold,
    )
    print(f"Public candidates (claimed_lb ≥ {claimed_lb_threshold:.5f}): {len(public_subs)}")
    if not public_subs:
        print(f"\n⚠ No candidates passed the filter. Try --claimed-lb-min lower.")
        return 1
    for s in public_subs[:8]:
        print(f"  {s.tag:<50}  claimed_lb={s.claimed_lb}  votes={s.votes}")
    print()

    # ---- Sweep ----
    K_values = [int(k) for k in args.k.split(",")]
    ratios = [float(r) for r in args.ratios.split(",")]
    print(f"Sweep: K ∈ {K_values} × ratios ∈ {ratios}")
    print(f"= {len(K_values) * len(ratios)} variants")
    print()

    results = ratio_sweep(
        baseline_test=baseline_test,
        public_subs=public_subs,
        test_ids=test_ids,
        baseline_lb_estimate=baseline_lb,
        K_values=K_values,
        ratios=ratios,
        name_prefix=args.name_prefix,
    )

    # ---- Save submissions ----
    print("=" * 70)
    print(f"Top 15 candidates by predicted LB")
    print("=" * 70)
    print(f"{'Name':<28} {'K':>2} {'Ratio':>5}  {'PubAvgLB':>9}  {'ρ':>6}  {'PredLB':>8}")
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results[:15], 1):
        print(f"{r.name:<28} {r.K:>2} {r.ratio:>5.2f}  {r.public_avg_claimed_lb:>9.5f}  {r.rho_with_baseline:>6.4f}  {r.predicted_lb:>8.5f}")
        if not args.dry_run:
            pd.DataFrame({ID: test_ids, TARGET: r.test_predictions}).to_csv(
                SUBMISSIONS / f"{r.name}.csv", index=False
            )

    # ---- Audit doc ----
    audit_path = ROOT / "curated_audit.md"
    write_audit_md(
        results, args.baseline, baseline_lb, audit_path,
        n_public_candidates=len(public_subs),
        claimed_lb_threshold=claimed_lb_threshold,
    )
    print(f"\nAudit written to: {audit_path}")
    print(f"\nTop pick: submissions/{results[0].name}.csv (predicted LB {results[0].predicted_lb:.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
