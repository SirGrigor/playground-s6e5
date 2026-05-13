"""v13b — Rank-blend our stack with harvested public submissions.

Pre-experiment checklist (per docs/pitfalls.md):
  Hypothesis: A rank-average of (our v11, our v12 if ready, plus all public
              submissions whose rank-correlation with v11 is < 0.95) beats
              pure v11 on private LB. Public sources contribute *uncorrelated
              noise structure* even when their absolute AUC is lower than ours,
              because the noise cancels in the average.
  Parent: v11_blend_4way (LB 0.94774)
  Predicted Δ holdout: N/A (no labels for public predictions).
                       Public-LB Δ proxy: +0.0005 to +0.002 if any harvested
                       submission has ρ_spearman(v11) < 0.95.
  Most relevant pitfall: #6 (don't trust public LB blindly), L18 (submission
                         selection: don't pick a blend by holdout AUC if only
                         tested 2-3 weight configs).
  Output: submissions/v13_*.csv (multiple candidates for human selection)
          + harvest/v13/diversity_report.md

Strategy:
  1. Load every CSV in submissions/ + harvest/v13/*/submission.csv.
  2. Align all on test id, rank-transform each column.
  3. Compute pairwise Spearman ρ matrix — diversity audit.
  4. Generate three candidate blends:
       a. uniform_all: rank-average of v11 + all valid public
       b. uniform_diverse: rank-average of v11 + public with ρ(v11) < 0.95
       c. weighted_lb: ρ-pruned, weighted by claimed LB if known else uniform
  5. Write candidates as v13_*.csv. Human picks 2 for the final Kaggle
     submission slots based on the diversity report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ID, ROOT, SUBMISSIONS, TARGET

HARVEST = ROOT / "harvest" / "v13"
OURS_BASELINE = "v11_blend_4way"
DIVERSITY_THRESHOLD = 0.95  # ρ_spearman above this → drop as redundant


def load_submission(path: Path, label: str) -> pd.Series:
    """Load a submission CSV and return a Series indexed by id, named `label`.
    Tolerates the prediction column being named anything (PitNextLap, target, etc.)."""
    df = pd.read_csv(path)
    pred_cols = [c for c in df.columns if c != ID]
    if len(pred_cols) != 1:
        raise ValueError(f"{path}: expected 1 prediction column, found {pred_cols}")
    return df.set_index(ID)[pred_cols[0]].rename(label)


def collect_predictions() -> pd.DataFrame:
    """Load ours + harvested into a single DataFrame indexed by id, columns=labels."""
    series: list[pd.Series] = []

    ours_path = SUBMISSIONS / f"{OURS_BASELINE}.csv"
    if not ours_path.exists():
        raise FileNotFoundError(f"{ours_path} not found — can't blend without our baseline")
    series.append(load_submission(ours_path, OURS_BASELINE))

    v12_path = SUBMISSIONS / "v12_factory.csv"
    if v12_path.exists():
        series.append(load_submission(v12_path, "v12_factory"))
        print(f"  + v12_factory included")
    else:
        print(f"  - v12_factory missing (skipped)")

    manifest_path = HARVEST / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found — run 15_v13_harvest.py first")
    manifest = json.loads(manifest_path.read_text())

    for entry in manifest:
        if not entry.get("valid"):
            print(f"  - {entry['tag']}: invalid, skipped ({entry.get('issues')})")
            continue
        sub_path = ROOT / entry["downloaded_csv"]
        series.append(load_submission(sub_path, entry["tag"]))
        print(f"  + {entry['tag']} included from {sub_path.relative_to(ROOT)}")

    df = pd.concat(series, axis=1).sort_index()
    if df.isna().any().any():
        missing = df.isna().sum()
        raise ValueError(f"NaNs after align — submissions disagree on test ids:\n{missing}")
    return df


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Each column → uniform [0,1] rank. Average-ranks for ties."""
    out = pd.DataFrame(index=df.index)
    n = len(df)
    for col in df.columns:
        out[col] = (rankdata(df[col].to_numpy(), method="average") - 1) / max(n - 1, 1)
    return out


def spearman_matrix(ranks: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ = Pearson on ranks. Reuses rank-normalized frame."""
    return ranks.corr(method="pearson")


def write_submission(name: str, ranks: np.ndarray, ids: np.ndarray) -> Path:
    """Write a rank-blend as a submission CSV."""
    out_path = SUBMISSIONS / f"{name}.csv"
    pd.DataFrame({ID: ids, TARGET: ranks}).to_csv(out_path, index=False)
    return out_path


def main() -> int:
    print("=== Collecting predictions ===")
    raw = collect_predictions()
    print(f"\nAligned matrix: {raw.shape[0]} rows × {raw.shape[1]} columns")
    print(f"Columns: {list(raw.columns)}\n")

    ranks = rank_normalize(raw)
    rho = spearman_matrix(ranks)
    print("=== Pairwise Spearman ρ ===")
    print(rho.round(4).to_string())
    print()

    # Diversity-pruned set: keep ours unconditionally, drop any public with
    # ρ(v11) ≥ threshold.
    public_cols = [c for c in ranks.columns if c not in (OURS_BASELINE, "v12_factory")]
    diverse_public = [c for c in public_cols if rho.at[c, OURS_BASELINE] < DIVERSITY_THRESHOLD]
    redundant = [c for c in public_cols if c not in diverse_public]

    print(f"Diversity filter (ρ({OURS_BASELINE}) < {DIVERSITY_THRESHOLD}):")
    print(f"  KEEP    public: {diverse_public}")
    print(f"  DROP    public: {redundant}  (too correlated to add signal)")
    print()

    ids = raw.index.to_numpy()
    ours = [OURS_BASELINE] + (["v12_factory"] if "v12_factory" in ranks.columns else [])

    # Candidate A — uniform across ours + ALL valid public
    cand_a_cols = ours + public_cols
    cand_a = ranks[cand_a_cols].mean(axis=1).to_numpy()
    pa = write_submission("v13_uniform_all", cand_a, ids)
    print(f"  → {pa.name}  ({len(cand_a_cols)} cols, uniform)")

    # Candidate B — uniform across ours + ρ-pruned public
    # SKIP if pruning leaves only our predictors — that produces a rank-identical
    # twin of v11 which would just consume a Kaggle submission slot.
    cand_b_cols = ours + diverse_public
    if not diverse_public:
        print(f"  → v13_uniform_diverse SKIPPED — all public ρ ≥ {DIVERSITY_THRESHOLD}, blend = rank(v11)")
    elif cand_b_cols == cand_a_cols:
        print(f"  → v13_uniform_diverse SKIPPED (identical to uniform_all)")
    else:
        cand_b = ranks[cand_b_cols].mean(axis=1).to_numpy()
        pb = write_submission("v13_uniform_diverse", cand_b, ids)
        print(f"  → {pb.name}  ({len(cand_b_cols)} cols, ρ-pruned)")

    # Candidate C — weighted: 2x weight to ours, 1x to ρ-pruned public.
    # Same degeneracy gate.
    if not diverse_public:
        print(f"  → v13_weighted_ours2x SKIPPED — no diverse public to weight against")
    else:
        weights = {c: 2.0 for c in ours} | {c: 1.0 for c in diverse_public}
        cand_c_cols = list(weights)
        w_vec = np.array([weights[c] for c in cand_c_cols])
        w_vec = w_vec / w_vec.sum()
        cand_c = (ranks[cand_c_cols].to_numpy() * w_vec).sum(axis=1)
        pc = write_submission("v13_weighted_ours2x", cand_c, ids)
        print(f"  → {pc.name}  ({len(cand_c_cols)} cols, ours 2x)")
    print()

    # Write diversity report
    report_path = HARVEST / "diversity_report.md"
    lines = [
        "# v13 Diversity Report",
        "",
        f"Test rows: {raw.shape[0]}",
        f"Aligned predictors: {raw.shape[1]} ({list(raw.columns)})",
        f"Diversity threshold ρ(v11) < {DIVERSITY_THRESHOLD}",
        "",
        "## Pairwise Spearman ρ",
        "",
        "```",
        rho.round(4).to_string(),
        "```",
        "",
        "## Candidates",
        "",
        f"- `v13_uniform_all` — uniform mean of {len(cand_a_cols)} predictors",
        f"- `v13_uniform_diverse` — uniform mean of {len(cand_b_cols)} (drops ρ≥{DIVERSITY_THRESHOLD})",
        f"- `v13_weighted_ours2x` — same as diverse, but ours weighted 2x",
        "",
        "## Decision",
        "",
        "Kaggle final submissions: pick 2 of the 3 candidates.",
        "Default pick if unsure: `v13_uniform_diverse` (most conservative).",
        "Second pick: whichever has the lowest ρ across its components.",
    ]
    report_path.write_text("\n".join(lines))
    print(f"Diversity report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
