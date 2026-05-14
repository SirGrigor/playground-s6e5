"""Curated public-mix blending utilities.

The key empirical lesson from S6E5 (L19): when stacking public submissions,
**filter aggressively by claimed LB**. Mean of top-K LB-curated submissions
gives +0.00017 LB over baseline; mean of all-N submissions gives only +0.00002.

This module provides the reusable math:
  - rank_norm: scale-invariant rank normalization
  - load_public_subset: filter manifest by claimed_lb, load test predictions
  - top_k_curated_mean: take top-K by claimed_lb, compute rank-space mean
  - curated_mix: (1-ratio)*baseline + ratio*public_mean (rank space)
  - ratio_sweep: generate K × ratio grid of candidates
  - predicted_lb_score: heuristic scoring for ranking candidates

Used by:
  - notebooks/26_curated_explorer.py (standalone)
  - notebooks/25_pipeline.py Phase C (integrated)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import ID, ROOT, TARGET


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Convert raw values to uniform [0,1] ranks.

    Scale-invariant: makes blending stable across models with different
    probability output ranges (e.g., one model outputs 0.001-0.5, another
    0.01-0.9 — rank space treats them equally).
    """
    return (rankdata(x, method="average") - 1) / max(len(x) - 1, 1)


def curated_mix(
    baseline_rank: np.ndarray,
    public_mean_rank: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Linear interpolation in rank space: (1-ratio)*baseline + ratio*public_mean.

    Args:
        baseline_rank: rank-normalized predictions from our baseline.
        public_mean_rank: rank-normalized mean of public submissions (already curated).
        ratio: weight on the public mean. ratio=0 → pure baseline; ratio=1 → pure public.

    Empirical peak from S6E5: ratio ≈ 0.70 (70% public, 30% baseline) gave best LB.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    return (1.0 - ratio) * baseline_rank + ratio * public_mean_rank


# ---------------------------------------------------------------------------
# Public submission loading + filtering
# ---------------------------------------------------------------------------


@dataclass
class PublicSubmission:
    """One harvested public submission with metadata."""
    tag: str
    slug: str
    claimed_lb: float | None
    test_path: Path
    votes: int = 0


def load_public_subset(
    manifest: list[dict],
    claimed_lb_threshold: float | None = None,
    only_verdict: str = "INCLUDE-TEST",
) -> list[PublicSubmission]:
    """Filter manifest entries by quality. Return PublicSubmission list sorted by LB desc.

    Args:
        manifest: parsed harvest/v18/manifest.json contents.
        claimed_lb_threshold: drop entries with claimed_lb below this (or unknown LB if set).
            None means accept all entries with the matching verdict.
        only_verdict: filter to entries with this verdict. Default INCLUDE-TEST.
    """
    out: list[PublicSubmission] = []
    for entry in manifest:
        if entry.get("verdict") != only_verdict:
            continue
        claimed_lb = entry.get("claimed_lb")
        if claimed_lb_threshold is not None:
            if claimed_lb is None or claimed_lb < claimed_lb_threshold:
                continue
        files = entry.get("files_found") or {}
        sub_path = files.get("submission")
        if sub_path is None:
            continue
        full_path = ROOT / sub_path
        if not full_path.exists():
            continue
        out.append(PublicSubmission(
            tag=entry.get("tag", ""),
            slug=entry.get("slug", ""),
            claimed_lb=claimed_lb,
            test_path=full_path,
            votes=int(entry.get("votes", 0)),
        ))
    # Sort by claimed_lb desc (None last), then votes desc
    out.sort(
        key=lambda p: (
            -(p.claimed_lb if p.claimed_lb is not None else -1.0),
            -p.votes,
        )
    )
    return out


def top_k_curated_mean(
    public_subs: list[PublicSubmission],
    K: int,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, list[PublicSubmission]]:
    """Take top-K by claimed_lb, load each, rank-normalize, return mean rank + the K selected.

    Returns:
        (mean_rank, selected_subs)
        mean_rank: 1D array of mean ranks aligned to test_ids.
        selected_subs: the K PublicSubmission objects used (for audit).
    """
    if not public_subs:
        raise ValueError("public_subs is empty — nothing to curate")
    K = min(K, len(public_subs))
    selected = public_subs[:K]
    rank_matrix = np.empty((K, len(test_ids)))
    for i, sub in enumerate(selected):
        df = pd.read_csv(sub.test_path).set_index(ID)
        pred_col = [c for c in df.columns if c != ID][0] if ID not in df.columns \
            else [c for c in df.columns if c != ID][0]
        # Re-read to get pred_col reliably
        df = pd.read_csv(sub.test_path)
        pred_col = [c for c in df.columns if c != ID][0]
        s = df.set_index(ID)[pred_col].loc[test_ids]
        rank_matrix[i] = rank_norm(s.to_numpy())
    return rank_matrix.mean(axis=0), list(selected)


# ---------------------------------------------------------------------------
# Ratio-sweep generation
# ---------------------------------------------------------------------------


@dataclass
class RatioSweepResult:
    """One (K, ratio) variant from the sweep."""
    name: str
    K: int
    ratio: float
    public_avg_claimed_lb: float
    selected_tags: list[str]
    predicted_lb: float
    rho_with_baseline: float
    test_predictions: np.ndarray


def ratio_sweep(
    baseline_test: np.ndarray,
    public_subs: list[PublicSubmission],
    test_ids: np.ndarray,
    baseline_lb_estimate: float,
    K_values: Iterable[int] = (3, 4, 5, 6),
    ratios: Iterable[float] = (0.20, 0.30, 0.50, 0.70, 0.90, 1.00),
    name_prefix: str = "curated",
) -> list[RatioSweepResult]:
    """Generate K × ratio grid of curated-mix candidates.

    Args:
        baseline_test: test predictions of the current baseline (raw, will be rank-normed).
        public_subs: already-filtered & sorted PublicSubmissions (descending claimed_lb).
        test_ids: test row ids in canonical order.
        baseline_lb_estimate: LB of the baseline (for predicted_lb computation).
        K_values: how many top-LB publics to include in the curated mean.
        ratios: ratio of public-mean weight. 0=pure baseline, 1=pure public.
        name_prefix: prefix for variant names (e.g., "curated_k4_r70").

    Returns:
        List of RatioSweepResult, ordered by descending predicted_lb.
    """
    baseline_rank = rank_norm(baseline_test)
    results: list[RatioSweepResult] = []
    for K in K_values:
        if K > len(public_subs):
            continue
        public_mean, selected = top_k_curated_mean(public_subs, K, test_ids)
        public_avg_lb = float(np.mean([
            s.claimed_lb for s in selected if s.claimed_lb is not None
        ])) if any(s.claimed_lb is not None for s in selected) else float("nan")
        rho_baseline_public = float(np.corrcoef(baseline_rank, public_mean)[0, 1])
        for ratio in ratios:
            mix = curated_mix(baseline_rank, public_mean, ratio)
            rho_with_baseline = float(np.corrcoef(mix, baseline_rank)[0, 1])
            predicted = predicted_lb_score(
                baseline_lb=baseline_lb_estimate,
                public_avg_lb=public_avg_lb,
                ratio=ratio,
                rho_baseline_public=rho_baseline_public,
            )
            results.append(RatioSweepResult(
                name=f"{name_prefix}_k{K}_r{int(ratio*100):02d}",
                K=K, ratio=ratio,
                public_avg_claimed_lb=public_avg_lb,
                selected_tags=[s.tag for s in selected],
                predicted_lb=predicted,
                rho_with_baseline=rho_with_baseline,
                test_predictions=mix,
            ))
    results.sort(key=lambda r: -r.predicted_lb)
    return results


def predicted_lb_score(
    baseline_lb: float,
    public_avg_lb: float,
    ratio: float,
    rho_baseline_public: float,
    diversity_bonus_scale: float = 0.0001,
) -> float:
    """Heuristic predicted LB for a curated mix variant.

    Combines:
      - Linear interpolation: (1-r)*baseline_lb + r*public_avg_lb
      - Diversity bonus: small upward correction when baseline and public mean
        have lower correlation (more noise diversity in the blend).

    Empirically (S6E5):
      - pure baseline (v18.007) LB = 0.95406
      - 30/70 curated (v19.006) LB = 0.95423, top4 avg ≈ 0.95417, ρ ≈ 0.99
      - pure curated (v19.007) LB = 0.95419

    The diversity bonus accounts for the fact that 30/70 beat both pure cases.
    """
    if np.isnan(public_avg_lb):
        public_avg_lb = baseline_lb  # no curation info; assume parity
    linear = (1.0 - ratio) * baseline_lb + ratio * public_avg_lb
    # Diversity bonus peaks when ratio is balanced AND correlation is < 1
    # ratio*(1-ratio) is symmetric around 0.5 (peaks at 0.25)
    diversity_bonus = (
        4.0 * ratio * (1.0 - ratio)  # 0 at extremes, 1 at 0.5
        * max(0.0, 1.0 - rho_baseline_public)  # 0 at ρ=1, grows as ρ drops
        * diversity_bonus_scale
    )
    return linear + diversity_bonus
