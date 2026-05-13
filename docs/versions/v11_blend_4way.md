# v11_blend_4way — hypothesis check

- **Parent**: `v9_realmlp_big`
- **Created**: 2026-05-13T07:35:00+00:00
- **Completed**: 2026-05-13T07:35:30+00:00
- **Cloud or local**: cloud
- **Git SHA**: `441122e`

## Hypothesis
> 4-way blend of v9 RealMLP big + v10 TabM_D + v1 LGB + v4 XGB. Tests architectural diversity within pytabkit. Predicted Δ vs v9 solo: +0.0005 to +0.002 depending on ρ(v9, v10).

- **Predicted Δ holdout**: `+0.00100`
- **Actual Δ holdout**: `+0.00025`
- **Match**: ⚠ off
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94820`
- **Per-fold AUC**: `0.94820`, `0.94820`, `0.94820`, `0.94820`, `0.94820`
- **Holdout AUC**: `0.94833`
- **Gap holdout−oof**: `+0.00013`
- **Runtime**: `0.0s`

## Changes from parent
**Pipeline:**
  - + 4-way Nelder-Mead blend
**Config:**
  - `blend_method` = `'weighted'`
  - `weights` = `'[0.696, 0.247, 0.057, 0.0]'`

## Flags
- ⚠ prediction_undershot(actual=+0.00025 vs pred=+0.00100, ratio=0.25)
- ⚠ multiple_changes(n=3) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-13T07:40:22+00:00] LB public: 0.94774 (Kaggle 2026-05-13). Δ vs v9 LB: +0.00017 — new best by a hair. CRITICAL FINDING: ρ(v9,v10)=0.988 — architectural diversity WITHIN pytabkit gives essentially the same correlation as tree-tree (0.989). All 4 internal models share too much signal. Cumulative gain since v1: +0.00489 LB. Internal diversity is exhausted; next big lever is external (public OOFs per L16 audit).
