# v8_blend_3way — hypothesis check

- **Parent**: `v7_realmlp`
- **Created**: 2026-05-13T07:02:30+00:00
- **Completed**: 2026-05-13T07:02:33+00:00
- **Cloud or local**: cloud
- **Git SHA**: `b40bc9a`

## Hypothesis
> Weighted 3-way blend of v1 LGB + v4 XGB + v7 RealMLP. v7 solo wins (holdout 0.9479) but trees may add decorrelated info. Nelder-Mead on OOF finds optimal weights; compare with simple 1/3 average. Predicted Δ holdout vs v7 solo: +0.001 to +0.003.

- **Predicted Δ holdout**: `+0.00150`
- **Actual Δ holdout**: `+0.00011`
- **Match**: ⚠ off
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94768`
- **Per-fold AUC**: `0.94768`, `0.94768`, `0.94768`, `0.94768`, `0.94768`
- **Holdout AUC**: `0.94803`
- **Gap holdout−oof**: `+0.00035`
- **Runtime**: `0.0s`

## Changes from parent
**Pipeline:**
  - + 3-way blend with Nelder-Mead weight tuning
**Config:**
  - `blend_method` = `'weighted'`
  - `weights` = `'[0.1377, 0.0289, 0.8333]'`

## Flags
- ⚠ prediction_undershot(actual=+0.00011 vs pred=+0.00150, ratio=0.08)
- ⚠ multiple_changes(n=3) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-13T07:04:26+00:00] LB public: 0.94735 (Kaggle 2026-05-13). Δ vs v7 LB: +0.00007 (noise floor). Nelder-Mead weights heavily favor v7 (0.83), v4 XGB nearly dropped (0.03). ρ(NN,tree)=0.97 — closer than tree-tree's 0.99 but still too high for big blend gain. Strategic lesson: with strongly correlated models, even optimal weighting can't extract much more.
