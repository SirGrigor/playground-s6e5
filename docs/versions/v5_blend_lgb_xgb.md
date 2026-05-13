# v5_blend_lgb_xgb — hypothesis check

- **Parent**: `v1_lgb`
- **Created**: 2026-05-13T05:42:37+00:00
- **Completed**: 2026-05-13T05:42:38+00:00
- **Cloud or local**: cloud
- **Git SHA**: `91e9c74`

## Hypothesis
> Simple average of v1 LGB + v4 XGB. Both tied solo (all metrics within 0.0003). Different decision algorithms (LGB Fisher vs XGB ordinal) → if OOF correlation < 0.97, decorrelated errors compound. Predicted Δ holdout vs best single solo: +0.0008 (mid of [0.0005, 0.0015]).

- **Predicted Δ holdout**: `+0.00080`
- **Actual Δ holdout**: `+0.00033`
- **Match**: ⚠ off
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94321`
- **Per-fold AUC**: `0.94321`, `0.94321`, `0.94321`, `0.94321`, `0.94321`
- **Holdout AUC**: `0.94412`
- **Gap holdout−oof**: `+0.00091`
- **Runtime**: `0.0s`

## Changes from parent
**Pipeline:**
  - + multi-model blend (v1_lgb + v4_xgb)
**Config:**
  - `blend_method` = `'simple_average'`
  - `weights` = `'0.5/0.5'`

## Flags
- ⚠ prediction_undershot(actual=+0.00033 vs pred=+0.00080, ratio=0.41)
- ⚠ multiple_changes(n=3) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-13T05:50:23+00:00] LB public: 0.94307 (Kaggle 2026-05-13). Δ vs v1 LB: +0.00022 — FIRST IMPROVEMENT. ρ_oof=0.9895 confirms L13: same features + different tree algo = nearly identical predictions. Blend lift is at noise floor. v6 needs GENUINE diversity: digit features OR RealMLP (different model class).
