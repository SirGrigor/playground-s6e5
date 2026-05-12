# v4_xgb — hypothesis check

- **Parent**: `v1_lgb`
- **Created**: 2026-05-12T20:04:55+00:00
- **Completed**: 2026-05-12T20:12:47+00:00
- **Cloud or local**: cloud
- **Git SHA**: `14d61cc`

## Hypothesis
> v2 and v3 proved LGB on Block 1 is plateaued. Switching to XGBoost (level-wise growth + ordinal cat encoding vs LGB leaf-wise + Fisher). Same features, different model family. Predicted Δ holdout vs v1 LGB: +0.0010 single-model; sets up v6 blend where decorrelated errors compound.

- **Predicted Δ holdout**: `+0.00100`
- **Actual Δ holdout**: `-0.00003`
- **Match**: ✗ sign mismatch
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94263`
- **Per-fold AUC**: `0.94257`, `0.94237`, `0.94242`, `0.94242`, `0.94337`
- **Holdout AUC**: `0.94375`
- **Gap holdout−oof**: `+0.00112`
- **Runtime**: `471.3s`

## Changes from parent
**Pipeline:**
  - + xgboost (replaces lightgbm as base model)
**Config:**
  - `algo` = `'xgb'`
  - `n_estimators` = `5000`
  - `max_depth` = `6`
  - `learning_rate` = `0.05`

## Flags
- ⚠ prediction_sign_mismatch(actual=-0.00003 vs pred=+0.00100)
- ⚠ multiple_changes(n=5) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-12T20:22:08+00:00] LB public: 0.94260 (Kaggle 2026-05-12). Δ vs v1 LB: -0.00025 (TIED within noise). All 3 metrics within 0.0003: OOF 0.94262, hold 0.94375, LB 0.94260. XGB and LGB equivalent solo — perfect blend setup if OOF correlation < 0.97.
