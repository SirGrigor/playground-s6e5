# v1_lgb — hypothesis check

- **Parent**: `—`
- **Created**: 2026-05-12T18:43:09+00:00
- **Completed**: 2026-05-12T18:52:31+00:00
- **Cloud or local**: local
- **Git SHA**: `55f567c`

## Hypothesis
> Plain LGB on raw + Block 1 (compound features + pit-window + tyre-life ratios + race-pace z-score). Expect OOF AUC in [0.91, 0.93].

- **Predicted Δ holdout**: `+0.00000`
- **Actual Δ**: (no parent or no result yet)
- **Match**: —
- **Confidence stated**: high

## Metrics
- **OOF AUC**: `0.94236`
- **Per-fold AUC**: `0.94247`, `0.94243`, `0.94229`, `0.94190`, `0.94271`
- **Holdout AUC**: `0.94379`
- **Gap holdout−oof**: `+0.00143`
- **Runtime**: `561.4s`

## Changes from parent
**Features:**
  - + block1 (compound, pit_window, tyre_life_ratio, race_pace_z)
**Pipeline:**
  - + sacred_holdout_v1
  - + stratified_kfold_5
**Config:**
  - `n_estimators` = `5000`
  - `early_stopping_rounds` = `100`
  - `learning_rate` = `0.05`

## Flags
- ⚠ multiple_changes(n=6) — attribution ambiguous, consider ablation

## Human notes
_None yet. Add with:_  `python -m src.diary flag v1_lgb "..."`
