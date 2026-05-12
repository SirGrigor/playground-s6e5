# v3_lgb_compxprog — hypothesis check

- **Parent**: `v1_lgb`
- **Created**: 2026-05-12T19:50:51+00:00
- **Completed**: 2026-05-12T19:55:28+00:00
- **Cloud or local**: cloud
- **Git SHA**: `456a5df`

## Hypothesis
> Compound × RaceProgress is the strongest EDA-confirmed interaction (different pit windows per compound). Adding it as a single native categorical (5 × 40 = up to 200 combos) lets LGB partition the joint pit-window space directly. Predicted Δ holdout vs v1: +0.0015.

- **Predicted Δ holdout**: `+0.00150`
- **Actual Δ holdout**: `-0.00277`
- **Match**: ✗ sign mismatch
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.93996`
- **Per-fold AUC**: `0.93975`, `0.93992`, `0.93998`, `0.93920`, `0.94095`
- **Holdout AUC**: `0.94102`
- **Gap holdout−oof**: `+0.00106`
- **Runtime**: `276.2s`

## Changes from parent
**Features:**
  - + compound_x_progress_bin (native cat)

## Flags
- ⚠ silent_regression(Δhold=-0.00277 vs v1_lgb)
- ⚠ prediction_sign_mismatch(actual=-0.00277 vs pred=+0.00150)

## Human notes
- [2026-05-12T20:01:12+00:00] LB public: 0.94035 (Kaggle 2026-05-12). Δ vs v1 LB: -0.00250. Confirmed across OOF/holdout/LB. Adding compound_x_progress_bin as native cat HURT more than TE (v2) did. Pattern: LGB on Block 1 is plateaued — adding features in either mechanism (TE numeric or native cat) crowds out productive splits via feature_fraction. DECISION: v1 is our best LGB-only config. v4 changes model family to XGB on same Block 1.
