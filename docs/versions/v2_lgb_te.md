# v2_lgb_te — hypothesis check

- **Parent**: `v1_lgb`
- **Created**: 2026-05-12T19:27:51+00:00
- **Completed**: 2026-05-12T19:37:01+00:00
- **Cloud or local**: cloud
- **Git SHA**: `af2bd26`

## Hypothesis
> Driver weak alone (MI=0.009) but (Driver,Compound) and (Race,Compound) encode strategic personality. sklearn TargetEncoder(cv=5) adds 4 fold-safe numeric features. Predicted Δ holdout vs v1: +0.004.

- **Predicted Δ holdout**: `+0.00400`
- **Actual Δ holdout**: `-0.00055`
- **Match**: ✗ sign mismatch
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94174`
- **Per-fold AUC**: `0.94167`, `0.94169`, `0.94155`, `0.94137`, `0.94242`
- **Holdout AUC**: `0.94324`
- **Gap holdout−oof**: `+0.00150`
- **Runtime**: `549.7s`

## Changes from parent
**Features:**
  - + TE_Driver, TE_Race, TE_Driver_x_Compound, TE_Race_x_Compound
**Pipeline:**
  - + sklearn_TargetEncoder(cv=5)

## Flags
- ⚠ prediction_sign_mismatch(actual=-0.00055 vs pred=+0.00400)
- ⚠ multiple_changes(n=2) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-12T19:42:29+00:00] LB public: 0.94223 (Kaggle 2026-05-12). Δ vs v1 LB: -0.00062. Confirmed across OOF/holdout/LB. TE features are redundant with LGB's native categorical handling on these cardinalities; the model has to fit them but they crowd out useful splits via feature_fraction=0.85. DECISION: drop TE block. v3 parent=v1_lgb.
