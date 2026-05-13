# v6_lgb_digits — hypothesis check

- **Parent**: `v1_lgb`
- **Created**: 2026-05-13T06:20:16+00:00
- **Completed**: 2026-05-13T06:29:43+00:00
- **Cloud or local**: cloud
- **Git SHA**: `aff455a`

## Hypothesis
> Block 4: 21 digit-extraction int8 features (yunsuxiaozi pattern) across 7 numeric source columns. These expose decimal-quantization patterns LGB cannot build via threshold splits (modular arithmetic). Orthogonal info — different from v2/v3 which were redundant cat encodings. Predicted Δ holdout vs v1: +0.0015.

- **Predicted Δ holdout**: `+0.00150`
- **Actual Δ holdout**: `-0.00043`
- **Match**: ✗ sign mismatch
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94178`
- **Per-fold AUC**: `0.94198`, `0.94146`, `0.94151`, `0.94154`, `0.94243`
- **Holdout AUC**: `0.94335`
- **Gap holdout−oof**: `+0.00157`
- **Runtime**: `566.7s`

## Changes from parent
**Features:**
  - + 21 digit features (Block 4 yunsuxiaozi)

## Flags
- ⚠ prediction_sign_mismatch(actual=-0.00043 vs pred=+0.00150)

## Human notes
- [2026-05-13T06:33:57+00:00] LB public: 0.94226 (Kaggle 2026-05-13). Δ vs v1 LB: -0.00059. Confirmed across OOF/holdout/LB. Third LGB feature-addition regression in a row (v2 -0.0006, v3 -0.0025, v6 -0.0006). EMPIRICAL LAW: LGB+Block 1 is a local maximum for this comp — adding features always regresses regardless of mechanism. DECISION: drop v6 solo. Move to v7 = RealMLP (different model class) for genuine diversity.
