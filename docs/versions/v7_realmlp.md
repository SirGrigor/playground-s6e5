# v7_realmlp — hypothesis check

- **Parent**: `v1_lgb`
- **Created**: 2026-05-13T06:51:15+00:00
- **Completed**: 2026-05-13T06:53:36+00:00
- **Cloud or local**: cloud
- **Git SHA**: `c314bff`

## Hypothesis
> RealMLP_TD (pytabkit) on Block 1 — genuine model-class diversity. Empirically, more LGB features and different tree algo both plateaued at ~0.943. Neural net w/ PLR embeddings has structurally different decision boundaries than trees. Expected solo Δ vs v1: 0 to +0.001. Real value: ρ(v7, v1) and ρ(v7, v4) should be 0.85-0.93 (vs 0.99 for tree-tree) → big blend payoff in v8 = blend(v1, v4, v7).

- **Predicted Δ holdout**: `+0.00050`
- **Actual Δ holdout**: `+0.00413`
- **Match**: ⚠ off
- **Confidence stated**: low

## Metrics
- **OOF AUC**: `0.94748`
- **Per-fold AUC**: `0.94708`, `0.94718`, `0.94727`, `0.94711`, `0.94875`
- **Holdout AUC**: `0.94792`
- **Gap holdout−oof**: `+0.00044`
- **Runtime**: `140.1s`

## Changes from parent
**Pipeline:**
  - + pytabkit.RealMLP_TD_Classifier (neural net)
**Config:**
  - `algo` = `'realmlp'`
  - `n_ens` = `8`
  - `n_epochs` = `5`
  - `hidden_sizes` = `[512, 256, 128]`

## Flags
- ⚠ prediction_overshot(actual=+0.00413 vs pred=+0.00050, ratio=8.26)
- ⚠ multiple_changes(n=5) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-13T06:56:24+00:00] LB public: 0.94728 (Kaggle 2026-05-13). Δ vs v1 LB: +0.00443 (MASSIVE WIN — 8x predicted!). Triangle all +0.004 aligned. RealMLP is genuinely better solo than trees on this data — PLR embeddings + learned cat embeddings + 5x ensemble. Gap to top LB closed from 0.0117 to 0.00752. v7 becomes new BEST SOLO. Next: v8 = weighted blend(v1, v4, v7) — ρ(v7, trees) likely 0.85-0.90 → expect another +0.001-0.003.
