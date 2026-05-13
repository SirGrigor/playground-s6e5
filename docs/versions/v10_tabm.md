# v10_tabm — hypothesis check

- **Parent**: `v9_realmlp_big`
- **Created**: 2026-05-13T07:25:23+00:00
- **Completed**: 2026-05-13T07:30:35+00:00
- **Cloud or local**: cloud
- **Git SHA**: `9ab97e3`

## Hypothesis
> TabM_D (pytabkit) — different neural architecture than RealMLP. Same Block 1 features, different network design (token-based / mixer-style). Predicted Δ vs v9 solo: -0.001 to +0.001 (architectural neutrality). Real value: ρ(RealMLP, TabM_D) should be 0.90-0.95 → blend payoff in v11.

- **Predicted Δ holdout**: `+0.00000`
- **Actual Δ holdout**: `-0.00098`
- **Match**: ✗ sign mismatch
- **Confidence stated**: low

## Metrics
- **OOF AUC**: `0.94660`
- **Per-fold AUC**: `0.94596`, `0.94640`, `0.94679`, `0.94611`, `0.94773`
- **Holdout AUC**: `0.94710`
- **Gap holdout−oof**: `+0.00050`
- **Runtime**: `310.8s`

## Changes from parent
**Pipeline:**
  - + pytabkit.TabM_D_Classifier (different neural architecture)
**Config:**
  - `algo` = `'tabm'`
  - `tabm_k` = `32`
  - `n_epochs` = `3`
  - `d_block` = `256`
  - `n_blocks` = `3`

## Flags
- ⚠ multiple_changes(n=6) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-13T07:33:05+00:00] LB public: 0.94632 (Kaggle 2026-05-13). Δ vs v9 LB: -0.00125 (solo underperforms RealMLP by ~0.001). Triangle aligned: OOF -0.00130, hold -0.00098, LB -0.00125 — within predicted neutral range. KEY TEST is the v11 blend: if ρ(v9, v10) < 0.95, decorrelated errors compound → blend lifts. If ρ > 0.97, redundant within neural family.
