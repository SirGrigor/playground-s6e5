# v9_realmlp_big — hypothesis check

- **Parent**: `v7_realmlp`
- **Created**: 2026-05-13T07:13:30+00:00
- **Completed**: 2026-05-13T07:19:30+00:00
- **Cloud or local**: cloud
- **Git SHA**: `58f20cd`

## Hypothesis
> Single change: n_ens 8 → 24. Same architecture, transforms, lr schedule. Bigger ensemble extracts more stable predictions via 1/sqrt(N) variance reduction. Predicted Δ holdout vs v7: +0.001.

- **Predicted Δ holdout**: `+0.00100`
- **Actual Δ holdout**: `+0.00016`
- **Match**: ⚠ off
- **Confidence stated**: medium

## Metrics
- **OOF AUC**: `0.94790`
- **Per-fold AUC**: `0.94754`, `0.94760`, `0.94787`, `0.94751`, `0.94897`
- **Holdout AUC**: `0.94808`
- **Gap holdout−oof**: `+0.00018`
- **Runtime**: `365.9s`

## Changes from parent
**Config:**
  - `n_ens` = `24`
  - `_prev_n_ens` = `8`

## Flags
- ⚠ prediction_undershot(actual=+0.00016 vs pred=+0.00100, ratio=0.16)
- ⚠ multiple_changes(n=2) — attribution ambiguous, consider ablation

## Human notes
- [2026-05-13T07:21:50+00:00] LB public: 0.94757 (Kaggle 2026-05-13). Δ vs v7 LB: +0.00029 — NEW BEST. Triangle aligned: OOF +0.00042, hold +0.00016, LB +0.00029. n_ens 8→24 delivered the expected variance reduction. Cumulative gain since v1: +0.00472.
