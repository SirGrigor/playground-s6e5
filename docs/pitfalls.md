# S6E5 Pitfalls — Lessons Carried from S6E4

**Read this before every modeling session.** Each item below is something we got wrong on S6E4 and the discipline we apply to avoid it on S6E5.

## 1. Sacred holdout — split BEFORE any work

**S6E4 mistake**: started building features and CV without a held-out test set; couldn't tell if our OOF was trustworthy until Phase 14C (very late).

**S6E5 protocol**:
- 20% sacred holdout, stratified on target, seed locked
- Row indices persisted to `data/splits/holdout_v1.parquet` and **tracked in git**
- `src/data.py::load_data()` returns `(train_pool, holdout, test)` — there is NO function returning the full train+holdout combined
- Touch holdout ONLY at end of each version to validate OOF estimate
- If `holdout_auc - oof_auc > 0.005`, methodology bug — investigate

## 2. Wrong metric — verify on day 1

**S6E4 mistake**: 4 hours optimizing accuracy before discovering the metric was `balanced_accuracy`.

**S6E5 status**: ✓ verified — ROC-AUC, confirmed in every baseline notebook + competition page.

## 3. Public LB > holdout inversion (L3)

**S6E4 lesson**: sometimes public LB scores BETTER than honest holdout. Don't trust the inversion — the holdout is the truer signal because the public LB uses only 20% of test rows and has sampling variance.

**S6E5 protocol**: track CV vs holdout vs LB per version. If `holdout < CV` more than 0.003, methodology is leaking. If `LB > holdout`, treat as noise — don't optimize toward it.

## 4. Pseudo-labeling on a well-learned problem (L4)

**S6E4 mistake (Phase 13)**: pseudo-labeled high-confidence test rows, retrained, watched OOF rise to 97.96% on augmented data. On real validation: 97.30% — same as before. The model just learned its own predictions better.

**S6E5 rule**: don't pseudo-label until the model is clearly underfitting or has a specific minority class we cannot learn directly. Default = NO pseudo-labels.

## 5. Manual target encoding causes fold collapse (Phase 12 lesson)

**S6E4 mistake**: built a manual OrderedTargetEncoder, fold 4 collapsed to 90.10% accuracy (vs ~97% normal) because of leakage between rows in mixed synthetic+original data.

**S6E5 rule**: ONLY use `sklearn.preprocessing.TargetEncoder(cv=5)` (available since sklearn 1.3). Never manual TE. If we need to extend, write tests against a 4-fold collapse pattern first.

## 6. Original dataset is for FE insights, NOT training (Phase 14B)

**S6E4 mistake**: trained XGB+LGB on the 10K original dataset alone, hit OOF 98.91%. LB on synthetic test was ~95%. Distribution shift > additive noise.

**S6E5 rule**: original dataset (`f1_strategy_dataset_v4.csv`) is read-only for FE insights — find which features matter, extract thresholds, validate decision boundaries. **Do NOT use as primary training data.** Adversarial AUC train↔original = 0.38 (significant drift confirmed in EDA).

## 7. Optuna threshold search on synthetic data (Phase 14)

**S6E4 mistake**: ran Optuna for 20 trials to find exact thresholds (soil, temp, rain, wind) on synthetic data. Returned different round numbers from defaults but scored identically. The threshold landscape on synthetic was flat — Optuna couldn't distinguish.

**S6E5 rule**: if doing threshold search, do it on the ORIGINAL dataset. Round numbers from EDA are usually fine for synthetic.

## 8. Rule engine overlay on already-good ML model (Phase 17)

**S6E4 mistake**: built a rule engine to force High/Low predictions on unambiguous cases. The tree models had already learned the same rules. Rules added nothing.

**S6E5 rule**: only use rules to IDENTIFY ambiguous cases for ML — not as post-processing on ML output.

## 9. Algorithm name is NOT diversity (L13)

**S6E4 mistake (Phase 6/v32)**: trained XGB and LGB on same features expecting blend gain. Correlation ρ > 0.99. Blend hurt.

**S6E5 rule**: model family is one of three valid diversity axes (basin regularization, feature space, model family). For trees, family choice DOES matter (XGB vs LGB vs CatBoost vs HistGB use different splitting). But also explore **pytabkit RealMLP** — confirmed L17 baseline — and HistGradientBoosting (sklearn) for genuine architectural diversity.

## 10. The "many bespoke versions" trap (S6E4 post-mortem L13 + grand summary)

**S6E4 mistake**: built v1, v2, ..., v34 — each a different hypothesis. Each took 30-60 min, total ~30 hours. Final solo LB only 0.97654.

**S6E5 rule**:
- ONE feature pipeline in `src/features.py`, versioned by config flags
- ONE training function in `src/train.py::train_variant(algo, ...)`
- Variants are configuration, not new scripts
- Save OOF + holdout + test probs to `probs/<version>/` for blending later
- Target: ~5-7 versions total, each ~10 min of effort

## 11. Compute parity wall — public-OOF aggregation closes it (L15 corrected, L16)

**S6E4 closure realization**: solo 0.97654 was below #20 LB (0.98070+). We attributed it to NVIDIA-tier compute. But the audit showed the winner (cstdy, 0.98158) stacked ~25 public OOFs from other notebooks.

**S6E5 rule**: Phase 6b — Day 14 onward, scan public notebooks for published OOFs. Build the meta-stack. Solo modeling is ONE voice in a 20+ voice ensemble, not the whole show.

## 12. Submission selection under shake-up risk (L18)

**S6E4 audit finding**: cstdy's strong-private submission was LOWER-public than alternatives. 763 row differences flipped the rank. The top-16 LB cluster on S6E5 is currently 0.00054 wide — same pattern.

**S6E5 protocol**: at submission lock time, pick TWO submissions with different stability profiles:
- A: max-public (gambles that public ≈ private)
- B: boundary-stable (smoother decisions on uncertain rows; trades a bit of public for shake-up resilience)

## Pre-experiment checklist (paste at the top of each `notebooks/0X_*.py`)

```
# Pre-experiment checklist
# Hypothesis: <ONE variable being changed>
# Predicted effect: <from theory>
# Most relevant pitfall: <one of the 12 above>
# Validation plan: 5-fold CV on train_pool + sacred holdout evaluation at end
# Abort signal: if 1-fold mini-test < <threshold>, stop
# Output: probs/<version>/oof.npy, probs/<version>/holdout.npy, probs/<version>/test.npy
# Code hygiene: features from src/features.py, seeds from src/config.py, no copy-paste
```
