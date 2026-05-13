# v13 Diversity Report

Test rows: 188165
Aligned predictors: 3 (['v11_blend_4way', 'nn_residual', 'catboost_10fold'])
Diversity threshold ρ(v11) < 0.95

## Pairwise Spearman ρ

```
                 v11_blend_4way  nn_residual  catboost_10fold
v11_blend_4way           1.0000       0.9821           0.9768
nn_residual              0.9821       1.0000           0.9906
catboost_10fold          0.9768       0.9906           1.0000
```

## Candidates

- `v13_uniform_all` — uniform mean of 3 predictors
- `v13_uniform_diverse` — uniform mean of 1 (drops ρ≥0.95)
- `v13_weighted_ours2x` — same as diverse, but ours weighted 2x

## Decision

Kaggle final submissions: pick 2 of the 3 candidates.
Default pick if unsure: `v13_uniform_diverse` (most conservative).
Second pick: whichever has the lowest ρ across its components.