# S6E5 — Phase 0 Reconnaissance (2026-05-11)

## Task

- Binary classification, target = `PitNextLap` (will the driver pit on the NEXT lap)
- Metric: **ROC-AUC** (confirmed via every baseline notebook)
- Deadline: 2026-05-31 23:59 UTC (20 days from start)
- 1,248 teams entered as of 2026-05-11

## Data shape (confirmed locally)

| File | Rows | Cols | Size |
|---|---|---|---|
| `train.csv` | 439,140 | 16 | 54 MB |
| `test.csv` | 188,165 | 15 | 22 MB |
| `f1_strategy_dataset_v4.csv` (original) | 101,371 | 16 | 13 MB |

Train ≈ 70% of total labeled rows; test ≈ 30%. Original dataset is ~4× smaller than synthetic train.

## Schema

### Train columns
| Column | Dtype | Notes |
|---|---|---|
| `id` | int64 | unique row identifier |
| `Driver` | object | **887 unique** (test: 801, all subset of train ✓) |
| `Compound` | object | 5 categories: SOFT, MEDIUM, HARD, INTERMEDIATE, WET |
| `Race` | object | 26 unique (same in train/test) |
| `Year` | int64 | 4 unique values |
| `PitStop` | int64 | **binary indicator** (NOT a counter — likely "has pitted before this lap") |
| `LapNumber` | int64 | lap within race |
| `Stint` | int64 | 8 unique values |
| `TyreLife` | float64 | laps on current tire |
| `Position` | int64 | 20 unique (race positions 1–20) |
| `LapTime (s)` | float64 | current lap time |
| `LapTime_Delta` | float64 | delta vs reference |
| `Cumulative_Degradation` | float64 | tire wear accumulated |
| `RaceProgress` | float64 | 0–1 fraction of race completed |
| `Position_Change` | float64 | position diff vs previous lap |
| `PitNextLap` | float64 | **target** (0.0 / 1.0) |

### Original dataset has extra column
- `Normalized_TyreLife` — derived feature not in train/test, candidate to recreate

## Target distribution

- Negative class: 351,759 (80.10%)
- Positive class: 87,381 (19.90%)
- **Imbalance ratio: 1 : 4** (moderate, NOT extreme — far less imbalanced than S6E4's 3.3% High class)

Implications:
- AUC is the metric — class imbalance affects it less than balanced_accuracy did on S6E4
- Class weighting still useful, but not critical
- No need for special imbalance handling (SMOTE, focal loss, etc.) at baseline

## Group structure & adversarial validation

### (Race, Driver, Year) combo overlap

| Set | Count |
|---|---|
| Unique combos in train | 40,869 |
| Unique combos in test | 37,038 |
| **Overlap (combos in BOTH)** | **35,674** |
| Test combos NOT in train | 1,364 (3.7%) |
| Train combos NOT in test | 5,195 (12.7%) |

**(Race, Driver, Year, LapNumber) overlap = 0** ✓ no row duplication.

**Critical interpretation**: 96.3% of test stints have laps in train. This is **NOT a group-holdout test set** — it's a within-stint random split by `LapNumber`. The same driver's same race has different laps in train vs test.

### Adversarial discriminator AUC

| Pair | AUC | Interpretation |
|---|---|---|
| Train vs Test | **0.5006** | ✓ NO distribution shift — i.i.d. from same generator |
| Train vs Original | **0.3819** | Weak drift (AUC ≠ 0.5 in either direction); not safe for direct concat as training data |

Driver overlap test → **all 801 test drivers appear in train**, no unseen-driver problem.

## CV strategy decision

Given the group structure (96.3% of test stints are seen in train) + zero adversarial signal train↔test:

**Primary: `StratifiedKFold(5)` on target.**
- Random splits simulate test distribution well (i.i.d. from same generator)
- Aligns with what top public notebooks use (pilkwang, kospintr both use plain StratifiedKFold)

**Secondary (sanity check): `GroupKFold(5)` by (Race, Driver, Year).**
- Use ONLY to detect potential target-encoding leakage on stint-level features
- Will be pessimistic vs LB (tests on completely-unseen stints, which is NOT the test distribution)

**Lock**: `CV_SEED=42`, `N_SPLITS=5` for primary.

## Top LB landscape (2026-05-11, 10 days into comp)

| Rank | Team | Score |
|---|---|---|
| 1 | MILANFX | 0.95480 |
| 2 | Chris Deotte | 0.95473 |
| 3 | Kaggler Sergio | 0.95466 |
| 4–16 | (various) | 0.95426 – 0.95455 |

Top-16 spread: **0.00054** — very tight cluster. **High shake-up risk per L18.**

## Insights from recon notebooks

### Chris Deotte EDA (`cdeotte/eda-predicting-f1-pit-stops`)
- `RaceProgress` acts as the "position around the lap-distance circuit"
- Pit windows are empirical zones — pit rate spikes at certain `RaceProgress` bins
- Different compounds have different pit-risk shapes (Compound × RaceProgress interaction is dominant)
- Tyre age + degradation explain WHY pit windows exist
- **Implication**: Compound × RaceProgress is the strongest feature interaction

### Pilkwang "Driver's High" (`pilkwang/s6e5-driver-s-high-driver-feature-eng`) — 46 votes
- "Driver feature must earn its place" — every block has a counterfactual; no counterfactual win, no promotion
- **Compound hardness mapping** (steal directly):
  - SOFT=1.0, MEDIUM=2.0, HARD=3.0, INTERMEDIATE=4.0, WET=5.0
- **Expected tire life per compound** (steal directly):
  - SOFT=25, MEDIUM=35, HARD=45, INTERMEDIATE=30, WET=40
- 5 modeling ideas to express:
  1. Where the car is in the race (RaceProgress)
  2. How old the tyre is RELATIVE to race state (TyreLife / expected_life)
  3. Compound changes tyre-life meaning (compound-aware features)
  4. Whether current lap is inside a plausible pit window (Compound + RaceProgress)
  5. Whether Driver carries repeated strategic context (TE on Driver, fold-safe)
- Feature blocks ordered by leakage risk: physical race-state first (target-free), TE later (fold-safe)

### Kospintr multi-model baseline (`kospintr/pitstop-catb-hgbc-xgb-lgbm-realmlp-baseline`) — 66 votes
- Casts low-card integers to categorical: `Year`, `PitStop`, `Stint`, `Position`
- Outlier clipping at >500 for `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`
- Uses 5 model families: CatBoost + HistGradientBoostingClassifier + XGB + LGB + RealMLP (pytabkit) → L17 confirmed
- Uses sklearn's `TargetEncoder` (CV-safe) → L17 audit gap module 10 validated

### Yekenot pytabkit (`yekenot/ps-s6-e5-realmlp-pytabkit`) — 82 votes (TOP)
- Pure RealMLP_TD_Classifier baseline already at top of public notebooks
- **L17 is fully validated by this comp's reality**: pytabkit IS the new tabular baseline

### Flexonafft blender (`flexonafft/f1-submission-blender-0-954`) — 62 votes
- Already 0.954+ from blending public outputs
- **OOF ecosystem is forming** — Phase 6b harvesting will have material to work with

## Tooling state (local)

| Library | Version | Status |
|---|---|---|
| sklearn | 1.8.0 | ✓ (has TargetEncoder ≥1.3 + HistGradientBoostingClassifier) |
| xgboost | 3.2.0 | ✓ GPU-capable |
| lightgbm | 4.6.0 | ✓ GPU-capable |
| catboost | 1.2.10 | ✓ GPU-capable |
| **pytabkit** | NOT INSTALLED | **GAP — need for L17 model family** |
| **torch** | NOT INSTALLED | **GAP — pytabkit dependency** |
| GPU | RTX 4070 Laptop 8GB | ✓ usable for XGB/LGB/CatBoost; marginal for RealMLP |

**Compute strategy** (per `feedback_cloud-first-compute.md`):
- Local 4070 used for: 10K-row sanity passes, EDA, quick 1-fold CV
- All full-data training, Optuna sweeps, multi-seed ensembles → Colab Pro
- Artifacts → Google Drive folder `https://drive.google.com/drive/folders/1N6PFShEtMj2KSYWxaQQz-6Kro1CTLilh`

## Outliers requiring cleaning (from baseline)

Apply at FE step (`features.py`):
- `LapTime (s)` > 500 → median impute
- `abs(LapTime_Delta)` > 500 → median impute
- `Cumulative_Degradation` > 500 → median impute

## Feature engineering plan (synthesizing recon + S6E4 audit lessons)

### Block 1: Physical race-state (target-free, ship Day 2)
- `tyre_life_ratio` = `TyreLife / COMPOUND_EXPECTED_LIFE[Compound]`
- `tyre_life_rel` = `TyreLife - COMPOUND_EXPECTED_LIFE[Compound]`
- `compound_hardness` (ordinal)
- `pit_window_indicator` per compound (derived from EDA pit-rate spikes)
- `race_progress_bin` (Compound × RaceProgress is dominant)
- `degradation_per_lap` = `Cumulative_Degradation / TyreLife`
- `lap_time_z` = (LapTime - median per Race) / std per Race
- `position_delta_3lap` (rolling mean of `Position_Change`)

### Block 2: Categorical interactions (cstdy pattern)
- `Compound × Stint`
- `Compound × RaceProgress_bin` (KEY per Deotte)
- `Race × Stint`
- Pairwise concat with `good_columns = {Compound, Race, Driver, Stint, Year}`

### Block 3: Target encoding (LAST, fold-safe only)
- Use **sklearn's `TargetEncoder(cv=5)`** (the L16-audit gap module — DO NOT use manual TE per S6E4 Phase 12 fold collapse lesson)
- Apply to: `Driver`, `Race`, `(Race, Compound)`, `(Driver, Compound)`

### Block 4: Digit extraction (yunsuxiaozi)
- For each float feature, extract digit-k for k ∈ {-4..3} as int8
- Apply to: `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `TyreLife`

## Action items before Phase 1

1. **Install pytabkit + torch** in a clean venv for s6e5:
   ```bash
   cd ~/IdeaProjects/kaggle/playground-s6e5
   uv venv && source .venv/bin/activate
   uv pip install pandas scikit-learn xgboost lightgbm catboost pytabkit torch
   ```
2. **Push competition data + original dataset to Google Drive** (`s6e5/data/raw/` and `s6e5/data/external/`)
3. **Create Colab notebook scaffold** that mounts Drive, loads from `s6e5/data/`, persists OOFs to `s6e5/probs/<version>/`

## Risk register

| Risk | Note |
|---|---|
| Tight LB cluster (top-16 spread 0.00054) | L18: submission selection critical — pair max-public + boundary-stable |
| Drivers in test that are rare in train | Use freq-thresholded label encoding (rare → "other") |
| Target encoding leakage on stint-level | Use sklearn's CV-safe TargetEncoder; never manual TE (S6E4 Phase 12 lesson) |
| Public OOF ecosystem may not be as rich as S6E4 | Fallback to solo strong-model approach (bronze still achievable) |
| pytabkit not in local venv | Install in fresh s6e5 venv before Phase 3 |

## Phase 0 status

- [x] A: Task & metric verification
- [x] B: Adversarial validation (train↔test=0.50, train↔orig=0.38)
- [x] C: Read recon notebooks (cdeotte EDA + pilkwang + kospintr + yekenot)
- [ ] D: Discussion forum sweep — pending (60 min — can defer to Day 2 morning)
- [x] E: CV strategy decision — StratifiedKFold(5), seed=42, with GroupKFold(R,D,Y) as sanity check
- [x] F: Tooling check — pytabkit + torch are gaps; will install in fresh venv

**Next**: Phase 1 (mental model + write v1 baseline notebook).
