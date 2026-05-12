# S6E5 — EDA Summary (2026-05-12)

Source: `notebooks/01_eda.py` run on full data (439K train, 188K test, 101K original).

## §1. Target distribution by group

### By Compound (pit rate)
| Compound | Pit rate | n |
|---|---|---|
| HARD | 32.75% | 170,518 |
| SOFT | 19.35% | 38,744 |
| INTERMEDIATE | 15.23% | 17,382 |
| MEDIUM | 10.11% | 211,141 |
| WET | 2.51% | 1,355 |

Spread is 3.3× between MEDIUM and HARD. Compound matters.

### By Stint (selected)
- Stint 1 → low pit rate (early stint, fresh tires)
- Stint 4-6 → highest pit rates (long into race)
- Stint 8 is rare (long races only)

### By Race (top 8 highest pit rate)
- Pre-Season Testing, Singapore GP, British GP, Italian GP, US GP, Miami GP, Mexico GP
- Bottom 8 are likely wet races or 1-stop strategy races

### By Year
- 4 unique years; Year is #3 in MI ranking — strategy evolves across seasons

### By PitStop
- Binary (0/1). Likely "has pitted at least once before this lap". Useless alone (MI=0.000) — needs interaction with Stint or TyreLife.

## §2. TyreLife vs pit rate (KEY)

Strong monotonic relationship. Pit rate rises smoothly from 1.7% (TyreLife≤2) to >80% (TyreLife≥56).

| TyreLife bin | Pit rate | n |
|---|---|---|
| (0, 2] | 1.7% | 33,379 |
| (2, 4] | 6.3% | 38,962 |
| (4, 6] | 9.5% | 38,960 |
| (8, 10] | 15.4% | 35,186 |
| (12, 14] | 20.4% | 34,277 |
| (16, 18] | 25.1% | 27,507 |
| (20, 22] | 29.6% | 20,270 |
| (24, 26] | 33.1% | 14,435 |
| (32, 34] | 41.0% | 5,246 |
| (40, 42] | 41.4% | 1,840 |
| (48, 50] | 49.5% | 519 |
| (52, 54] | 63.0% | 208 |
| (60, 62] | 90.9% | 66 |

**Median TyreLife per compound**:
- HARD: 17.0 (long-life tire, used for length)
- MEDIUM: 11.0
- INTERMEDIATE: 12.0
- SOFT: 10.0
- WET: 9.0

Plot: `figs/01_tyrelife_per_compound.png`

## §3. RaceProgress pit windows (Deotte confirmed)

**Each compound has a DIFFERENT pit window** — this is THE feature interaction.

| Compound | Pit window (RaceProgress) | Peak pit rate in window |
|---|---|---|
| HARD | 0.50–0.70 | 44–48% |
| MEDIUM | 0.55–0.725 | 33–37% |
| SOFT | **0.05–0.225** (EARLY!) | 25–29% |
| INTERMEDIATE | 0.175–0.325 | 14–17% |
| WET | (sample too small) | — |

Strategic interpretation:
- **SOFT** is the qualifying/start tire — used briefly, pitted early
- **MEDIUM** is the middle compound — pitted middle-late
- **HARD** is the long-stint tire — pitted late
- **INTERMEDIATE** is wet-dry transition — pitted in middle of race when conditions shift

Plot: `figs/02_raceprogress_per_compound.png`

## §4. Mutual Information ranking

Subsampled 100K rows.

```
1. RaceProgress       0.095
2. Stint              0.081
3. Year               0.073  ← surprising
4. LapTime_Delta      0.060
5. Cumulative_Degr    0.057
6. LapNumber          0.050
7. TyreLife           0.046
8. Compound           0.043
9. Position_Change    0.039
10. LapTime (s)       0.027
11. Race              0.017
12. Driver            0.009  ← weak alone
13. Position          0.002  ← near-zero
14. PitStop           0.000  ← useless alone
```

**Notes**:
- `RaceProgress` + `Stint` + `Year` are the top 3 — together they encode "where in the race, which stint of the race, which season's strategy"
- `Compound` is only #8 in raw MI but creates the strongest *interactions* — MI underestimates interaction-driven signal
- `Driver` is weak alone (matches pilkwang's "Driver must earn its place" framing)
- `Position` and `PitStop` are useless alone — only useful in interactions

## §5. Stint structure

42,233 unique (Race, Driver, Year) stints.

```
count    42,233
mean         14.85 laps/stint
std           9.14
min           1
25%           6
50%          16
75%          22
max          51
```

## §6. Train/test split pattern

**Train/test laps within each stint are INTERLEAVED, not sequential.**

Example: `MAS @ Monaco 2024`
- train laps: `[1, 3, 5, 7, 10, 12, 13, 15, 19, 23, 25, 27, 29, 30, 34, …]`
- test laps:  `[6, 8, 17, 18, 21, 22, 24, 26, 31, 44, 46, 56, 63, 64, 68, …]`

Per-stint train_frac distribution:
```
mean    0.701
std     0.199
25%     0.615
50%     0.714
75%     0.800
```

**Implication**: random `StratifiedKFold(5)` simulates test well. BUT any feature computed from rows within a stint (rolling means, lag features, target encoding by Driver/Race) WILL leak unless fold-safe. Use sklearn `TargetEncoder(cv=5)` per recon plan.

## §7. Outlier check (cleaning hygiene)

| Column | Max | Median | Std | Pct > 500 |
|---|---|---|---|---|
| LapTime (s) | 2507.61 | 90.52 | 19.77 | 0.005% (20 rows) |
| LapTime_Delta (abs) | 2423.93 | -0.30 | 43.95 | 0.031% (138 rows) |
| Cumulative_Degradation | 2412.03 | -20.99 | 54.77 | 0.004% (18 rows) |

Tiny fraction → median impute is hygiene, not signal-bearing.

## Surprises (vs recon doc hypotheses)

1. **Year is #3 in MI** (0.073) — didn't expect. Suggests yearly strategy evolution; build Year × Compound and Year × Race features.
2. **Compound is only #8 in raw MI** but #1 in INTERACTION strength (different pit windows per compound). MI for solo features understates interaction value.
3. **SOFT compound pits VERY EARLY** (0.05–0.225 RaceProgress). The interaction window is fundamentally different from HARD/MEDIUM.
4. **Train/test split is interleaved within stints** — random KFold is fine; group-CV would be too pessimistic.
5. **Driver alone is weak signal** (MI=0.009) — pilkwang's "earn its place" framing is empirically correct. Don't add Driver-naked. Add Driver-in-interaction.
6. **Position is near-zero alone** (MI=0.002) — only useful in interaction.

## Feature engineering blueprint (refined from EDA)

### Block 1: Physical race-state (target-free, ship first)
- `tyre_life_ratio` = `TyreLife / COMPOUND_EXPECTED_LIFE[Compound]`
- `tyre_life_minus_expected` = `TyreLife - COMPOUND_EXPECTED_LIFE[Compound]`
- `tyre_life_over_median` per compound (using EDA medians: HARD=17, MEDIUM=11, INTER=12, SOFT=10, WET=9)
- `in_pit_window` per compound (derived from EDA peaks):
  - SOFT: RaceProgress ∈ [0.05, 0.225]
  - MEDIUM: RaceProgress ∈ [0.55, 0.725]
  - HARD: RaceProgress ∈ [0.50, 0.70]
  - INTERMEDIATE: RaceProgress ∈ [0.175, 0.325]
- `degradation_per_lap` = `Cumulative_Degradation / max(TyreLife, 1)`
- `lap_time_z_per_race` = (LapTime - mean per Race) / std per Race
- `race_progress_bin` (40 bins) for non-linear modeling

### Block 2: Categorical interactions (cstdy pattern)
- `Compound × RaceProgress_bin` (the KEY interaction)
- `Compound × Stint`
- `Compound × Year`
- `Year × Race`
- `Stint × Position` (front-grid pit patterns differ)

### Block 3: Target encoding (LAST, fold-safe)
- Use **sklearn `TargetEncoder(cv=5)`**
- Apply to: `Driver`, `Race`, `(Race, Year)`, `(Driver, Compound)`, `(Driver, Stint)`

### Block 4: Driver-in-interaction (since Driver alone is weak)
- `(Driver, Compound)` target-encoded pit rate
- `(Driver, Stint)` target-encoded pit rate
- Frequency-thresholded label encoding for Driver (rare → "other")

## Phase 2 status

- [x] Class distribution + cross-tabs by main categoricals
- [x] TyreLife vs target (monotonic predictor confirmed)
- [x] RaceProgress vs target per Compound (pit windows confirmed, compound-specific)
- [x] MI ranking
- [x] Stint structure analysis
- [x] Train/test split pattern verified (interleaved, not sequential)
- [x] Outlier cleaning thresholds confirmed
- [ ] Save FE blueprint into `src/features.py` skeleton — next step

**Next**: Phase 3 — write v1 LGB baseline notebook with Block 1 features only (target-free, fastest path to a submitted LB score).
