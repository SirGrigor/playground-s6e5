"""Feature engineering pipeline.

Block 1 (target-free, ship first):
  - Outlier clipping
  - Compound-aware tyre features
  - Pit-window indicators
  - Per-race lap-time z-score
  - Degradation rate

Block 2 (categorical interactions):  TODO
Block 3 (sklearn TargetEncoder, CV-safe — NEVER manual TE):  TODO
Block 4 (digit extraction):  TODO

Add blocks one at a time, validating each via holdout delta before adding the next.
"""
import numpy as np
import pandas as pd

from .config import (
    COMPOUND_HARDNESS,
    COMPOUND_EXPECTED_LIFE,
    COMPOUND_PIT_WINDOW,
    OUTLIER_ABS_THRESHOLD,
)


def clip_outliers(df, inplace=False):
    df = df if inplace else df.copy()
    for col, thresh in OUTLIER_ABS_THRESHOLD.items():
        if col in df.columns:
            med = df.loc[df[col].abs() <= thresh, col].median()
            mask = df[col].abs() > thresh
            df.loc[mask, col] = med
    return df


def add_compound_features(df, inplace=False):
    df = df if inplace else df.copy()
    df["compound_hardness"] = df["Compound"].map(COMPOUND_HARDNESS).astype("float32")
    df["compound_expected_life"] = df["Compound"].map(COMPOUND_EXPECTED_LIFE).astype("float32")
    df["tyre_life_ratio"] = df["TyreLife"] / df["compound_expected_life"]
    df["tyre_life_minus_expected"] = df["TyreLife"] - df["compound_expected_life"]
    return df


def add_pit_window_features(df, inplace=False):
    df = df if inplace else df.copy()
    lo = df["Compound"].map({c: w[0] for c, w in COMPOUND_PIT_WINDOW.items()}).astype("float32")
    hi = df["Compound"].map({c: w[1] for c, w in COMPOUND_PIT_WINDOW.items()}).astype("float32")
    df["pit_window_lo"] = lo
    df["pit_window_hi"] = hi
    df["in_pit_window"] = ((df["RaceProgress"] >= lo) & (df["RaceProgress"] <= hi)).astype("int8")
    df["dist_to_pit_window"] = np.where(
        df["RaceProgress"] < lo,
        lo - df["RaceProgress"],
        np.where(df["RaceProgress"] > hi, df["RaceProgress"] - hi, 0.0),
    ).astype("float32")
    return df


def add_race_pace_features(df, inplace=False):
    df = df if inplace else df.copy()
    race_stats = df.groupby("Race")["LapTime (s)"].agg(["mean", "std"])
    df = df.join(race_stats.rename(columns={"mean": "race_lap_mean", "std": "race_lap_std"}), on="Race")
    df["lap_time_z_per_race"] = (
        (df["LapTime (s)"] - df["race_lap_mean"]) / df["race_lap_std"].replace(0, 1)
    ).astype("float32")
    df["degradation_per_lap"] = (df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)).astype("float32")
    return df


def build_block1(df, inplace=False):
    """Apply Block 1 transformations (target-free physical race-state features).

    Safe to call on train, holdout, test — uses no target information.
    Race-pace features compute per-race aggregates WITHIN the dataframe passed,
    so callers should pass `pd.concat([train, holdout, test])` or similar if
    they want global statistics; for now, compute per-frame.
    """
    df = clip_outliers(df, inplace=inplace)
    df = add_compound_features(df, inplace=True)
    df = add_pit_window_features(df, inplace=True)
    df = add_race_pace_features(df, inplace=True)
    return df


# =========================================================================
# Block 2 — categorical interactions (one at a time, per attribution discipline).
# Each interaction is built as a NEW categorical column that LGB consumes via
# its native Fisher-algorithm categorical splits. This is structurally different
# from Block 3 (TargetEncoder) — Block 2 gives LGB MORE freedom (find best
# splits in the joint space), Block 3 gave LGB LESS freedom (collapse to one
# numeric per combo). v2 showed TE-style was redundant; Block 2 tests if
# explicit interaction categoricals deliver where TE didn't.
# =========================================================================


def add_compound_x_progress_bin(df, n_bins=40, inplace=False):
    """Compound × RaceProgress bin (40-way) as a categorical interaction.

    From EDA: pit-rate-by-RaceProgress curve is dramatically different per
    Compound — SOFT pits at 0.05–0.225 (early), HARD pits at 0.50–0.70 (late),
    MEDIUM pits at 0.55–0.725, etc. A single feature that combines compound +
    bin lets LGB partition the joint pit-window space directly.

    Cardinality: 5 compounds × 40 bins = up to 200 categories.
    On 350K rows: ~1,750 obs/category — plenty for stable splits.
    """
    df = df if inplace else df.copy()
    bins = np.linspace(0, 1, n_bins + 1)
    progress_bin = pd.cut(
        df["RaceProgress"], bins=bins, include_lowest=True, labels=False
    ).astype("int16")
    df["compound_x_progress_bin"] = (
        df["Compound"].astype(str) + "_b" + progress_bin.astype(str)
    ).astype("category")
    return df


# Block 3: sklearn TargetEncoder(cv=5) — IMPLEMENTED in train.py _apply_target_encoding()
#   (kept inside fold loop because TE is target-dependent → must be fit-per-fold)
#   v2 result: NEGATIVE on this feature set (redundant with LGB native cat).


# =========================================================================
# Block 4 — Digit-extraction features (yunsuxiaozi pattern).
# Extracts the decimal digit of each numeric feature at specific positions.
# Examples:  LapTime=78.491 → digit(-1)=4, digit(-2)=9, digit(-3)=1
#            TyreLife=27    → digit(0)=7, digit(1)=2
# Why it might help: LGB partitions continuous values at thresholds, but it
# CANNOT discover modular-arithmetic patterns (e.g. "all values ending in 7").
# If the synthetic data generator quantized values to specific precisions,
# digit features expose orthogonal information trees can't build implicitly.
# =========================================================================


# Per-feature useful digit positions (chosen by value range)
# k >= 0: integer-part digit (k=0 → ones, k=1 → tens, k=2 → hundreds)
# k < 0: decimal-part digit (k=-1 → tenths, k=-2 → hundredths, k=-3 → thousandths)
DIGIT_POSITIONS = {
    "TyreLife":               [0, 1],
    "LapNumber":              [0, 1],
    "LapTime (s)":            [-2, -1, 0, 1],
    "LapTime_Delta":          [-2, -1, 0, 1],
    "Cumulative_Degradation": [-2, -1, 0, 1, 2],
    "RaceProgress":           [-3, -2, -1],
    "Position_Change":        [0],
}


def _digit_at(arr: np.ndarray, k: int) -> np.ndarray:
    """Extract decimal digit at position k.

    k=0 → ones, k=1 → tens, k=-1 → tenths, k=-2 → hundredths.

    Uses np.round AFTER scaling, per S6E4 L10 lesson (IEEE 754 imprecision).
    Plain `(val // 10**k) % 10` failed 60% of the time on S6E4 due to 0.01
    being inexact in float64; this version rounds before integer cast.
    """
    if k >= 0:
        # Integer part, position k from ones
        return ((np.floor(arr).astype("int64") // (10 ** k)) % 10).astype("int8")
    else:
        # Decimal part — shift left by -k places, round to handle float imprecision
        shift = -k
        scaled = np.round(arr * (10 ** shift)).astype("int64")
        return (scaled % 10).astype("int8")


def add_digit_features(df, inplace=False, positions=None):
    """Add digit-extraction features per DIGIT_POSITIONS config.

    Adds columns named e.g. 'LapTime (s)_digit-1' for the tenths digit of LapTime.
    Total: 21 new features (sum of DIGIT_POSITIONS list lengths) given the default config.
    """
    df = df if inplace else df.copy()
    positions = positions or DIGIT_POSITIONS
    for col, ks in positions.items():
        if col not in df.columns:
            continue
        arr = df[col].to_numpy()
        for k in ks:
            df[f"{col}_digit{k}"] = _digit_at(arr, k)
    return df
