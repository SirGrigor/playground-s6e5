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


# Block 2: categorical interactions — TODO
# Block 3: sklearn TargetEncoder(cv=5) — TODO (NEVER manual TE, see docs/pitfalls.md #5)
# Block 4: digit extraction — TODO
