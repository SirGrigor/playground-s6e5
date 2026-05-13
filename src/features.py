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


# =========================================================================
# Block Yekenot — feature engineering reproduced from
# https://www.kaggle.com/code/yekenot/ps-s6-e5-realmlp-pytabkit (OOF 0.9537).
#
# Target-free transforms (the TE on Race×Compound / Race×Year combos is
# handled separately by train_variant's te_pairs argument, so it stays
# fold-safe). The transforms here are deterministic given the training
# distribution; we fit on pool, transform on holdout/test.
#
# Components:
#   1. Arithmetic interactions  (LapNumber/RaceProgress, TyreLife/LapNumber)
#   2. Floor-factorize numericals → categorical codes
#   3. Count encoding for categoricals
#   4. KBins discretization (RaceProgress×200, LapTime×7)
# =========================================================================

YEKENOT_NUM_COLS = [
    "Year", "PitStop", "LapNumber", "Stint", "TyreLife", "Position",
    "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
    "RaceProgress", "Position_Change",
]
YEKENOT_CAT_COLS = ["Driver", "Compound", "Race"]
YEKENOT_KBIN_CONFIG = {"RaceProgress": 200, "LapTime (s)": 7}


def _yekenot_arith(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_LapNumber_/_RaceProgress"] = (
        df["LapNumber"] / (df["RaceProgress"] + 1e-6)
    ).astype("float32")
    df["_TyreLife_/_LapNumber"] = (
        df["TyreLife"] / df["LapNumber"].clip(lower=1)
    ).astype("float32")
    return df


def yekenot_fe_fit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply yekenot FE in fit mode, return (transformed_df, state_dict).

    Pass `state_dict` to `yekenot_fe_transform` for downstream frames.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    df = _yekenot_arith(df)
    state: dict = {"factorize": {}, "counts": {}, "kbins": {}, "arith_cols": [
        "_LapNumber_/_RaceProgress", "_TyreLife_/_LapNumber"
    ]}

    # Floor-factorize numericals (and the two arith cols) → categorical codes
    floor_cols = YEKENOT_NUM_COLS + state["arith_cols"]
    for col in floor_cols:
        cat_name = f"{col}_cat_" if col in YEKENOT_NUM_COLS else f"{col[1:]}_cat_"
        codes, uniques = pd.factorize(np.floor(df[col]), sort=True)
        state["factorize"][col] = (cat_name, uniques)
        df[cat_name] = codes.astype("int32")

    # Count encoding on (cat cols + year_cat / pitstop_cat from above)
    count_target_cols = list(YEKENOT_CAT_COLS) + ["Year_cat_", "PitStop_cat_"]
    for col in count_target_cols:
        count_map = df[col].value_counts()
        state["counts"][col] = count_map.to_dict()
        new_name = f"_{col}_count" if col in YEKENOT_CAT_COLS else f"_{col[:-1]}_count"
        df[new_name] = df[col].map(state["counts"][col]).fillna(0).astype("int32")

    # KBins discretization
    for col, n_bins in YEKENOT_KBIN_CONFIG.items():
        kb = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
        )
        binned = kb.fit_transform(df[[col]]).ravel().astype("int32")
        state["kbins"][col] = kb
        df[f"{col}_{n_bins}_quantile_bin_"] = binned

    return df, state


def yekenot_fe_transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply pre-fitted yekenot FE to new data (holdout / test)."""
    df = _yekenot_arith(df)

    for col, (cat_name, uniques) in state["factorize"].items():
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes = np.floor(df[col]).map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes

    for col, count_map in state["counts"].items():
        new_name = f"_{col}_count" if col in YEKENOT_CAT_COLS else f"_{col[:-1]}_count"
        df[new_name] = df[col].map(count_map).fillna(0).astype("int32")

    for col, kb in state["kbins"].items():
        n_bins = state["kbins"][col].n_bins
        binned = kb.transform(df[[col]]).ravel().astype("int32")
        df[f"{col}_{n_bins}_quantile_bin_"] = binned

    return df


def yekenot_feature_lists(state: dict) -> tuple[list[str], list[str]]:
    """After running yekenot_fe_fit, derive (feature_cols, categorical_cols).

    Returns:
      feature_cols: numeric features to feed to RealMLP (arith + counts + bins)
      categorical_cols: cat features for native embedding (Driver/Compound/Race + the *_cat_)
    """
    floor_cats = [name for (name, _) in state["factorize"].values()]
    counts = [
        f"_{col}_count" if col in YEKENOT_CAT_COLS else f"_{col[:-1]}_count"
        for col in (list(YEKENOT_CAT_COLS) + ["Year_cat_", "PitStop_cat_"])
    ]
    bins = [
        f"{col}_{n}_quantile_bin_" for col, n in YEKENOT_KBIN_CONFIG.items()
    ]
    numeric_feats = (
        ["LapNumber", "Stint", "TyreLife", "LapTime (s)", "LapTime_Delta",
         "Cumulative_Degradation", "RaceProgress", "Position_Change"]
        + state["arith_cols"] + counts
    )
    cat_feats = list(YEKENOT_CAT_COLS) + floor_cats + bins
    return numeric_feats, cat_feats
