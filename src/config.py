"""Centralized configuration: seeds, paths, constants.

Single source of truth. Imported by every notebook and module so that
seeds and paths cannot drift across the project.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
SPLITS = DATA / "splits"
PROBS = ROOT / "probs"
SUBMISSIONS = ROOT / "submissions"
DOCS = ROOT / "docs"

# Seeds — locked across the entire competition.
CV_SEED = 42
HOLDOUT_SEED = 11
MODEL_SEED = 7
N_FOLDS = 5
HOLDOUT_FRAC = 0.20

# Problem definition
TARGET = "PitNextLap"
ID = "id"
METRIC = "roc_auc"

# Schema
NUMERIC_COLS = [
    "TyreLife",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
    "LapNumber",
]
LOWCARD_INT_AS_CAT = ["Year", "PitStop", "Stint", "Position"]
CATEGORICAL_STR_COLS = ["Driver", "Compound", "Race"]

# Compound knowledge (from pilkwang's notebook + verified in our EDA)
COMPOUND_HARDNESS = {
    "SOFT": 1.0,
    "MEDIUM": 2.0,
    "HARD": 3.0,
    "INTERMEDIATE": 4.0,
    "WET": 5.0,
}
COMPOUND_EXPECTED_LIFE = {
    "SOFT": 25.0,
    "MEDIUM": 35.0,
    "HARD": 45.0,
    "INTERMEDIATE": 30.0,
    "WET": 40.0,
}
# Pit windows derived from our own EDA (notebooks/figs/eda_summary.md §3)
COMPOUND_PIT_WINDOW = {
    "SOFT": (0.05, 0.225),
    "MEDIUM": (0.55, 0.725),
    "HARD": (0.50, 0.70),
    "INTERMEDIATE": (0.175, 0.325),
    "WET": (0.0, 1.0),  # too few rows to derive a window
}

# Outlier cleaning thresholds (from EDA §7 + baseline notebooks)
OUTLIER_ABS_THRESHOLD = {
    "LapTime (s)": 500.0,
    "LapTime_Delta": 500.0,
    "Cumulative_Degradation": 500.0,
}
