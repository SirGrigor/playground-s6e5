"""Data loading with the sacred-holdout protocol enforced.

There is intentionally NO function that returns `train_pool + holdout` combined
for training. The only way to access the holdout is via `load_holdout()` —
explicitly, and never during model fitting.
"""
from pathlib import Path
import pandas as pd
import numpy as np

from .config import RAW, EXTERNAL, SPLITS, TARGET, ID, HOLDOUT_SEED, HOLDOUT_FRAC


def _read_train_full():
    """Internal: load the raw training CSV. Use load_train_pool() / load_holdout() instead."""
    return pd.read_csv(RAW / "train.csv")


def load_test():
    return pd.read_csv(RAW / "test.csv")


def load_original():
    """Read-only — for FE insights, NOT training (S6E4 Phase 14B lesson)."""
    return pd.read_csv(EXTERNAL / "f1_strategy_dataset_v4.csv")


def load_holdout_indices():
    """Load the locked holdout row indices.

    These were generated once by notebooks/02_holdout_split.py and persisted
    to data/splits/holdout_v1.parquet. The split is git-tracked so the
    holdout assignment is reproducible across sessions and contributors.
    """
    path = SPLITS / "holdout_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Holdout split missing at {path}. Run notebooks/02_holdout_split.py first."
        )
    return pd.read_parquet(path)["id"].to_numpy()


def load_train_pool():
    """Load the 80% training pool (everything NOT in holdout). Use this for ALL training."""
    full = _read_train_full()
    holdout_ids = set(load_holdout_indices().tolist())
    mask = ~full[ID].isin(holdout_ids)
    return full.loc[mask].reset_index(drop=True)


def load_holdout():
    """Load the 20% sacred holdout. ONLY call this at the END of each version to validate.

    Never use during feature engineering, CV, or hyperparameter tuning.
    """
    full = _read_train_full()
    holdout_ids = set(load_holdout_indices().tolist())
    mask = full[ID].isin(holdout_ids)
    return full.loc[mask].reset_index(drop=True)


# Intentional missing: NO load_train_combined() / load_full_train() etc.
# If you ever feel the urge to add one, you are about to repeat S6E4 Phase 14C.
# The sacred holdout exists precisely so it can never leak into training.
