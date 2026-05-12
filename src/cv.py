"""Cross-validation setup. StratifiedKFold(5) seed=42 — locked per `src/config.py`."""
from sklearn.model_selection import StratifiedKFold

from .config import CV_SEED, N_FOLDS, TARGET


def make_cv():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)


def fold_indices(df):
    """Yield (train_idx, val_idx) for the locked CV setup."""
    cv = make_cv()
    y = df[TARGET].to_numpy()
    yield from cv.split(df.index.to_numpy(), y)
