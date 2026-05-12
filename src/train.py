"""Universal train_variant() — one interface, multiple algo backends.

Currently implements: lgb. Will add xgb, catboost, histgb, realmlp incrementally.

Returns OOF + holdout + test predictions along with per-fold AUCs and
runtime — exactly the shape the observer module expects.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import CV_SEED, MODEL_SEED, N_FOLDS, TARGET
from .cv import make_cv


def _prep_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        if c in out.columns and not pd.api.types.is_categorical_dtype(out[c]):
            out[c] = out[c].astype("category")
    return out


def _align_category_codes(train: pd.DataFrame, *others: pd.DataFrame, cat_cols: list[str]):
    """Align category dictionaries across train/holdout/test so codes match."""
    train = train.copy()
    aligned_others = [df.copy() for df in others]
    for c in cat_cols:
        if c not in train.columns:
            continue
        # Combine categories from ALL frames (train + each other) to capture all levels
        all_vals = pd.concat([train[c].astype("object")] + [df[c].astype("object") for df in aligned_others])
        cats = pd.Index(sorted(all_vals.dropna().unique()))
        train[c] = pd.Categorical(train[c], categories=cats)
        for df in aligned_others:
            df[c] = pd.Categorical(df[c], categories=cats)
    return (train, *aligned_others)


def _lgb_defaults(use_gpu: bool) -> dict:
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "verbose": -1,
        "n_estimators": 5000,
        "random_state": MODEL_SEED,
    }
    if use_gpu:
        params["device_type"] = "gpu"
    return params


def _train_lgb(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    X_holdout: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    n_folds: int,
    cv_seed: int,
    early_stopping_rounds: int,
) -> dict:
    import lightgbm as lgb

    cv = make_cv()
    oof = np.zeros(len(X_pool))
    holdout_pred_folds = np.zeros((n_folds, len(X_holdout)))
    test_pred_folds = np.zeros((n_folds, len(X_test)))
    fold_aucs: list[float] = []

    # Align categorical dictionaries up front so each fold sees consistent codes
    X_pool, X_holdout, X_test = _align_category_codes(
        X_pool, X_holdout, X_test, cat_cols=categorical_cols
    )

    X_pool_view = X_pool[feature_cols]
    X_holdout_view = X_holdout[feature_cols]
    X_test_view = X_test[feature_cols]

    for fold, (tr_idx, val_idx) in enumerate(cv.split(np.arange(len(X_pool)), y_pool)):
        Xtr, ytr = X_pool_view.iloc[tr_idx], y_pool[tr_idx]
        Xva, yva = X_pool_view.iloc[val_idx], y_pool[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_pred = model.predict_proba(Xva)[:, 1]
        oof[val_idx] = val_pred
        fold_auc = roc_auc_score(yva, val_pred)
        fold_aucs.append(fold_auc)
        holdout_pred_folds[fold] = model.predict_proba(X_holdout_view)[:, 1]
        test_pred_folds[fold] = model.predict_proba(X_test_view)[:, 1]
        print(f"  fold {fold+1}/{n_folds}  AUC={fold_auc:.5f}  best_iter={model.best_iteration_}")

    holdout_pred = holdout_pred_folds.mean(axis=0)
    test_pred = test_pred_folds.mean(axis=0)
    return {
        "oof_pred": oof,
        "holdout_pred": holdout_pred,
        "test_pred": test_pred,
        "fold_aucs": fold_aucs,
    }


def train_variant(
    *,
    algo: str,
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    X_holdout: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str] | None = None,
    params: dict | None = None,
    n_folds: int = N_FOLDS,
    cv_seed: int = CV_SEED,
    use_gpu: bool = False,
    early_stopping_rounds: int = 100,
) -> dict[str, Any]:
    """Train one variant via 5-fold CV; return OOF + holdout + test predictions."""
    categorical_cols = categorical_cols or []
    t0 = time.time()
    algo = algo.lower()
    if algo == "lgb":
        params = {**_lgb_defaults(use_gpu), **(params or {})}
        result = _train_lgb(
            X_pool=X_pool, y_pool=y_pool, X_holdout=X_holdout, X_test=X_test,
            feature_cols=feature_cols, categorical_cols=categorical_cols,
            params=params, n_folds=n_folds, cv_seed=cv_seed,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        raise NotImplementedError(f"algo={algo!r} not yet implemented in train_variant().")
    result["runtime_sec"] = time.time() - t0
    result["algo"] = algo
    result["params"] = params
    return result
