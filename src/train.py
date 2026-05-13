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


def _make_pair_col(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Build a stable string-pair feature like 'D086_HARD' from two categoricals."""
    return df[a].astype(str) + "_" + df[b].astype(str)


def _apply_target_encoding(
    Xtr: pd.DataFrame,
    Xva: pd.DataFrame,
    Xho: pd.DataFrame,
    Xte: pd.DataFrame,
    ytr: np.ndarray,
    te_cols: list[str],
    te_pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Fold-safe target encoding via sklearn.preprocessing.TargetEncoder(cv=5).

    Per S6E5 pitfall #5 — NEVER use manual TE (S6E4 Phase 12 fold collapse).
    sklearn's TargetEncoder handles fold-safe encoding internally via cv-fitting.

    Returns the four frames with NEW TE_xxx columns added, plus the list of new column names.
    """
    from sklearn.preprocessing import TargetEncoder

    Xtr = Xtr.copy()
    Xva = Xva.copy()
    Xho = Xho.copy()
    Xte = Xte.copy()

    # Build pair columns first (in all frames consistently)
    new_pair_names = []
    for a, b in te_pairs:
        name = f"{a}_x_{b}"
        Xtr[name] = _make_pair_col(Xtr, a, b)
        Xva[name] = _make_pair_col(Xva, a, b)
        Xho[name] = _make_pair_col(Xho, a, b)
        Xte[name] = _make_pair_col(Xte, a, b)
        new_pair_names.append(name)

    encode_cols = list(te_cols) + new_pair_names
    out_cols = [f"TE_{c}" for c in encode_cols]

    te = TargetEncoder(cv=5, smooth="auto", random_state=42, target_type="binary")
    Xtr_te = te.fit_transform(Xtr[encode_cols].astype(str), ytr)
    Xva_te = te.transform(Xva[encode_cols].astype(str))
    Xho_te = te.transform(Xho[encode_cols].astype(str))
    Xte_te = te.transform(Xte[encode_cols].astype(str))

    for i, c in enumerate(out_cols):
        Xtr[c] = Xtr_te[:, i].astype("float32")
        Xva[c] = Xva_te[:, i].astype("float32")
        Xho[c] = Xho_te[:, i].astype("float32")
        Xte[c] = Xte_te[:, i].astype("float32")

    # Drop the intermediate pair string columns — model uses only TE_xxx versions
    Xtr = Xtr.drop(columns=new_pair_names)
    Xva = Xva.drop(columns=new_pair_names)
    Xho = Xho.drop(columns=new_pair_names)
    Xte = Xte.drop(columns=new_pair_names)

    return Xtr, Xva, Xho, Xte, out_cols


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
    te_cols: list[str] | None = None,
    te_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    import lightgbm as lgb

    te_cols = te_cols or []
    te_pairs = te_pairs or []

    cv = make_cv()
    oof = np.zeros(len(X_pool))
    holdout_pred_folds = np.zeros((n_folds, len(X_holdout)))
    test_pred_folds = np.zeros((n_folds, len(X_test)))
    fold_aucs: list[float] = []

    # Align categorical dictionaries up front so each fold sees consistent codes
    X_pool, X_holdout, X_test = _align_category_codes(
        X_pool, X_holdout, X_test, cat_cols=categorical_cols
    )

    # Use the full frames (need raw categoricals for both LGB native cats AND TE source)
    use_te = bool(te_cols or te_pairs)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(np.arange(len(X_pool)), y_pool)):
        Xtr = X_pool.iloc[tr_idx].copy()
        Xva = X_pool.iloc[val_idx].copy()
        ytr = y_pool[tr_idx]
        yva = y_pool[val_idx]
        Xho = X_holdout.copy()
        Xte = X_test.copy()

        fold_feature_cols = list(feature_cols)
        if use_te:
            Xtr, Xva, Xho, Xte, te_added = _apply_target_encoding(
                Xtr, Xva, Xho, Xte, ytr, te_cols, te_pairs
            )
            fold_feature_cols = list(feature_cols) + te_added

        Xtr_view = Xtr[fold_feature_cols]
        Xva_view = Xva[fold_feature_cols]
        Xho_view = Xho[fold_feature_cols]
        Xte_view = Xte[fold_feature_cols]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr_view, ytr,
            eval_set=[(Xva_view, yva)],
            categorical_feature=categorical_cols,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        val_pred = model.predict_proba(Xva_view)[:, 1]
        oof[val_idx] = val_pred
        fold_auc = roc_auc_score(yva, val_pred)
        fold_aucs.append(fold_auc)
        holdout_pred_folds[fold] = model.predict_proba(Xho_view)[:, 1]
        test_pred_folds[fold] = model.predict_proba(Xte_view)[:, 1]
        print(f"  fold {fold+1}/{n_folds}  AUC={fold_auc:.5f}  best_iter={model.best_iteration_}")

    holdout_pred = holdout_pred_folds.mean(axis=0)
    test_pred = test_pred_folds.mean(axis=0)
    return {
        "oof_pred": oof,
        "holdout_pred": holdout_pred,
        "test_pred": test_pred,
        "fold_aucs": fold_aucs,
    }


def _xgb_defaults(use_gpu: bool) -> dict:
    """XGBoost defaults — tuned for tabular binary classification, ~350K rows.

    Different from LGB: XGB uses level-wise growth + ordinal categorical encoding
    (with enable_categorical=True since XGB 2.0). Decision boundaries genuinely
    differ from LGB's leaf-wise + Fisher splits → real diversity for blending later.
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "n_estimators": 5000,
        "tree_method": "hist",
        "enable_categorical": True,
        "verbosity": 0,
        "random_state": MODEL_SEED,
    }
    if use_gpu:
        params["device"] = "cuda"
    return params


def _train_xgb(
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
    import xgboost as xgb

    cv = make_cv()
    oof = np.zeros(len(X_pool))
    holdout_pred_folds = np.zeros((n_folds, len(X_holdout)))
    test_pred_folds = np.zeros((n_folds, len(X_test)))
    fold_aucs: list[float] = []

    # Align categorical dictionaries so XGB sees consistent codes across folds
    X_pool, X_holdout, X_test = _align_category_codes(
        X_pool, X_holdout, X_test, cat_cols=categorical_cols
    )

    X_pool_view = X_pool[feature_cols]
    X_holdout_view = X_holdout[feature_cols]
    X_test_view = X_test[feature_cols]

    # XGB needs early_stopping_rounds in the constructor (no callback API like LGB)
    xgb_params = {**params, "early_stopping_rounds": early_stopping_rounds}

    for fold, (tr_idx, val_idx) in enumerate(cv.split(np.arange(len(X_pool)), y_pool)):
        Xtr, ytr = X_pool_view.iloc[tr_idx], y_pool[tr_idx]
        Xva, yva = X_pool_view.iloc[val_idx], y_pool[val_idx]
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
        )
        val_pred = model.predict_proba(Xva)[:, 1]
        oof[val_idx] = val_pred
        fold_auc = roc_auc_score(yva, val_pred)
        fold_aucs.append(fold_auc)
        holdout_pred_folds[fold] = model.predict_proba(X_holdout_view)[:, 1]
        test_pred_folds[fold] = model.predict_proba(X_test_view)[:, 1]
        print(f"  fold {fold+1}/{n_folds}  AUC={fold_auc:.5f}  best_iter={model.best_iteration}")

    holdout_pred = holdout_pred_folds.mean(axis=0)
    test_pred = test_pred_folds.mean(axis=0)
    return {
        "oof_pred": oof,
        "holdout_pred": holdout_pred,
        "test_pred": test_pred,
        "fold_aucs": fold_aucs,
    }


def _realmlp_defaults(use_gpu: bool) -> dict:
    """pytabkit RealMLP_TD_Classifier defaults — derived from yekenot's top-voted
    public S6E5 notebook (n_ens=24 → 8 for cost). Different architecture than trees:
    neural net with PLR embeddings + learned categorical embeddings. Decision
    boundaries genuinely differ → low correlation with LGB/XGB → big blend payoff.
    """
    return {
        "random_state": MODEL_SEED,
        "verbosity": 1,
        "val_metric_name": "1-auc_ovr",

        # Ensembling — yekenot used n_ens=24, we use 8 for cost/quality balance
        "n_ens": 8,
        "n_epochs": 5,
        "batch_size": 1024,
        "use_early_stopping": False,

        # Optimization
        "lr": 0.03,
        "wd": 0.018,
        "sq_mom": 0.98,
        "lr_sched": "lin_cos_log_15",
        "first_layer_lr_factor": 0.25,

        # Architecture
        "embedding_size": 6,
        "max_one_hot_cat_size": 18,
        "hidden_sizes": [512, 256, 128],
        "act": "silu",
        "p_drop": 0.05,
        "p_drop_sched": "expm4t",

        # PLR (Periodic Linear ReLU) embeddings — the key reason RealMLP_TD works
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_act_name": "gelu",
        "plr_lr_factor": 0.1151,
        "plr_sigma": 2.33,

        # Regularization
        "ls_eps": 0.01,
        "ls_eps_sched": "sqrt_cos",

        # Transforms
        "add_front_scale": False,
        "bias_init_mode": "neg-uniform-dynamic-2",
        "tfms": ["one_hot", "median_center", "robust_scale",
                 "smooth_clip", "embedding", "l2_normalize"],
    }


def _tabm_defaults(use_gpu: bool) -> dict:
    """pytabkit TabM_D_Classifier defaults — same family as RealMLP but
    different architecture (different paper / network design). Used as the
    "architectural diversity" companion to RealMLP for blending.
    Config patterned on cstdy's S6E4 winning notebook (which used TabM_D).
    """
    return {
        "random_state": MODEL_SEED,
        "verbosity": 1,
        "val_metric_name": "1-auc_ovr",

        # TabM-specific ensembling parameter
        "tabm_k": 32,
        "num_emb_type": "pwl",
        "d_embedding": 12,
        "batch_size": 256,
        "lr": 3e-3,
        "weight_decay": 2e-2,
        "n_epochs": 3,
        "dropout": 0.1,
        "d_block": 256,
        "n_blocks": 3,
    }


def _train_tabm(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    X_holdout: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    n_folds: int,
    cv_seed: int,
) -> dict:
    from pytabkit import TabM_D_Classifier

    cv = make_cv()
    oof = np.zeros(len(X_pool))
    holdout_pred_folds = np.zeros((n_folds, len(X_holdout)))
    test_pred_folds = np.zeros((n_folds, len(X_test)))
    fold_aucs: list[float] = []

    X_pool, X_holdout, X_test = _align_category_codes(
        X_pool, X_holdout, X_test, cat_cols=categorical_cols
    )

    X_pool_view = X_pool[feature_cols]
    X_holdout_view = X_holdout[feature_cols]
    X_test_view = X_test[feature_cols]

    for fold, (tr_idx, val_idx) in enumerate(cv.split(np.arange(len(X_pool)), y_pool)):
        Xtr = X_pool_view.iloc[tr_idx]
        ytr = y_pool[tr_idx]
        Xva = X_pool_view.iloc[val_idx]
        yva = y_pool[val_idx]

        model = TabM_D_Classifier(**params)
        model.fit(Xtr, ytr, Xva, yva)

        val_pred = model.predict_proba(Xva)[:, 1]
        oof[val_idx] = val_pred
        fold_auc = roc_auc_score(yva, val_pred)
        fold_aucs.append(fold_auc)
        holdout_pred_folds[fold] = model.predict_proba(X_holdout_view)[:, 1]
        test_pred_folds[fold] = model.predict_proba(X_test_view)[:, 1]
        print(f"  fold {fold+1}/{n_folds}  AUC={fold_auc:.5f}")

    holdout_pred = holdout_pred_folds.mean(axis=0)
    test_pred = test_pred_folds.mean(axis=0)
    return {
        "oof_pred": oof,
        "holdout_pred": holdout_pred,
        "test_pred": test_pred,
        "fold_aucs": fold_aucs,
    }


def _train_realmlp(
    X_pool: pd.DataFrame,
    y_pool: np.ndarray,
    X_holdout: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    params: dict,
    n_folds: int,
    cv_seed: int,
    te_cols: list[str] | None = None,
    te_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    from pytabkit import RealMLP_TD_Classifier

    te_cols = te_cols or []
    te_pairs = te_pairs or []
    use_te = bool(te_cols or te_pairs)

    cv = make_cv()
    oof = np.zeros(len(X_pool))
    holdout_pred_folds = np.zeros((n_folds, len(X_holdout)))
    test_pred_folds = np.zeros((n_folds, len(X_test)))
    fold_aucs: list[float] = []

    # Align category dicts so pytabkit's embedding sees consistent codes
    X_pool, X_holdout, X_test = _align_category_codes(
        X_pool, X_holdout, X_test, cat_cols=categorical_cols
    )

    for fold, (tr_idx, val_idx) in enumerate(cv.split(np.arange(len(X_pool)), y_pool)):
        Xtr = X_pool.iloc[tr_idx].copy()
        Xva = X_pool.iloc[val_idx].copy()
        ytr = y_pool[tr_idx]
        yva = y_pool[val_idx]
        Xho = X_holdout.copy()
        Xte = X_test.copy()

        fold_feature_cols = list(feature_cols)
        if use_te:
            Xtr, Xva, Xho, Xte, te_added = _apply_target_encoding(
                Xtr, Xva, Xho, Xte, ytr, te_cols, te_pairs
            )
            fold_feature_cols = list(feature_cols) + te_added

        Xtr_view = Xtr[fold_feature_cols]
        Xva_view = Xva[fold_feature_cols]
        Xho_view = Xho[fold_feature_cols]
        Xte_view = Xte[fold_feature_cols]

        model = RealMLP_TD_Classifier(**params)
        # pytabkit's fit takes val set as positional args (different from sklearn)
        model.fit(Xtr_view, ytr, Xva_view, yva)

        val_pred = model.predict_proba(Xva_view)[:, 1]
        oof[val_idx] = val_pred
        fold_auc = roc_auc_score(yva, val_pred)
        fold_aucs.append(fold_auc)
        holdout_pred_folds[fold] = model.predict_proba(Xho_view)[:, 1]
        test_pred_folds[fold] = model.predict_proba(Xte_view)[:, 1]
        print(f"  fold {fold+1}/{n_folds}  AUC={fold_auc:.5f}")

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
    te_cols: list[str] | None = None,
    te_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Train one variant via 5-fold CV; return OOF + holdout + test predictions.

    Args:
        algo: 'lgb' or 'xgb' (more to come: catboost, histgb, realmlp)
        te_cols: list of categorical columns to fold-safely target-encode
        te_pairs: list of (col_a, col_b) tuples to fold-safely target-encode
    """
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
            te_cols=te_cols, te_pairs=te_pairs,
        )
    elif algo == "xgb":
        params = {**_xgb_defaults(use_gpu), **(params or {})}
        # XGB doesn't have fold-safe TE built into our pipeline yet — guard
        if te_cols or te_pairs:
            raise NotImplementedError("TE for XGB not implemented yet (would need fold-safe loop).")
        result = _train_xgb(
            X_pool=X_pool, y_pool=y_pool, X_holdout=X_holdout, X_test=X_test,
            feature_cols=feature_cols, categorical_cols=categorical_cols,
            params=params, n_folds=n_folds, cv_seed=cv_seed,
            early_stopping_rounds=early_stopping_rounds,
        )
    elif algo == "realmlp":
        params = {**_realmlp_defaults(use_gpu), **(params or {})}
        result = _train_realmlp(
            X_pool=X_pool, y_pool=y_pool, X_holdout=X_holdout, X_test=X_test,
            feature_cols=feature_cols, categorical_cols=categorical_cols,
            params=params, n_folds=n_folds, cv_seed=cv_seed,
            te_cols=te_cols, te_pairs=te_pairs,
        )
    elif algo == "tabm":
        params = {**_tabm_defaults(use_gpu), **(params or {})}
        if te_cols or te_pairs:
            raise NotImplementedError("TE for TabM not wired yet.")
        result = _train_tabm(
            X_pool=X_pool, y_pool=y_pool, X_holdout=X_holdout, X_test=X_test,
            feature_cols=feature_cols, categorical_cols=categorical_cols,
            params=params, n_folds=n_folds, cv_seed=cv_seed,
        )
    else:
        raise NotImplementedError(f"algo={algo!r} not yet implemented in train_variant().")
    result["runtime_sec"] = time.time() - t0
    result["algo"] = algo
    result["params"] = params
    return result
