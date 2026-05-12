"""Evaluation utilities — AUC on OOF and sacred holdout."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)


def report(oof, y_train, holdout_pred=None, y_holdout=None, label=""):
    out = {"oof_auc": auc(y_train, oof)}
    if holdout_pred is not None and y_holdout is not None:
        out["holdout_auc"] = auc(y_holdout, holdout_pred)
        out["gap_oof_minus_holdout"] = out["oof_auc"] - out["holdout_auc"]
    msg = f"[{label}] " if label else ""
    msg += "  ".join(f"{k}={v:.5f}" for k, v in out.items())
    print(msg)
    return out
