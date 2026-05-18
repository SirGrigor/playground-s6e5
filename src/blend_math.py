"""Step-4 blend math helpers.

Tools for math-driven blend weight selection without burning LB submission slots.

- fit_quadratic_lb: 3+ LB points → analytical optimum via quadratic fit
- nm_optimize_holdout: Nelder-Mead on holdout AUC for N-way blend
- predict_blend_lb: predict blend LB given solo LBs + rho (calibrated formula)
- rank_normalize: rank-uniform conversion for math-comparable inputs

Usage in notebook:
    from src.blend_math import fit_quadratic_lb
    w_opt, lb_pred = fit_quadratic_lb(
        weights=[0.0, 0.5, 1.0],
        lbs=[0.95446, 0.95448, 0.95449],
    )
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def rank_normalize(a: np.ndarray) -> np.ndarray:
    """Map array to rank-uniform on [0, 1]."""
    r = rankdata(a)
    return (r - 1.0) / (len(a) - 1.0)


def fit_quadratic_lb(
    weights: list[float] | np.ndarray,
    lbs: list[float] | np.ndarray,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float, dict]:
    """Fit LB(w) = a + b·w + c·w² and return (w_optimal, predicted_lb, debug).

    Requires ≥3 measured points. If c<0 (concave-down), the analytical optimum
    is w* = -b/(2c); clipped to bounds if outside. If c≥0 (concave-up or flat),
    optimum is at a boundary (whichever bound has higher predicted LB).

    Args:
        weights: weight values tested (e.g., [0.0, 0.5, 1.0])
        lbs:     LB scores at those weights
        bounds:  valid weight range, defaults to [0, 1]

    Returns:
        (w_optimal, lb_at_optimum, debug_dict with {a, b, c, w_unconstrained})
    """
    w = np.asarray(weights, dtype=float)
    y = np.asarray(lbs, dtype=float)
    if len(w) < 3:
        raise ValueError(f"Need ≥3 LB points, got {len(w)}")

    A = np.vstack([np.ones_like(w), w, w**2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = coef

    def lb(x: float) -> float:
        return a + b * x + c * x**2

    if abs(c) < 1e-12:
        # Linear — optimum at boundary
        w_uncon = bounds[1] if b > 0 else bounds[0]
        w_opt = w_uncon
    else:
        w_uncon = -b / (2.0 * c)
        if c < 0:
            # Concave-down — w_uncon is a maximum
            w_opt = float(np.clip(w_uncon, bounds[0], bounds[1]))
        else:
            # Concave-up — w_uncon is a minimum; optimum is whichever bound is higher
            w_opt = bounds[0] if lb(bounds[0]) >= lb(bounds[1]) else bounds[1]

    return w_opt, lb(w_opt), {
        "a": a, "b": b, "c": c,
        "w_unconstrained": w_uncon,
        "shape": "concave-down" if c < 0 else ("concave-up" if c > 0 else "linear"),
    }


def nm_optimize_holdout(
    preds_holdout: dict[str, np.ndarray],
    y_holdout: np.ndarray,
    init_weights: dict[str, float] | None = None,
    rank_norm: bool = True,
) -> tuple[dict[str, float], float, dict]:
    """Find blend weights maximizing holdout AUC via Nelder-Mead.

    Args:
        preds_holdout:  {model_name: predictions_on_holdout}. All must share length.
        y_holdout:      ground truth (binary)
        init_weights:   starting weights {name: w}. Defaults to uniform.
        rank_norm:      if True, rank-normalize each model's preds before blending
                        (recommended when models have different output scales)

    Returns:
        (optimal_weights, holdout_auc_at_opt, debug)
    """
    names = list(preds_holdout.keys())
    if rank_norm:
        preds = np.column_stack([rank_normalize(preds_holdout[n]) for n in names])
    else:
        preds = np.column_stack([preds_holdout[n] for n in names])

    if init_weights is None:
        init = np.ones(len(names)) / len(names)
    else:
        init = np.array([init_weights.get(n, 1.0 / len(names)) for n in names])

    def loss(w: np.ndarray) -> float:
        w_clip = np.clip(w, 0.0, None)
        s = w_clip.sum()
        if s < 1e-9:
            return 1.0  # degenerate
        w_norm = w_clip / s
        blend = preds @ w_norm
        return -roc_auc_score(y_holdout, blend)

    result = minimize(
        loss, init,
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 1000},
    )
    w_final = np.clip(result.x, 0.0, None)
    w_final = w_final / w_final.sum()
    auc_final = -result.fun

    return dict(zip(names, w_final.tolist())), float(auc_final), {
        "niter": result.nit,
        "init_auc": -loss(init),
        "rank_normalized": rank_norm,
    }


def predict_blend_lb(
    lb_a: float,
    lb_b: float,
    rho: float,
    weight_b: float,
    bonus_coef: float = 0.022,
) -> float:
    """Predict blend LB from solo LBs + rank correlation + weight.

    Model:  LB(w) ≈ (1-w)·lb_a + w·lb_b + bonus_coef·(1-ρ²)·w·(1-w)

    bonus_coef=0.022 was calibrated on the S6E5 giov+TabPFN pair (one LB point at
    w=0.20, ρ=0.972, gap=0.005, observed bonus=0.00085). Use as a rough guide;
    actual blend bonus can vary by ±30% depending on tail behavior.

    Args:
        lb_a:        LB of model A (at weight 1-w)
        lb_b:        LB of model B (at weight w)
        rho:         rank-correlation between A and B
        weight_b:    weight on B in the blend
        bonus_coef:  blend-bonus calibration constant (default S6E5-fitted)

    Returns:
        Predicted blend LB
    """
    w = weight_b
    linear = (1.0 - w) * lb_a + w * lb_b
    bonus = bonus_coef * (1.0 - rho**2) * w * (1.0 - w)
    return linear + bonus


def recommend_next_weight(
    measured_w: list[float],
    measured_lb: list[float],
    rho: float | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, str]:
    """Suggest the next weight to test for maximum information gain.

    If <3 points: pick the most extreme untested point (0, 1, or 0.5).
    If ≥3 points: fit quadratic, recommend the analytical optimum if it's
                  inside bounds and far from any measured point; else exit signal.

    Returns:
        (recommended_weight, reason_string). reason='stop' means math says we're done.
    """
    measured_w = sorted(set(measured_w))
    if len(measured_w) == 0:
        return 0.0, "no data — submit pure A first"
    if len(measured_w) == 1:
        return 1.0, "submit pure B to get the second endpoint"
    if len(measured_w) == 2:
        return 0.5, "submit interior point (50/50) to fit quadratic"

    w_opt, lb_pred, dbg = fit_quadratic_lb(measured_w, measured_lb, bounds=bounds)
    # If w_opt is close (within 0.05) to a measured point, math is converged
    if min(abs(w_opt - wm) for wm in measured_w) < 0.05:
        return w_opt, f"converged at w={w_opt:.3f} (curve shape: {dbg['shape']}) — stop"
    return w_opt, f"submit math-optimal w={w_opt:.3f}, predicted LB {lb_pred:.5f}"
