"""Statistical recovery metrics — for cells where ground truth is known.

These quantify how well an estimator recovers the true support and
coefficients on synthetic data, independent of cross-package agreement.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def support_metrics(
    beta_hat: NDArray[np.float64],
    beta_true: NDArray[np.float64],
    rtol: float = 1e-6,
) -> dict[str, float]:
    """Support-recovery precision/recall/F1.

    Active = |β| > rtol * max(|β|). At λ_max all coefficients are 0,
    so a near-zero estimate against a non-zero truth is "no support
    found" (recall 0, precision is 1 vacuously → reported as 0.0).
    """
    scale_true = float(np.max(np.abs(beta_true)))
    scale_hat  = float(np.max(np.abs(beta_hat)))
    s_true = np.abs(beta_true) > rtol * scale_true if scale_true > 0 else np.zeros_like(beta_true, dtype=bool)
    s_hat  = np.abs(beta_hat)  > rtol * scale_hat  if scale_hat  > 0 else np.zeros_like(beta_hat,  dtype=bool)
    tp = int(np.sum(s_true & s_hat))
    fp = int(np.sum(~s_true & s_hat))
    fn = int(np.sum(s_true & ~s_hat))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "support_precision": precision,
        "support_recall":    recall,
        "support_f1":        f1,
        "true_support_size": int(np.sum(s_true)),
        "hat_support_size":  int(np.sum(s_hat)),
    }


def beta_rmse(beta_hat: NDArray[np.float64], beta_true: NDArray[np.float64]) -> float:
    """Root-mean-square error of coefficient estimate."""
    diff = beta_hat - beta_true
    return float(np.sqrt(np.mean(diff * diff)))


def prediction_mse(
    x: NDArray[np.float64],
    beta_hat: NDArray[np.float64],
    beta_true: NDArray[np.float64],
) -> float:
    """E[(x β̂ − x β*)²] on the provided design."""
    pred_hat  = x @ beta_hat
    pred_true = x @ beta_true
    diff = pred_hat - pred_true
    return float(np.mean(diff * diff))


def per_lambda(
    coef_path: NDArray[np.float64],
    beta_true: NDArray[np.float64],
    x_eval: NDArray[np.float64] | None = None,
    rtol: float = 1e-6,
) -> dict[str, list[float]]:
    """Recovery metrics along an entire (n_lambdas, p) coefficient path."""
    out: dict[str, list[float]] = {
        "support_f1": [], "support_precision": [], "support_recall": [],
        "beta_rmse": [], "hat_support_size": [],
    }
    if x_eval is not None:
        out["prediction_mse"] = []
    for bh in coef_path:
        sm = support_metrics(bh, beta_true, rtol)
        out["support_f1"].append(sm["support_f1"])
        out["support_precision"].append(sm["support_precision"])
        out["support_recall"].append(sm["support_recall"])
        out["hat_support_size"].append(float(sm["hat_support_size"]))
        out["beta_rmse"].append(beta_rmse(bh, beta_true))
        if x_eval is not None:
            out["prediction_mse"].append(prediction_mse(x_eval, bh, beta_true))
    return out
