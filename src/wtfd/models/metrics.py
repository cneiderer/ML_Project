"""
metrics.py

Purpose
-------
Provide reusable evaluation utilities for binary classification in the WTDF
modeling workflow.

This module centralizes metric computation so that model evaluation is:
- consistent across notebooks and scripts
- reusable across different model families
- independent from trainer implementation details
- easier to compare across experiments and time horizons

Core Responsibilities
---------------------
- Compute binary classification metrics from labels and probabilities.
- Derive confusion-matrix-based counts and rates.
- Evaluate model performance at a specific probability threshold.
- Sweep across thresholds for comparison and threshold selection.
- Safely handle edge cases such as single-class evaluation subsets.

Design Notes
------------
- Task-agnostic: functions operate only on y_true, y_pred, and y_prob.
- Trainer-independent: no assumptions about model classes or pipelines.
- Safe by default: AUC metrics return NaN rather than failing when undefined.
- Comparison-friendly: outputs are returned as flat dictionaries or DataFrames.

Typical Usage
-------------
>>> y_prob = trainer.predict_proba(X_val)
>>> best = find_best_threshold(y_val, y_prob, optimize_for="f1")
>>> test_metrics = evaluate_at_threshold(y_test, y_prob_test, threshold=best["best_threshold"])
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Input normalization and validation helpers
# ---------------------------------------------------------------------
def _as_1d_numpy(
    values: pd.Series | np.ndarray | list[float] | list[int],
) -> np.ndarray:
    """
    Convert a vector-like input to a flattened NumPy array.

    Parameters
    ----------
    values : pandas.Series or numpy.ndarray or list
        Input array-like values.

    Returns
    -------
    numpy.ndarray
        Flattened one-dimensional NumPy array.

    Notes
    -----
    This helper standardizes downstream metric functions so they do not need
    to handle multiple input container types separately.
    """
    arr = np.asarray(values).ravel()
    logger.debug("Converted input to 1D NumPy array with length=%d", len(arr))
    return arr


def _validate_binary_inputs(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: Optional[pd.Series | np.ndarray | list[float]] = None,
    y_pred: Optional[pd.Series | np.ndarray | list[int] | list[float]] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate and normalize binary classification inputs.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like or None, default=None
        Positive-class probabilities or scores.
    y_pred : array-like or None, default=None
        Predicted binary labels.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray | None, numpy.ndarray | None]
        Normalized arrays for y_true, y_prob, and y_pred.

    Raises
    ------
    ValueError
        If lengths are inconsistent, if y_true contains NaN values, or if
        y_true / y_pred contain non-binary values.
    """
    y_true_arr = _as_1d_numpy(y_true)

    if pd.isna(y_true_arr).any():
        logger.error("y_true contains NaN values.")
        raise ValueError("y_true contains NaN values. Drop missing targets before evaluation.")

    unique_true = set(pd.Series(y_true_arr).unique())
    if not unique_true.issubset({0, 1}):
        logger.error("y_true contains non-binary values: %s", sorted(unique_true))
        raise ValueError(
            f"y_true must be binary with values in {{0, 1}}, got {sorted(unique_true)}"
        )

    y_prob_arr = None
    if y_prob is not None:
        y_prob_arr = _as_1d_numpy(y_prob)
        if len(y_prob_arr) != len(y_true_arr):
            logger.error(
                "Length mismatch between y_true (%d) and y_prob (%d)",
                len(y_true_arr),
                len(y_prob_arr),
            )
            raise ValueError("y_prob must have the same length as y_true")

    y_pred_arr = None
    if y_pred is not None:
        y_pred_arr = _as_1d_numpy(y_pred)
        if len(y_pred_arr) != len(y_true_arr):
            logger.error(
                "Length mismatch between y_true (%d) and y_pred (%d)",
                len(y_true_arr),
                len(y_pred_arr),
            )
            raise ValueError("y_pred must have the same length as y_true")

        unique_pred = set(pd.Series(y_pred_arr).unique())
        if not unique_pred.issubset({0, 1}):
            logger.error("y_pred contains non-binary values: %s", sorted(unique_pred))
            raise ValueError(
                f"y_pred must be binary with values in {{0, 1}}, got {sorted(unique_pred)}"
            )

    logger.debug(
        "Validated binary inputs successfully | n=%d | has_y_prob=%s | has_y_pred=%s",
        len(y_true_arr),
        y_prob_arr is not None,
        y_pred_arr is not None,
    )
    return (
        y_true_arr.astype(int),
        y_prob_arr,
        None if y_pred_arr is None else y_pred_arr.astype(int),
    )


# ---------------------------------------------------------------------
# Safe AUC helpers
# ---------------------------------------------------------------------
def safe_roc_auc(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Compute ROC AUC safely.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Positive-class probabilities or scores.

    Returns
    -------
    float
        ROC AUC value, or NaN if y_true contains fewer than two classes.

    Notes
    -----
    ROC AUC is undefined when only one class is present in y_true. In that
    case this function returns NaN rather than raising an exception.
    """
    y_true_arr, y_prob_arr, _ = _validate_binary_inputs(y_true=y_true, y_prob=y_prob)

    if len(np.unique(y_true_arr)) < 2:
        logger.warning("ROC AUC undefined because y_true contains fewer than two classes.")
        return float("nan")

    score = float(roc_auc_score(y_true_arr, y_prob_arr))
    logger.debug("Computed ROC AUC=%.6f", score)
    return score


def safe_pr_auc(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: pd.Series | np.ndarray | list[float],
) -> float:
    """
    Compute PR AUC safely.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Positive-class probabilities or scores.

    Returns
    -------
    float
        PR AUC value, or NaN if y_true contains fewer than two classes.

    Notes
    -----
    PR AUC is undefined in meaningful terms when only one class is present in
    y_true. In that case this function returns NaN rather than raising.
    """
    y_true_arr, y_prob_arr, _ = _validate_binary_inputs(y_true=y_true, y_prob=y_prob)

    if len(np.unique(y_true_arr)) < 2:
        logger.warning("PR AUC undefined because y_true contains fewer than two classes.")
        return float("nan")

    score = float(average_precision_score(y_true_arr, y_prob_arr))
    logger.debug("Computed PR AUC=%.6f", score)
    return score


# ---------------------------------------------------------------------
# Confusion-matrix-based metrics
# ---------------------------------------------------------------------
def compute_confusion_metrics(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_pred: pd.Series | np.ndarray | list[int] | list[float],
) -> dict[str, float | int | np.ndarray]:
    """
    Compute confusion-matrix-derived metrics for binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    dict[str, float | int | numpy.ndarray]
        Dictionary containing:
        - confusion_matrix
        - tn, fp, fn, tp
        - specificity
        - negative_predictive_value
        - false_positive_rate
        - false_negative_rate
        - support_negative
        - support_positive

    Notes
    -----
    Confusion matrix values are computed using the label order [0, 1] to ensure
    consistent interpretation across experiments.
    """
    y_true_arr, _, y_pred_arr = _validate_binary_inputs(y_true=y_true, y_pred=y_pred)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    results = {
        "confusion_matrix": cm,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "specificity": specificity,
        "negative_predictive_value": npv,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "support_negative": int((y_true_arr == 0).sum()),
        "support_positive": int((y_true_arr == 1).sum()),
    }

    logger.debug(
        "Computed confusion metrics | tn=%d fp=%d fn=%d tp=%d",
        tn,
        fp,
        fn,
        tp,
    )
    return results


# ---------------------------------------------------------------------
# Standard binary classification metric bundles
# ---------------------------------------------------------------------
def compute_binary_classification_metrics(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_pred: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: Optional[pd.Series | np.ndarray | list[float]] = None,
) -> dict[str, Any]:
    """
    Compute a standardized set of binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    y_prob : array-like or None, default=None
        Positive-class probabilities or scores.

    Returns
    -------
    dict[str, Any]
        Metric dictionary containing:
        - precision
        - recall
        - f1
        - balanced_accuracy
        - accuracy
        - prevalence
        - classification_report
        - confusion-matrix-based metrics
        - roc_auc
        - pr_auc

    Notes
    -----
    If y_prob is not provided, AUC metrics are returned as NaN.
    """
    y_true_arr, y_prob_arr, y_pred_arr = _validate_binary_inputs(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
    )

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    accuracy = float((y_true_arr == y_pred_arr).mean())
    prevalence = float((y_true_arr == 1).mean())

    metrics: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "accuracy": accuracy,
        "prevalence": prevalence,
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            output_dict=True,
            zero_division=0,
        ),
    }

    # Add confusion-matrix-derived counts and rates.
    metrics.update(compute_confusion_metrics(y_true_arr, y_pred_arr))

    # Add probability-based ranking metrics when probabilities are available.
    if y_prob_arr is not None:
        metrics["roc_auc"] = safe_roc_auc(y_true_arr, y_prob_arr)
        metrics["pr_auc"] = safe_pr_auc(y_true_arr, y_prob_arr)
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    logger.info(
        "Computed binary metrics | precision=%.4f recall=%.4f f1=%.4f balanced_accuracy=%.4f",
        precision,
        recall,
        f1,
        balanced_acc,
    )
    return metrics


# ---------------------------------------------------------------------
# Threshold evaluation / selection
# ---------------------------------------------------------------------
def evaluate_at_threshold(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: pd.Series | np.ndarray | list[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate binary classification performance at a specified threshold.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Positive-class probabilities or scores.
    threshold : float, default=0.5
        Probability threshold used to convert y_prob into binary predictions.

    Returns
    -------
    dict[str, Any]
        Standardized metric dictionary for the supplied threshold, including
        the threshold value itself.

    Raises
    ------
    ValueError
        If the threshold is outside [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        logger.error("Invalid threshold %.6f. Must be in [0, 1].", threshold)
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    y_true_arr, y_prob_arr, _ = _validate_binary_inputs(y_true=y_true, y_prob=y_prob)

    # Convert continuous probabilities into binary predictions.
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    results = compute_binary_classification_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_prob=y_prob_arr,
    )
    results["threshold"] = float(threshold)

    logger.debug(
        "Evaluated threshold=%.4f | precision=%.4f recall=%.4f f1=%.4f",
        threshold,
        float(results["precision"]),
        float(results["recall"]),
        float(results["f1"]),
    )
    return results


def _score_from_metric_name(metrics: dict[str, Any], optimize_for: str) -> float:
    """
    Extract a threshold-selection score from a metric dictionary.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metric dictionary.
    optimize_for : str
        Metric name to optimize.

    Returns
    -------
    float
        Extracted score value.

    Raises
    ------
    ValueError
        If the requested optimization metric is unsupported.
    """
    supported = {
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "specificity",
    }

    if optimize_for not in supported:
        logger.error(
            "Unsupported optimize_for '%s'. Supported metrics: %s",
            optimize_for,
            sorted(supported),
        )
        raise ValueError(
            f"Unsupported optimize_for='{optimize_for}'. Supported values are: {sorted(supported)}"
        )

    score = float(metrics[optimize_for])
    logger.debug("Extracted optimization score %s=%.6f", optimize_for, score)
    return score


def build_threshold_sweep_table(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: pd.Series | np.ndarray | list[float],
    thresholds: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of threshold-dependent performance metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Positive-class probabilities or scores.
    thresholds : iterable of float or None, default=None
        Thresholds to evaluate. If None, a default threshold grid is used.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per threshold and columns including:
        - threshold
        - precision
        - recall
        - f1
        - balanced_accuracy
        - specificity
        - false_positive_rate
        - false_negative_rate
        - tp, fp, tn, fn
        - support_positive, support_negative
        - roc_auc, pr_auc

    Notes
    -----
    ROC AUC and PR AUC do not vary with the threshold because they are based
    on ranking/probability information, but they are included for convenience
    when comparing tables across runs.
    """
    y_true_arr, y_prob_arr, _ = _validate_binary_inputs(y_true=y_true, y_prob=y_prob)

    if thresholds is None:
        # Use a moderately dense default grid with a few especially common
        # thresholds included explicitly.
        thresholds_arr = np.unique(
            np.concatenate(
                [
                    np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
                    np.linspace(0.02, 0.98, 49),
                ]
            )
        )
    else:
        thresholds_arr = np.array(sorted({float(t) for t in thresholds}))

    rows: list[dict[str, Any]] = []

    for threshold in thresholds_arr:
        result = evaluate_at_threshold(
            y_true=y_true_arr,
            y_prob=y_prob_arr,
            threshold=float(threshold),
        )

        rows.append(
            {
                "threshold": result["threshold"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "balanced_accuracy": result["balanced_accuracy"],
                "specificity": result["specificity"],
                "false_positive_rate": result["false_positive_rate"],
                "false_negative_rate": result["false_negative_rate"],
                "tp": result["tp"],
                "fp": result["fp"],
                "tn": result["tn"],
                "fn": result["fn"],
                "support_positive": result["support_positive"],
                "support_negative": result["support_negative"],
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    logger.info(
        "Built threshold sweep table with %d candidate thresholds",
        len(sweep_df),
    )
    return sweep_df


def find_best_threshold(
    y_true: pd.Series | np.ndarray | list[int] | list[float],
    y_prob: pd.Series | np.ndarray | list[float],
    optimize_for: str = "f1",
    thresholds: Optional[Iterable[float]] = None,
) -> dict[str, Any]:
    """
    Select the best decision threshold for a chosen optimization metric.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Positive-class probabilities or scores.
    optimize_for : str, default='f1'
        Metric to optimize. Supported values:
        - 'f1'
        - 'precision'
        - 'recall'
        - 'balanced_accuracy'
        - 'specificity'
    thresholds : iterable of float or None, default=None
        Optional candidate thresholds to evaluate.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - best_threshold
        - best_score
        - optimize_for
        - threshold_table
        - best_metrics

    Notes
    -----
    Tie-breaking logic prefers:
    1. higher recall
    2. higher precision
    3. smaller threshold

    This makes the selection slightly recall-friendly, which is often desirable
    in leading-fault detection tasks where missed positives can be costly.
    """
    sweep_df = build_threshold_sweep_table(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
    )

    if sweep_df.empty:
        logger.error("Threshold sweep produced no candidate rows.")
        raise ValueError("Threshold sweep produced no candidate rows.")

    supported = {
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "specificity",
    }
    if optimize_for not in supported:
        logger.error(
            "Unsupported optimize_for '%s'. Supported metrics: %s",
            optimize_for,
            sorted(supported),
        )
        raise ValueError(
            f"Unsupported optimize_for='{optimize_for}'. Supported values are: {sorted(supported)}"
        )

    max_score = float(sweep_df[optimize_for].max())
    best_rows = sweep_df[sweep_df[optimize_for] == max_score].copy()

    # Tie-break among equally scoring thresholds in a way that remains slightly
    # recall-friendly for early warning use cases.
    best_rows = best_rows.sort_values(
        by=["recall", "precision", "threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_threshold = float(best_rows.loc[0, "threshold"])
    best_metrics = evaluate_at_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=best_threshold,
    )

    logger.info(
        "Selected best threshold=%.6f using optimize_for=%s with best_score=%.6f",
        best_threshold,
        optimize_for,
        max_score,
    )

    return {
        "best_threshold": best_threshold,
        "best_score": max_score,
        "optimize_for": optimize_for,
        "threshold_table": sweep_df,
        "best_metrics": best_metrics,
    }