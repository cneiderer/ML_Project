"""
trainer.py

Purpose
-------
Training and evaluation utilities for the WTDF modeling workflow.

This module supports:
- Logistic Regression
- Random Forest
- XGBoost

Key design choices
------------------
- Threshold tuning is performed on a validation set, not the training set.
- Scaling is applied only when appropriate (e.g., logistic regression).
- Evaluation is recall-aware and includes richer diagnostics to support the
  project objective of leading fault detection.
- Feature-importance extraction is supported for all model families.
"""

from __future__ import annotations

import gc
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WindFaultTrainer:
    """
    Trainer and evaluator for leading-fault classification models.

    Parameters
    ----------
    model_type : str, default="xgboost"
        Model family to train. Supported values:
        - "xgboost"
        - "rf"
        - "logistic"
    params : dict or None, default=None
        Optional model hyperparameter overrides.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    SUPPORTED_MODELS = {"xgboost", "rf", "logistic"}

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type='{model_type}'. "
                f"Supported values are: {sorted(self.SUPPORTED_MODELS)}"
            )

        self.model_type = model_type
        self.params = params or {}
        self.random_state = random_state
        self.best_threshold = 0.5
        self.model = self._initialize_model()

        logger.info(
            "Initialized WindFaultTrainer with model_type=%s, random_state=%d",
            self.model_type,
            self.random_state,
        )

    def _initialize_model(self) -> Pipeline | RandomForestClassifier | XGBClassifier:
        """
        Initialize the requested model.

        Returns
        -------
        sklearn estimator
            Model instance ready for fitting.
        """
        if self.model_type == "xgboost":
            config = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_state,
                "tree_method": "hist",
                "device": "cpu",
                "eval_metric": "logloss",
            }
            config.update(self.params)
            model = XGBClassifier(**config)

        elif self.model_type == "rf":
            config = {
                "n_estimators": 100,
                "max_depth": 10,
                "class_weight": "balanced_subsample",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            config.update(self.params)
            model = RandomForestClassifier(**config)

        else:  # logistic
            config = {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": self.random_state,
            }
            config.update(self.params)
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(**config)),
                ]
            )

        logger.debug("Initialized model for model_type=%s", self.model_type)
        return model

    def _get_estimator(self) -> Any:
        """
        Return the underlying estimator, whether wrapped in a pipeline or not.
        """
        if isinstance(self.model, Pipeline):
            return self.model.named_steps["model"]
        return self.model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the model on the training set.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Training labels.
        """
        logger.info(
            "Fitting model_type=%s on %d rows and %d features",
            self.model_type,
            len(X_train),
            X_train.shape[1],
        )

        if self.model_type == "xgboost":
            pos = int(np.sum(y_train))
            neg = int(len(y_train) - pos)
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0
            self._get_estimator().set_params(scale_pos_weight=scale_pos_weight)

            logger.debug(
                "Set XGBoost scale_pos_weight=%.4f using pos=%d, neg=%d",
                scale_pos_weight,
                pos,
                neg,
            )

        self.model.fit(X_train, y_train)

        logger.info("Model fitting complete for model_type=%s", self.model_type)

    def tune_threshold(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_for: str = "f1",
    ) -> float:
        """
        Tune the decision threshold using validation-set probabilities.

        Parameters
        ----------
        X_val : pandas.DataFrame
            Validation features.
        y_val : pandas.Series
            Validation labels.
        optimize_for : str, default="f1"
            Metric to optimize when selecting threshold.
            Currently supports:
            - "f1"

        Returns
        -------
        float
            Selected decision threshold.
        """
        if optimize_for != "f1":
            raise ValueError(
                f"Unsupported optimize_for='{optimize_for}'. Only 'f1' is currently supported."
            )

        logger.info(
            "Tuning threshold on validation set | rows=%d | optimize_for=%s",
            len(X_val),
            optimize_for,
        )

        y_probs = self.predict_proba(X_val)
        self.best_threshold = self._tune_threshold_from_probs(y_val, y_probs)

        logger.info("Selected best threshold=%.6f", self.best_threshold)
        return self.best_threshold

    def fit_and_tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_for: str = "f1",
    ) -> float:
        """
        Fit the model on training data and tune threshold on validation data.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training features.
        y_train : pandas.Series
            Training labels.
        X_val : pandas.DataFrame
            Validation features.
        y_val : pandas.Series
            Validation labels.
        optimize_for : str, default="f1"
            Metric used for threshold tuning.

        Returns
        -------
        float
            Selected decision threshold.
        """
        self.fit(X_train, y_train)
        return self.tune_threshold(X_val, y_val, optimize_for=optimize_for)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return positive-class probabilities.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Positive-class probabilities.
        """
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict binary labels using the specified or tuned threshold.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        threshold : float or None, default=None
            Threshold to apply. If None, uses self.best_threshold.

        Returns
        -------
        numpy.ndarray
            Predicted binary labels.
        """
        threshold_to_use = self.best_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        preds = (probs >= threshold_to_use).astype(int)
        return preds

    def _tune_threshold_from_probs(
        self,
        y_true: pd.Series,
        y_probs: np.ndarray,
    ) -> float:
        """
        Select threshold that maximizes F1-score on validation data.
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = int(np.argmax(f1_scores))

        # precision_recall_curve returns one more precision/recall element
        # than threshold elements.
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        del precisions, recalls, thresholds, f1_scores
        gc.collect()

        return float(best_threshold)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Evaluate the model on a test set with a compact metric summary.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test features.
        y_test : pandas.Series
            Test labels.
        threshold : float or None, default=None
            Threshold to use for binary predictions. If None, uses the tuned
            threshold stored in self.best_threshold.

        Returns
        -------
        dict[str, float]
            Compact evaluation metrics.
        """
        detailed = self.evaluate_detailed(X_test, y_test, threshold=threshold)

        metrics = {
            "precision": detailed["precision"],
            "recall": detailed["recall"],
            "f1": detailed["f1"],
            "roc_auc": detailed["roc_auc"],
            "pr_auc": detailed["pr_auc"],
            "threshold": detailed["threshold"],
        }

        return metrics

    def evaluate_detailed(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Evaluate the model with richer diagnostics.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test features.
        y_test : pandas.Series
            Test labels.
        threshold : float or None, default=None
            Threshold to use for binary predictions. If None, uses the tuned
            threshold stored in self.best_threshold.

        Returns
        -------
        dict[str, Any]
            Detailed evaluation results including confusion matrix and report.
        """
        threshold_to_use = self.best_threshold if threshold is None else threshold

        logger.info(
            "Evaluating model_type=%s on %d rows using threshold=%.6f",
            self.model_type,
            len(X_test),
            threshold_to_use,
        )

        y_probs = self.predict_proba(X_test)
        y_pred = (y_probs >= threshold_to_use).astype(int)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        roc_auc = np.nan
        pr_auc = np.nan

        unique_classes = pd.Series(y_test).dropna().unique()
        if len(unique_classes) >= 2:
            roc_auc = roc_auc_score(y_test, y_probs)
            pr_auc = average_precision_score(y_test, y_probs)
        else:
            logger.warning(
                "AUC metrics undefined because y_test contains fewer than 2 classes"
            )

        results = {
            "precision": report.get("1", {}).get("precision", 0.0),
            "recall": report.get("1", {}).get("recall", 0.0),
            "f1": report.get("1", {}).get("f1-score", 0.0),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": float(threshold_to_use),
            "confusion_matrix": cm,
            "classification_report": report,
            "support_positive": int((pd.Series(y_test) == 1).sum()),
            "support_negative": int((pd.Series(y_test) == 0).sum()),
        }

        del y_probs, y_pred, report
        gc.collect()

        return results

    def get_feature_importance(
        self,
        feature_names: list[str] | pd.Index,
        sort: bool = True,
    ) -> pd.DataFrame:
        """
        Return model-specific feature importance information.

        Parameters
        ----------
        feature_names : list[str] or pandas.Index
            Ordered feature names used during model fitting.
        sort : bool, default=True
            If True, sort by descending absolute importance / coefficient.

        Returns
        -------
        pandas.DataFrame
            DataFrame with feature importance information.
        """
        feature_names = list(feature_names)

        if self.model_type == "logistic":
            estimator = self._get_estimator()
            values = estimator.coef_.ravel()
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": values,
                    "abs_importance": np.abs(values),
                }
            )
            sort_col = "abs_importance"

        elif self.model_type in {"rf", "xgboost"}:
            estimator = self._get_estimator()
            values = estimator.feature_importances_
            importance_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": values,
                }
            )
            sort_col = "importance"

        else:
            raise ValueError(
                f"Feature importance not supported for model_type={self.model_type}"
            )

        if sort:
            importance_df = importance_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

        logger.debug(
            "Computed feature importance for model_type=%s with %d features",
            self.model_type,
            len(feature_names),
        )

        return importance_df

    def refit_on_train_plus_val(
        self,
        X_train_val: pd.DataFrame,
        y_train_val: pd.Series,
    ) -> None:
        """
        Refit the model on the combined training + validation set.

        This is useful after model selection and threshold tuning are complete.

        Parameters
        ----------
        X_train_val : pandas.DataFrame
            Combined training and validation features.
        y_train_val : pandas.Series
            Combined training and validation labels.
        """
        logger.info(
            "Refitting final model on train+val data | rows=%d | features=%d",
            len(X_train_val),
            X_train_val.shape[1],
        )
        self.fit(X_train_val, y_train_val)