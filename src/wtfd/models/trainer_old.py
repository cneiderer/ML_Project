"""
trainer.py

Purpose
-------
Training and evaluation utilities for the WTDF modeling workflow.

This module supports:
- Logistic Regression
- Random Forest
- XGBoost

Design Notes
------------
- Threshold tuning is performed on a validation set, not the training set.
- Scaling is applied only when appropriate (e.g., logistic regression).
- Evaluation is recall-aware and includes richer diagnostics to support the
  project objective of leading fault detection.
- Feature-importance extraction is supported for all model families.
- Threshold search and metric computation are delegated to wtfd.models.metrics.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]

from wtfd.models.metrics import evaluate_at_threshold, find_best_threshold
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
        self.threshold_tuning_summary: Optional[dict[str, Any]] = None
        self.model = self._initialize_model()

        logger.info(
            "Initialized WindFaultTrainer with model_type=%s, random_state=%d",
            self.model_type,
            self.random_state,
        )

    def _initialize_model(self) -> Pipeline | RandomForestClassifier | Any:
        """
        Initialize the requested model.

        Returns
        -------
        sklearn estimator
            Model instance ready for fitting.
        """
        if self.model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError(
                    "model_type='xgboost' requested but xgboost is not installed. "
                    "Install xgboost or choose model_type='rf' or 'logistic'."
                )

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

        if pd.Series(y_train).isna().any():
            raise ValueError("y_train contains NaN values. Drop missing targets before fitting.")

        unique_y = set(pd.Series(y_train).unique())
        if not unique_y.issubset({0, 1}):
            raise ValueError(f"y_train must be binary with values in {{0, 1}}, got {sorted(unique_y)}")

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
        thresholds: Optional[list[float] | np.ndarray] = None,
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
            Metric to optimize when selecting threshold. Supported values:
            - "f1"
            - "precision"
            - "recall"
            - "balanced_accuracy"
            - "specificity"
        thresholds : list[float] or numpy.ndarray or None, default=None
            Optional candidate threshold values.

        Returns
        -------
        float
            Selected decision threshold.
        """
        logger.info(
            "Tuning threshold on validation set | rows=%d | optimize_for=%s",
            len(X_val),
            optimize_for,
        )

        y_probs = self.predict_proba(X_val)
        tuning_results = find_best_threshold(
            y_true=y_val,
            y_prob=y_probs,
            optimize_for=optimize_for,
            thresholds=thresholds,
        )

        self.best_threshold = float(tuning_results["best_threshold"])
        self.threshold_tuning_summary = tuning_results

        logger.info(
            "Selected best threshold=%.6f using optimize_for=%s with score=%.6f",
            self.best_threshold,
            optimize_for,
            float(tuning_results["best_score"]),
        )

        return self.best_threshold

    def fit_and_tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_for: str = "f1",
        thresholds: Optional[list[float] | np.ndarray] = None,
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
        thresholds : list[float] or numpy.ndarray or None, default=None
            Optional candidate threshold values.

        Returns
        -------
        float
            Selected decision threshold.
        """
        self.fit(X_train, y_train)
        return self.tune_threshold(
            X_val,
            y_val,
            optimize_for=optimize_for,
            thresholds=thresholds,
        )

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

        return {
            "precision": float(detailed["precision"]),
            "recall": float(detailed["recall"]),
            "f1": float(detailed["f1"]),
            "roc_auc": float(detailed["roc_auc"]),
            "pr_auc": float(detailed["pr_auc"]),
            "balanced_accuracy": float(detailed["balanced_accuracy"]),
            "specificity": float(detailed["specificity"]),
            "threshold": float(detailed["threshold"]),
        }

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
        results = evaluate_at_threshold(
            y_true=y_test,
            y_prob=y_probs,
            threshold=float(threshold_to_use),
        )

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