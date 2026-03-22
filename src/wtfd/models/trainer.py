"""
trainer.py

Purpose
-------
Provide model training, threshold tuning, prediction, and evaluation utilities
for the WTDF modeling workflow.

This module defines the `WindFaultTrainer`, a reusable trainer class that wraps
supported binary classification models and exposes a consistent interface for:
- model initialization
- fitting on training data
- threshold tuning on validation data
- probability prediction
- thresholded prediction
- compact and detailed evaluation
- feature importance extraction
- final refitting on combined train + validation data

Supported Model Families
------------------------
- logistic : Logistic Regression with standard scaling
- rf       : Random Forest
- xgboost  : XGBoost (when the xgboost package is installed)

Design Notes
------------
- Provide a consistent interface across model types.
- Keep evaluation logic out of the trainer when possible.
- Use logging to make training and evaluation runs traceable.
- Keep model-specific branching isolated to initialization and a few targeted
  behaviors.
- Preserve compatibility with both notebooks and CLI scripts.

Typical Usage
-------------
>>> trainer = WindFaultTrainer(model_type="rf", random_state=42)
>>> trainer.fit(X_train, y_train)
>>> trainer.tune_threshold(X_val, y_val, optimize_for="f1")
>>> metrics = trainer.evaluate(X_test, y_test)

Or, as a combined fit+tune workflow:

>>> trainer = WindFaultTrainer(model_type="logistic")
>>> trainer.fit_and_tune(X_train, y_train, X_val, y_val)
>>> preds = trainer.predict(X_test)
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
    Trainer and evaluator for binary wind fault prediction models.

    This class provides a common interface across supported model families for
    training, threshold tuning, prediction, evaluation, and feature importance
    extraction.

    Parameters
    ----------
    model_type : str, default='xgboost'
        Model family to initialize. Supported values are:
        - 'xgboost'
        - 'rf'
        - 'logistic'
    params : dict[str, Any] or None, default=None
        Optional model hyperparameter overrides. These are merged into the
        model-family-specific defaults during initialization.
    random_state : int, default=42
        Random seed used for reproducibility in supported models.

    Attributes
    ----------
    model_type : str
        Selected model family.
    params : dict[str, Any]
        Hyperparameter overrides supplied at initialization.
    random_state : int
        Reproducibility seed.
    best_threshold : float
        Currently selected decision threshold. Defaults to 0.5 and may be
        updated after threshold tuning.
    threshold_tuning_summary : dict[str, Any] or None
        Detailed threshold tuning results returned by
        `wtfd.models.metrics.find_best_threshold`.
    model : sklearn-compatible estimator
        Initialized model instance or pipeline.
    """

    SUPPORTED_MODELS = {"xgboost", "rf", "logistic"}

    def __init__(
        self,
        model_type: str = "xgboost",
        params: Optional[dict[str, Any]] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize a `WindFaultTrainer`.

        Parameters
        ----------
        model_type : str, default='xgboost'
            Model family to initialize.
        params : dict[str, Any] or None, default=None
            Optional hyperparameter overrides.
        random_state : int, default=42
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If `model_type` is not one of the supported values.
        """
        if model_type not in self.SUPPORTED_MODELS:
            logger.error(
                "Unsupported model_type '%s'. Supported values: %s",
                model_type,
                sorted(self.SUPPORTED_MODELS),
            )
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
            "Initialized WindFaultTrainer | model_type=%s | random_state=%d",
            self.model_type,
            self.random_state,
        )

    def _initialize_model(self) -> Pipeline | RandomForestClassifier | Any:
        """
        Initialize the requested model family using default settings plus any
        caller-provided overrides.

        Returns
        -------
        sklearn estimator
            Initialized estimator or pipeline.

        Raises
        ------
        ImportError
            If `model_type='xgboost'` is requested but xgboost is not installed.

        Notes
        -----
        - Logistic regression is wrapped in a pipeline with `StandardScaler`.
        - Random forest uses balanced subsampling by default.
        - XGBoost uses CPU histogram mode by default and later receives
          `scale_pos_weight` during fitting based on the training target ratio.
        """
        if self.model_type == "xgboost":
            if XGBClassifier is None:
                logger.error(
                    "model_type='xgboost' requested but xgboost is not installed."
                )
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
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(**config)),
                ]
            )

        logger.debug(
            "Model initialized | model_type=%s | params=%s",
            self.model_type,
            self.params,
        )
        return model

    def _get_estimator(self) -> Any:
        """
        Return the underlying estimator, whether wrapped in a pipeline or not.

        Returns
        -------
        Any
            The underlying fitted or unfitted estimator object.

        Notes
        -----
        Logistic regression is wrapped in a pipeline, so this helper provides a
        consistent way to access the actual model object for tasks such as
        coefficient extraction.
        """
        if isinstance(self.model, Pipeline):
            estimator = self.model.named_steps["model"]
        else:
            estimator = self.model

        logger.debug("Retrieved underlying estimator for model_type=%s", self.model_type)
        return estimator

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the model on the training set.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature matrix.
        y_train : pandas.Series
            Training binary target labels.

        Raises
        ------
        ValueError
            If `y_train` contains NaN values or non-binary values.

        Notes
        -----
        For XGBoost, `scale_pos_weight` is computed from the training class
        distribution and applied before fitting. This helps address class
        imbalance without requiring the caller to provide it manually.
        """
        logger.info(
            "Fitting model | model_type=%s | rows=%d | features=%d",
            self.model_type,
            len(X_train),
            X_train.shape[1],
        )

        if pd.Series(y_train).isna().any():
            logger.error("y_train contains NaN values.")
            raise ValueError("y_train contains NaN values. Drop missing targets before fitting.")

        unique_y = set(pd.Series(y_train).unique())
        if not unique_y.issubset({0, 1}):
            logger.error("y_train contains non-binary values: %s", sorted(unique_y))
            raise ValueError(
                f"y_train must be binary with values in {{0, 1}}, got {sorted(unique_y)}"
            )

        if self.model_type == "xgboost":
            pos = int(np.sum(y_train))
            neg = int(len(y_train) - pos)
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0

            self._get_estimator().set_params(scale_pos_weight=scale_pos_weight)

            logger.debug(
                "Set XGBoost scale_pos_weight=%.6f using pos=%d and neg=%d",
                scale_pos_weight,
                pos,
                neg,
            )

        self.model.fit(X_train, y_train)

        logger.info("Model fitting complete | model_type=%s", self.model_type)

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
            Validation feature matrix.
        y_val : pandas.Series
            Validation binary target labels.
        optimize_for : str, default='f1'
            Metric used to select the best threshold. Supported values are
            defined by `wtfd.models.metrics.find_best_threshold`.
        thresholds : list[float] or numpy.ndarray or None, default=None
            Optional candidate thresholds. If None, the metrics module uses its
            default threshold grid.

        Returns
        -------
        float
            Best threshold selected on the validation set.

        Notes
        -----
        The selected threshold is stored in `self.best_threshold`, and the full
        tuning summary is stored in `self.threshold_tuning_summary`.
        """
        logger.info(
            "Tuning threshold | model_type=%s | rows=%d | optimize_for=%s",
            self.model_type,
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
            "Threshold tuning complete | best_threshold=%.6f | optimize_for=%s | best_score=%.6f",
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
        Fit the model on training data and tune the threshold on validation data.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training feature matrix.
        y_train : pandas.Series
            Training binary target labels.
        X_val : pandas.DataFrame
            Validation feature matrix.
        y_val : pandas.Series
            Validation binary target labels.
        optimize_for : str, default='f1'
            Metric used for threshold selection.
        thresholds : list[float] or numpy.ndarray or None, default=None
            Optional threshold candidates.

        Returns
        -------
        float
            Best threshold selected on the validation set.

        Notes
        -----
        This is the standard workflow for most experiments: first fit the model,
        then tune the threshold on a held-out validation split.
        """
        logger.info("Running combined fit-and-tune workflow | model_type=%s", self.model_type)
        self.fit(X_train, y_train)
        return self.tune_threshold(
            X_val=X_val,
            y_val=y_val,
            optimize_for=optimize_for,
            thresholds=thresholds,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict positive-class probabilities.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Predicted probabilities for the positive class.

        Notes
        -----
        This method assumes the underlying model supports `predict_proba`, which
        is true for all currently supported model families.
        """
        probs = self.model.predict_proba(X)[:, 1]
        logger.debug(
            "Predicted probabilities | model_type=%s | rows=%d",
            self.model_type,
            len(X),
        )
        return probs

    def predict(
        self,
        X: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict binary labels using a specified or tuned threshold.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        threshold : float or None, default=None
            Threshold used to convert probabilities to class labels. If None,
            `self.best_threshold` is used.

        Returns
        -------
        numpy.ndarray
            Predicted binary labels.
        """
        threshold_to_use = self.best_threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        preds = (probs >= threshold_to_use).astype(int)

        logger.debug(
            "Predicted binary labels | model_type=%s | rows=%d | threshold=%.6f",
            self.model_type,
            len(X),
            threshold_to_use,
        )
        return preds

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Evaluate the model on a test set and return a compact summary.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test feature matrix.
        y_test : pandas.Series
            Test binary target labels.
        threshold : float or None, default=None
            Threshold used for binary prediction. If None, uses
            `self.best_threshold`.

        Returns
        -------
        dict[str, float]
            Compact evaluation summary containing:
            - precision
            - recall
            - f1
            - roc_auc
            - pr_auc
            - balanced_accuracy
            - specificity
            - threshold

        Notes
        -----
        This method is intended for comparison tables and concise reporting.
        For full diagnostics, use `evaluate_detailed`.
        """
        detailed = self.evaluate_detailed(X_test, y_test, threshold=threshold)

        compact_metrics = {
            "precision": float(detailed["precision"]),
            "recall": float(detailed["recall"]),
            "f1": float(detailed["f1"]),
            "roc_auc": float(detailed["roc_auc"]),
            "pr_auc": float(detailed["pr_auc"]),
            "balanced_accuracy": float(detailed["balanced_accuracy"]),
            "specificity": float(detailed["specificity"]),
            "threshold": float(detailed["threshold"]),
        }

        logger.info(
            "Compact evaluation complete | model_type=%s | f1=%.4f | recall=%.4f | precision=%.4f",
            self.model_type,
            compact_metrics["f1"],
            compact_metrics["recall"],
            compact_metrics["precision"],
        )
        return compact_metrics

    def evaluate_detailed(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Evaluate the model on a test set with richer diagnostics.

        Parameters
        ----------
        X_test : pandas.DataFrame
            Test feature matrix.
        y_test : pandas.Series
            Test binary target labels.
        threshold : float or None, default=None
            Threshold used for binary prediction. If None, uses
            `self.best_threshold`.

        Returns
        -------
        dict[str, Any]
            Detailed evaluation output from
            `wtfd.models.metrics.evaluate_at_threshold`, which may include:
            - scalar performance metrics
            - confusion matrix
            - threshold value
            - classification report

        Notes
        -----
        This method is intended for deeper diagnostics, artifact saving, and
        notebook inspection.
        """
        threshold_to_use = self.best_threshold if threshold is None else threshold

        logger.info(
            "Evaluating model | model_type=%s | rows=%d | threshold=%.6f",
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

        logger.debug(
            "Detailed evaluation complete | model_type=%s | keys=%s",
            self.model_type,
            sorted(results.keys()),
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
            If True, sort the output by descending importance magnitude.

        Returns
        -------
        pandas.DataFrame
            Feature importance DataFrame.

            For logistic regression, columns are:
            - feature
            - coefficient
            - abs_importance

            For random forest and XGBoost, columns are:
            - feature
            - importance

        Raises
        ------
        ValueError
            If feature importance is unsupported for the current model type.

        Notes
        -----
        - Logistic regression uses coefficient values and absolute coefficient
          magnitude.
        - Tree-based models use the estimator's `feature_importances_`.
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
            logger.error(
                "Feature importance not supported for model_type=%s",
                self.model_type,
            )
            raise ValueError(
                f"Feature importance not supported for model_type={self.model_type}"
            )

        if sort:
            importance_df = (
                importance_df.sort_values(sort_col, ascending=False)
                .reset_index(drop=True)
            )

        logger.info(
            "Computed feature importance | model_type=%s | n_features=%d",
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
        Refit the model on the combined training + validation dataset.

        Parameters
        ----------
        X_train_val : pandas.DataFrame
            Combined training and validation feature matrix.
        y_train_val : pandas.Series
            Combined training and validation binary target labels.

        Notes
        -----
        This method is useful after model selection and threshold tuning are
        complete, when you want to train the final model on all development
        data prior to final holdout evaluation or deployment-style export.
        """
        logger.info(
            "Refitting model on combined train+val data | model_type=%s | rows=%d | features=%d",
            self.model_type,
            len(X_train_val),
            X_train_val.shape[1],
        )
        self.fit(X_train_val, y_train_val)