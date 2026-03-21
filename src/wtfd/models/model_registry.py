"""
model_registry.py

Purpose
-------
Define the canonical set of supported model configurations for the WTDF
modeling workflow.

This module acts as the central registry for model definitions used throughout
the project. It specifies:
- which model names are supported
- the corresponding trainer-facing `model_type`
- default hyperparameter values
- brief human-readable descriptions

Design Notes
------------
- Keep model definitions explicit and easy to inspect.
- Return copies of configurations so callers cannot accidentally mutate the
  registry in place.
- Use lightweight logging for retrieval and validation events.
- Keep the registry trainer-friendly: each config should map directly to a
  `WindFaultTrainer(...)` initialization call.

Typical Usage
-------------
>>> from wtfd.models.model_registry import get_model_config
>>> cfg = get_model_config("rf")
>>> trainer = WindFaultTrainer(
...     model_type=cfg["model_type"],
...     params=cfg["params"],
... )
"""

from __future__ import annotations

from typing import Any

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Canonical model registry
# ---------------------------------------------------------------------
# Each entry in MODEL_REGISTRY defines a named model configuration that can be
# referenced consistently across notebooks, scripts, and experiment configs.
#
# Expected structure for each model entry:
# {
#     "model_type": <trainer-supported model type>,
#     "params": {<default hyperparameters>},
#     "description": <human-readable description>,
# }
#
# Notes:
# - `model_type` must be compatible with WindFaultTrainer.
# - `params` should contain only model-specific hyperparameters.
# - Random seed handling belongs in the trainer, not here.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "logistic": {
        "model_type": "logistic",
        "params": {
            "C": 1.0,
            "penalty": "l2",
        },
        "description": "Logistic Regression (interpretable linear baseline)",
    },
    "rf": {
        "model_type": "rf",
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 5,
        },
        "description": "Random Forest (robust non-linear tree baseline)",
    },
    "xgboost": {
        "model_type": "xgboost",
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "description": "XGBoost (gradient boosting performance candidate)",
    },
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_model_config(model_name: str) -> dict[str, Any]:
    """
    Retrieve a copy of a registered model configuration.

    Parameters
    ----------
    model_name : str
        Name of the model configuration to retrieve. This must exist in
        `MODEL_REGISTRY`.

    Returns
    -------
    dict[str, Any]
        Copy of the model configuration dictionary. The returned dictionary
        contains:
        - "model_type": trainer-supported model family name
        - "params": dictionary of default model hyperparameters
        - "description": human-readable description

    Raises
    ------
    ValueError
        If `model_name` is not present in the registry.

    Notes
    -----
    A copy is returned instead of the original registry entry so callers can
    safely modify fields such as `params` without mutating the shared global
    registry.

    Examples
    --------
    >>> cfg = get_model_config("logistic")
    >>> cfg["model_type"]
    'logistic'
    >>> cfg["params"]["C"]
    1.0
    """
    if model_name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        logger.error(
            "Unknown model '%s'. Available models: %s",
            model_name,
            available,
        )
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    # Return shallow copies so the caller can safely customize the returned
    # config for an experiment without modifying the registry itself.
    config = MODEL_REGISTRY[model_name].copy()
    config["params"] = config.get("params", {}).copy()

    logger.info(
        "Retrieved model config '%s' | model_type=%s",
        model_name,
        config["model_type"],
    )
    logger.debug(
        "Model config details for '%s': params=%s",
        model_name,
        config["params"],
    )

    return config


def list_available_models() -> list[str]:
    """
    Return the sorted list of available model names in the registry.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Sorted list of registered model names.

    Notes
    -----
    This helper is useful for:
    - CLI argument validation
    - notebook display / inspection
    - experiment configuration checks

    Examples
    --------
    >>> list_available_models()
    ['logistic', 'rf', 'xgboost']
    """
    models = sorted(MODEL_REGISTRY.keys())
    logger.debug("Listing available models: %s", models)
    return models