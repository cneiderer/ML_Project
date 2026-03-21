"""
experiments.py

Purpose
-------
Define the canonical experiment configurations for the WTDF modeling workflow.

This module is the single source of truth for *what experiments are run*.
Each experiment represents a specific modeling problem definition, including:

- which canonical state labels should be treated as positive
- which registered models should be evaluated
- which metric should be optimized during threshold tuning
- which data split strategy should be used

Design Notes
------------
- Keep experiment definitions explicit and easy to inspect.
- Use descriptive experiment names and comments so prediction horizons are
  immediately understandable.
- Return copies of configurations so callers cannot accidentally mutate the
  canonical experiment registry.
- Keep logging lightweight and focused on retrieval and validation.

Typical Usage
-------------
>>> from wtfd.models.experiments import get_experiment_config
>>> exp = get_experiment_config("pre_24h")
>>> exp["positive_states"]
['pre_0_24h', 'event_occurring']
>>> exp["models"]
['logistic', 'rf', 'xgboost']
"""

from __future__ import annotations

from typing import Any

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Canonical experiment registry
# ---------------------------------------------------------------------
# Each entry defines a modeling problem configuration.
#
# Expected structure for each experiment entry:
# {
#     "description": <human-readable summary>,
#     "positive_states": [<canonical state labels treated as positive>],
#     "models": [<registered model names>],
#     "optimize_for": <threshold tuning metric>,
#     "split_method": <supported split strategy>,
# }
#
# Notes:
# - `positive_states` should match canonical state names produced by the
#   preprocessing / target-construction pipeline.
# - `models` should reference names defined in `model_registry.py`.
# - `optimize_for` should be supported by `metrics.py` / `trainer.py`.
# - `split_method` should be supported by `splitter.py`.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    # --------------------------------------------------
    # 24-hour prediction horizon
    # Positive rows include:
    # - pre_0_24h
    # - event_occurring
    # --------------------------------------------------
    "pre_24h": {
        "description": "Predict faults within the next 24 hours.",
        "positive_states": ["pre_0_24h", "event_occurring"],
        "models": ["logistic", "rf", "xgboost"],
        "optimize_for": "f1",
        "split_method": "event_chronological",
    },

    # --------------------------------------------------
    # 48-hour prediction horizon
    # Positive rows include:
    # - pre_24_48h
    # - pre_0_24h
    # - event_occurring
    # --------------------------------------------------
    "pre_48h": {
        "description": "Predict faults within the next 48 hours.",
        "positive_states": ["pre_24_48h", "pre_0_24h", "event_occurring"],
        "models": ["logistic", "rf", "xgboost"],
        "optimize_for": "f1",
        "split_method": "event_chronological",
    },

    # --------------------------------------------------
    # 72-hour prediction horizon
    # Positive rows include:
    # - pre_48_72h
    # - pre_24_48h
    # - pre_0_24h
    # - event_occurring
    # --------------------------------------------------
    "pre_72h": {
        "description": "Predict faults within the next 72 hours.",
        "positive_states": [
            "pre_48_72h",
            "pre_24_48h",
            "pre_0_24h",
            "event_occurring",
        ],
        "models": ["logistic", "rf", "xgboost"],
        "optimize_for": "f1",
        "split_method": "event_chronological",
    },
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_experiment_config(experiment_name: str) -> dict[str, Any]:
    """
    Retrieve a copy of a registered experiment configuration.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment configuration to retrieve. This must exist in
        `EXPERIMENTS`.

    Returns
    -------
    dict[str, Any]
        Copy of the experiment configuration dictionary. The returned
        dictionary contains:

        - "description": human-readable experiment summary
        - "positive_states": list of canonical positive state labels
        - "models": list of registered model names to run
        - "optimize_for": threshold tuning objective
        - "split_method": split strategy name

    Raises
    ------
    ValueError
        If `experiment_name` is not present in the registry.

    Notes
    -----
    A copy is returned so that callers can safely modify experiment fields
    locally for a single run without mutating the shared canonical registry.

    Examples
    --------
    >>> exp = get_experiment_config("pre_48h")
    >>> exp["positive_states"]
    ['pre_24_48h', 'pre_0_24h', 'event_occurring']
    >>> exp["optimize_for"]
    'f1'
    """
    if experiment_name not in EXPERIMENTS:
        available = sorted(EXPERIMENTS.keys())
        logger.error(
            "Unknown experiment '%s'. Available experiments: %s",
            experiment_name,
            available,
        )
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {available}"
        )

    # Return shallow copies of nested lists so callers can override local
    # values safely without mutating the shared registry.
    config = EXPERIMENTS[experiment_name].copy()
    config["positive_states"] = list(config.get("positive_states", []))
    config["models"] = list(config.get("models", []))

    logger.info(
        "Retrieved experiment config '%s' | models=%s | optimize_for=%s | split_method=%s",
        experiment_name,
        config["models"],
        config.get("optimize_for"),
        config.get("split_method"),
    )
    logger.debug(
        "Experiment '%s' details | positive_states=%s | description=%s",
        experiment_name,
        config["positive_states"],
        config.get("description"),
    )

    return config


def list_available_experiments() -> list[str]:
    """
    Return the sorted list of available experiment names.

    Returns
    -------
    list[str]
        Sorted list of registered experiment names.

    Notes
    -----
    This helper is useful for:

    - CLI argument validation
    - notebook inspection and display
    - experiment selection menus
    - validation of external configuration inputs

    Examples
    --------
    >>> list_available_experiments()
    ['pre_24h', 'pre_48h', 'pre_72h']
    """
    experiments = sorted(EXPERIMENTS.keys())
    logger.debug("Listing available experiments: %s", experiments)
    return experiments