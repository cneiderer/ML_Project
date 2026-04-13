"""
experiments.py

Purpose
-------
Define the canonical experiment configurations for the WTDF modeling workflow.

This module is the single source of truth for *what experiments are run*.
Each experiment represents a specific modeling problem definition, including:

- which canonical state labels should be treated as positive
- which canonical state labels should be treated as negative
- which canonical state labels should be dropped
- which registered models should be evaluated
- which metric should be optimized during threshold tuning
- which data split strategy should be used
"""

from __future__ import annotations

from typing import Any

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Canonical experiment registry
# ---------------------------------------------------------------------
# Expected structure for each experiment entry:
# {
#     "description": <human-readable summary>,
#     "positive_states": [<canonical state labels treated as positive>],
#     "negative_states": [<canonical state labels treated as negative>],
#     "drop_states": [<canonical state labels excluded from training/eval>],
#     "models": [<registered model names>],
#     "optimize_for": <threshold tuning metric>,
#     "split_method": <supported split strategy>,
# }
#
# Notes:
# - `positive_states`, `negative_states`, and `drop_states` should match
#   canonical state names produced by preprocessing.
# - These experiment definitions intentionally exclude `event_occurring`
#   from the predictive target to avoid training on the actual event window.
EXPERIMENTS: dict[str, dict[str, Any]] = {
    # --------------------------------------------------
    # 24-hour prediction horizon
    # Positive rows:
    # - pre_0_24h
    # Negative rows:
    # - normal
    # Dropped rows:
    # - pre_48_72h
    # - pre_24_48h
    # - event_occurring
    # - excluded_buffer
    # --------------------------------------------------
    "pre_24h": {
        "description": "Predict faults within the next 24 hours.",
        "positive_states": ["pre_0_24h"],
        "negative_states": ["normal"],
        "drop_states": ["pre_48_72h", "pre_24_48h", "event_occurring", "excluded_buffer"],
        "models": ["logistic", "rf", "xgboost"],
        "optimize_for": "f1",
        "split_method": "event_chronological",
    },

    # --------------------------------------------------
    # 48-hour prediction horizon
    # Positive rows:
    # - pre_24_48h
    # - pre_0_24h
    # Negative rows:
    # - normal
    # Dropped rows:
    # - pre_48_72h
    # - event_occurring
    # - excluded_buffer
    # --------------------------------------------------
    "pre_48h": {
        "description": "Predict faults within the next 48 hours.",
        "positive_states": ["pre_24_48h", "pre_0_24h"],
        "negative_states": ["normal"],
        "drop_states": ["pre_48_72h", "event_occurring", "excluded_buffer"],
        "models": ["logistic", "rf", "xgboost"],
        "optimize_for": "f1",
        "split_method": "event_chronological",
    },

    # --------------------------------------------------
    # 72-hour prediction horizon
    # Positive rows:
    # - pre_48_72h
    # - pre_24_48h
    # - pre_0_24h
    # Negative rows:
    # - normal
    # Dropped rows:
    # - event_occurring
    # - excluded_buffer
    # --------------------------------------------------
    "pre_72h": {
        "description": "Predict faults within the next 72 hours.",
        "positive_states": ["pre_48_72h", "pre_24_48h", "pre_0_24h"],
        "negative_states": ["normal"],
        "drop_states": ["event_occurring", "excluded_buffer"],
        "models": ["logistic", "rf", "xgboost"],
        "optimize_for": "f1",
        "split_method": "event_chronological",
    },
}


def get_experiment_config(experiment_name: str) -> dict[str, Any]:
    """
    Retrieve a copy of a registered experiment configuration.
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

    config = EXPERIMENTS[experiment_name].copy()
    config["positive_states"] = list(config.get("positive_states", []))
    config["negative_states"] = list(config.get("negative_states", []))
    config["drop_states"] = list(config.get("drop_states", []))
    config["models"] = list(config.get("models", []))

    logger.info(
        "Retrieved experiment config '%s' | models=%s | optimize_for=%s | split_method=%s",
        experiment_name,
        config["models"],
        config.get("optimize_for"),
        config.get("split_method"),
    )
    logger.debug(
        "Experiment '%s' details | positive_states=%s | negative_states=%s | drop_states=%s | description=%s",
        experiment_name,
        config["positive_states"],
        config["negative_states"],
        config["drop_states"],
        config.get("description"),
    )

    return config


def list_available_experiments() -> list[str]:
    """
    Return the sorted list of available experiment names.
    """
    experiments = sorted(EXPERIMENTS.keys())
    logger.debug("Listing available experiments: %s", experiments)
    return experiments