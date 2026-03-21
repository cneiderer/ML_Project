"""
feature_selector.py

Purpose
-------
Define and enforce the selection of valid modeling features from the processed
WTDF dataset.

This module is responsible for determining which columns from the processed
dataset should be used as model inputs (X). It explicitly avoids performing
any feature engineering, which belongs in `wtfd.data.preprocessing`.

Core Responsibilities
--------------------
- Exclude identifiers, timestamps, labels, and other non-feature columns.
- Prevent target leakage by default.
- Optionally restrict features to numeric columns only.
- Provide a consistent, reusable interface for feature selection.
- Support future feature subsets without requiring a separate feature_builder.

Design Notes
------------
- Separation of concerns: feature engineering happens upstream.
- Safety first: default exclusions prevent leakage and invalid inputs.
- Reproducibility: centralize feature selection logic in one place.
- Extensibility: support future feature subsets without refactoring.

Typical Usage
-------------
>>> from wtfd.models.feature_selector import build_feature_matrix
>>> X = build_feature_matrix(df)
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Default column exclusions
# ---------------------------------------------------------------------
# These columns should never be used as model features.
# This list is intentionally conservative to prevent leakage and misuse.
DEFAULT_EXCLUDED_COLUMNS: set[str] = {
    # Identifiers / metadata
    "asset_id",
    "turbine_id",
    "farm_id",
    "wind_farm",
    "event_id",
    "event_type",
    "event_code",

    # Time / ordering
    "time_stamp",
    "timestamp",
    "date",
    "datetime",

    # Targets / labels / states
    "target",
    "label",
    "state_label",
    "state_name",
    "binary_target",

    # Buffer / exclusion flags
    "is_excluded_buffer",
    "exclude_from_training",

    # Split / CV helpers (future-proofing)
    "split",
    "fold",
    "cv_fold",
    "set_type",
}


# ---------------------------------------------------------------------
# Optional named feature subsets (future-facing)
# ---------------------------------------------------------------------
# These allow experiments to reference "feature sets" without introducing
# a separate feature_builder module yet.
FEATURE_SUBSETS: dict[str, dict[str, object]] = {
    "all": {
        "description": "All valid modeling features after exclusions.",
        "include_prefixes": None,
        "exclude_prefixes": None,
    },
}


# ---------------------------------------------------------------------
# Helper utilities (internal)
# ---------------------------------------------------------------------
def _normalize_column_iterable(columns: Optional[Iterable[str]]) -> set[str]:
    """
    Normalize an iterable of column names into a set of strings.

    Parameters
    ----------
    columns : iterable of str or None
        Input column names.

    Returns
    -------
    set[str]
        Normalized column name set.
    """
    if columns is None:
        return set()

    normalized = {str(col) for col in columns}
    logger.debug("Normalized %d column names.", len(normalized))
    return normalized


def _validate_feature_subset_name(feature_subset: Optional[str]) -> None:
    """
    Validate that a feature subset exists.

    Parameters
    ----------
    feature_subset : str or None
        Feature subset name.

    Raises
    ------
    ValueError
        If subset is not defined.
    """
    if feature_subset is None:
        return

    if feature_subset not in FEATURE_SUBSETS:
        available = sorted(FEATURE_SUBSETS.keys())
        logger.error("Invalid feature_subset '%s'. Available: %s", feature_subset, available)
        raise ValueError(
            f"Unknown feature_subset='{feature_subset}'. "
            f"Available subsets: {available}"
        )


def _apply_feature_subset_rules(columns: list[str], feature_subset: Optional[str]) -> list[str]:
    """
    Apply include/exclude rules for a named feature subset.

    Parameters
    ----------
    columns : list[str]
        Candidate feature columns.
    feature_subset : str or None
        Feature subset name.

    Returns
    -------
    list[str]
        Filtered columns.
    """
    _validate_feature_subset_name(feature_subset)

    if feature_subset is None or feature_subset == "all":
        return columns

    config = FEATURE_SUBSETS[feature_subset]
    include_prefixes = config.get("include_prefixes")
    exclude_prefixes = config.get("exclude_prefixes")

    filtered = columns

    if include_prefixes:
        include_prefixes = tuple(include_prefixes)  # type: ignore
        filtered = [c for c in filtered if c.startswith(include_prefixes)]

    if exclude_prefixes:
        exclude_prefixes = tuple(exclude_prefixes)  # type: ignore
        filtered = [c for c in filtered if not c.startswith(exclude_prefixes)]

    logger.info(
        "Applied feature subset '%s': %d → %d columns",
        feature_subset,
        len(columns),
        len(filtered),
    )
    return filtered


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_feature_columns(
    df: pd.DataFrame,
    extra_excluded_columns: Optional[Iterable[str]] = None,
    include_columns: Optional[Iterable[str]] = None,
    numeric_only: bool = True,
    feature_subset: Optional[str] = None,
    require_non_empty: bool = True,
) -> list[str]:
    """
    Determine valid model feature columns from a processed DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    extra_excluded_columns : iterable of str, optional
        Additional columns to exclude.
    include_columns : iterable of str, optional
        Explicit allow-list (applied after exclusions).
    numeric_only : bool, default=True
        Restrict to numeric features.
    feature_subset : str, optional
        Named feature subset.
    require_non_empty : bool, default=True
        Raise error if no features remain.

    Returns
    -------
    list[str]
        Selected feature column names.

    Raises
    ------
    ValueError
        If no features remain.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Expected DataFrame, got %s", type(df))
        raise TypeError("df must be a pandas DataFrame")

    logger.info(
        "Selecting features | shape=%s | numeric_only=%s | subset=%s",
        df.shape,
        numeric_only,
        feature_subset,
    )

    excluded = set(DEFAULT_EXCLUDED_COLUMNS)
    excluded.update(_normalize_column_iterable(extra_excluded_columns))

    candidate_columns = [c for c in df.columns if c not in excluded]

    if include_columns is not None:
        include_set = _normalize_column_iterable(include_columns)
        candidate_columns = [c for c in candidate_columns if c in include_set]

    if numeric_only:
        numeric_cols = set(df.select_dtypes(include="number").columns)
        candidate_columns = [c for c in candidate_columns if c in numeric_cols]

    candidate_columns = _apply_feature_subset_rules(candidate_columns, feature_subset)

    if require_non_empty and not candidate_columns:
        logger.error("No features remain after selection.")
        raise ValueError("No feature columns remain after filtering.")

    logger.info("Selected %d feature columns.", len(candidate_columns))
    return candidate_columns


def build_feature_matrix(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Build feature matrix X from DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    **kwargs
        Passed to get_feature_columns.

    Returns
    -------
    pandas.DataFrame
        Feature matrix.
    """
    cols = get_feature_columns(df, **kwargs)
    X = df.loc[:, cols]

    logger.info("Feature matrix shape: %s", X.shape)
    return X.copy()


def summarize_feature_selection(df: pd.DataFrame, **kwargs) -> dict:
    """
    Summarize feature selection results.

    Returns
    -------
    dict
        Summary statistics and selected columns.
    """
    selected = get_feature_columns(df, require_non_empty=False, **kwargs)

    summary = {
        "n_total_columns": len(df.columns),
        "n_selected_features": len(selected),
        "selected_features": selected,
    }

    logger.info(
        "Feature summary: total=%d | selected=%d",
        summary["n_total_columns"],
        summary["n_selected_features"],
    )
    return summary


def validate_no_leakage_columns_in_features(
    feature_columns: Sequence[str],
    extra_forbidden_columns: Optional[Iterable[str]] = None,
) -> None:
    """
    Validate feature list does not contain forbidden columns.

    Raises
    ------
    ValueError
        If leakage columns are found.
    """
    forbidden = set(DEFAULT_EXCLUDED_COLUMNS)
    forbidden.update(_normalize_column_iterable(extra_forbidden_columns))

    offending = [c for c in feature_columns if c in forbidden]

    if offending:
        logger.error("Leakage columns detected: %s", offending)
        raise ValueError(f"Forbidden columns in features: {offending}")

    logger.debug("Feature validation passed (%d features).", len(feature_columns))