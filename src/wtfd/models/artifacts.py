"""
artifacts.py

Purpose
-------
Manage persistence of modeling run artifacts for the WTDF project.

This module provides standardized helpers for saving and loading the outputs of
modeling experiments, including:
- run output directories
- tabular artifacts (CSV)
- JSON metadata / metric payloads

Design Notes
------------
- Keep persistence logic reusable and lightweight.
- Use clear, explicit file formats.
- Log all major save/load operations.
- Avoid overengineering until more artifact types are truly needed.

Typical usage
-------------
>>> run_dir = ensure_run_output_dir(Path("outputs/modeling"), "pre_24h")
>>> save_dataframe_artifact(summary_df, run_dir / "model_comparison_summary.csv")
>>> save_json_artifact({"experiment": "pre_24h"}, run_dir / "run_metadata.json")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Directory / run management
# ---------------------------------------------------------------------
def ensure_directory(path: Path) -> Path:
    """
    Ensure that a directory exists.

    Parameters
    ----------
    path : pathlib.Path
        Directory path to create if it does not already exist.

    Returns
    -------
    pathlib.Path
        The same path, guaranteed to exist as a directory.
    """
    path.mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured directory exists: %s", path)
    return path


def generate_run_id(prefix: Optional[str] = None) -> str:
    """
    Generate a timestamp-based run identifier.

    Parameters
    ----------
    prefix : str or None, default=None
        Optional prefix to prepend to the timestamp.

    Returns
    -------
    str
        Run identifier string.

    Examples
    --------
    >>> generate_run_id()
    '20260321_143015'
    >>> generate_run_id("pre_24h")
    'pre_24h_20260321_143015'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{prefix}_{timestamp}" if prefix else timestamp
    logger.debug("Generated run_id=%s", run_id)
    return run_id


def ensure_run_output_dir(
    base_output_dir: Path,
    experiment_name: str,
    run_id: Optional[str] = None,
) -> Path:
    """
    Create and return a standardized output directory for a modeling run.

    Parameters
    ----------
    base_output_dir : pathlib.Path
        Root directory for modeling outputs.
    experiment_name : str
        Experiment identifier, such as 'pre_24h'.
    run_id : str or None, default=None
        Optional run identifier. If not provided, one is generated
        automatically.

    Returns
    -------
    pathlib.Path
        Path to the created run output directory.

    Directory structure
    -------------------
    The created structure is:

        <base_output_dir>/
            <experiment_name>/
                <run_id>/

    Examples
    --------
    >>> ensure_run_output_dir(Path("outputs/modeling"), "pre_24h")
    PosixPath('outputs/modeling/pre_24h/pre_24h_20260321_143015')
    """
    run_id = run_id or generate_run_id(prefix=experiment_name)
    run_dir = base_output_dir / experiment_name / run_id
    ensure_directory(run_dir)

    logger.info(
        "Created modeling run output directory: %s",
        run_dir,
    )
    return run_dir


# ---------------------------------------------------------------------
# Tabular artifact persistence (CSV)
# ---------------------------------------------------------------------
def save_dataframe_artifact(
    df: pd.DataFrame,
    output_path: Path,
    index: bool = False,
) -> Path:
    """
    Save a pandas DataFrame as a CSV artifact.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save.
    output_path : pathlib.Path
        Destination file path. Expected to end in '.csv'.
    index : bool, default=False
        Whether to include the DataFrame index in the saved file.

    Returns
    -------
    pathlib.Path
        Path to the saved artifact.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error(
            "save_dataframe_artifact expected DataFrame, got %s",
            type(df),
        )
        raise TypeError("df must be a pandas DataFrame")

    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=index)

    logger.info(
        "Saved DataFrame artifact to %s | shape=%s | index=%s",
        output_path,
        df.shape,
        index,
    )
    return output_path


def load_dataframe_artifact(
    input_path: Path,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV artifact into a pandas DataFrame.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the CSV file.
    **read_csv_kwargs : Any
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    """
    if not input_path.exists():
        logger.error("CSV artifact not found: %s", input_path)
        raise FileNotFoundError(f"CSV artifact not found: {input_path}")

    df = pd.read_csv(input_path, **read_csv_kwargs)

    logger.info(
        "Loaded DataFrame artifact from %s | shape=%s",
        input_path,
        df.shape,
    )
    return df


# ---------------------------------------------------------------------
# JSON / metadata persistence
# ---------------------------------------------------------------------
def save_json_artifact(
    payload: dict[str, Any],
    output_path: Path,
    indent: int = 2,
) -> Path:
    """
    Save a dictionary payload as a JSON artifact.

    Parameters
    ----------
    payload : dict[str, Any]
        Dictionary to serialize to JSON.
    output_path : pathlib.Path
        Destination JSON file path.
    indent : int, default=2
        Indentation level for pretty-printed JSON.

    Returns
    -------
    pathlib.Path
        Path to the saved JSON artifact.

    Raises
    ------
    TypeError
        If `payload` is not a dictionary.
    """
    if not isinstance(payload, dict):
        logger.error(
            "save_json_artifact expected dict payload, got %s",
            type(payload),
        )
        raise TypeError("payload must be a dictionary")

    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, default=_json_default_serializer)

    logger.info(
        "Saved JSON artifact to %s | keys=%s",
        output_path,
        sorted(payload.keys()),
    )
    return output_path


def load_json_artifact(input_path: Path) -> dict[str, Any]:
    """
    Load a JSON artifact into a Python dictionary.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the JSON file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    TypeError
        If the loaded JSON content is not a dictionary.
    """
    if not input_path.exists():
        logger.error("JSON artifact not found: %s", input_path)
        raise FileNotFoundError(f"JSON artifact not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        logger.error(
            "Expected JSON object/dict in %s, got %s",
            input_path,
            type(payload),
        )
        raise TypeError(f"Expected dict in JSON artifact, got {type(payload)}")

    logger.info(
        "Loaded JSON artifact from %s | keys=%s",
        input_path,
        sorted(payload.keys()),
    )
    return payload


# ---------------------------------------------------------------------
# Modeling-specific convenience wrappers
# ---------------------------------------------------------------------
def save_feature_importance(
    feature_importance_df: pd.DataFrame,
    model_dir: Path,
    filename: str = "feature_importance.csv",
) -> Path:
    """
    Save feature importance results for a model.

    Parameters
    ----------
    feature_importance_df : pandas.DataFrame
        Feature importance DataFrame.
    model_dir : pathlib.Path
        Directory for model-specific artifacts.
    filename : str, default='feature_importance.csv'
        Output filename.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    output_path = model_dir / filename
    logger.debug("Saving feature importance artifact to %s", output_path)
    return save_dataframe_artifact(feature_importance_df, output_path)


def save_threshold_sweep(
    threshold_df: pd.DataFrame,
    model_dir: Path,
    filename: str = "threshold_sweep.csv",
) -> Path:
    """
    Save threshold sweep results for a model.

    Parameters
    ----------
    threshold_df : pandas.DataFrame
        Threshold sweep DataFrame.
    model_dir : pathlib.Path
        Directory for model-specific artifacts.
    filename : str, default='threshold_sweep.csv'
        Output filename.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    output_path = model_dir / filename
    logger.debug("Saving threshold sweep artifact to %s", output_path)
    return save_dataframe_artifact(threshold_df, output_path)


def save_model_metrics(
    metrics_payload: dict[str, Any],
    model_dir: Path,
    filename: str = "summary_metrics.json",
) -> Path:
    """
    Save model metric payload as JSON.

    Parameters
    ----------
    metrics_payload : dict[str, Any]
        Metric dictionary to save.
    model_dir : pathlib.Path
        Directory for model-specific artifacts.
    filename : str, default='summary_metrics.json'
        Output filename.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    output_path = model_dir / filename
    logger.debug("Saving model metrics artifact to %s", output_path)
    return save_json_artifact(metrics_payload, output_path)


def save_run_metadata(
    metadata_payload: dict[str, Any],
    run_dir: Path,
    filename: str = "run_metadata.json",
) -> Path:
    """
    Save top-level run metadata as JSON.

    Parameters
    ----------
    metadata_payload : dict[str, Any]
        Run metadata dictionary.
    run_dir : pathlib.Path
        Run output directory.
    filename : str, default='run_metadata.json'
        Output filename.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    output_path = run_dir / filename
    logger.debug("Saving run metadata artifact to %s", output_path)
    return save_json_artifact(metadata_payload, output_path)


# ---------------------------------------------------------------------
# Internal serialization helpers
# ---------------------------------------------------------------------
def _json_default_serializer(obj: Any) -> Any:
    """
    Provide fallback serialization for JSON dumping.

    Parameters
    ----------
    obj : Any
        Object that `json.dump` cannot serialize directly.

    Returns
    -------
    Any
        JSON-serializable representation of the object.

    Notes
    -----
    This helper currently handles common pandas-related scalar types by
    converting them to strings when necessary. It can be extended later if
    the project starts storing more complex metadata objects.
    """
    try:
        # Handle pandas / NumPy scalar-like objects that support item().
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass

    # Fall back to string representation for unsupported objects.
    return str(obj)