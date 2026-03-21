"""
run_modeling.py

Purpose
-------
Command-line entrypoint for running end-to-end WTDF modeling experiments.

This script orchestrates the modeling workflow by:
- loading a processed dataset
- selecting an experiment definition
- constructing an experiment-specific binary target
- splitting the data into train/validation/test sets
- selecting valid model features
- training and tuning one or more models
- evaluating final performance
- saving results and artifacts for later comparison

This script is intentionally thin relative to the reusable modeling package.
All core logic should live in reusable modules under `wtfd.models`, while this
script handles orchestration and command-line interaction.

Workflow Overview
-----------------
1. Parse command-line arguments.
2. Load experiment configuration from `wtfd.models.experiments`.
3. Load the processed dataset from disk.
4. Create the experiment-specific binary target from canonical state labels.
5. Split the dataset according to the experiment split strategy.
6. Select modeling features.
7. Train and tune each requested model.
8. Evaluate each model on the holdout set.
9. Save metrics, threshold sweeps, feature importance, and metadata.

Design Principles
-----------------
- Reproducible: experiment definitions and model configs are centralized.
- Transparent: logging captures each major step of the workflow.
- Reusable: this script delegates business logic to reusable package modules.
- Safe: feature leakage is prevented through centralized feature selection.

Typical Usage
-------------
Run the default models for the 24-hour horizon experiment:

    python scripts/run_modeling.py --data-path data/processed/master_dataset.parquet --experiment pre_24h

Run only logistic regression for the 48-hour horizon experiment:

    python scripts/run_modeling.py \
        --data-path data/processed/master_dataset.parquet \
        --experiment pre_48h \
        --models logistic

Specify an output directory and disable artifact saving:

    python scripts/run_modeling.py \
        --data-path data/processed/master_dataset.parquet \
        --experiment pre_72h \
        --output-dir outputs/modeling \
        --no-save
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from wtfd.models.artifacts import (
    ensure_run_output_dir,
    save_dataframe_artifact,
    save_json_artifact,
)
from wtfd.models.experiments import get_experiment_config, list_available_experiments
from wtfd.models.feature_selector import (
    get_feature_columns,
    validate_no_leakage_columns_in_features,
)
from wtfd.models.model_registry import get_model_config, list_available_models
from wtfd.models.splitter import WindFarmSplitter
from wtfd.models.trainer import WindFaultTrainer
from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the modeling runner.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run WTDF modeling experiments from the command line."
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the processed master dataset (Parquet or CSV).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list_available_experiments(),
        help="Experiment name defined in wtfd.models.experiments.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional override list of model names to run. "
            f"Available models: {', '.join(list_available_models())}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/modeling",
        help="Directory where modeling artifacts will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible model training and splits.",
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Restrict selected features to numeric columns only.",
    )
    parser.add_argument(
        "--feature-subset",
        type=str,
        default=None,
        help="Optional named feature subset from feature_selector.py.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="If provided, do not save artifacts to disk.",
    )

    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    """
    Load a processed dataset from disk.

    Supported file formats:
    - Parquet (.parquet)
    - CSV (.csv)

    Parameters
    ----------
    data_path : pathlib.Path
        Path to the processed dataset.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If the file extension is unsupported.
    """
    if not data_path.exists():
        logger.error("Dataset not found: %s", data_path)
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    logger.info("Loading dataset from: %s", data_path)

    suffix = data_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        logger.error("Unsupported dataset format: %s", suffix)
        raise ValueError(
            f"Unsupported dataset format '{suffix}'. Expected .parquet or .csv."
        )

    logger.info("Loaded dataset with shape=%s", df.shape)
    return df


def build_experiment_dataset(
    df: pd.DataFrame,
    experiment_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Build the experiment-specific dataset by creating a binary target from
    canonical state labels and dropping rows that should not be used for
    training.

    Parameters
    ----------
    df : pandas.DataFrame
        Full processed dataset.
    experiment_config : dict[str, Any]
        Experiment configuration dictionary from experiments.py.

    Returns
    -------
    pandas.DataFrame
        Experiment-ready dataset with binary target column populated.

    Notes
    -----
    This function assumes `WindFarmSplitter.create_binary_target_from_state`
    creates or overwrites the `target` column using the provided positive states.
    Rows with NaN targets are dropped because they are not eligible for model
    training/evaluation (for example, excluded buffer rows).
    """
    positive_states = experiment_config["positive_states"]

    logger.info(
        "Creating experiment-specific binary target using positive_states=%s",
        positive_states,
    )

    df_exp = WindFarmSplitter.create_binary_target_from_state(
        df=df.copy(),
        positive_states=positive_states,
    )

    pre_drop_rows = len(df_exp)
    df_exp = df_exp[df_exp["target"].notna()].copy()
    post_drop_rows = len(df_exp)

    logger.info(
        "Dropped %d rows with missing target values. Remaining rows=%d",
        pre_drop_rows - post_drop_rows,
        post_drop_rows,
    )

    return df_exp


def split_dataset(
    df: pd.DataFrame,
    experiment_config: dict[str, Any],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the experiment dataset into train/validation/test sets according to
    the configured split strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment-ready dataset.
    experiment_config : dict[str, Any]
        Experiment configuration dictionary.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Train, validation, and test DataFrames.

    Raises
    ------
    ValueError
        If the requested split method is unsupported.
    """
    split_method = experiment_config.get("split_method", "event_chronological")

    logger.info("Splitting dataset using split_method='%s'", split_method)

    splitter = WindFarmSplitter(random_state=random_state)

    if split_method == "event_chronological":
        train_df, val_df, test_df = splitter.event_chronological_split(df)
    elif split_method == "turbine_grouped":
        train_df, val_df, test_df = splitter.turbine_grouped_split(df)
    elif split_method == "random":
        train_df, val_df, test_df = splitter.random_split(df)
    else:
        logger.error("Unsupported split method: %s", split_method)
        raise ValueError(f"Unsupported split_method='{split_method}'")

    logger.info(
        "Split complete | train=%d | val=%d | test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    return train_df, val_df, test_df


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_only: bool,
    feature_subset: Optional[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare aligned feature matrices and target vectors for train/val/test sets.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training split.
    val_df : pandas.DataFrame
        Validation split.
    test_df : pandas.DataFrame
        Test split.
    numeric_only : bool
        Whether to keep only numeric columns.
    feature_subset : str or None
        Optional named feature subset.

    Returns
    -------
    tuple
        A tuple containing:
        - X_train
        - y_train
        - X_val
        - y_val
        - X_test
        - y_test
        - feature_columns

    Notes
    -----
    Feature columns are determined from the training set and then aligned across
    validation and test splits to prevent accidental schema drift.
    """
    logger.info(
        "Selecting feature columns | numeric_only=%s | feature_subset=%s",
        numeric_only,
        feature_subset,
    )

    feature_columns = get_feature_columns(
        train_df,
        numeric_only=numeric_only,
        feature_subset=feature_subset,
    )
    validate_no_leakage_columns_in_features(feature_columns)

    X_train = train_df.loc[:, feature_columns].copy()
    y_train = train_df["target"].astype(int).copy()

    X_val = val_df.loc[:, feature_columns].copy()
    y_val = val_df["target"].astype(int).copy()

    X_test = test_df.loc[:, feature_columns].copy()
    y_test = test_df["target"].astype(int).copy()

    logger.info(
        "Prepared feature matrices | n_features=%d | train_shape=%s | val_shape=%s | test_shape=%s",
        len(feature_columns),
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_columns


def run_single_model(
    model_name: str,
    model_config: dict[str, Any],
    experiment_name: str,
    experiment_config: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: list[str],
    random_state: int,
) -> dict[str, Any]:
    """
    Train, tune, and evaluate a single model for the given experiment.

    Parameters
    ----------
    model_name : str
        Registered model name.
    model_config : dict[str, Any]
        Model configuration dictionary from model_registry.py.
    experiment_name : str
        Experiment identifier.
    experiment_config : dict[str, Any]
        Experiment configuration dictionary.
    X_train, y_train, X_val, y_val, X_test, y_test :
        Modeling splits.
    feature_columns : list[str]
        Ordered feature column names used for training.
    random_state : int
        Random seed.

    Returns
    -------
    dict[str, Any]
        Result bundle containing:
        - summary metrics
        - detailed metrics
        - threshold tuning summary
        - feature importance DataFrame (if available)
        - metadata
    """
    logger.info(
        "Running model '%s' for experiment '%s'",
        model_name,
        experiment_name,
    )

    trainer = WindFaultTrainer(
        model_type=model_config["model_type"],
        params=model_config.get("params", {}),
        random_state=random_state,
    )

    trainer.fit_and_tune(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        optimize_for=experiment_config.get("optimize_for", "f1"),
    )

    test_summary = trainer.evaluate(X_test, y_test)
    test_detailed = trainer.evaluate_detailed(X_test, y_test)

    try:
        feature_importance_df = trainer.get_feature_importance(feature_columns)
        logger.info(
            "Computed feature importance for model '%s' with %d features",
            model_name,
            len(feature_columns),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Feature importance unavailable for model '%s': %s",
            model_name,
            exc,
        )
        feature_importance_df = None

    results = {
        "model_name": model_name,
        "experiment_name": experiment_name,
        "summary_metrics": test_summary,
        "detailed_metrics": test_detailed,
        "threshold_tuning_summary": trainer.threshold_tuning_summary,
        "feature_importance_df": feature_importance_df,
        "metadata": {
            "model_type": model_config["model_type"],
            "optimize_for": experiment_config.get("optimize_for", "f1"),
            "split_method": experiment_config.get("split_method"),
            "positive_states": experiment_config.get("positive_states"),
            "best_threshold": trainer.best_threshold,
            "n_features": len(feature_columns),
            "feature_columns": feature_columns,
        },
    }

    logger.info(
        "Completed model '%s' | F1=%.4f | Recall=%.4f | Precision=%.4f | ROC_AUC=%.4f | PR_AUC=%.4f",
        model_name,
        float(test_summary["f1"]),
        float(test_summary["recall"]),
        float(test_summary["precision"]),
        float(test_summary["roc_auc"]),
        float(test_summary["pr_auc"]),
    )

    return results


def flatten_summary_metrics(
    model_results: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Flatten model summary metrics into a comparison DataFrame.

    Parameters
    ----------
    model_results : list[dict[str, Any]]
        Per-model result bundles from `run_single_model`.

    Returns
    -------
    pandas.DataFrame
        One-row-per-model summary table.
    """
    rows: list[dict[str, Any]] = []

    for result in model_results:
        row = {
            "experiment_name": result["experiment_name"],
            "model_name": result["model_name"],
            **result["summary_metrics"],
            "best_threshold": result["metadata"]["best_threshold"],
            "n_features": result["metadata"]["n_features"],
            "split_method": result["metadata"]["split_method"],
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "precision"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    logger.info("Built comparison summary DataFrame with %d rows", len(summary_df))
    return summary_df


def save_run_outputs(
    run_dir: Path,
    experiment_name: str,
    experiment_config: dict[str, Any],
    model_results: list[dict[str, Any]],
    summary_df: pd.DataFrame,
) -> None:
    """
    Save run artifacts to disk.

    Parameters
    ----------
    run_dir : pathlib.Path
        Output directory for the run.
    experiment_name : str
        Experiment identifier.
    experiment_config : dict[str, Any]
        Experiment configuration.
    model_results : list[dict[str, Any]]
        Per-model result bundles.
    summary_df : pandas.DataFrame
        Comparison summary DataFrame.
    """
    logger.info("Saving artifacts to: %s", run_dir)

    save_dataframe_artifact(summary_df, run_dir / "model_comparison_summary.csv")

    metadata = {
        "experiment_name": experiment_name,
        "experiment_config": experiment_config,
        "n_models": len(model_results),
        "models": [result["model_name"] for result in model_results],
    }
    save_json_artifact(metadata, run_dir / "run_metadata.json")

    for result in model_results:
        model_name = result["model_name"]
        model_dir = run_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        save_json_artifact(
            result["summary_metrics"],
            model_dir / "summary_metrics.json",
        )

        # Remove large/non-JSON-friendly fields before serializing detailed metrics.
        detailed_metrics = {
            key: value
            for key, value in result["detailed_metrics"].items()
            if key not in {"confusion_matrix", "classification_report"}
        }
        save_json_artifact(
            detailed_metrics,
            model_dir / "detailed_metrics.json",
        )

        save_json_artifact(
            result["metadata"],
            model_dir / "model_metadata.json",
        )

        threshold_summary = result.get("threshold_tuning_summary")
        if threshold_summary is not None:
            threshold_table = threshold_summary.get("threshold_table")
            if threshold_table is not None:
                save_dataframe_artifact(
                    threshold_table,
                    model_dir / "threshold_sweep.csv",
                )

            threshold_json = {
                key: value
                for key, value in threshold_summary.items()
                if key != "threshold_table"
            }
            save_json_artifact(
                threshold_json,
                model_dir / "threshold_tuning_summary.json",
            )

        feature_importance_df = result.get("feature_importance_df")
        if feature_importance_df is not None:
            save_dataframe_artifact(
                feature_importance_df,
                model_dir / "feature_importance.csv",
            )

    logger.info("Artifact saving complete.")


def main() -> None:
    """
    Run the end-to-end modeling workflow from the command line.
    """
    args = parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    experiment_name = args.experiment

    logger.info("Starting modeling run for experiment='%s'", experiment_name)
    logger.info("Command-line arguments: %s", json.dumps(vars(args), indent=2))

    experiment_config = get_experiment_config(experiment_name)

    requested_models = args.models if args.models else experiment_config["models"]
    logger.info("Models selected for this run: %s", requested_models)

    df = load_dataset(data_path)
    df_exp = build_experiment_dataset(df, experiment_config)

    train_df, val_df, test_df = split_dataset(
        df=df_exp,
        experiment_config=experiment_config,
        random_state=args.random_state,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_feature_matrices(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        numeric_only=args.numeric_only,
        feature_subset=args.feature_subset,
    )

    model_results: list[dict[str, Any]] = []

    for model_name in requested_models:
        model_config = get_model_config(model_name)

        result = run_single_model(
            model_name=model_name,
            model_config=model_config,
            experiment_name=experiment_name,
            experiment_config=experiment_config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_columns=feature_columns,
            random_state=args.random_state,
        )
        model_results.append(result)

    summary_df = flatten_summary_metrics(model_results)

    logger.info("Top model comparison results:\n%s", summary_df.to_string(index=False))

    if not args.no_save:
        run_dir = ensure_run_output_dir(
            base_output_dir=output_dir,
            experiment_name=experiment_name,
        )
        save_run_outputs(
            run_dir=run_dir,
            experiment_name=experiment_name,
            experiment_config=experiment_config,
            model_results=model_results,
            summary_df=summary_df,
        )
    else:
        logger.info("Artifact saving disabled via --no-save")

    logger.info("Modeling run complete.")


if __name__ == "__main__":
    main()