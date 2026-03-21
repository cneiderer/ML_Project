"""
run_preprocessing.py

Purpose
-------
Command-line entrypoint for the WTDF preprocessing pipeline.

This script:
1. Instantiates the WindFarmProcessor using the provided feature-map config.
2. Processes all raw turbine event CSV files into event-level Parquet files.
3. Optionally consolidates those event-level Parquet files into a single
   master dataset Parquet file.
4. Supports optional pre-event and post-event exclusion buffers used during
   canonical state labeling.

Typical usage
-------------
Process all raw files into event-level Parquet outputs only:
    python scripts/run_preprocessing.py \
        --config config/feature_map.yaml \
        --raw-data-root data/raw/zenodo_windfarm_data \
        --processed-dir data/processed

Process all raw files and also create a master dataset:
    python scripts/run_preprocessing.py \
        --config config/feature_map.yaml \
        --raw-data-root data/raw/zenodo_windfarm_data \
        --processed-dir data/processed \
        --create-master \
        --master-output data/processed/master_dataset.parquet

Run with exclusion buffers:
    python scripts/run_preprocessing.py \
        --config config/feature_map.yaml \
        --raw-data-root data/raw/zenodo_windfarm_data \
        --processed-dir data/processed \
        --buffer-before-hours 6 \
        --buffer-after-hours 6 \
        --create-master

Force overwrite of existing processed outputs:
    python scripts/run_preprocessing.py \
        --config config/feature_map.yaml \
        --raw-data-root data/raw/zenodo_windfarm_data \
        --processed-dir data/processed \
        --overwrite

Notes
-----
- This script is a thin operational wrapper around the preprocessing module.
  Core preprocessing logic should remain in src/wtfd/data/preprocessing.py.
- Useful for long-running jobs with tools like nohup, screen, or tmux.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from wtfd.data.preprocessing import WindFarmProcessor
from wtfd.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the preprocessing workflow.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the WTDF preprocessing pipeline and optionally build a master dataset."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/feature_map.yaml",
        help="Path to the YAML feature-mapping configuration file.",
    )
    parser.add_argument(
        "--raw-data-root",
        type=str,
        required=True,
        help="Root directory containing raw wind farm data folders.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory where processed event-level Parquet files will be written.",
    )
    parser.add_argument(
        "--buffer-before-hours",
        type=float,
        default=0.0,
        help="Optional exclusion buffer before the earliest pre-event window.",
    )
    parser.add_argument(
        "--buffer-after-hours",
        type=float,
        default=0.0,
        help="Optional exclusion buffer after event end.",
    )
    parser.add_argument(
        "--create-master",
        action="store_true",
        help="If provided, create a consolidated master_dataset.parquet after preprocessing.",
    )
    parser.add_argument(
        "--master-output",
        type=str,
        default=None,
        help=(
            "Output path for the consolidated master dataset. "
            "If omitted and --create-master is used, defaults to "
            "<processed-dir>/master_dataset.parquet."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing Parquet files in the processed output directory.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a log file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser.parse_args()


def main() -> int:
    """
    Execute the preprocessing pipeline from the command line.

    Returns
    -------
    int
        Exit status code. Returns 0 on success and 1 on failure.
    """
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = get_logger(__name__, log_file=args.log_file, level=log_level)

    try:
        config_path = Path(args.config)
        raw_data_root = Path(args.raw_data_root)
        processed_dir = Path(args.processed_dir)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if not raw_data_root.exists():
            raise FileNotFoundError(f"Raw data root not found: {raw_data_root}")

        processed_dir.mkdir(parents=True, exist_ok=True)

        existing_parquet_files = list(processed_dir.glob("*.parquet"))
        if existing_parquet_files:
            logger.warning(
                "Processed directory already contains %d parquet file(s): %s",
                len(existing_parquet_files),
                processed_dir,
            )
            if not args.overwrite:
                raise FileExistsError(
                    f"Processed directory {processed_dir} already contains "
                    f"{len(existing_parquet_files)} parquet file(s). "
                    "Use --overwrite to allow overwriting existing outputs."
                )

        logger.info("Starting WTDF preprocessing run")
        logger.info("Config path: %s", config_path)
        logger.info("Raw data root: %s", raw_data_root)
        logger.info("Processed output directory: %s", processed_dir)
        logger.info("Buffer before hours: %s", args.buffer_before_hours)
        logger.info("Buffer after hours: %s", args.buffer_after_hours)
        logger.info("Overwrite existing files: %s", args.overwrite)

        processor = WindFarmProcessor(
            config_path=config_path,
            buffer_before_hours=args.buffer_before_hours,
            buffer_after_hours=args.buffer_after_hours,
        )

        processor.process_all_turbines(
            raw_data_root=raw_data_root,
            output_dir=processed_dir,
        )

        if args.create_master:
            master_output = (
                Path(args.master_output)
                if args.master_output is not None
                else processed_dir / "master_dataset.parquet"
            )

            logger.info("Creating consolidated master dataset at: %s", master_output)

            master_dataset_path = processor.create_master_dataset(
                processed_dir=processed_dir,
                output_path=master_output,
            )

            try:
                row_count = processor.get_parquet_row_count(master_dataset_path)
                logger.info(
                    "Master dataset created at %s | rows=%d",
                    master_dataset_path,
                    row_count,
                )
            except Exception:
                logger.warning(
                    "Master dataset created at %s but row count could not be determined",
                    master_dataset_path,
                )

        logger.info("WTDF preprocessing completed successfully")
        return 0

    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())