"""
preprocessing.py

Purpose
-------
Preprocess raw SCADA event files from multiple wind farms into a unified,
model-ready tabular dataset for wind turbine fault detection.

This module:
1. Reads farm-specific raw CSV files.
2. Selects only required source columns using the YAML feature map.
3. Maps farm-specific sensors into a standardized feature schema.
4. Normalizes units across farms.
5. Computes derived physics-based and temporal features.
6. Applies event-based state labels and a convenience binary target.
7. Validates expected features and label fields.
8. Supports memory-aware master dataset creation via PyArrow.

Labeling Design
---------------
Canonical state labels preserve time-to-event structure:

    0 = normal
    1 = pre_48_72h
    2 = pre_24_48h
    3 = pre_0_24h
    4 = event_occurring
    5 = excluded_buffer

A convenience binary target column is also created:
- target = 1 for states {1, 2, 3, 4}
- target = 0 for state {0}
- target = NaN for state {5}

This allows the processed dataset to remain flexible for:
- binary classification
- alternate horizon definitions
- future sequence modeling
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from wtfd.utils.logging_utils import get_logger


PathLike = Union[str, Path]

logger = get_logger(__name__)


class WindFarmProcessor:
    """
    End-to-end preprocessing pipeline for harmonizing wind farm SCADA data.

    Parameters
    ----------
    config_path : str or Path, default="feature_map.yaml"
        Path to the YAML configuration file defining per-farm sensor mappings,
        units, and expected standardized feature names.
    buffer_before_hours : float, default=0.0
        Optional exclusion buffer immediately before the earliest pre-event
        window (i.e., before 72 hours to event start).
    buffer_after_hours : float, default=0.0
        Optional exclusion buffer after event_end to avoid recovery / edge
        effects being treated as normal.
    """

    STATE_LABELS = {
        "normal": 0,
        "pre_48_72h": 1,
        "pre_24_48h": 2,
        "pre_0_24h": 3,
        "event_occurring": 4,
        "excluded_buffer": 5,
    }

    STATE_NAMES = {v: k for k, v in STATE_LABELS.items()}

    def __init__(
        self,
        config_path: PathLike = "feature_map.yaml",
        buffer_before_hours: float = 0.0,
        buffer_after_hours: float = 0.0,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.standard_features = self.config["standard_features"]
        self.buffer_before_hours = buffer_before_hours
        self.buffer_after_hours = buffer_after_hours

        logger.info(
            "Initialized WindFarmProcessor with config_path=%s, buffer_before_hours=%s, buffer_after_hours=%s",
            config_path,
            buffer_before_hours,
            buffer_after_hours,
        )

    def _compute_derived_features(self, df: pd.DataFrame, farm_id: str) -> pd.DataFrame:
        """
        Compute physics-based and temporal features.
        """
        logger.debug("Computing derived features for farm_id=%s", farm_id)

        df = df.sort_values("time_stamp").copy()

        if "yaw_error" not in df.columns or df["yaw_error"].isnull().all():
            if "wind_direction" in df.columns and "nacelle_direction" in df.columns:
                error = np.abs(df["wind_direction"] - df["nacelle_direction"])
                df["yaw_error"] = np.where(error > 180, 360 - error, error)
                logger.debug("Backfilled yaw_error from directional channels for farm_id=%s", farm_id)

        if "gearbox_oil_temp" in df.columns and "amb_temp" in df.columns:
            df["temp_delta_gearbox"] = (df["gearbox_oil_temp"] - df["amb_temp"]).fillna(0)

        if "gearbox_oil_temp" in df.columns:
            df["temp_trend_24h"] = df["gearbox_oil_temp"].diff(144)

        if "gen_speed" in df.columns:
            df["rpm_volatility"] = df["gen_speed"].rolling(window=36).std()

        if "nacelle_temp" in df.columns and "hub_temp" in df.columns:
            df["temp_divergence"] = df["nacelle_temp"] - df["hub_temp"]

        if "wind_speed" in df.columns and "active_power" in df.columns:
            v3 = df["wind_speed"] ** 3
            df["power_efficiency"] = np.where(v3 > 0.1, df["active_power"] / v3, 0.0)

        if "vibration_raw" in df.columns:
            df["vibration_magnitude"] = df["vibration_raw"].abs()

        lag_targets = ["gearbox_oil_temp", "generator_temp", "active_power", "yaw_error"]
        windows = [6, 24]

        for col in [c for c in lag_targets if c in df.columns]:
            for w in windows:
                df[f"{col}_delta_{w}"] = df[col] - df[col].shift(w)
                df[f"{col}_volatility_{w}"] = df[col].rolling(window=w).std()

        logger.debug("Completed derived feature computation for farm_id=%s", farm_id)
        return df.fillna(0)

    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that expected mapped and derived features are present.
        """
        expected = set(self.standard_features) | set(self.config.get("derived_features", []))
        missing = expected - set(df.columns)

        if missing:
            logger.warning("Missing features detected: %s", sorted(missing))
        else:
            logger.debug("Feature validation passed with no missing features")

        return df

    def _validate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate canonical state labels and convenience binary target.
        """
        required_cols = {
            "event_id",
            "event_label",
            "event_start",
            "event_end",
            "state_label",
            "state_name",
            "is_excluded_buffer",
            "target",
        }
        missing = required_cols - set(df.columns)
        if missing:
            logger.warning("Missing label-related columns: %s", sorted(missing))
            return df

        valid_states = set(self.STATE_LABELS.values())
        observed_states = set(df["state_label"].dropna().unique())
        if not observed_states.issubset(valid_states):
            logger.warning("Unexpected state_label values found: %s", sorted(observed_states - valid_states))

        non_buffer_target = df.loc[~df["is_excluded_buffer"], "target"].dropna().unique()
        if not set(non_buffer_target).issubset({0, 1}):
            logger.warning("Unexpected target values found outside buffer rows: %s", sorted(set(non_buffer_target)))

        return df

    def pipeline(
        self,
        farm_id: str,
        csv_path: PathLike,
        event_path: Optional[PathLike] = None,
    ) -> pd.DataFrame:
        """
        Run the preprocessing pipeline for a single raw event CSV.
        """
        csv_path = Path(csv_path)
        event_path = Path(event_path) if event_path is not None else None

        logger.info("Starting pipeline for farm_id=%s, file=%s", farm_id, csv_path.name)

        needed_cols = self._get_required_columns(farm_id, csv_path)
        logger.debug("Identified %d required columns for %s", len(needed_cols), csv_path.name)

        df = pd.read_csv(csv_path, sep=";", usecols=needed_cols, low_memory=False)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])

        logger.debug("Loaded raw data for %s with shape=%s", csv_path.name, df.shape)

        df = self._map_sensors(df, farm_id)
        df = self._normalize_physics(df, farm_id)
        df = self._compute_derived_features(df, farm_id)
        df = self._validate_features(df)

        if event_path is not None and event_path.exists():
            df = self._label_by_event_id(df, event_path, csv_path.stem)
            df = self._validate_labels(df)
        else:
            logger.info("No event metadata provided or found for %s; skipping labeling", csv_path.name)

        logger.info("Completed pipeline for farm_id=%s, file=%s, output_shape=%s", farm_id, csv_path.name, df.shape)
        return df

    def process_all_turbines(self, raw_data_root: PathLike, output_dir: PathLike) -> None:
        """
        Process all turbine event CSVs for all configured farms and save each
        processed file as Parquet.
        """
        raw_root = Path(raw_data_root)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        logger.info("Beginning batch preprocessing from %s to %s", raw_root, out_root)

        for farm_id in self.config["farms"].keys():
            farm_path = raw_root / f"Wind Farm {farm_id}"
            csv_folder = farm_path / "datasets"
            event_path = farm_path / "event_info.csv"

            if not csv_folder.exists():
                logger.warning("Skipping farm %s because dataset folder does not exist: %s", farm_id, csv_folder)
                continue

            csv_files = list(csv_folder.glob("*.csv"))
            logger.info("Found %d raw files for farm %s", len(csv_files), farm_id)

            for csv_file in tqdm(csv_files, desc=f"Farm {farm_id}"):
                processed_df = self.pipeline(farm_id, csv_file, event_path)

                if processed_df is not None:
                    file_name = f"{farm_id}_event_{csv_file.stem}.parquet"
                    output_path = out_root / file_name
                    processed_df.to_parquet(output_path, index=False)
                    logger.info("Saved processed file: %s", output_path)

                    del processed_df
                    gc.collect()

        logger.info("Completed batch preprocessing for all configured farms")

    def _get_required_columns(self, farm_id: str, csv_path: PathLike) -> list[str]:
        """
        Identify the minimal set of raw columns required for a given farm file.
        """
        farm_cfg = self.config["farms"][farm_id]
        actual_cols = pd.read_csv(csv_path, sep=";", nrows=0).columns.tolist()

        requested_cols = ["time_stamp", "asset_id"]

        for sensor in farm_cfg["sensors"].values():
            if isinstance(sensor, list):
                requested_cols.extend(sensor)
            elif sensor:
                requested_cols.append(sensor)

        selected_cols = [c for c in set(requested_cols) if c in actual_cols]
        missing_requested = sorted(set(requested_cols) - set(actual_cols))

        if missing_requested:
            logger.debug(
                "Some requested columns were not present in %s: %s",
                Path(csv_path).name,
                missing_requested,
            )

        return selected_cols

    def _map_sensors(self, raw_df: pd.DataFrame, farm_id: str) -> pd.DataFrame:
        """
        Map farm-specific raw sensor channels into a unified standardized schema.
        """
        farm_cfg = self.config["farms"][farm_id]

        mapped = pd.DataFrame(index=raw_df.index)
        mapped["time_stamp"] = raw_df["time_stamp"]
        mapped["farm_id"] = farm_id
        mapped["asset_id"] = raw_df["asset_id"].astype(str) if "asset_id" in raw_df.columns else "Unknown"

        for feature in self.standard_features:
            sensor_ref = farm_cfg["sensors"].get(feature)

            if isinstance(sensor_ref, list):
                available = [s for s in sensor_ref if s in raw_df.columns]
                if available:
                    mapped[feature] = (
                        pd.to_numeric(raw_df[available].stack(), errors="coerce")
                        .unstack()
                        .mean(axis=1)
                    )
                else:
                    logger.debug("No available source columns found for feature '%s' in farm %s", feature, farm_id)
                    mapped[feature] = np.nan
            else:
                if sensor_ref in raw_df.columns:
                    mapped[feature] = pd.to_numeric(raw_df[sensor_ref], errors="coerce")
                else:
                    logger.debug(
                        "Source column '%s' not found for feature '%s' in farm %s",
                        sensor_ref,
                        feature,
                        farm_id,
                    )
                    mapped[feature] = np.nan

        return mapped

    def _normalize_physics(self, df: pd.DataFrame, farm_id: str) -> pd.DataFrame:
        """
        Normalize cross-farm physical units into a common representation.
        """
        units = self.config["farms"][farm_id]["units"]

        if units["power"] == "Wh":
            df["active_power"] *= 0.006
            logger.debug("Converted active_power from Wh-based units for farm %s", farm_id)

        if "gen_speed" in df.columns and units["speed"] == "rad/s":
            df["gen_speed"] *= (60 / (2 * np.pi))
            logger.debug("Converted gen_speed from rad/s to RPM for farm %s", farm_id)

        return df

    def _label_by_event_id(self, df: pd.DataFrame, event_path: PathLike, filename_id: str) -> pd.DataFrame:
        """
        Label rows using canonical event-state labels.

        Canonical states
        ----------------
        0 = normal
        1 = pre_48_72h
        2 = pre_24_48h
        3 = pre_0_24h
        4 = event_occurring
        5 = excluded_buffer

        The method also creates a convenience binary target:
        - target = 1 for states {1, 2, 3, 4}
        - target = 0 for state {0}
        - target = NaN for state {5}
        """
        events = pd.read_csv(event_path, sep=";")

        # ------------------------------------------------------------------
        # Initialize canonical event/state fields
        # ------------------------------------------------------------------
        df["event_id"] = pd.Series([pd.NA] * len(df), dtype="Int64")
        df["event_label"] = pd.NA
        df["event_start"] = pd.NaT
        df["event_end"] = pd.NaT

        df["state_label"] = self.STATE_LABELS["normal"]
        df["state_name"] = self.STATE_NAMES[self.STATE_LABELS["normal"]]
        df["is_excluded_buffer"] = False

        # Convenience binary target; may be overwritten below.
        df["target"] = 0.0

        try:
            target_event_id = int("".join(filter(str.isdigit, filename_id)))
        except Exception:
            logger.warning("Could not parse event_id from filename '%s'; keeping default normal labels", filename_id)
            return df

        df["event_id"] = target_event_id

        event_info = events[events["event_id"] == target_event_id]
        if event_info.empty:
            logger.warning("No matching event metadata found for event_id=%s", target_event_id)
            return df

        row = event_info.iloc[0]

        event_label = str(row["event_label"]).lower()
        event_start = pd.to_datetime(row["event_start"])
        event_end = pd.to_datetime(row["event_end"])

        df["event_label"] = row["event_label"]
        df["event_start"] = event_start
        df["event_end"] = event_end

        # If not anomaly, keep default normal labels but preserve metadata.
        if event_label != "anomaly":
            logger.debug("Event_id=%s is not labeled as anomaly; rows remain normal", target_event_id)
            return df

        # ------------------------------------------------------------------
        # Define canonical temporal regions relative to event_start
        # ------------------------------------------------------------------
        pre_72h_start = event_start - pd.Timedelta(hours=72)
        pre_48h_start = event_start - pd.Timedelta(hours=48)
        pre_24h_start = event_start - pd.Timedelta(hours=24)

        buffer_before_start = pre_72h_start - pd.Timedelta(hours=self.buffer_before_hours)
        buffer_after_end = event_end + pd.Timedelta(hours=self.buffer_after_hours)

        # Base masks
        mask_event = (df["time_stamp"] >= event_start) & (df["time_stamp"] <= event_end)
        mask_pre_0_24 = (df["time_stamp"] >= pre_24h_start) & (df["time_stamp"] < event_start)
        mask_pre_24_48 = (df["time_stamp"] >= pre_48h_start) & (df["time_stamp"] < pre_24h_start)
        mask_pre_48_72 = (df["time_stamp"] >= pre_72h_start) & (df["time_stamp"] < pre_48h_start)

        # Optional exclusion buffers
        mask_buffer_before = pd.Series(False, index=df.index)
        if self.buffer_before_hours > 0:
            mask_buffer_before = (
                (df["time_stamp"] >= buffer_before_start)
                & (df["time_stamp"] < pre_72h_start)
            )

        mask_buffer_after = pd.Series(False, index=df.index)
        if self.buffer_after_hours > 0:
            mask_buffer_after = (
                (df["time_stamp"] > event_end)
                & (df["time_stamp"] <= buffer_after_end)
            )

        mask_buffer = mask_buffer_before | mask_buffer_after

        # ------------------------------------------------------------------
        # Apply state labels with clear precedence
        # ------------------------------------------------------------------
        df.loc[mask_pre_48_72, "state_label"] = self.STATE_LABELS["pre_48_72h"]
        df.loc[mask_pre_24_48, "state_label"] = self.STATE_LABELS["pre_24_48h"]
        df.loc[mask_pre_0_24, "state_label"] = self.STATE_LABELS["pre_0_24h"]
        df.loc[mask_event, "state_label"] = self.STATE_LABELS["event_occurring"]

        # Buffer should only override rows that are otherwise still normal.
        df.loc[mask_buffer & (df["state_label"] == self.STATE_LABELS["normal"]), "state_label"] = (
            self.STATE_LABELS["excluded_buffer"]
        )

        df["state_name"] = df["state_label"].map(self.STATE_NAMES)
        df["is_excluded_buffer"] = df["state_label"] == self.STATE_LABELS["excluded_buffer"]

        # ------------------------------------------------------------------
        # Convenience binary target
        # ------------------------------------------------------------------
        positive_states = {
            self.STATE_LABELS["pre_48_72h"],
            self.STATE_LABELS["pre_24_48h"],
            self.STATE_LABELS["pre_0_24h"],
            self.STATE_LABELS["event_occurring"],
        }

        df["target"] = np.where(
            df["state_label"] == self.STATE_LABELS["excluded_buffer"],
            np.nan,
            np.where(df["state_label"].isin(positive_states), 1.0, 0.0),
        )

        logger.debug(
            "Applied canonical state labels for event_id=%s | counts=%s",
            target_event_id,
            df["state_name"].value_counts(dropna=False).to_dict(),
        )

        return df

    def load_processed_events(self, processed_dir: PathLike) -> pd.DataFrame:
        """
        Load and concatenate all processed event-level Parquet files from a directory.

        Warning
        -------
        This method loads all event parquet files into memory and is intended
        primarily for smaller debugging workflows.
        """
        processed_path = Path(processed_dir)

        parquet_files = [
            f for f in sorted(processed_path.glob("*.parquet"))
            if f.name != "master_dataset.parquet"
        ]

        logger.info(
            "Loading %d processed event files from %s",
            len(parquet_files),
            processed_path,
        )

        if not parquet_files:
            raise FileNotFoundError(
                f"No event-level parquet files found in {processed_path}"
            )

        logger.warning(
            "load_processed_events() loads all event parquet files into memory. "
            "Use create_master_dataset() for larger datasets."
        )

        dfs = []
        for f in parquet_files:
            df = pd.read_parquet(f)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True).reset_index(drop=True)

        del dfs
        gc.collect()

        logger.info(
            "Loaded consolidated event dataset with shape=%s",
            combined.shape,
        )
        return combined

    def create_master_dataset(
        self,
        processed_dir: PathLike,
        output_path: PathLike,
        batch_size: int = 65536,
    ) -> Path:
        """
        Create and save a consolidated master dataset parquet file from all
        processed event-level parquet files using a memory-aware PyArrow stream.
        """
        processed_path = Path(processed_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parquet_files = [
            f
            for f in sorted(processed_path.glob("*.parquet"))
            if f.name != "master_dataset.parquet"
        ]

        if not parquet_files:
            raise FileNotFoundError(
                f"No event-level parquet files found in {processed_path}"
            )

        logger.info(
            "Creating master dataset from %d event parquet files using PyArrow dataset",
            len(parquet_files),
        )

        dataset = ds.dataset([str(f) for f in parquet_files], format="parquet")

        writer = None
        total_rows = 0
        total_batches = 0

        try:
            scanner = dataset.scanner(batch_size=batch_size)

            for record_batch in scanner.to_batches():
                if writer is None:
                    writer = pq.ParquetWriter(output_path, record_batch.schema)
                    logger.info("Initialized ParquetWriter for %s", output_path)

                table = pa.Table.from_batches([record_batch])
                writer.write_table(table)

                total_rows += record_batch.num_rows
                total_batches += 1

                del table, record_batch
                gc.collect()

            if writer is None:
                raise ValueError(
                    "Dataset scan produced no record batches; master dataset was not created."
                )

        finally:
            if writer is not None:
                writer.close()

        logger.info(
            "Saved master dataset to %s | rows=%d | batches=%d",
            output_path,
            total_rows,
            total_batches,
        )

        gc.collect()
        return output_path

    def get_parquet_row_count(self, parquet_path: PathLike) -> int:
        """
        Return the total number of rows in a parquet file using PyArrow metadata.
        """
        parquet_path = Path(parquet_path)
        parquet_file = pq.ParquetFile(parquet_path)
        return parquet_file.metadata.num_rows