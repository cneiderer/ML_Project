"""
splitter.py

Purpose
-------
Utilities for preparing WTDF datasets for modeling while reducing leakage risk.

This module supports:

1. Event-level time splitting
   Recommended primary strategy. Entire events are assigned to train,
   validation, or test based on event chronology.

2. Global row-level time splitting
   Rows are sorted by timestamp and split into contiguous time blocks.

3. Grouped time splitting by turbine
   Each turbine's history is split chronologically into train, validation,
   and test segments, then recombined.

4. Group-aware train/test splitting by turbine
   Entire turbines are assigned to train or test while stratifying by farm.

It also supports converting the canonical state labels produced during
preprocessing into experiment-specific binary classification targets for
24h / 48h / 72h leading-fault experiments.

Canonical state labels expected from preprocessing
--------------------------------------------------
0 = normal
1 = pre_48_72h
2 = pre_24_48h
3 = pre_0_24h
4 = event_occurring
5 = excluded_buffer

Design Notes
------------
- The canonical processed dataset should retain all rows and state labels.
- Experiment-specific binary targets are derived here, not hardcoded during
  preprocessing.
- A globally unique turbine group key is built from farm_id + asset_id.
- prepare_xy() only separates features and target; scaling belongs in the
  trainer/model pipeline, not in the splitter.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WindFarmSplitter:
    """
    Splitting helper for the WTDF modeling workflow.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds for GroupKFold cross-validation.
    random_state : int, default=42
        Random seed used in randomized split methods.
    """

    STATE_LABELS = {
        "normal": 0,
        "pre_48_72h": 1,
        "pre_24_48h": 2,
        "pre_0_24h": 3,
        "event_occurring": 4,
        "excluded_buffer": 5,
    }

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

        logger.info(
            "Initialized WindFarmSplitter with n_splits=%d, random_state=%d",
            n_splits,
            random_state,
        )

    @staticmethod
    def _build_group_id(df: pd.DataFrame) -> pd.Series:
        """
        Build a globally unique turbine grouping key from farm_id and asset_id.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset containing farm_id and asset_id columns.

        Returns
        -------
        pandas.Series
            Combined group key in the form '<farm_id>__<asset_id>'.
        """
        required_cols = {"farm_id", "asset_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for group key creation: {sorted(missing)}"
            )

        return df["farm_id"].astype(str) + "__" + df["asset_id"].astype(str)

    @staticmethod
    def _validate_split_sizes(
        train_size: float,
        val_size: float,
        test_size: float,
    ) -> None:
        """
        Validate that split sizes sum to 1.0.
        """
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train_size + val_size + test_size must equal 1.0, got {total:.6f}"
            )

    def create_binary_target_from_state(
        self,
        df: pd.DataFrame,
        horizon_hours: int,
        state_col: str = "state_label",
        include_event: bool = False,
        drop_buffer: bool = True,
        target_col: str = "target",
        keep_only_relevant_rows: bool = True,
    ) -> pd.DataFrame:
        """
        Derive an experiment-specific binary target from canonical state labels.

        Horizon mapping
        ---------------
        24h:
            positive = {pre_0_24h}
            dropped  = {pre_24_48h, pre_48_72h, excluded_buffer}
        48h:
            positive = {pre_0_24h, pre_24_48h}
            dropped  = {pre_48_72h, excluded_buffer}
        72h:
            positive = {pre_0_24h, pre_24_48h, pre_48_72h}
            dropped  = {excluded_buffer}

        Parameters
        ----------
        df : pandas.DataFrame
            Canonical processed dataset.
        horizon_hours : int
            Leading-fault horizon to model. Supported values are 24, 48, 72.
        state_col : str, default="state_label"
            Name of the canonical state label column.
        include_event : bool, default=False
            If True, event_occurring rows are treated as positive.
            If False, event_occurring rows are dropped from the binary dataset.
        drop_buffer : bool, default=True
            If True, excluded_buffer rows are removed.
        target_col : str, default="target"
            Name of the derived binary target column.
        keep_only_relevant_rows : bool, default=True
            If True, rows outside the chosen horizon are dropped rather than
            being relabeled as negative. This is the recommended setting.

        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame with a derived binary target column.

        Raises
        ------
        ValueError
            If the required state column is missing or the horizon is unsupported.
        """
        if state_col not in df.columns:
            raise ValueError(f"Missing required state label column: {state_col}")

        if horizon_hours not in {24, 48, 72}:
            raise ValueError(
                f"horizon_hours must be one of {{24, 48, 72}}, got {horizon_hours}"
            )

        logger.info(
            "Creating binary target from canonical states | horizon=%dh | include_event=%s | drop_buffer=%s | keep_only_relevant_rows=%s",
            horizon_hours,
            include_event,
            drop_buffer,
            keep_only_relevant_rows,
        )

        normal = self.STATE_LABELS["normal"]
        pre_48_72 = self.STATE_LABELS["pre_48_72h"]
        pre_24_48 = self.STATE_LABELS["pre_24_48h"]
        pre_0_24 = self.STATE_LABELS["pre_0_24h"]
        event = self.STATE_LABELS["event_occurring"]
        buffer_state = self.STATE_LABELS["excluded_buffer"]

        positive_by_horizon = {
            24: {pre_0_24},
            48: {pre_24_48, pre_0_24},
            72: {pre_48_72, pre_24_48, pre_0_24},
        }

        positive_states = set(positive_by_horizon[horizon_hours])
        if include_event:
            positive_states.add(event)

        keep_states = {normal} | positive_states

        # Outside-horizon pre-event states remain valid canonical states, but
        # they are not part of the current binary task and are best dropped.
        if not keep_only_relevant_rows:
            keep_states = {normal, pre_48_72, pre_24_48, pre_0_24}
            if include_event:
                keep_states.add(event)

        if drop_buffer:
            keep_states.discard(buffer_state)
        else:
            keep_states.add(buffer_state)

        df_out = df.copy()

        if keep_only_relevant_rows:
            df_out = df_out[df_out[state_col].isin(keep_states)].copy()
        elif drop_buffer:
            df_out = df_out[df_out[state_col] != buffer_state].copy()

        # If event rows are excluded, remove them explicitly.
        if not include_event:
            df_out = df_out[df_out[state_col] != event].copy()

        df_out[target_col] = df_out[state_col].isin(positive_states).astype(int)

        logger.info(
            "Derived binary target created | rows=%d | positives=%d | negatives=%d",
            len(df_out),
            int(df_out[target_col].sum()),
            int((df_out[target_col] == 0).sum()),
        )

        return df_out.reset_index(drop=True)

    def get_group_aware_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a group-aware train/test split stratified by farm.

        Entire turbines (farm_id + asset_id) are assigned to train or test.
        This is useful when the goal is cross-turbine generalization, but it
        is not time-aware.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        test_size : float, default=0.2
            Fraction of unique turbine groups allocated to the test set.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Train and test DataFrames.
        """
        logger.info(
            "Creating group-aware train/test split with test_size=%.3f on %d rows",
            test_size,
            len(df),
        )

        asset_meta = df[["farm_id", "asset_id"]].drop_duplicates().copy()
        asset_meta["group_id"] = (
            asset_meta["farm_id"].astype(str) + "__" + asset_meta["asset_id"].astype(str)
        )

        train_groups, test_groups = train_test_split(
            asset_meta["group_id"],
            test_size=test_size,
            stratify=asset_meta["farm_id"],
            random_state=self.random_state,
        )

        group_id = self._build_group_id(df)

        train_df = df[group_id.isin(train_groups)].copy()
        test_df = df[group_id.isin(test_groups)].copy()

        logger.info(
            "Group-aware split created | train_rows=%d | test_rows=%d",
            len(train_df),
            len(test_df),
        )

        return train_df, test_df

    def get_train_val_test_split_by_time(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        time_col: str = "time_stamp",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create a global chronological train/validation/test split.

        Rows are sorted by time, then partitioned into contiguous blocks.
        This is a simple time-aware baseline strategy.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        train_size : float, default=0.7
            Fraction of rows assigned to training.
        val_size : float, default=0.15
            Fraction of rows assigned to validation.
        test_size : float, default=0.15
            Fraction of rows assigned to testing.
        time_col : str, default="time_stamp"
            Name of the timestamp column used for chronological ordering.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            Train, validation, and test DataFrames.
        """
        if time_col not in df.columns:
            raise ValueError(f"Missing required time column: {time_col}")

        self._validate_split_sizes(train_size, val_size, test_size)

        logger.info(
            "Creating global time-based split with train=%.2f, val=%.2f, test=%.2f on %d rows",
            train_size,
            val_size,
            test_size,
            len(df),
        )

        df_sorted = df.sort_values(time_col).reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        logger.info(
            "Global time split created | train_rows=%d | val_rows=%d | test_rows=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return train_df, val_df, test_df

    def get_grouped_time_split_by_turbine(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        time_col: str = "time_stamp",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create a time-aware split within each turbine group, then recombine.

        For each unique turbine (farm_id + asset_id), rows are sorted by time
        and split into train/validation/test segments. The segments from all
        turbines are then concatenated back together.

        Important
        ---------
        This preserves chronology within each turbine, but the same turbine
        will appear in train, validation, and test across different time
        windows.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        train_size : float, default=0.7
            Fraction of rows per turbine assigned to training.
        val_size : float, default=0.15
            Fraction of rows per turbine assigned to validation.
        test_size : float, default=0.15
            Fraction of rows per turbine assigned to testing.
        time_col : str, default="time_stamp"
            Name of the timestamp column used for chronological ordering.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            Train, validation, and test DataFrames.
        """
        if time_col not in df.columns:
            raise ValueError(f"Missing required time column: {time_col}")

        self._validate_split_sizes(train_size, val_size, test_size)

        logger.info(
            "Creating grouped time split by turbine with train=%.2f, val=%.2f, test=%.2f on %d rows",
            train_size,
            val_size,
            test_size,
            len(df),
        )

        df_work = df.copy()
        df_work["_group_id"] = self._build_group_id(df_work)

        train_parts: list[pd.DataFrame] = []
        val_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []

        for group_name, group_df in df_work.groupby("_group_id", sort=False):
            group_df = group_df.sort_values(time_col)

            n = len(group_df)
            train_end = int(n * train_size)
            val_end = train_end + int(n * val_size)

            if train_end == 0 or val_end == train_end or val_end >= n:
                logger.debug(
                    "Small group encountered during grouped turbine split | group=%s | rows=%d",
                    group_name,
                    n,
                )

            train_parts.append(group_df.iloc[:train_end])
            val_parts.append(group_df.iloc[train_end:val_end])
            test_parts.append(group_df.iloc[val_end:])

        train_df = (
            pd.concat(train_parts, ignore_index=True)
            .drop(columns="_group_id")
            .sort_values(time_col)
            .reset_index(drop=True)
        )
        val_df = (
            pd.concat(val_parts, ignore_index=True)
            .drop(columns="_group_id")
            .sort_values(time_col)
            .reset_index(drop=True)
        )
        test_df = (
            pd.concat(test_parts, ignore_index=True)
            .drop(columns="_group_id")
            .sort_values(time_col)
            .reset_index(drop=True)
        )

        logger.info(
            "Grouped turbine time split created | train_rows=%d | val_rows=%d | test_rows=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return train_df, val_df, test_df

    def get_event_level_time_split(
        self,
        df: pd.DataFrame,
        event_id_col: str = "event_id",
        time_col: str = "time_stamp",
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create an event-level chronological train/validation/test split.

        Entire events are assigned to train, validation, or test based on
        event chronology. All rows belonging to the same event remain in the
        same split.

        This is the recommended primary strategy for the project.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        event_id_col : str, default="event_id"
            Column identifying event membership for each row.
        time_col : str, default="time_stamp"
            Timestamp column used to determine event chronology.
        train_size : float, default=0.7
            Fraction of events assigned to training.
        val_size : float, default=0.15
            Fraction of events assigned to validation.
        test_size : float, default=0.15
            Fraction of events assigned to testing.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            Train, validation, and test DataFrames.
        """
        required_cols = {event_id_col, time_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for event-level split: {sorted(missing)}"
            )

        self._validate_split_sizes(train_size, val_size, test_size)

        logger.info(
            "Creating event-level time split using event_id_col=%s with train=%.2f, val=%.2f, test=%.2f on %d rows",
            event_id_col,
            train_size,
            val_size,
            test_size,
            len(df),
        )

        event_meta = (
            df.groupby(event_id_col, as_index=False)
            .agg(
                event_start=(time_col, "min"),
                event_end=(time_col, "max"),
                n_rows=(event_id_col, "size"),
            )
            .sort_values("event_start")
            .reset_index(drop=True)
        )

        n_events = len(event_meta)
        train_end = int(n_events * train_size)
        val_end = train_end + int(n_events * val_size)

        train_event_ids = set(event_meta.iloc[:train_end][event_id_col])
        val_event_ids = set(event_meta.iloc[train_end:val_end][event_id_col])
        test_event_ids = set(event_meta.iloc[val_end:][event_id_col])

        train_df = df[df[event_id_col].isin(train_event_ids)].copy()
        val_df = df[df[event_id_col].isin(val_event_ids)].copy()
        test_df = df[df[event_id_col].isin(test_event_ids)].copy()

        logger.info(
            "Event-level time split created | train_events=%d | val_events=%d | test_events=%d | train_rows=%d | val_rows=%d | test_rows=%d",
            len(train_event_ids),
            len(val_event_ids),
            len(test_event_ids),
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return train_df, val_df, test_df

    def get_cv_iter(self, df: pd.DataFrame, target_col: str = "target") -> Iterator[tuple[pd.Index, pd.Index]]:
        """
        Return a GroupKFold iterator using unique turbine group IDs.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        target_col : str, default="target"
            Name of the target column used for grouped cross-validation.

        Returns
        -------
        iterator
            GroupKFold split iterator yielding train and validation indices.
        """
        if target_col not in df.columns:
            raise ValueError(f"Input DataFrame must contain target column: {target_col}")

        logger.info(
            "Creating GroupKFold iterator with n_splits=%d on %d rows using target_col=%s",
            self.n_splits,
            len(df),
            target_col,
        )

        gkf = GroupKFold(n_splits=self.n_splits)
        group_id = self._build_group_id(df)

        return gkf.split(df, df[target_col], groups=group_id)

    def prepare_xy(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        drop_cols: Optional[Sequence[str]] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target without scaling.

        Scaling should be handled inside the training pipeline rather than in
        the splitter to avoid duplicate transformations and leakage.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        target_col : str, default="target"
            Name of the target column.
        drop_cols : sequence of str or None, default=None
            Non-feature columns to exclude from X. If None, sensible defaults
            matching the canonical processed dataset are used.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            Feature matrix X and target vector y.
        """
        if target_col not in df.columns:
            raise ValueError(f"Input DataFrame is missing target column: {target_col}")

        if drop_cols is None:
            drop_cols = [
                "time_stamp",
                "asset_id",
                "farm_id",
                "event_id",
                "event_label",
                "event_start",
                "event_end",
                "state_label",
                "state_name",
                "is_excluded_buffer",
            ]

        existing_drops = [c for c in drop_cols if c in df.columns]

        X = df.drop(columns=[target_col] + existing_drops)
        y = df[target_col]

        logger.debug(
            "Prepared X/y split | rows=%d | features=%d | target_col=%s",
            len(X),
            X.shape[1],
            target_col,
        )

        return X, y