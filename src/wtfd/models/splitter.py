"""
splitter.py

Purpose
-------
Provide dataset preparation and split utilities for the WTDF modeling workflow.

This module is responsible for two closely related tasks:

1. Converting canonical preprocessing state labels into experiment-specific
   binary classification targets for leading-fault prediction.

2. Splitting the processed dataset into train / validation / test partitions
   using strategies designed to reduce leakage risk and preserve temporal
   structure where appropriate.

Supported Split Strategies
--------------------------
This module supports several splitting strategies, each useful for different
diagnostic or experimental purposes:

1. Event-level chronological splitting
   Entire events are assigned to train, validation, or test based on event
   chronology. This is the recommended primary strategy for the project.

2. Global row-level chronological splitting
   Rows are sorted globally by timestamp and split into contiguous time blocks.

3. Grouped chronological splitting by turbine
   Each turbine's history is split chronologically into train, validation,
   and test segments, then all segments are recombined.

4. Group-aware train/test splitting by turbine
   Entire turbines are assigned to train and test while preserving farm-level
   grouping as much as possible.

5. GroupKFold cross-validation iterator
   Yields grouped folds based on unique turbine identities.

Canonical State Labels Expected from Preprocessing
-------------------------------------------------
The canonical processed dataset is expected to include state labels created
during preprocessing:

    0 = normal
    1 = pre_48_72h
    2 = pre_24_48h
    3 = pre_0_24h
    4 = event_occurring
    5 = excluded_buffer

The convenience `target` column produced during preprocessing may be useful
for sanity checks, but experiment-specific binary targets should be derived
here so that different prediction horizons remain configurable. 

Design Notes
------------
- Preserve the canonical processed dataset without hardcoding a single target
  definition into preprocessing.
- Keep split logic explicit and leakage-aware.
- Use globally unique turbine group IDs built from farm_id + asset_id.
- Keep scaling and other model transformations out of this module.
- Keep feature/target separation simple and reusable.

Typical Usage
-------------
>>> splitter = WindFarmSplitter(random_state=42)
>>> df_exp = splitter.create_binary_target_from_state(
...     df,
...     positive_states=["pre_0_24h", "event_occurring"],
... )
>>> train_df, val_df, test_df = splitter.get_event_level_time_split(df_exp)
>>> X_train, y_train = splitter.prepare_xy(train_df)

Notes
-----
- This module assumes the processed dataset already contains canonical columns
  such as `state_label`, `farm_id`, `asset_id`, and typically `time_stamp`.
- Rows in the excluded buffer are assigned NaN targets and should generally be
  dropped before modeling.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

from wtfd.utils.logging_utils import get_logger

logger = get_logger(__name__)


class WindFarmSplitter:
    """
    Splitting and target-construction helper for the WTDF modeling workflow.

    This class provides:
    - experiment-specific binary target creation from canonical state labels
    - several train/validation/test split strategies
    - grouped cross-validation utilities
    - simple feature/target separation for downstream model training

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds used for GroupKFold cross-validation.
    random_state : int, default=42
        Random seed used by randomized split methods.

    Attributes
    ----------
    n_splits : int
        Number of GroupKFold folds.
    random_state : int
        Random seed used by applicable split methods.
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

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        """
        Initialize the splitter.

        Parameters
        ----------
        n_splits : int, default=5
            Number of folds used for GroupKFold cross-validation.
        random_state : int, default=42
            Random seed used by randomized split methods.
        """
        self.n_splits = n_splits
        self.random_state = random_state

        logger.info(
            "Initialized WindFarmSplitter with n_splits=%d, random_state=%d",
            n_splits,
            random_state,
        )

    # ------------------------------------------------------------------
    # Internal validation / helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _build_group_id(df: pd.DataFrame) -> pd.Series:
        """
        Build a globally unique turbine grouping key from `farm_id` and
        `asset_id`.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset containing `farm_id` and `asset_id`.

        Returns
        -------
        pandas.Series
            Combined group key in the form '<farm_id>__<asset_id>'.

        Raises
        ------
        ValueError
            If either `farm_id` or `asset_id` is missing.

        Notes
        -----
        Using both farm and asset identifiers avoids accidental group collisions
        when asset IDs are only unique within a single farm.
        """
        required_cols = {"farm_id", "asset_id"}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(
                "Missing required columns for group key creation: %s",
                sorted(missing),
            )
            raise ValueError(
                f"Missing required columns for group key creation: {sorted(missing)}"
            )

        group_id = df["farm_id"].astype(str) + "__" + df["asset_id"].astype(str)
        logger.debug("Built group_id series for %d rows.", len(group_id))
        return group_id

    @staticmethod
    def _validate_split_sizes(
        train_size: float,
        val_size: float,
        test_size: float,
    ) -> None:
        """
        Validate that train/validation/test fractions sum to 1.0.

        Parameters
        ----------
        train_size : float
            Fraction assigned to training.
        val_size : float
            Fraction assigned to validation.
        test_size : float
            Fraction assigned to testing.

        Raises
        ------
        ValueError
            If any split size is negative or if the values do not sum to 1.0
            within a small numerical tolerance.
        """
        split_sizes = [train_size, val_size, test_size]

        if any(size < 0 for size in split_sizes):
            logger.error(
                "Split sizes must be non-negative. Received train=%.4f, val=%.4f, test=%.4f",
                train_size,
                val_size,
                test_size,
            )
            raise ValueError("Split sizes must be non-negative.")

        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-8:
            logger.error(
                "Split sizes must sum to 1.0. Received train=%.4f, val=%.4f, test=%.4f, total=%.8f",
                train_size,
                val_size,
                test_size,
                total,
            )
            raise ValueError(
                "Split sizes must sum to 1.0. "
                f"Received total={total:.8f} from "
                f"train={train_size}, val={val_size}, test={test_size}"
            )

        logger.debug(
            "Validated split sizes successfully: train=%.4f, val=%.4f, test=%.4f",
            train_size,
            val_size,
            test_size,
        )

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
        """
        Validate that required columns are present in a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset to validate.
        required_cols : sequence of str
            Column names that must be present.

        Raises
        ------
        ValueError
            If any required columns are missing.
        """
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.error("Missing required columns: %s", sorted(missing))
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------
    @classmethod
    def create_binary_target_from_state(
        cls,
        df: pd.DataFrame,
        positive_states: Sequence[str | int],
        state_col: str = "state_label",
        target_col: str = "target",
        buffer_state: str | int = "excluded_buffer",
    ) -> pd.DataFrame:
        """
        Create an experiment-specific binary target from canonical state labels.

        Parameters
        ----------
        df : pandas.DataFrame
            Input processed dataset containing canonical state labels.
        positive_states : sequence of str or int
            Canonical states to treat as positive for the experiment.
            Values may be provided as state names (for example,
            `'pre_0_24h'`, `'event_occurring'`) or as integer state labels.
        state_col : str, default='state_label'
            Column containing canonical state labels.
        target_col : str, default='target'
            Name of the binary target column to create or overwrite.
        buffer_state : str or int, default='excluded_buffer'
            Canonical state that should map to NaN target values rather than 0
            or 1.

        Returns
        -------
        pandas.DataFrame
            Copy of the input DataFrame with the experiment-specific target
            column created or overwritten.

        Raises
        ------
        ValueError
            If `state_col` is missing, if a requested state name is unknown,
            or if observed state labels are outside the canonical label set.

        Notes
        -----
        Mapping behavior is:

        - positive experiment states -> 1.0
        - normal / other non-buffer states -> 0.0
        - excluded buffer state -> NaN

        This preserves the ability to exclude buffer rows from downstream
        modeling while still allowing experiments to redefine what counts as a
        positive leading-fault window. 
        """
        if state_col not in df.columns:
            logger.error("Input DataFrame is missing required state column: %s", state_col)
            raise ValueError(f"Input DataFrame is missing required state column: {state_col}")

        valid_state_values = set(cls.STATE_LABELS.values())
        observed_state_values = set(pd.Series(df[state_col]).dropna().unique())

        if not observed_state_values.issubset(valid_state_values):
            invalid = sorted(observed_state_values - valid_state_values)
            logger.error("Unexpected canonical state values found: %s", invalid)
            raise ValueError(f"Unexpected canonical state values found: {invalid}")

        def _resolve_state(state: str | int) -> int:
            """Resolve a state name or integer state label to its canonical integer value."""
            if isinstance(state, int):
                if state not in cls.STATE_NAMES:
                    logger.error("Unknown integer state label requested: %s", state)
                    raise ValueError(f"Unknown integer state label: {state}")
                return state

            if state not in cls.STATE_LABELS:
                logger.error("Unknown state name requested: %s", state)
                raise ValueError(f"Unknown state name: {state}")

            return cls.STATE_LABELS[state]

        positive_state_values = {_resolve_state(state) for state in positive_states}
        buffer_state_value = _resolve_state(buffer_state)

        logger.info(
            "Creating experiment target from canonical states | positive_states=%s | buffer_state=%s",
            sorted(positive_state_values),
            buffer_state_value,
        )

        df_out = df.copy()

        def _map_state_to_target(state_value: int | float | pd._libs.missing.NAType) -> float:
            """
            Map a canonical state label to an experiment-specific binary target.

            Returns
            -------
            float
                1.0 for positive states, 0.0 for non-positive non-buffer states,
                and NaN for excluded buffer rows.
            """
            if pd.isna(state_value):
                return float("nan")

            state_int = int(state_value)

            if state_int == buffer_state_value:
                return float("nan")
            if state_int in positive_state_values:
                return 1.0
            return 0.0

        df_out[target_col] = df_out[state_col].map(_map_state_to_target)

        logger.info(
            "Created binary target column '%s' | positive_rows=%d | negative_rows=%d | missing_rows=%d",
            target_col,
            int((df_out[target_col] == 1).sum()),
            int((df_out[target_col] == 0).sum()),
            int(df_out[target_col].isna().sum()),
        )

        return df_out

    # ------------------------------------------------------------------
    # Split strategies
    # ------------------------------------------------------------------
    def get_global_time_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        time_col: str = "time_stamp",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create a global chronological train/validation/test split.

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
        time_col : str, default='time_stamp'
            Timestamp column used for chronological ordering.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            Train, validation, and test DataFrames.

        Raises
        ------
        ValueError
            If `time_col` is missing or split sizes are invalid.

        Notes
        -----
        This method preserves global chronology but does not guarantee that
        rows from the same event or turbine remain isolated within a single
        split. It is therefore mainly useful for diagnostics rather than as the
        primary project split strategy.
        """
        if time_col not in df.columns:
            logger.error("Missing required time column: %s", time_col)
            raise ValueError(f"Missing required time column: {time_col}")

        self._validate_split_sizes(train_size, val_size, test_size)

        logger.info(
            "Creating global time split with train=%.2f, val=%.2f, test=%.2f on %d rows",
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
        Create a chronological split within each turbine group, then recombine.

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
        time_col : str, default='time_stamp'
            Timestamp column used for chronological ordering within each group.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            Train, validation, and test DataFrames.

        Raises
        ------
        ValueError
            If `time_col` is missing or split sizes are invalid.

        Important
        ---------
        This preserves chronology *within each turbine*, but the same turbine
        can still appear in train, validation, and test across different time
        windows. That makes it useful for certain temporal diagnostics, but not
        necessarily as the strictest leakage-control strategy.
        """
        if time_col not in df.columns:
            logger.error("Missing required time column: %s", time_col)
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

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        event_id_col : str, default='event_id'
            Column identifying event membership for each row.
        time_col : str, default='time_stamp'
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

        Raises
        ------
        ValueError
            If required columns are missing or split sizes are invalid.

        Notes
        -----
        Entire events remain within the same split, which makes this the
        recommended primary strategy for the project because it reduces event
        leakage while preserving event chronology. 
        """
        required_cols = {event_id_col, time_col}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(
                "Missing required columns for event-level split: %s",
                sorted(missing),
            )
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

    def get_grouped_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_by_farm: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a group-aware train/test split using entire turbines as groups.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        test_size : float, default=0.2
            Fraction of turbine groups assigned to the test split.
        stratify_by_farm : bool, default=True
            If True, stratify turbine groups by farm_id when possible.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Train and test DataFrames.

        Raises
        ------
        ValueError
            If required grouping columns are missing.

        Notes
        -----
        This split keeps entire turbines together, which can be useful when you
        want stronger generalization tests across assets. However, because it
        does not explicitly create a validation split, it is more naturally
        suited to train/test studies or as a building block for custom split
        logic.
        """
        self._validate_required_columns(df, ["farm_id", "asset_id"])

        logger.info(
            "Creating grouped train/test split with test_size=%.2f on %d rows",
            test_size,
            len(df),
        )

        df_work = df.copy()
        df_work["_group_id"] = self._build_group_id(df_work)

        group_meta = (
            df_work.groupby("_group_id", as_index=False)
            .agg(farm_id=("farm_id", "first"))
        )

        stratify = group_meta["farm_id"] if stratify_by_farm and group_meta["farm_id"].nunique() > 1 else None

        train_groups, test_groups = train_test_split(
            group_meta["_group_id"],
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        train_groups = set(train_groups)
        test_groups = set(test_groups)

        train_df = df_work[df_work["_group_id"].isin(train_groups)].drop(columns="_group_id").copy()
        test_df = df_work[df_work["_group_id"].isin(test_groups)].drop(columns="_group_id").copy()

        logger.info(
            "Grouped train/test split created | train_groups=%d | test_groups=%d | train_rows=%d | test_rows=%d",
            len(train_groups),
            len(test_groups),
            len(train_df),
            len(test_df),
        )

        return train_df, test_df

    def get_random_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create a randomized row-level train/validation/test split.

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

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            Train, validation, and test DataFrames.

        Notes
        -----
        This method is generally not the preferred final evaluation strategy
        for a temporal event-detection problem, but it can still be useful for
        sanity checks, quick baselines, or debugging data flow.
        """
        self._validate_split_sizes(train_size, val_size, test_size)

        logger.info(
            "Creating random split with train=%.2f, val=%.2f, test=%.2f on %d rows",
            train_size,
            val_size,
            test_size,
            len(df),
        )

        train_df, temp_df = train_test_split(
            df,
            test_size=(1.0 - train_size),
            random_state=self.random_state,
            shuffle=True,
        )

        relative_test_size = test_size / (val_size + test_size)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info(
            "Random split created | train_rows=%d | val_rows=%d | test_rows=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Cross-validation utilities
    # ------------------------------------------------------------------
    def get_cv_iter(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
    ) -> Iterator[tuple[pd.Index, pd.Index]]:
        """
        Return a GroupKFold iterator using turbine-level group IDs.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        target_col : str, default='target'
            Target column used alongside grouping for cross-validation.

        Returns
        -------
        iterator
            GroupKFold iterator yielding train and validation indices.

        Raises
        ------
        ValueError
            If `target_col` is missing.

        Notes
        -----
        Grouping is based on the combined turbine identity formed from
        `farm_id` and `asset_id`, which prevents the same turbine from
        appearing in both training and validation within a single fold.
        """
        if target_col not in df.columns:
            logger.error("Input DataFrame must contain target column: %s", target_col)
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

    # ------------------------------------------------------------------
    # Feature / target separation
    # ------------------------------------------------------------------
    def prepare_xy(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        drop_cols: Optional[Sequence[str]] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target without applying scaling.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataset.
        target_col : str, default='target'
            Name of the target column.
        drop_cols : sequence of str or None, default=None
            Non-feature columns to exclude from X. If None, a conservative
            default set matching the canonical processed dataset is used.

        Returns
        -------
        tuple[pandas.DataFrame, pandas.Series]
            Feature matrix X and target vector y.

        Raises
        ------
        ValueError
            If the target column is missing.

        Notes
        -----
        This method intentionally performs only simple feature/target
        separation. Scaling and other model transformations should happen
        inside the trainer/model pipeline to avoid leakage and duplicated
        preprocessing logic. 
        """
        if target_col not in df.columns:
            logger.error("Input DataFrame is missing target column: %s", target_col)
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

        existing_drops = [col for col in drop_cols if col in df.columns]

        X = df.drop(columns=[target_col, *existing_drops], errors="ignore").copy()
        y = df[target_col].copy()

        logger.info(
            "Prepared X/y from DataFrame | rows=%d | n_features=%d | target_col=%s",
            len(df),
            X.shape[1],
            target_col,
        )
        logger.debug(
            "Dropped non-feature columns during prepare_xy: %s",
            [target_col, *existing_drops],
        )

        return X, y