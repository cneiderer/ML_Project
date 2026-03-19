import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

class WindFarmProcessor:
    def __init__(self, config_path="feature_map.yaml", window_days=3):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.standard_features = self.config['standard_features']
        self.window_days = window_days

    def process_all_turbines(self, raw_data_root, output_dir):
        """Iterates through the project structure, processes each turbine, and saves as Parquet."""
        raw_root = Path(raw_data_root)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        for farm_id in self.config['farms'].keys():
            farm_path = raw_root / f"Wind Farm {farm_id}"
            csv_folder = farm_path / "datasets"
            event_path = farm_path / "event_info.csv"
            
            if not csv_folder.exists():
                print(f"Directory not found: {csv_folder}")
                continue

            # Get list of files to set the total for tqdm
            csv_files = list(csv_folder.glob("*.csv"))
            print(f"\nProcessing Farm {farm_id}:")
            for csv_file in tqdm(csv_files, desc=f"Farm {farm_id}", unit="turbine"):
                # pipeline now returns None if a file is empty or missing columns
                turbine_df = self.pipeline(farm_id, csv_file, event_path)
                
                if turbine_df is not None:
                    file_name = f"{farm_id}_{csv_file.stem}.parquet"
                    turbine_df.to_parquet(out_root / file_name, index=False)
                    del turbine_df

    def pipeline(self, farm_id, csv_path, event_path=None):
        """Coordinates the transformation steps for a single turbine CSV."""
        needed_cols = self._get_required_columns(farm_id, csv_path)
        
        # Read only what's available
        raw_df = pd.read_csv(csv_path, sep=';', usecols=needed_cols)
        raw_df['time_stamp'] = pd.to_datetime(raw_df['time_stamp'])
        
        # Core Transformation Steps
        df = self._map_sensors(raw_df, farm_id)
        df = self._normalize_physics(df, farm_id)
        df = self._compute_derived_features(df, farm_id, raw_df)
        
        if event_path and event_path.exists():
            df = self._label_failure_windows(df, farm_id, event_path, csv_path.stem)
        
        return df

    def _get_required_columns(self, farm_id, csv_path):
        """Identifies columns needed and verifies they exist in the file to avoid ValueErrors."""
        farm_cfg = self.config['farms'][farm_id]
        actual_cols = pd.read_csv(csv_path, sep=';', nrows=0).columns.tolist()
        
        requested_cols = ['time_stamp', 'asset_id']
        for sensor in farm_cfg['sensors'].values():
            if isinstance(sensor, list):
                requested_cols.extend(sensor)
            elif sensor:
                requested_cols.append(sensor)
        
        return [c for c in set(requested_cols) if c in actual_cols]

    def _map_sensors(self, raw_df, farm_id):
        """Maps site-specific sensor IDs to standardized feature names."""
        farm_cfg = self.config['farms'][farm_id]
        mapped = pd.DataFrame(index=raw_df.index)
        mapped['time_stamp'] = raw_df['time_stamp']
        mapped['asset_id'] = raw_df['asset_id']
        mapped['farm_id'] = farm_id

        for feature in self.standard_features:
            sensor_ref = farm_cfg['sensors'].get(feature)
            
            # Handle missing sensors or lists of sensors (like 3-phase temps)
            if sensor_ref is None or (isinstance(sensor_ref, str) and sensor_ref not in raw_df.columns):
                mapped[feature] = np.nan
                mapped[f"has_{feature}"] = 0
            elif isinstance(sensor_ref, list):
                # Site B Yaw handled in derived; others averaged here
                if feature == "yaw_error" and farm_id == "B":
                    mapped[feature] = np.nan 
                    mapped["has_yaw_error"] = 0
                else:
                    available_sensors = [s for s in sensor_ref if s in raw_df.columns]
                    mapped[feature] = raw_df[available_sensors].mean(axis=1) if available_sensors else np.nan
                mapped[f"has_{feature}"] = 1 if feature in mapped and not mapped[feature].isna().all() else 0
            else:
                mapped[feature] = raw_df[sensor_ref]
                mapped[f"has_{feature}"] = 1
        
        return mapped

    def _normalize_physics(self, df, farm_id):
        """Converts units (e.g., Wh to kW, rad/s to RPM)."""
        units = self.config['farms'][farm_id]['units']
        if units['power'] == 'Wh':
            df['active_power'] = df['active_power'] * 0.006
        if units['speed'] == 'rad/s':
            df['gen_speed'] = df['gen_speed'] * (60 / (2 * np.pi))
        return df

    def _compute_derived_features(self, df, farm_id, raw_df):
        """Calculates deltas, trends, and complex engineering features."""
        
        # Standardize and Sort
        df = df.sort_values(['asset_id', 'time_stamp'])
        
        # Physics-Based Deltas
        df['temp_delta_gearbox'] = df['gearbox_oil_temp'] - df['amb_temp']
        df['power_efficiency'] = df['active_power'] / (df['wind_speed']**3).replace(0, np.nan)

        # Vibration Magnitude (Euclidean Norm)
        # Specifically for Farm C which has x, y, z components
        vibe_cols = [c for c in raw_df.columns if 'vibration' in c.lower()]
        if len(vibe_cols) >= 2:
            df['vibration_magnitude'] = np.sqrt((raw_df[vibe_cols]**2).sum(axis=1))
        else:
            df['vibration_magnitude'] = np.nan

        # Yaw Error Calculation (Farm B specific)
        if farm_id == "B":
            sensor_ref = self.config['farms']['B']['sensors']['yaw_error']
            if isinstance(sensor_ref, list) and all(s in raw_df.columns for s in sensor_ref):
                # result is wrapped to [-180, 180]
                error = raw_df[sensor_ref[0]] - raw_df[sensor_ref[1]]
                df['yaw_error'] = (error + 180) % 360 - 180
        
        # Instantaneous Stress Index
        df['yaw_stress_index'] = df['yaw_error'].abs()

        # Temperature Divergence (Fleet-wide deviation)
        # How much hotter is this turbine than the average turbine at this timestamp?
        fleet_avg_temp = df.groupby('time_stamp')['gearbox_oil_temp'].transform('mean')
        df['temp_divergence'] = df['gearbox_oil_temp'] - fleet_avg_temp

        # Time-Series Trends
        # 24-hour delta (144 rows at 10-min intervals)
        df['temp_trend_24h'] = df.groupby('asset_id')['gearbox_oil_temp'].diff(144)

        # 6-hour volatility (36 intervals) - Captures immediate mechanical jitter
        df['rpm_volatility_6h'] = df.groupby('asset_id')['gen_speed'].transform(
            lambda x: x.rolling(36).std()
        )
        # 24-hour volatility (144 intervals) - Captures long-term bearing degradation
        df['rpm_volatility_24h'] = df.groupby('asset_id')['gen_speed'].transform(
            lambda x: x.rolling(144).std()
        )

        # 6-hour persistent yaw stress (36 intervals) -
        # Identifies if the turbine is stuck in a misaligned state
        df['yaw_stress_persistent_6h'] = df.groupby('asset_id')['yaw_stress_index'].transform(
            lambda x: x.rolling(36).mean()
        )

        # Drop initial NaN rows
        # Since our largest window is 24h (144 intervals), the first 144 rows 
        # of every turbine will have NaNs in the trend/volatility/persistence columns.
        # We group by asset and skip the first 144 rows of each
        # df = df.groupby('asset_id').apply(lambda x: x.iloc[144:], include_groups=False).reset_index()
        df = df.groupby('asset_id').tail(-144).reset_index(drop=True)
        
        return df

    def _label_failure_windows(self, df, farm_id, event_path, turbine_filename):
        """Labels failure precursor window based on inconsistent event log schemas."""
        events = pd.read_csv(event_path, sep=';')
        df['target'] = 0
        
        # Handle Inconsistent ID Columns (asset vs asset_id)
        id_col = 'asset' if 'asset' in events.columns else 'asset_id'
        
        # Handle Inconsistent Time Columns (event_start vs event_time)
        time_col = 'event_start' if 'event_start' in events.columns else 'event_time'
        
        # Extract turbine numeric ID from filename (e.g., '14.csv' -> '14')
        turbine_id_str = ''.join(filter(str.isdigit, turbine_filename))
        
        # Filter for the specific turbine and failure keywords
        critical = events[
            (events[id_col].astype(str) == turbine_id_str) & 
            (events['event_description'].str.contains('failure|damage|defect', case=False, na=False))
        ].copy()
        
        # Apply the window labeling
        for _, row in critical.iterrows():
            fail_time = pd.to_datetime(row[time_col])
            start_time = fail_time - pd.Timedelta(days=self.window_days)
            
            # Mask the rows for this turbine that fall in the window
            mask = (df['time_stamp'] >= start_time) & (df['time_stamp'] <= fail_time)
            df.loc[mask, 'target'] = 1
            
        return df

    def impute_missing_sensors(self, full_df):
        """Fills missing fleet sensors with global averages."""
        for feature in self.standard_features:
            if full_df[feature].isnull().any():
                full_df[feature] = full_df[feature].fillna(full_df[feature].mean())
        return full_df

    def load_processed_dataset(self, processed_dir):
        """
        Gathers all processed Parquet files from disk into a single 
        harmonized DataFrame and prunes useless constant feature flags.
        """
        processed_path = Path(processed_dir)
        parquet_files = list(processed_path.glob("*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {processed_dir}. "
                                    "Did you run process_all_turbines() first?")

        print(f"Combining {len(parquet_files)} turbines into master dataset...")
        
        # Load and concatenate
        dfs = [pd.read_parquet(f) for f in parquet_files]
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Global Imputation
        # Use data from other sensors to impute missing feature
        full_df = self.impute_missing_sensors(full_df)
        
        # Prune constant 'has_' feature flags
        # Columns that are 100% the same across the whole fleet provide no info
        cols_to_drop = [
            c for c in full_df.columns 
            if c.startswith('has_') and full_df[c].nunique() <= 1
        ]
        
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} constant indicators: {cols_to_drop}")
            full_df = full_df.drop(columns=cols_to_drop)
        
        # Final Sort
        full_df = full_df.sort_values(['farm_id', 'asset_id', 'time_stamp']).reset_index(drop=True)
        
        print(f"Success. Final dataset shape: {full_df.shape}")
        return full_df