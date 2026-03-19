import pandas as pd
import numpy as np
import yaml
import gc
from pathlib import Path
from tqdm import tqdm

class WindFarmProcessor:
    def __init__(self, config_path="feature_map.yaml", lead_in_days=0):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.standard_features = self.config['standard_features']
        self.lead_in_days = lead_in_days

    def process_all_turbines(self, raw_data_root, output_dir):
        raw_root = Path(raw_data_root)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        for farm_id in self.config['farms'].keys():
            farm_path = raw_root / f"Wind Farm {farm_id}"
            csv_folder = farm_path / "datasets"
            event_path = farm_path / "event_info.csv"
            
            if not csv_folder.exists():
                continue

            csv_files = list(csv_folder.glob("*.csv"))
            print(f"\nProcessing Farm {farm_id}:")
            for csv_file in tqdm(csv_files, desc=f"Farm {farm_id}"):
                processed_df = self.pipeline(farm_id, csv_file, event_path)
                
                if processed_df is not None:
                    # We save using farm + event_id to ensure unique filenames
                    file_name = f"{farm_id}_event_{csv_file.stem}.parquet"
                    processed_df.to_parquet(out_root / file_name, index=False)
                    del processed_df
                    gc.collect()

    def pipeline(self, farm_id, csv_path, event_path=None):
        needed_cols = self._get_required_columns(farm_id, csv_path)
        
        # 1. Load the data for this specific event
        df = pd.read_csv(csv_path, sep=';', usecols=needed_cols, low_memory=False)
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        
        # 2. Map and Normalize Sensors
        df = self._map_sensors(df, farm_id)
        df = self._normalize_physics(df, farm_id)
        
        # 3. Feature Engineering
        df = self._compute_derived_features(df, farm_id)
        
        # 4. Labeling: Filename is the event_id
        if event_path and event_path.exists():
            df = self._label_by_event_id(df, event_path, csv_path.stem)
        
        return df

    def _get_required_columns(self, farm_id, csv_path):
        farm_cfg = self.config['farms'][farm_id]
        actual_cols = pd.read_csv(csv_path, sep=';', nrows=0).columns.tolist()
        requested_cols = ['time_stamp', 'asset_id']
        for sensor in farm_cfg['sensors'].values():
            if isinstance(sensor, list): requested_cols.extend(sensor)
            elif sensor: requested_cols.append(sensor)
        return [c for c in set(requested_cols) if c in actual_cols]

    def _map_sensors(self, raw_df, farm_id):
        farm_cfg = self.config['farms'][farm_id]
        mapped = pd.DataFrame(index=raw_df.index)
        mapped['time_stamp'] = raw_df['time_stamp']
        mapped['farm_id'] = farm_id
        # Note: The asset_id is inside the file, but we'll verify it against the log
        mapped['asset_id'] = raw_df['asset_id'].astype(str) if 'asset_id' in raw_df.columns else "Unknown"

        for feature in self.standard_features:
            sensor_ref = farm_cfg['sensors'].get(feature)
            if isinstance(sensor_ref, list):
                avail = [s for s in sensor_ref if s in raw_df.columns]
                mapped[feature] = pd.to_numeric(raw_df[avail].stack(), errors='coerce').unstack().mean(axis=1) if avail else np.nan
            else:
                mapped[feature] = pd.to_numeric(raw_df[sensor_ref], errors='coerce') if sensor_ref in raw_df.columns else np.nan
        return mapped

    def _normalize_physics(self, df, farm_id):
        units = self.config['farms'][farm_id]['units']
        if units['power'] == 'Wh': df['active_power'] *= 0.006
        if units['speed'] == 'rad/s': df['gen_speed'] *= (60 / (2 * np.pi))
        return df

    def _compute_derived_features(self, df, farm_id):
        df = df.sort_values('time_stamp')
        # Technical health indicators
        if 'gearbox_oil_temp' in df and 'amb_temp' in df:
            df['temp_delta_gearbox'] = (df['gearbox_oil_temp'] - df['amb_temp']).fillna(0)
        
        # Power Efficiency (P / v^3)
        if 'wind_speed' in df and 'active_power' in df:
            v3 = df['wind_speed']**3
            df['power_efficiency'] = np.where(v3 > 0.1, df['active_power'] / v3, 0.0)
        
        # Moving average to detect "drift"
        if 'gearbox_oil_temp' in df:
            df['temp_rolling_mean'] = df['gearbox_oil_temp'].rolling(144, min_periods=1).mean()
            df['temp_divergence'] = df['gearbox_oil_temp'] - df['temp_rolling_mean']
        
        return df.fillna(0)

    def _label_by_event_id(self, df, event_path, filename_id):
        """Uses filename as event_id to find the specific time-window in the log."""
        events = pd.read_csv(event_path, sep=';')
        
        # 1. Clean the event_id from filename
        try:
            target_event_id = int(''.join(filter(str.isdigit, filename_id)))
        except:
            df['target'] = 0
            return df

        # 2. Find the specific event record
        event_info = events[events['event_id'] == target_event_id]
        
        df['target'] = 0
        if not event_info.empty:
            row = event_info.iloc[0]
            # Only label as 1 if the event is an anomaly
            if str(row['event_label']).lower() == 'anomaly':
                start_dt = pd.to_datetime(row['event_start']) - pd.Timedelta(days=self.lead_in_days)
                end_dt = pd.to_datetime(row['event_end'])
                
                mask = (df['time_stamp'] >= start_dt) & (df['time_stamp'] <= end_dt)
                df.loc[mask, 'target'] = 1
        
        return df

    def load_processed_dataset(self, processed_dir):
        processed_path = Path(processed_dir)
        parquet_files = list(processed_path.glob("*.parquet"))
        print(f"Combining {len(parquet_files)} event files...")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True).reset_index(drop=True)