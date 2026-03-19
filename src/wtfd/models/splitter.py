import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold

class WindFarmSplitter:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state

    def get_train_test_split(self, df, test_size=0.2):
        """
        Splits data so that the Test set contains entirely unique turbines.
        Ensures each farm is represented in both sets.
        """
        # Identify unique turbines and their parent farm
        # This allows us to stratify by farm while splitting by asset_id
        asset_meta = df[['asset_id', 'farm_id']].drop_duplicates()
        
        train_assets, test_assets = train_test_split(
            asset_meta['asset_id'],
            test_size=test_size,
            stratify=asset_meta['farm_id'],
            random_state=self.random_state
        )
        
        train_df = df[df['asset_id'].isin(train_assets)].copy()
        test_df = df[df['asset_id'].isin(test_assets)].copy()
        
        print(f"Split Summary:")
        print(f"  Train: {len(train_assets)} turbines ({len(train_df)} rows)")
        print(f"  Test:  {len(test_assets)} turbines ({len(test_df)} rows)")
        
        return train_df, test_df

    def get_cv_iter(self, df):
        """
        Returns a GroupKFold iterator. 
        Use this in your training loop to prevent data leakage.
        """
        gkf = GroupKFold(n_splits=self.n_splits)
        
        # df and df['target'] are just placeholders for the split indices
        # groups=df['asset_id'] ensures assets are never split across folds
        return gkf.split(df, df['target'], groups=df['asset_id'])

    def prepare_xy(self, df, target_col='target', drop_cols=None):
        """
        Separates features from target and removes non-feature columns 
        to prevent the model from 'cheating' or crashing.
        """
        # Default columns that should NEVER be used as training features
        if drop_cols is None:
            drop_cols = ['time_stamp', 'asset_id', 'farm_id']
            
        # Ensure we don't try to drop columns that aren't there
        existing_drops = [c for c in drop_cols if c in df.columns]
        
        X = df.drop(columns=[target_col] + existing_drops)
        y = df[target_col]
        
        return X, y