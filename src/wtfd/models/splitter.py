import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler

class WindFarmSplitter:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.scaler = StandardScaler()

    def get_train_test_split(self, df, test_size=0.2):
        # Group by asset to get unique turbine list + their farm origin
        asset_meta = df[['asset_id', 'farm_id']].drop_duplicates()
        
        # Stratified split ensures Farm A, B, and C are represented in both sets
        train_assets, test_assets = train_test_split(
            asset_meta['asset_id'],
            test_size=test_size,
            stratify=asset_meta['farm_id'], # Crucial for cross-farm generalization
            random_state=self.random_state
        )
        
        train_df = df[df['asset_id'].isin(train_assets)].copy()
        test_df = df[df['asset_id'].isin(test_assets)].copy()
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

    def prepare_xy(self, df, target_col='target', drop_cols=None, fit_scaler=False):
        """
        Separates features from target and optionally scales the data.
        Set fit_scaler=True ONLY for your Training set.
        """
        if drop_cols is None:
            drop_cols = ['time_stamp', 'asset_id', 'farm_id']
            
        existing_drops = [c for c in drop_cols if c in df.columns]
        
        # 1. Separate Features and Target
        X = df.drop(columns=[target_col] + existing_drops)
        y = df[target_col]
        
        # 2. Handle Scaling (Crucial for Logistic Regression)
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        # Convert back to DataFrame to keep column names (helpful for XGBoost)
        X_final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_final, y