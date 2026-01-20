"""
Module 3: Feature Engineering
Week 3-4 Implementation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for energy consumption prediction
    """
    
    def __init__(self, df):
        """
        Initialize feature engineer
        
        Args:
            df: Cleaned dataframe with datetime index
        """
        self.df = df.copy()
        self.scaler = None
        
    def extract_time_features(self):
        """Extract time-based features from datetime index"""
        print("\n⏰ Extracting time-based features...")
        
        # Ensure index is datetime
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        
        # Extract features
        self.df['hour'] = self.df.index.hour
        self.df['day'] = self.df.index.day
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['dayofweek'] = self.df.index.dayofweek
        self.df['weekofyear'] = self.df.index.isocalendar().week
        self.df['quarter'] = self.df.index.quarter
        self.df['is_weekend'] = (self.df.index.dayofweek >= 5).astype(int)
        
        # Time of day categories
        self.df['time_of_day'] = pd.cut(self.df['hour'], 
                                         bins=[0, 6, 12, 18, 24],
                                         labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                         include_lowest=True)
        
        # Cyclical encoding for hour (to capture circular nature)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # Cyclical encoding for month
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        print(f"   ✓ Added {14} time-based features")
        
        return self.df
    
    def create_lag_features(self, columns, lags=[1, 2, 3, 6, 12, 24]):
        """
        Create lag features for time series prediction
        
        Args:
            columns: List of columns to create lags for
            lags: List of lag periods
        """
        print(f"\n📊 Creating lag features for {len(columns)} columns...")
        
        for col in columns:
            if col in self.df.columns:
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        # Drop rows with NaN values created by lagging
        rows_before = len(self.df)
        self.df.dropna(inplace=True)
        rows_after = len(self.df)
        
        print(f"   ✓ Created lag features for lags: {lags}")
        print(f"   ✓ Rows: {rows_before:,} → {rows_after:,} (dropped {rows_before - rows_after:,} NaN rows)")
        
        return self.df
    
    def create_rolling_features(self, columns, windows=[3, 6, 12, 24]):
        """
        Create rolling window statistics
        
        Args:
            columns: List of columns to create rolling features for
            windows: List of window sizes
        """
        print(f"\n📈 Creating rolling window features...")
        
        for col in columns:
            if col in self.df.columns:
                for window in windows:
                    # Rolling mean
                    self.df[f'{col}_rolling_mean_{window}'] = \
                        self.df[col].rolling(window=window).mean()
                    
                    # Rolling std
                    self.df[f'{col}_rolling_std_{window}'] = \
                        self.df[col].rolling(window=window).std()
                    
                    # Rolling min/max
                    self.df[f'{col}_rolling_min_{window}'] = \
                        self.df[col].rolling(window=window).min()
                    
                    self.df[f'{col}_rolling_max_{window}'] = \
                        self.df[col].rolling(window=window).max()
        
        # Drop NaN values
        rows_before = len(self.df)
        self.df.dropna(inplace=True)
        rows_after = len(self.df)
        
        print(f"   ✓ Created rolling features for windows: {windows}")
        print(f"   ✓ Rows: {rows_before:,} → {rows_after:,}")
        
        return self.df
    
    def create_device_aggregations(self, device_columns):
        """
        Create aggregated features from device-level data
        
        Args:
            device_columns: List of device column names
        """
        print(f"\n🏠 Creating device aggregation features...")
        
        # Total energy consumption
        self.df['total_energy'] = self.df[device_columns].sum(axis=1)
        
        # Average energy per device
        self.df['avg_device_energy'] = self.df[device_columns].mean(axis=1)
        
        # Max device consumption
        self.df['max_device_energy'] = self.df[device_columns].max(axis=1)
        
        # Number of active devices (consumption > 0)
        self.df['active_devices'] = (self.df[device_columns] > 0).sum(axis=1)
        
        # Energy variance across devices
        self.df['energy_variance'] = self.df[device_columns].var(axis=1)
        
        # Percentage of total for each device
        for col in device_columns:
            if col in self.df.columns:
                self.df[f'{col}_pct'] = (self.df[col] / self.df['total_energy']) * 100
                self.df[f'{col}_pct'].fillna(0, inplace=True)
        
        print(f"   ✓ Created {5 + len(device_columns)} aggregation features")
        
        return self.df
    
    def create_interaction_features(self, weather_cols, energy_col='total_energy'):
        """
        Create interaction features between weather and energy
        
        Args:
            weather_cols: List of weather feature columns
            energy_col: Energy column to interact with
        """
        print(f"\n🌤️ Creating weather-energy interaction features...")
        
        count = 0
        for weather_col in weather_cols:
            if weather_col in self.df.columns and energy_col in self.df.columns:
                # Interaction term
                self.df[f'{energy_col}_x_{weather_col}'] = \
                    self.df[energy_col] * self.df[weather_col]
                count += 1
        
        print(f"   ✓ Created {count} interaction features")
        
        return self.df
    
    def normalize_features(self, columns_to_scale, method='minmax'):
        """
        Normalize/standardize features
        
        Args:
            columns_to_scale: List of columns to scale
            method: 'minmax' or 'standard'
        """
        print(f"\n📏 Normalizing features using {method} scaling...")
        
        # Select scaler
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        # Fit and transform
        scaled_data = self.scaler.fit_transform(self.df[columns_to_scale])
        
        # Create scaled dataframe
        df_scaled = pd.DataFrame(
            scaled_data,
            columns=[f'{col}_scaled' for col in columns_to_scale],
            index=self.df.index
        )
        
        # Concatenate with original
        self.df = pd.concat([self.df, df_scaled], axis=1)
        
        print(f"   ✓ Scaled {len(columns_to_scale)} features")
        
        return self.df, self.scaler
    
    def save_scaler(self, path='models/scaler.pkl'):
        """Save the fitted scaler"""
        if self.scaler is not None:
            joblib.dump(self.scaler, path)
            print(f"   ✓ Scaler saved to {path}")
    
    def get_feature_importance_data(self, target_col):
        """
        Prepare data for feature importance analysis
        
        Args:
            target_col: Target variable column name
        """
        # Separate features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        # Remove non-numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        
        return X_numeric, y
    
    def create_sequences_for_lstm(self, target_col, sequence_length=24):
        """
        Create sequences for LSTM model
        
        Args:
            target_col: Target variable
            sequence_length: Length of input sequences
        """
        print(f"\n🔄 Creating sequences for LSTM (length={sequence_length})...")
        
        data = self.df[target_col].values
        
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"   ✓ Created {len(X):,} sequences")
        print(f"   ✓ X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y


def main():
    """Main execution function"""
    print("="*60)
    print("SMART ENERGY ANALYSIS - FEATURE ENGINEERING")
    print("="*60)
    
    # Load cleaned data
    print("\n📂 Loading cleaned data...")
    df = pd.read_csv('data/processed/energy_data_clean.csv', 
                     index_col='time', parse_dates=True)
    print(f"   ✓ Loaded {len(df):,} records")
    
    # Initialize feature engineer
    fe = FeatureEngineer(df)
    
    # Extract time features
    df_features = fe.extract_time_features()
    
    # Define device columns (adjust based on your dataset)
    device_cols = [col for col in df.columns if any(device in col.lower() 
                   for device in ['dishwasher', 'fridge', 'microwave', 'furnace', 
                                  'kitchen', 'office', 'living', 'barn', 'well'])]
    
    # Create device aggregations
    if device_cols:
        df_features = fe.create_device_aggregations(device_cols[:5])  # Use first 5 for demo
    
    # Create lag features for main target
    target_col = 'use_HO' if 'use_HO' in df.columns else device_cols[0]
    df_features = fe.create_lag_features([target_col], lags=[1, 2, 3, 6, 12, 24])
    
    # Create rolling features
    df_features = fe.create_rolling_features([target_col], windows=[3, 6, 12, 24])
    
    # Save engineered features
    print(f"\n💾 Saving engineered features...")
    df_features.to_csv('data/processed/energy_data_features.csv')
    print(f"   ✓ Saved to data/processed/energy_data_features.csv")
    
    print("\n" + "="*60)
    print("✓ FEATURE ENGINEERING COMPLETE!")
    print(f"  Total features: {len(df_features.columns)}")
    print("="*60)
    
    return fe, df_features


if __name__ == "__main__":
    fe, df_features = main()
