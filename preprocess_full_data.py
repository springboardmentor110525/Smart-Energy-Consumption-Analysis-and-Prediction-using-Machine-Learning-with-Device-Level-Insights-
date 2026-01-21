"""
Full Data Preprocessing Script
Processes ALL rows from HomeC.csv for model training.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("FULL DATA PREPROCESSING")
print("=" * 60)

# Step 1: Load raw data
print("\n[1/6] Loading HomeC.csv...")
df = pd.read_csv('HomeC.csv', low_memory=False)
df = df[:-1]  # Remove last incomplete row
print(f"     Raw data: {len(df):,} rows")

# Step 2: Convert timestamp
print("\n[2/6] Converting timestamps...")
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['Timestamp'] = pd.to_datetime(df['time'], unit='s')
df.set_index('Timestamp', inplace=True)
df = df.sort_index()

# Remove any duplicate timestamps
df = df[~df.index.duplicated(keep='first')]
print(f"     Date range: {df.index.min()} to {df.index.max()}")

# Step 3: Resample to 15-minute intervals
print("\n[3/6] Resampling to 15-minute intervals...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_hourly = df[numeric_cols].resample('15min').mean()

# Remove rows with all NaN and handle missing values
df_hourly = df_hourly.dropna(how='all')
df_hourly = df_hourly.ffill().bfill()
print(f"     Resampled data: {len(df_hourly):,} rows")

# Save clean data
df_hourly.to_csv('clean_energy_data.csv')
print(f"     Saved: clean_energy_data.csv")

# Step 4: Scale the data
print("\n[4/6] Scaling data (MinMax 0-1)...")
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled_array = scaler.fit_transform(df_hourly)
df_scaled = pd.DataFrame(df_scaled_array, columns=df_hourly.columns, index=df_hourly.index)

df_scaled.to_csv('scaled_energy_data.csv')
print(f"     Saved: scaled_energy_data.csv")

# Step 5: Create features
print("\n[5/6] Engineering features...")

def create_features(df):
    """Create time-based and lag features."""
    df = df.copy()
    
    # Time-based features (normalized 0-1)
    df['hour'] = df.index.hour / 23.0
    df['day_of_week'] = df.index.dayofweek / 6.0
    df['day_of_month'] = df.index.day / 31.0
    df['month'] = df.index.month / 12.0
    
    # Aggregate device features
    if all(col in df.columns for col in ['Kitchen 12 [kW]', 'Kitchen 14 [kW]', 'Kitchen 38 [kW]']):
        df['Kitchen_Total_kW'] = df['Kitchen 12 [kW]'] + df['Kitchen 14 [kW]'] + df['Kitchen 38 [kW]']
    if all(col in df.columns for col in ['Furnace 1 [kW]', 'Furnace 2 [kW]']):
        df['Furnace_Total_kW'] = df['Furnace 1 [kW]'] + df['Furnace 2 [kW]']
    
    # Lag features (adjusted for 15-min intervals: 4 = 1h, 96 = 24h)
    df['lag_1h'] = df['use [kW]'].shift(4)      # 4 x 15min = 1 hour
    df['lag_24h'] = df['use [kW]'].shift(96)    # 96 x 15min = 24 hours
    
    # Rolling features (adjusted for 15-min intervals)
    df['rolling_mean_3h'] = df['use [kW]'].rolling(window=12).mean()   # 12 x 15min = 3h
    df['rolling_mean_24h'] = df['use [kW]'].rolling(window=96).mean()  # 96 x 15min = 24h
    
    # Drop NaN rows (first 96 rows due to lag_24h)
    df = df.dropna()
    
    return df

df_features = create_features(df_scaled)
print(f"     Features created: {len(df_features):,} rows")

# Step 6: Train/Test split (80/20)
print("\n[6/6] Creating train/test split...")
train_size = int(len(df_features) * 0.8)
train_features = df_features.iloc[:train_size]
test_features = df_features.iloc[train_size:]

train_features.to_csv('train_features.csv')
test_features.to_csv('test_features.csv')

print(f"     Training set: {len(train_features):,} rows")
print(f"     Testing set:  {len(test_features):,} rows")

# Also save raw train/test for reference
train_data = df_scaled.iloc[:train_size]
test_data = df_scaled.iloc[train_size:]
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print("=" * 60)
print(f"""
Summary:
  - Original data:    {len(df):,} rows (minute-level)
  - Hourly data:      {len(df_hourly):,} rows
  - Training samples: {len(train_features):,} rows
  - Testing samples:  {len(test_features):,} rows
  
Files created:
  - clean_energy_data.csv
  - scaled_energy_data.csv
  - train_features.csv
  - test_features.csv
  - train_data.csv
  - test_data.csv
""")
