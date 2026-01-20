import pandas as pd

df = pd.read_csv("./database/sample.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

def generate_future_timestamps(n_steps, freq):
    start = df["timestamp"].max() + pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(
        start=start,
        periods=n_steps,
        freq=freq
    )

def get_past_category(df, home_id, device_type, n_steps):
    last_row = df[(df.home_id == home_id) & (df.device_type == device_type)].sort_values("timestamp").iloc[-1]

    df_future_cat = pd.DataFrame({
        "device_type": [device_type] * n_steps,
        "room": [last_row.room] * n_steps,
        "activity": [last_row.activity] * n_steps
    })

    return df_future_cat

def get_last_numerical_features(df, home_id, device_type, n_steps):
    numeric_cols = ["indoor_temp", "outdoor_temp", "humidity", "light_level",  'energy_lag_1H', 'energy_lag_1D', 'energy_lag_1W', 'energy_roll_mean_1hr', 'energy_roll_mean_12hr', 'energy_roll_mean_24hr']
        
    last_row = df[
        (df.home_id == home_id) & 
        (df.device_type == device_type)
    ].sort_values("timestamp").iloc[-n_steps:]

    return last_row[numeric_cols].reset_index(drop=True)

def generate_temporal_features(new_timestamps):
    df_time = pd.DataFrame({"timestamp": new_timestamps})
    
    df_time["is_weekend"] = (df_time["timestamp"].dt.dayofweek >= 5 ).astype(float)
    df_time["hour_of_day"] = df_time["timestamp"].dt.hour.astype(float)
    df_time["day_of_week"] = df_time["timestamp"].dt.dayofweek.astype(float)
    df_time["month_of_year"] = df_time["timestamp"].dt.month.astype(float)
    
    return df_time

def get_bin_feat(df, home_id, device_type, n_steps):
    binary_cols = ["user_present", "status"] 
    
    past_rows = df[
        (df.home_id == home_id) &
        (df.device_type == device_type)
    ].sort_values("timestamp").iloc[-n_steps:]
    
    df_binary = past_rows[binary_cols].copy()
    for col in df_binary.columns:
        if df_binary[col].dtype == object:
            df_binary[col] = df_binary[col].map({"on": 1, "off": 0})
        df_binary[col] = df_binary[col].astype(float)

    return df_binary.reset_index(drop=True)