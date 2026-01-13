import os
import joblib
import pandas as pd

def load_artifacts():
    base_path = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(base_path, "artifacts", "scaler.joblib"))
    encoder = joblib.load(os.path.join(base_path, "artifacts", "ohe.joblib"))
    feature_order = joblib.load(os.path.join(base_path, "artifacts", "feature_order.joblib"))
    return scaler, encoder, feature_order

def preprocess(df):
    scaler, encoder, feature_order = load_artifacts()
    features_to_encode = ['device_type', 'room', 'activity']
    features_to_scale = ["indoor_temp", "outdoor_temp", "humidity", "light_level"]

    df_encoded = pd.DataFrame(
        encoder.transform(df[features_to_encode]),
        columns=encoder.get_feature_names_out(features_to_encode)
    )

    df_scaled = pd.DataFrame(
        scaler.transform(df[features_to_scale]),
        columns=features_to_scale
    )

    int_cols = ["user_present","status","is_weekend","hour_of_day","day_of_week"]
    int32_cols = ["month_of_year"]
    float_cols = [
        "indoor_temp","outdoor_temp","humidity","light_level",
        "device_type_air_conditioner","device_type_fridge","device_type_light","device_type_tv","device_type_washer",
        "room_bedroom","room_kitchen","room_laundry_room","room_living_room",
        "activity_away","activity_cooking","activity_idle","activity_sleeping","activity_watching_tv",
        "energy_lag_1H","energy_lag_1D","energy_lag_1W","energy_roll_mean_1hr","energy_roll_mean_12hr","energy_roll_mean_24hr"
    ]

    other_cols = [c for c in df.columns if c not in features_to_encode + features_to_scale]
    df_other = df[other_cols].reset_index(drop=True)

    for col in int_cols:
        if col in df_other:
            df_other[col] = df_other[col].astype(int)

    for col in int32_cols:
        if col in df_other:
            df_other[col] = df_other[col].astype('int32')

    for col in float_cols:
        if col in df_other:
            df_other[col] = df_other[col].astype(float)
    
    df_preprocessed = pd.concat([df_other.reset_index(drop=True), df_encoded, df_scaled], axis=1)

    df_preprocessed = df_preprocessed[feature_order]

    return df_preprocessed