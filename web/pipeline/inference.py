import numpy as np
import pandas as pd
from utils.helpers import generate_future_timestamps, get_past_category, get_last_numerical_features, generate_temporal_features, get_bin_feat
from pipeline.preprocess import preprocess

def make_input_seq(df, home_id, device_types, n_steps):
    all_sequences = []

    for device_type in device_types:
        # obtain new timestamps
        new_timestamps = generate_future_timestamps(n_steps, "15min")
        
        # get temporal feature from timestamps
        temporal_feat = generate_temporal_features(new_timestamps)

        # get categorical feature values corresponding to given device
        past_rows_categorical = get_past_category(df, home_id, device_type, n_steps)

        # assuming numerical feature values from past values
        past__rows_numeric = get_last_numerical_features(df, home_id, device_type, n_steps)

        # assuming binary features value
        past_bin_feat = get_bin_feat(df, home_id, device_type, n_steps)

        X_future = pd.concat([past_bin_feat.reset_index(drop=True),
                        temporal_feat.reset_index(drop=True),
                        past_rows_categorical.reset_index(drop=True),
                        past__rows_numeric.reset_index(drop=True)], axis=1)

        X_future_preprocessed = preprocess(X_future)
        seq_len = 12
        n_features = X_future_preprocessed.shape[1]
        n_sequences = len(X_future_preprocessed) // seq_len

        X_sequences = X_future_preprocessed.values.reshape(n_sequences, seq_len, n_features)
        all_sequences.append(X_sequences)
    
    return np.vstack(all_sequences)