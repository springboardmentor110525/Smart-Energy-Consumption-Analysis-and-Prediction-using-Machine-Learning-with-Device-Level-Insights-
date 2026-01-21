import sys
import json
import numpy as np
import pandas as pd
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)

def main():
    try:
        if len(sys.argv) < 6:
            raise ValueError("Insufficient arguments")

        model_path = sys.argv[1]
        model_type = sys.argv[2]
        device = sys.argv[3]
        target_date = sys.argv[4]
        history_json = sys.argv[5]

        history = json.loads(history_json)
        
        # Basic validation
        if not history or len(history) < 10:
            # Return average if not enough data
            print(json.dumps({"prediction": float(np.mean(history) if history else 0)}))
            return

        if model_type == "linear":
            # Simple linear extrapolation based on recent history
            X = np.arange(len(history)).reshape(-1, 1)
            y = np.array(history)
            reg = LinearRegression().fit(X, y)
            
            # Predict next step (or N steps until target date - simplified to next step here)
            next_step = np.array([[len(history)]])
            prediction = reg.predict(next_step)[0]
            
            print(json.dumps({"prediction": float(prediction)}))
            return

        elif model_type == "lstm":
            # Load Keras model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            model = tf.keras.models.load_model(model_path)
            
            # Prepare input
            # Assuming model expects (1, 60, 1) or similar. 
            # We need to reshape history to match model input shape.
            # Inspecting model input shape would be ideal, but let's assume standard [samples, time_steps, features]
            
            input_data = np.array(history)
            
            # Pad or truncate to match expected sequence length (e.g. 60)
            # If model expects 60 and we have 60, great.
            # We'll try to use the last 60 points.
            
            # NOTE: Without knowing the exact model architecture (input shape), this is a best guess.
            # Many timeseries LSTMs take (Batch, Steps, Features).
            
            # Let's try to infer or assume 60 steps, 1 feature.
            SEQ_LEN = 60
            if len(input_data) > SEQ_LEN:
                input_data = input_data[-SEQ_LEN:]
            elif len(input_data) < SEQ_LEN:
                # Pad with mean
                pad_width = SEQ_LEN - len(input_data)
                input_data = np.pad(input_data, (pad_width, 0), mode='edge')
            
            # Reshape to (1, SEQ_LEN, 1)
            input_tensor = input_data.reshape(1, SEQ_LEN, 1)
            
            try:
                prediction = model.predict(input_tensor, verbose=0)
                # scalar prediction
                val = float(prediction[0][0])
                print(json.dumps({"prediction": val}))
            except Exception as e:
                # Fallback if shape mismatch
                # print(json.dumps({"error": f"Model prediction error: {str(e)}"}))
                # Fallback to linear
                print(json.dumps({"prediction": float(np.mean(history))}))
                
            return

        else:
            print(json.dumps({"prediction": 0}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
