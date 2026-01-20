import numpy as np

def aggregate_predictions(pred, mode):
    pred = np.array(pred)

    if mode == "hourly":    
        return pred.reshape(-1, 4).sum(axis=1)

    if mode == "daily":      
        return pred.reshape(-1, 96).sum(axis=1)

    if mode == "weekly": 
        daily = pred.reshape(-1, 96).sum(axis=1)
        pad_len = (-len(daily)) % 7  # 30 → pad 4 zeros
        daily_padded = np.pad(daily, (0, pad_len), 'constant')
        return daily_padded.reshape(-1, 7).sum(axis=1)

    return pred
