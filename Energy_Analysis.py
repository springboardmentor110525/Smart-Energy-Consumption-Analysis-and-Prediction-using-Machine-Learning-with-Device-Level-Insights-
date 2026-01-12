#!/usr/bin/env python
# coding: utf-8

# In[8]:


print(df.dtypes)
print(df.isnull().sum())
df.ffill(inplace=True)
print(df.duplicated().sum())
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['use [kW]'], bins=50, kde=True)
plt.subplot(1, 2, 2)
sns.boxplot(x=df['use [kW]'])
plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv('HomeC.csv', low_memory=False)
df = df[:-1]
df['time'] = pd.to_numeric(df['time'])
df['Timestamp'] = pd.to_datetime(df['time'], unit='s')
df.set_index('Timestamp', inplace=True)
print(df.head())


# In[10]:


numeric_cols = df.select_dtypes(include=[np.number])
df_hourly = numeric_cols.resample('h').mean()
print(df.shape)
print(df_hourly.shape)
df_hourly[['use [kW]']].plot(figsize=(15, 5))
plt.show()


# In[11]:


df_hourly.to_csv('clean_energy_data.csv')
print("Success! Your clean data is saved as 'clean_energy_data.csv'")


# In[12]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled_array = scaler.fit_transform(df_hourly)
df_scaled = pd.DataFrame(df_scaled_array, columns=df_hourly.columns, index=df_hourly.index)

print(df_scaled.max().head())
print(df_scaled.min().head())

df_scaled.to_csv('scaled_energy_data.csv')


# In[13]:


train_size = int(len(df_scaled) * 0.8)
train_data = df_scaled.iloc[:train_size]
test_data = df_scaled.iloc[train_size:]

print(train_data.shape)
print(test_data.shape)

train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')


# In[14]:


import pandas as pd

# 1. Load the Scaled Data from Module 2
# We use index_col=0 and parse_dates=True so pandas understands the Timestamp index
train_df = pd.read_csv('train_data.csv', index_col=0, parse_dates=True)
test_df = pd.read_csv('test_data.csv', index_col=0, parse_dates=True)

def create_features(df):
    """
    Module 3: Feature Engineering
    - Extracts time features (Task 1)
    - Aggregates devices (Task 2)
    - Creates lags/rolling means (Task 3)
    """
    df = df.copy()
    # --- Task 1: Extract Relevant Time-Based Features ---
    # We divide by the max value (e.g., 23 for hour) to keep features between 0 and 1.
    # This matches the scale of your energy data, preventing model confusion.
    df['hour'] = df.index.hour / 23.0
    df['day_of_week'] = df.index.dayofweek / 6.0
    df['day_of_month'] = df.index.day / 31.0
    df['month'] = df.index.month / 12.0
    # --- Task 2: Aggregate Device-Level Consumption Statistics ---
    # Combine related circuits into meaningful groups so the model sees stronger patterns.
    # (Note: These column names must match your CSV headers exactly)
    df['Kitchen_Total_kW'] = df['Kitchen 12 [kW]'] + df['Kitchen 14 [kW]'] + df['Kitchen 38 [kW]']
    df['Furnace_Total_kW'] = df['Furnace 1 [kW]'] + df['Furnace 2 [kW]']
    # --- Task 3: Create Lag Features and Moving Averages ---
    # Lag 1: Usage 1 hour ago (Immediate history)
    df['lag_1h'] = df['use [kW]'].shift(1)
    # Lag 24: Usage 24 hours ago (Daily cycle pattern)
    df['lag_24h'] = df['use [kW]'].shift(24)
    # Rolling Mean: Average usage of the previous 3 hours (Trend smoothing)
    df['rolling_mean_3h'] = df['use [kW]'].rolling(window=3).mean()
    # --- Task 4: Prepare Final Feature Set ---
    # Drop rows with NaN (Not a Number) created by shifting/rolling
    # (The first 24 rows will be lost because they don't have enough history for lag_24h)
    df.dropna(inplace=True)
    return df
# Apply the function to both Train and Test datasets
train_features = create_features(train_df)
test_features = create_features(test_df)
# Print verification to show the work is done
print("Final Training Set Shape:", train_features.shape)
print("Final Testing Set Shape:", test_features.shape)
# Show all columns to prove nothing was removed
print("\nAll Available Features:")
print(train_features.columns.tolist())
# Save the final processed files for Module 4
train_features.to_csv('train_features.csv')
test_features.to_csv('test_features.csv')
print("\nSuccess: Module 3 complete. Data saved for modeling.")


# In[ ]:




