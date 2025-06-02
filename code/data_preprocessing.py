# data_preprocessing.py
import pandas as pd
import numpy as np
import os
import sys
from config import *

def load_data(data_paths, date_column, price_column):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data = {}
    for filename in data_paths:
        file_path = os.path.join(script_dir, filename)
        try:
            df = pd.read_csv(file_path)
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df['Log_Return'] = np.log(df[price_column] / df[price_column].shift(1))
            df = df.dropna()
            data[filename.split('.')[0]] = df
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Ensure the file exists in the project directory.")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Column {e} not found in {file_path}. Ensure 'Date' and 'Close' columns exist.")
            sys.exit(1)
    return data

def preprocess_features(df, lag_days):
    features = []
    for lag in range(1, lag_days + 1):
        features.append(df['Log_Return'].shift(lag))
    feature_df = pd.concat(features, axis=1).dropna()
    feature_df.columns = [f'Lag_{i}' for i in range(1, lag_days + 1)]
    
    for col in feature_df.columns:
        median = feature_df[col].median()
        mad = (feature_df[col] - median).abs().median()
        feature_df[col] = (feature_df[col] - median) / mad if mad != 0 else feature_df[col]
    
    mean = feature_df.mean()
    std = feature_df.std()
    feature_df = (feature_df - mean) / std
    feature_df = feature_df.dropna()
    
    target = df['Log_Return'].loc[feature_df.index]
    target = np.clip(target * 10, -1, 1)
    
    return feature_df.values, target.values, feature_df.index

def create_sliding_windows(data, start_date, end_date, window_years, train_years, validation_years, test_years):
    windows = []
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    current_start = start
    while current_start + pd.Timedelta(days=365 * window_years) <= end:
        train_end = current_start + pd.Timedelta(days=365 * train_years)
        validation_end = train_end + pd.Timedelta(days=365 * validation_years)
        test_end = validation_end + pd.Timedelta(days=365 * test_years)
        if test_end > end:
            break
        window_data = {}
        for key, df in data.items():
            window_data[key] = {
                'train': df[(df.index >= current_start) & (df.index < train_end)],
                'validation': df[(df.index >= train_end) & (df.index < validation_end)],
                'test': df[(df.index >= validation_end) & (df.index < test_end)]
            }
        windows.append(window_data)
        current_start += pd.Timedelta(days=365)
    return windows