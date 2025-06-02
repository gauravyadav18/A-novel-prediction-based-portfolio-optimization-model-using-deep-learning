# main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from data_preprocessing import load_data, preprocess_features, create_sliding_windows
from autoencoders import Autoencoder
from lstm_model import LSTMModel
from omega_optimization import worst_case_omega_optimization
from portfolio_metrics import compute_metrics
from config import *

def main():
    data = load_data(DATA_PATHS, DATE_COLUMN, PRICE_COLUMN)
    windows = create_sliding_windows(data, TRAIN_START_DATE, TEST_END_DATE, 
                                    WINDOW_YEARS, TRAIN_YEARS, VALIDATION_YEARS, TEST_YEARS)
    
    all_metrics = []
    for window_idx, window_data in enumerate(windows):
        print(f"Processing window {window_idx + 1}/{len(windows)}")
        
        train_features, train_targets, val_features, val_targets, test_features, test_returns, test_indices = [], [], [], [], [], [], None
        for idx_name, idx_data in window_data.items():
            train_X, train_y, _ = preprocess_features(idx_data['train'], LAG_DAYS)
            val_X, val_y, _ = preprocess_features(idx_data['validation'], LAG_DAYS)
            test_X, _, test_idx = preprocess_features(idx_data['test'], LAG_DAYS)
            test_ret = idx_data['test']['Log_Return']
            train_features.append(train_X)
            train_targets.append(train_y)
            val_features.append(val_X)
            val_targets.append(val_y)
            test_features.append(test_X)
            test_returns.append(test_ret)
            if test_indices is None:
                test_indices = test_idx
        
        # Stack features horizontally
        train_X = np.hstack(train_features)
        val_X = np.hstack(val_features)
        test_X = np.hstack(test_features)
        
        # Stack targets into (num_samples, num_indices)
        num_indices = len(DATA_PATHS)  # e.g., 4
        train_y = np.stack(train_targets, axis=1)  # Shape: (num_samples, 4)
        val_y = np.stack(val_targets, axis=1)      # Shape: (num_samples, 4)
        
        # Align test_returns with test_X
        test_returns = np.stack([ret.loc[test_indices].values for ret in test_returns], axis=1)  # Shape: (num_test_samples, 4)
        
        # Train Autoencoder
        ae = Autoencoder(input_dim=train_X.shape[1], hidden_units=AE_HIDDEN_UNITS)
        encoder = ae.train(train_X, val_X)
        
        train_X_ae = encoder.predict(train_X)
        val_X_ae = encoder.predict(val_X)
        test_X_ae = encoder.predict(test_X)
        
        # Train LSTM with output shape (num_indices,)
        lstm = LSTMModel(input_dim=AE_HIDDEN_UNITS, output_dim=num_indices)
        lstm.train(train_X_ae, train_y, val_X_ae, val_y)
        
        # Predict returns for each time step and each asset
        predicted_returns_matrix = lstm.predict(test_X_ae)  # Shape: (num_test_samples, 4)
        
        # Average predicted returns over the test period for each asset
        predicted_returns = np.mean(predicted_returns_matrix, axis=0)  # Shape: (4,)
        
        # Select assets based on predicted returns
        selected_indices = predicted_returns > THRESHOLD
        if not np.any(selected_indices):
            selected_indices = np.ones(len(predicted_returns), dtype=bool)
        
        # Compute predictive errors for each time step
        predictive_errors = test_returns - predicted_returns_matrix  # Shape: (num_test_samples, 4)
        predictive_errors_3d = np.repeat(predictive_errors[:, :, np.newaxis], NUM_DISTRIBUTIONS, axis=2)
        
        # Compute historical returns for each asset
        historical_returns = np.zeros((num_indices, 2))
        for i in range(num_indices):
            historical_returns[i, 0] = np.mean(test_returns[-20:, i])
            historical_returns[i, 1] = np.mean(test_returns[-5:, i])
        
        # Optimize portfolio using selected assets
        weights = worst_case_omega_optimization(predicted_returns[selected_indices],  historical_returns[selected_indices], predictive_errors_3d[:, selected_indices, :],   np.sum(selected_indices))
        
        # Compute portfolio returns
        portfolio_returns = np.sum(test_returns[:, selected_indices] * weights, axis=1)
        benchmark_returns = np.mean(test_returns, axis=1)
        
        metrics = compute_metrics(portfolio_returns, benchmark_returns)
        metrics['Window'] = window_idx + 1
        all_metrics.append(metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    print("\nPortfolio Performance Metrics:")
    print(metrics_df)
    metrics_df.to_csv("portfolio_metrics.csv", index=False)

if __name__ == "__main__":
    main()