import os
import pickle
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from stacking_models import StackedEnsembleRegressor, StackedEnsembleClassifier

def main():
    print("Starting Model Parity Verification check...")
    
    # Files paths
    spread_model_path = 'wnba_spread_model.pkl'
    totals_model_path = 'wnba_total_model.pkl'
    spread_metadata_path = 'model_metadata.json'
    totals_metadata_path = 'total_model_metadata.json'
    data_path = 'ml_ready_data.csv'
    
    # Assert files exist
    for f in [spread_model_path, totals_model_path, spread_metadata_path, totals_metadata_path, data_path]:
        if not os.path.exists(f):
            print(f"ERROR: File {f} does not exist. Parity check failed.")
            return
            
    # 1. Load data
    df = pd.read_csv(data_path)
    
    # Load metadata first
    with open(spread_metadata_path, 'r') as f:
        spread_meta = json.load(f)
    with open(totals_metadata_path, 'r') as f:
        totals_meta = json.load(f)
        
    baseline_features = spread_meta['baseline_features']
    full_features = spread_meta['full_features']
    t_baseline_features = totals_meta['baseline_features']
    t_full_features = totals_meta['full_features']
    
    # Filter validation batch to drop NaNs in required features
    all_needed = list(set(baseline_features + full_features + t_baseline_features + t_full_features))
    data_columns = [col for col in all_needed if col != 's1_total_pred']
    val_batch = df.dropna(subset=data_columns).iloc[:50].copy()
    
    if len(val_batch) < 10:
        # Fallback to simple mean fill if too few rows
        print("Warning: Few clean rows found, falling back to filling NaNs with column means.")
        val_batch = df.iloc[:50].copy()
        for col in all_needed:
            val_batch[col] = val_batch[col].fillna(df[col].mean()).fillna(0.0)
            
    print(f"Loaded validation batch of {len(val_batch)} matches from {data_path}.")
    
    # 2. Verify Spread Model
    print("\nVerifying Spread Model...")
    with open(spread_model_path, 'rb') as f:
        spread_model = pickle.load(f)
    
    # Check baseline features exist in data
    missing_baseline = [c for c in baseline_features if c not in val_batch.columns]
    if missing_baseline:
        print(f"ERROR: Missing baseline features in data: {missing_baseline}")
        return
        
    # Predict using Stage 1 ELO fallback
    X_val_baseline = val_batch[baseline_features]
    predicted_spreads_s1 = spread_model['stage1_regressor'].predict(X_val_baseline)
    print(f"Stage 1 spread predictions generated successfully. Shape: {predicted_spreads_s1.shape}")
    print(f"Stage 1 Spreads Mean: {np.mean(predicted_spreads_s1):.4f}, Std: {np.std(predicted_spreads_s1):.4f}")
    
    # Check bounds
    assert not np.isnan(predicted_spreads_s1).any(), "NaN values found in Stage 1 spread predictions!"
    assert np.std(predicted_spreads_s1) > 2.0, "Stage 1 spread prediction variance is suspiciously low!"
    
    # 3. Verify Totals Model
    print("\nVerifying Totals Model...")
    with open(totals_model_path, 'rb') as f:
        totals_model = pickle.load(f)
    with open(totals_metadata_path, 'r') as f:
        totals_meta = json.load(f)
        
    t_baseline_features = totals_meta['baseline_features']
    t_full_features = totals_meta['full_features']
    
    # Predict totals Stage 1
    X_t_baseline = val_batch[t_baseline_features]
    s1_pace = totals_model['stage1_pace_regressor'].predict(X_t_baseline)
    s1_home_eff = totals_model['stage1_home_eff_regressor'].predict(X_t_baseline)
    s1_away_eff = totals_model['stage1_away_eff_regressor'].predict(X_t_baseline)
    s1_totals = s1_pace * (s1_home_eff + s1_away_eff) / 100.0
    print(f"Stage 1 pace & efficiency predictions generated successfully.")
    print(f"Stage 1 Totals Mean: {np.mean(s1_totals):.4f}, Std: {np.std(s1_totals):.4f}")
    
    assert not np.isnan(s1_totals).any(), "NaN values found in Stage 1 totals predictions!"
    
    # Predict totals Stage 2 (Residuals)
    val_batch_totals = val_batch.copy()
    val_batch_totals['s1_total_pred'] = s1_totals
    X_t_full = val_batch_totals[t_full_features]
    
    totals_dist = totals_model['stage2_regressor'].pred_dist(X_t_full)
    residuals_mean = totals_dist.mean()
    residuals_std = totals_dist.std()
    
    predicted_totals_s2 = val_batch['OverUnder'].values + residuals_mean
    print(f"Stage 2 totals predictions generated successfully.")
    print(f"Stage 2 Totals Mean: {np.mean(predicted_totals_s2):.4f}, Std: {np.std(predicted_totals_s2):.4f}")
    print(f"Predictive volatility (sigma) Mean: {np.mean(residuals_std):.4f}, Std: {np.std(residuals_std):.4f}")
    
    assert not np.isnan(predicted_totals_s2).any(), "NaN values found in Stage 2 totals predictions!"
    assert np.std(predicted_totals_s2) > 2.0, "Stage 2 totals prediction variance is suspiciously low!"
    assert (residuals_std > 0).all(), "Negative or zero volatility σ detected!"
    
    print("\n==================================================")
    print("  MODEL PARITY VERIFICATION PASSED SUCCESSFULLY!  ")
    print("==================================================")

if __name__ == '__main__':
    main()
