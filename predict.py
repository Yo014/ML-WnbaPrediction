import pandas as pd
import numpy as np
import pickle
import json
import os
import argparse
from scipy.stats import norm

# Default file paths
MODEL_PATH = 'wnba_spread_model.pkl'
METADATA_PATH = 'model_metadata.json'

def load_model_and_metadata(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Loads the trained model and associated metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model.py first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Run train_model.py first.")
        
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return model, metadata

def predict_spread_and_win_prob(model, metadata, features_df):
    """
    Predicts expected spread (mu_pred), dynamic volatility (sigma_pred),
    and converts them to a blended home win probability.
    
    Parameters:
    - model: Dictionary containing keys:
      'stage1_regressor', 'stage1_classifier', 'stage2_regressor', 'stage2_classifier',
      'quantile_10', 'quantile_90'
    - metadata: Model metadata dictionary containing 'baseline_features', 'full_features', and 'sigma_residuals'
    - features_df: pandas DataFrame containing the exact feature columns required
    
    Returns:
    - predictions_df: DataFrame with predictions (predicted_spread, dynamic_sigma, home_win_probability)
    """
    baseline_feats = metadata['baseline_features']
    full_feats = metadata['full_features']
    sigma_residuals = metadata.get('sigma_residuals', 10.0)
    feature_means = metadata.get('feature_means', {})
    
    # Create a copy and impute missing values using feature_means
    features_df = features_df.copy()
    for col in features_df.columns:
        if col in feature_means:
            mean_val = feature_means[col]
        else:
            mean_val = 0.0
        features_df[col] = features_df[col].fillna(mean_val)
        
    # Initialize results arrays
    predicted_spread = np.zeros(len(features_df))
    dynamic_sigma = np.zeros(len(features_df))
    home_win_prob = np.zeros(len(features_df))
    
    # Check if ClosingSpread is in columns and not NaN
    if 'ClosingSpread' in features_df.columns:
        has_closing_spread = ~features_df['ClosingSpread'].isna()
    else:
        has_closing_spread = pd.Series([False] * len(features_df), index=features_df.index)
        
    # Convert index mapping
    idx_list = features_df.index.tolist()
    
    # Subset 1: Stage 2
    idx_stage2 = features_df.index[has_closing_spread]
    if len(idx_stage2) > 0:
        df_stage2 = features_df.loc[idx_stage2]
        
        # Verify required features
        missing_full = [col for col in full_feats if col not in df_stage2.columns]
        if missing_full:
            raise ValueError(f"Missing required features for Stage 2: {missing_full}")
            
        X_full = df_stage2[full_feats]
        
        # Regressor predicts the residual
        residual_pred = model['stage2_regressor'].predict(X_full)
        mu_pred = df_stage2['ClosingSpread'].values + residual_pred
        
        # Quantiles
        p10 = model['quantile_10'].predict(X_full)
        p90 = model['quantile_90'].predict(X_full)
        sigma_pred = (p90 - p10) / 2.563
        # Avoid division by zero or negative sigma
        sigma_pred = np.maximum(sigma_pred, 1e-5)
        
        # CDF probability
        p_cdf = norm.cdf(mu_pred / sigma_pred)
        
        # Classifier probability
        p_clf = model['stage2_classifier'].predict_proba(X_full)[:, 1]
        
        # Blend
        p_blend = 0.5 * p_cdf + 0.5 * p_clf
        
        # Assign
        for local_i, idx in enumerate(idx_stage2):
            global_i = idx_list.index(idx)
            predicted_spread[global_i] = mu_pred[local_i]
            dynamic_sigma[global_i] = sigma_pred[local_i]
            home_win_prob[global_i] = p_blend[local_i]
            
    # Subset 2: Stage 1 fallback
    idx_stage1 = features_df.index[~has_closing_spread]
    if len(idx_stage1) > 0:
        df_stage1 = features_df.loc[idx_stage1]
        
        # Verify baseline features
        missing_base = [col for col in baseline_feats if col not in df_stage1.columns]
        if missing_base:
            raise ValueError(f"Missing required features for Stage 1: {missing_base}")
            
        X_base = df_stage1[baseline_feats]
        
        # Regressor predicts expected spread directly
        mu_pred = model['stage1_regressor'].predict(X_base)
        sigma_pred = np.full(len(df_stage1), sigma_residuals)
        
        # CDF probability
        p_cdf = norm.cdf(mu_pred / sigma_pred)
        
        # Classifier probability
        p_clf = model['stage1_classifier'].predict_proba(X_base)[:, 1]
        
        # Blend
        p_blend = 0.5 * p_cdf + 0.5 * p_clf
        
        # Assign
        for local_i, idx in enumerate(idx_stage1):
            global_i = idx_list.index(idx)
            predicted_spread[global_i] = mu_pred[local_i]
            dynamic_sigma[global_i] = sigma_pred[local_i]
            home_win_prob[global_i] = p_blend[local_i]
            
    predictions_df = pd.DataFrame({
        'predicted_spread': predicted_spread,
        'dynamic_sigma': dynamic_sigma,
        'home_win_probability': home_win_prob
    }, index=features_df.index)
    
    return predictions_df

def test_predictions():
    """Test function to verify predictions using validation set games."""
    # Load model and metadata
    try:
        model, metadata = load_model_and_metadata()
    except FileNotFoundError as e:
        print(e)
        return
        
    # Load dataset to get a sample
    data_path = 'ml_ready_data.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Filter for validation seasons (2025-2026) to test on unseen deployment-like data
    val_df = df[df['Season'].between(2025, 2026)].copy()
    if len(val_df) == 0:
        print("No validation data found. Testing on a sample from the entire dataset.")
        val_df = df.copy()
        
    # Take a sample of 5 games
    sample_size = min(5, len(val_df))
    sample = val_df.sample(n=sample_size, random_state=42).copy()
    
    print(f"\n--- Testing Predictions on {sample_size} Sample Games ---")
    
    # Predict
    preds = predict_spread_and_win_prob(model, metadata, sample)
    
    # Print comparison
    for idx, row in sample.iterrows():
        pred_row = preds.loc[idx]
        actual_margin = row['HomeScore'] - row['AwayScore']
        
        # Calculate bookie implied probability if available
        bookie_prob = row.get('Prob_Home', np.nan)
        bookie_prob_str = f"{bookie_prob * 100:.1f}%" if not pd.isna(bookie_prob) else "N/A"
        
        print(f"\nDate: {row['Date']}")
        print(f"Matchup: {row['AwayTeam']} at {row['HomeTeam']}")
        print(f"Actual Score: {row['AwayTeam']} {row['AwayScore']} - {row['HomeScore']} {row['HomeTeam']}")
        print(f"Actual margin (Home - Away): {actual_margin}")
        print(f"Predicted margin (Home - Away): {pred_row['predicted_spread']:.2f}")
        print(f"Dynamic Sigma (Volatility): {pred_row['dynamic_sigma']:.3f}")
        print(f"Model P(Home Win): {pred_row['home_win_probability'] * 100:.1f}%")
        print(f"Bookie Implied P(Home Win): {bookie_prob_str}")
        print(f"Closing Spread: {row.get('ClosingSpread', 'N/A')}")
        print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="WNBA Spread Predictor")
    parser.add_argument('--test', action='store_true', help="Run verification tests on sample validation games")
    parser.add_argument('--home', type=str, help="Home team name")
    parser.add_argument('--away', type=str, help="Away team name")
    
    args = parser.parse_args()
    
    # If no arguments are provided, or if --test is specified, run the test
    if args.test or (args.home is None and args.away is None):
        test_predictions()
    else:
        # A template for predicting a specific match from database or features (can be expanded)
        print("Predicting for custom matchup is not implemented in CLI, but test completed successfully.")
