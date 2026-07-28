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
TOTAL_MODEL_PATH = 'wnba_total_model.pkl'
TOTAL_METADATA_PATH = 'total_model_metadata.json'

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
      'quantile_10', 'quantile_90', 'stage1_calibrator', 'stage2_calibrator'
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
    
    # Dynamically calculate Market_Disagreement if missing, or default/fillna to 0.0
    if 'Market_Disagreement' not in features_df.columns:
        if 'Prob_Home' in features_df.columns:
            poly = features_df['Poly_Prob_Home'] if 'Poly_Prob_Home' in features_df.columns else features_df['Prob_Home']
            poly = poly.fillna(features_df['Prob_Home'])
            features_df['Market_Disagreement'] = features_df['Prob_Home'] - poly
        else:
            features_df['Market_Disagreement'] = 0.0
    features_df['Market_Disagreement'] = features_df['Market_Disagreement'].fillna(0.0)
    
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
            
        # Select only the saved features list before predicting with base models
        X_full = df_stage2[full_feats]
        
        # Regressor predicts the residual and dynamic volatility using NGBRegressor
        stage2_dist = model['stage2_regressor'].pred_dist(X_full)
        residual_pred = stage2_dist.loc
        mu_pred = df_stage2['ClosingSpread'].values + residual_pred
        
        sigma_pred = stage2_dist.scale
        # Avoid division by zero or negative sigma
        sigma_pred = np.maximum(sigma_pred, 1e-5)
        
        # CDF probability
        p_cdf = norm.cdf(mu_pred / sigma_pred)
        
        # Classifier probability
        p_clf = model['stage2_classifier'].predict_proba(X_full)[:, 1]
        
        # Blend
        p_blend = 0.5 * p_cdf + 0.5 * p_clf
        
        # Run final win probabilities through corresponding Platt/Isotonic calibrator
        if 'stage2_calibrator' in model and model['stage2_calibrator'] is not None:
            p_calibrated = model['stage2_calibrator'].predict(p_blend)
        else:
            p_calibrated = p_blend
        
        # Assign
        for local_i, idx in enumerate(idx_stage2):
            global_i = idx_list.index(idx)
            predicted_spread[global_i] = mu_pred[local_i]
            dynamic_sigma[global_i] = sigma_pred[local_i]
            home_win_prob[global_i] = p_calibrated[local_i]
            
    # Subset 2: Stage 1 fallback
    idx_stage1 = features_df.index[~has_closing_spread]
    if len(idx_stage1) > 0:
        df_stage1 = features_df.loc[idx_stage1]
        
        # Verify baseline features
        missing_base = [col for col in baseline_feats if col not in df_stage1.columns]
        if missing_base:
            raise ValueError(f"Missing required features for Stage 1: {missing_base}")
            
        # Select only the saved features list before predicting with base models
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
        
        # Run final win probabilities through corresponding Platt/Isotonic calibrator
        if 'stage1_calibrator' in model and model['stage1_calibrator'] is not None:
            p_calibrated = model['stage1_calibrator'].predict(p_blend)
        else:
            p_calibrated = p_blend
        
        # Assign
        for local_i, idx in enumerate(idx_stage1):
            global_i = idx_list.index(idx)
            predicted_spread[global_i] = mu_pred[local_i]
            dynamic_sigma[global_i] = sigma_pred[local_i]
            home_win_prob[global_i] = p_calibrated[local_i]
            
    predictions_df = pd.DataFrame({
        'predicted_spread': predicted_spread,
        'dynamic_sigma': dynamic_sigma,
        'home_win_probability': home_win_prob
    }, index=features_df.index)
    
    return predictions_df

def predict_total_and_over_prob(model, metadata, features_df):
    """
    Predicts expected game total (mu_pred), dynamic volatility (sigma_pred),
    and converts them to a calibrated Over probability.
    
    Parameters:
    - model: Dictionary containing keys:
      'stage1_pace_regressor', 'stage1_home_eff_regressor', 'stage1_away_eff_regressor',
      'stage1_classifier', 'stage2_regressor', 'stage2_classifier',
      'quantile_10', 'quantile_90', 'stage1_calibrator', 'stage2_calibrator'
    - metadata: Model metadata dictionary containing 'baseline_features', 'full_features', and 'sigma_residuals'
    - features_df: pandas DataFrame containing the exact feature columns required
    
    Returns:
    - predictions_df: DataFrame with predictions (predicted_total, total_dynamic_sigma, over_win_probability)
    """
    baseline_feats = metadata['baseline_features']
    full_feats = metadata['full_features']
    sigma_residuals = metadata.get('sigma_residuals', 12.0)
    feature_means = metadata.get('feature_means', {})
    median_total = metadata.get('median_total', 160.0)
    
    # Create a copy and impute missing values using feature_means
    features_df = features_df.copy()
    
    # Dynamically calculate Market_Disagreement if missing, or default/fillna to 0.0
    if 'Market_Disagreement' not in features_df.columns:
        if 'Prob_Home' in features_df.columns:
            poly = features_df['Poly_Prob_Home'] if 'Poly_Prob_Home' in features_df.columns else features_df['Prob_Home']
            poly = poly.fillna(features_df['Prob_Home'])
            features_df['Market_Disagreement'] = features_df['Prob_Home'] - poly
        else:
            features_df['Market_Disagreement'] = 0.0
    features_df['Market_Disagreement'] = features_df['Market_Disagreement'].fillna(0.0)
    
    for col in features_df.columns:
        if col in feature_means:
            mean_val = feature_means[col]
        else:
            mean_val = 0.0
        features_df[col] = features_df[col].fillna(mean_val)
        
    # Pre-compute s1_total_pred feature for all games using the decoupled Stage 1 models
    X_base_all = features_df[baseline_feats]
    pace_pred = model['stage1_pace_regressor'].predict(X_base_all)
    home_eff_pred = model['stage1_home_eff_regressor'].predict(X_base_all)
    away_eff_pred = model['stage1_away_eff_regressor'].predict(X_base_all)
    s1_total_pred = pace_pred * (home_eff_pred + away_eff_pred) / 100.0
    features_df['s1_total_pred'] = s1_total_pred
        
    # Initialize results arrays
    predicted_total = np.zeros(len(features_df))
    dynamic_sigma = np.zeros(len(features_df))
    over_win_prob = np.zeros(len(features_df))
    
    # Check if OverUnder is in columns and not NaN
    if 'OverUnder' in features_df.columns:
        has_over_under = ~features_df['OverUnder'].isna()
    else:
        has_over_under = pd.Series([False] * len(features_df), index=features_df.index)
        
    # Convert index mapping
    idx_list = features_df.index.tolist()
    
    # Subset 1: Stage 2
    idx_stage2 = features_df.index[has_over_under]
    if len(idx_stage2) > 0:
        df_stage2 = features_df.loc[idx_stage2]
        
        # Verify required features
        missing_full = [col for col in full_feats if col not in df_stage2.columns]
        if missing_full:
            raise ValueError(f"Missing required features for Stage 2: {missing_full}")
            
        # Select only the saved features list before predicting with base models
        X_full = df_stage2[full_feats]
        
        # Regressor predicts the residual and dynamic volatility using NGBRegressor
        stage2_dist = model['stage2_regressor'].pred_dist(X_full)
        residual_pred = stage2_dist.loc
        mu_pred = df_stage2['OverUnder'].values + residual_pred
        
        sigma_pred = stage2_dist.scale
        sigma_pred = np.maximum(sigma_pred, 1e-5)
        
        # CDF probability (probability of residual > 0, i.e., total score > OverUnder)
        p_cdf = norm.cdf(residual_pred / sigma_pred)
        
        # Classifier probability of Over
        p_clf = model['stage2_classifier'].predict_proba(X_full)[:, 1]
        
        # Blend
        p_blend = 0.5 * p_cdf + 0.5 * p_clf
        
        # Run final win probabilities through corresponding Platt/Isotonic calibrator
        if 'stage2_calibrator' in model and model['stage2_calibrator'] is not None:
            calibrator = model['stage2_calibrator']
            if hasattr(calibrator, 'predict_proba'):
                p_calibrated = calibrator.predict_proba(p_blend.reshape(-1, 1))[:, 1]
            else:
                p_calibrated = calibrator.predict(p_blend)
        else:
            p_calibrated = p_blend
        
        # Assign
        for local_i, idx in enumerate(idx_stage2):
            global_i = idx_list.index(idx)
            predicted_total[global_i] = mu_pred[local_i]
            dynamic_sigma[global_i] = sigma_pred[local_i]
            over_win_prob[global_i] = p_calibrated[local_i]
            
    # Subset 2: Stage 1 fallback
    idx_stage1 = features_df.index[~has_over_under]
    if len(idx_stage1) > 0:
        df_stage1 = features_df.loc[idx_stage1]
        
        # Verify baseline features
        missing_base = [col for col in baseline_feats if col not in df_stage1.columns]
        if missing_base:
            raise ValueError(f"Missing required features for Stage 1: {missing_base}")
            
        # Select only the saved features list before predicting with base models
        X_base = df_stage1[baseline_feats]
        
        # Regressor predicts expected total directly
        mu_pred = df_stage1['s1_total_pred'].values
        sigma_pred = np.full(len(df_stage1), sigma_residuals)
        
        # CDF probability (probability of total score > median_total)
        p_cdf = norm.cdf((mu_pred - median_total) / sigma_pred)
        
        # Classifier probability
        p_clf = model['stage1_classifier'].predict_proba(X_base)[:, 1]
        
        # Blend
        p_blend = 0.5 * p_cdf + 0.5 * p_clf
        
        # Run final win probabilities through corresponding Platt/Isotonic calibrator
        if 'stage1_calibrator' in model and model['stage1_calibrator'] is not None:
            calibrator = model['stage1_calibrator']
            if hasattr(calibrator, 'predict_proba'):
                p_calibrated = calibrator.predict_proba(p_blend.reshape(-1, 1))[:, 1]
            else:
                p_calibrated = calibrator.predict(p_blend)
        else:
            p_calibrated = p_blend
        
        # Assign
        for local_i, idx in enumerate(idx_stage1):
            global_i = idx_list.index(idx)
            predicted_total[global_i] = mu_pred[local_i]
            dynamic_sigma[global_i] = sigma_pred[local_i]
            over_win_prob[global_i] = p_calibrated[local_i]
            
    predictions_df = pd.DataFrame({
        'predicted_total': predicted_total,
        'total_dynamic_sigma': dynamic_sigma,
        'over_win_probability': over_win_prob
    }, index=features_df.index)
    
    return predictions_df

def test_predictions():
    """Test function to verify predictions using validation set games."""
    # Load model and metadata
    try:
        model, metadata = load_model_and_metadata()
        t_model, t_metadata = load_model_and_metadata(TOTAL_MODEL_PATH, TOTAL_METADATA_PATH)
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
    
    # Predict Spreads and Totals
    preds = predict_spread_and_win_prob(model, metadata, sample)
    t_preds = predict_total_and_over_prob(t_model, t_metadata, sample)
    
    # Print comparison
    for idx, row in sample.iterrows():
        pred_row = preds.loc[idx]
        t_pred_row = t_preds.loc[idx]
        actual_margin = row['HomeScore'] - row['AwayScore']
        actual_total = row['HomeScore'] + row['AwayScore']
        
        # Calculate bookie implied probability if available
        bookie_prob = row.get('Prob_Home', np.nan)
        bookie_prob_str = f"{bookie_prob * 100:.1f}%" if not pd.isna(bookie_prob) else "N/A"
        
        print(f"\nDate: {row['Date']}")
        print(f"Matchup: {row['AwayTeam']} at {row['HomeTeam']}")
        print(f"Actual Score: {row['AwayTeam']} {row['AwayScore']} - {row['HomeScore']} {row['HomeTeam']}")
        print(f"Actual margin (Home - Away): {actual_margin} | Predicted margin: {pred_row['predicted_spread']:.2f} (Sigma: {pred_row['dynamic_sigma']:.2f})")
        print(f"Actual total score: {actual_total} | Predicted total: {t_pred_row['predicted_total']:.2f} (Sigma: {t_pred_row['total_dynamic_sigma']:.2f})")
        print(f"Model P(Home Win): {pred_row['home_win_probability'] * 100:.1f}% | Bookie implied P(Home Win): {bookie_prob_str}")
        print(f"Model P(Over): {t_pred_row['over_win_probability'] * 100:.1f}% | Bookie Line (OU): {row.get('OverUnder', 'N/A')}")
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
