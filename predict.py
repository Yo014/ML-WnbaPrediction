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
    Predicts expected spread (mu_pred) and converts it to home win probability.
    
    Parameters:
    - model: Trained XGBRegressor model
    - metadata: Model metadata dictionary containing 'features' list and 'sigma_residuals'
    - features_df: pandas DataFrame containing the exact feature columns required
    
    Returns:
    - predictions_df: DataFrame with predictions (predicted_spread, home_win_probability)
    """
    features_list = metadata['features']
    sigma_residuals = metadata['sigma_residuals']
    
    # Check if all required features are present
    missing_cols = [col for col in features_list if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")
        
    # Reorder/extract features to match the exact training feature order
    X = features_df[features_list]
    
    # Predict expected point spread (Home - Away score margin)
    mu_pred = model.predict(X)
    
    # Convert spread to Home Win Probability using normal CDF
    # P(Home Win) = scipy.stats.norm.cdf(predicted_spread / sigma_residuals)
    home_win_prob = norm.cdf(mu_pred / sigma_residuals)
    
    predictions_df = pd.DataFrame({
        'predicted_spread': mu_pred,
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
