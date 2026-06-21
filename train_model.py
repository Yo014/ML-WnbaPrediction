import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import mean_absolute_error, r2_score

def train_model():
    # 1. Read the engineered dataset
    data_path = 'ml_ready_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sort chronologically to make sure it is in order
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 2. Define target and features
    # Target = Home_Score - Away_Score
    df['Target'] = df['HomeScore'] - df['AwayScore']
    
    # Confirm target aligns with Score_Diff
    if not np.allclose(df['Target'], df['Score_Diff']):
        print("Warning: Calculated Target does not match Score_Diff in some cases. Using calculated Target.")
    
    # Define features
    features = [
        # Ratings (EMA)
        'Home_Offensive_Rating_EMA_5', 'Home_Defensive_Rating_EMA_5',
        'Home_Offensive_Rating_EMA_10', 'Home_Defensive_Rating_EMA_10',
        'Away_Offensive_Rating_EMA_5', 'Away_Defensive_Rating_EMA_5',
        'Away_Offensive_Rating_EMA_10', 'Away_Defensive_Rating_EMA_10',
        'Home_Net_Rating_EMA_5', 'Away_Net_Rating_EMA_5',
        'Home_Net_Rating_EMA_10', 'Away_Net_Rating_EMA_10',
        
        # Four Factors (EMA)
        'Home_eFG%_EMA_5', 'Home_TOV%_EMA_5', 'Home_ORB%_EMA_5', 'Home_FT_Rate_EMA_5',
        'Home_eFG%_EMA_10', 'Home_TOV%_EMA_10', 'Home_ORB%_EMA_10', 'Home_FT_Rate_EMA_10',
        'Away_eFG%_EMA_5', 'Away_TOV%_EMA_5', 'Away_ORB%_EMA_5', 'Away_FT_Rate_EMA_5',
        'Away_eFG%_EMA_10', 'Away_TOV%_EMA_10', 'Away_ORB%_EMA_10', 'Away_FT_Rate_EMA_10',
        
        # Rest / Schedule
        'Home_Days_Rest', 'Home_Back_To_Back', 'Home_Three_In_Four',
        'Away_Days_Rest', 'Away_Back_To_Back', 'Away_Three_In_Four',
        'Rest_Diff',
        
        # Referee Stats (EMA)
        'Ref_Pts_EMA', 'Ref_Fouls_EMA', 'Ref_HomeWin_EMA',
        
        # Talent Floor
        'Home_Talent_Floor', 'Away_Talent_Floor', 'Talent_Floor_Diff',
        
        # Market Data
        'BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder',
        'Prob_Home',
        
        # Squad Health
        'Home_Missing_Usage_Pct', 'Away_Missing_Usage_Pct',
        'Home_Missing_BPM_Pct', 'Away_Missing_BPM_Pct',
        'Home_Missing_Minutes_Pct', 'Away_Missing_Minutes_Pct',
        'Home_Injured_Players_Count', 'Away_Injured_Players_Count',
        'Missing_Usage_Diff',
        
        # Differentials and Bias
        'Net_Rating_Diff_5', 'eFG%_Diff_5', 'TOV%_Diff_5', 'ORB%_Diff_5', 'FT_Rate_Diff_5',
        'Net_Rating_Diff_10', 'eFG%_Diff_10', 'TOV%_Diff_10', 'ORB%_Diff_10', 'FT_Rate_Diff_10',
        'H2H_Bias'
    ]
    
    print(f"Total features defined: {len(features)}")
    
    # Verify features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    # 3. Split chronologically
    # Train: seasons 2018 to 2024
    # Validate: seasons 2025 to 2026
    train_mask = df['Season'].between(2018, 2024)
    val_mask = df['Season'].between(2025, 2026)
    
    X_train = df.loc[train_mask, features].reset_index(drop=True)
    y_train = df.loc[train_mask, 'Target'].reset_index(drop=True)
    
    X_val = df.loc[val_mask, features].reset_index(drop=True)
    y_val = df.loc[val_mask, 'Target'].reset_index(drop=True)
    
    print(f"Train set size: {len(X_train)} matches (Seasons 2018-2024)")
    print(f"Validation set size: {len(X_val)} matches (Seasons 2025-2026)")
    
    # Combine train and val for a custom PredefinedSplit
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    
    # PredefinedSplit indicator: -1 for train, 0 for validation
    split_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
    ps = PredefinedSplit(test_fold=split_fold)
    
    # 4. Tune XGBRegressor hyperparameters using RandomizedSearchCV
    param_distributions = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
        'n_estimators': [50, 100, 150, 200, 300, 400],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        'reg_lambda': [1.0, 2.0, 5.0, 10.0, 20.0],
        'min_child_weight': [1, 3, 5, 10]
    }
    
    base_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    
    print("Fine-tuning hyperparameters using RandomizedSearchCV...")
    # Using MAE (neg_mean_absolute_error) for scoring
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=150,
        scoring='neg_mean_absolute_error',
        cv=ps,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_combined, y_combined)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"\nBest Hyperparameters found:")
    print(json.dumps(best_params, indent=4))
    
    # Evaluate model performance
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_wl_accuracy = np.mean((y_train > 0) == (y_train_pred > 0)) * 100.0
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_wl_accuracy = np.mean((y_val > 0) == (y_val_pred > 0)) * 100.0
    
    print(f"\nEvaluation Metrics:")
    print(f"Train MAE:                 {train_mae:.4f}")
    print(f"Train R2:                  {train_r2:.4f}")
    print(f"Train Win/Loss Accuracy:   {train_wl_accuracy:.2f}%")
    print(f"Validation MAE:            {val_mae:.4f}")
    print(f"Validation R2:             {val_r2:.4f}")
    print(f"Validation Win/Loss Acc:   {val_wl_accuracy:.2f}%")
    
    # 5. Compute validation residuals standard deviation
    residuals = y_val - y_val_pred
    sigma_residuals = float(np.std(residuals))
    print(f"Validation Residuals Standard Deviation (sigma): {sigma_residuals:.4f}")
    
    # 6. Save the trained model to wnba_spread_model.pkl
    model_filename = 'wnba_spread_model.pkl'
    print(f"Saving model to {model_filename}...")
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
        
    # Save the feature importances
    importances = best_model.feature_importances_
    feature_importances = {feat: float(imp) for feat, imp in zip(features, importances)}
    # Sort feature importances
    sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
    
    # 7. Save metadata to model_metadata.json
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'features': features,
        'best_hyperparameters': best_params,
        'metrics': {
            'train_mae': float(train_mae),
            'train_r2': float(train_r2),
            'train_wl_accuracy': float(train_wl_accuracy),
            'val_mae': float(val_mae),
            'val_r2': float(val_r2),
            'val_wl_accuracy': float(val_wl_accuracy)
        },
        'sigma_residuals': sigma_residuals,
        'feature_importances': sorted_importances[:20]  # Store top 20 features
    }
    
    metadata_filename = 'model_metadata.json'
    print(f"Saving metadata to {metadata_filename}...")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nFeature Importances (Top 10):")
    for feat, imp in sorted_importances[:10]:
        print(f"  {feat}: {imp:.4f}")
        
    print("\nTraining workflow complete!")

if __name__ == '__main__':
    train_model()
