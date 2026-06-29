import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import os
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, log_loss
from scipy.stats import norm

class WalkForwardSeasonSplitter:
    def __init__(self, seasons_series):
        self.seasons = np.array(seasons_series)
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return 4
        
    def split(self, X, y=None, groups=None):
        # Fold 1: Train 2018-2021, Validate 2022
        # Fold 2: Train 2018-2022, Validate 2023
        # Fold 3: Train 2018-2023, Validate 2024
        # Fold 4: Train 2018-2024, Validate 2025
        folds = [
            ((2018, 2021), 2022),
            ((2018, 2022), 2023),
            ((2018, 2023), 2024),
            ((2018, 2024), 2025)
        ]
        for train_range, val_season in folds:
            train_idx = np.where((self.seasons >= train_range[0]) & (self.seasons <= train_range[1]))[0]
            val_idx = np.where(self.seasons == val_season)[0]
            yield train_idx, val_idx

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
    df['Target'] = df['HomeScore'] - df['AwayScore']
    
    # Define features (including market features, excluding referee features)
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
        
        # Talent Floor
        'Home_Talent_Floor', 'Away_Talent_Floor', 'Talent_Floor_Diff',
        
        # Squad Health
        'Home_Missing_Usage_Pct', 'Away_Missing_Usage_Pct',
        'Home_Missing_BPM_Pct', 'Away_Missing_BPM_Pct',
        'Home_Missing_Minutes_Pct', 'Away_Missing_Minutes_Pct',
        'Home_Injured_Players_Count', 'Away_Injured_Players_Count',
        'Missing_Usage_Diff',
        
        # Differentials and Bias
        'Net_Rating_Diff_5', 'eFG%_Diff_5', 'TOV%_Diff_5', 'ORB%_Diff_5', 'FT_Rate_Diff_5',
        'Net_Rating_Diff_10', 'eFG%_Diff_10', 'TOV%_Diff_10', 'ORB%_Diff_10', 'FT_Rate_Diff_10',
        'H2H_Bias',
        
        # Referee Features
        'Ref_Pts_EMA', 'Ref_Fouls_EMA', 'Ref_HomeWin_EMA',
        
        # Market Features
        'BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder', 'Prob_Home'
    ]
    
    print(f"Total features defined: {len(features)}")
    
    # Verify features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
        
    # 3. Prepare data for tuning on seasons 2018-2026 (split at 2026-06-01)
    train_val_mask = df['Date'] < '2026-06-01'
    train_val_df = df[train_val_mask].copy().reset_index(drop=True)
    test_df = df[~train_val_mask].copy().reset_index(drop=True)
    
    if test_df.empty:
        raise ValueError("Held-out test set (Date >= 2026-06-01) is empty.")
        
    X_train_val = train_val_df[features]
    y_reg_train_val = train_val_df['Target']
    y_clf_train_val = (y_reg_train_val > 0).astype(int)
    
    # Calculate sample weights
    max_train_date = train_val_df['Date'].max()
    days_diff = (pd.to_datetime(max_train_date) - pd.to_datetime(train_val_df['Date'])).dt.days
    train_val_weights = np.maximum(0.2, np.exp(-0.000551 * days_diff)).values
    
    print(f"Tuning dataset size (Date < 2026-06-01): {len(train_val_df)} matches")
    print(f"June 2026 test set size (Date >= 2026-06-01): {len(test_df)} matches")
    
    # 4. Hyperparameter tuning using RandomizedSearchCV and Custom Splitter
    cv_splitter = WalkForwardSeasonSplitter(train_val_df['Season'])
    
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
    
    print("Fine-tuning XGBRegressor...")
    base_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    random_search_reg = RandomizedSearchCV(
        estimator=base_reg,
        param_distributions=param_distributions,
        n_iter=60,
        scoring='neg_mean_absolute_error',
        cv=cv_splitter,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search_reg.fit(X_train_val, y_reg_train_val, sample_weight=train_val_weights)
    best_reg = random_search_reg.best_estimator_
    best_reg_params = random_search_reg.best_params_
    print(f"Best Regressor Params: {best_reg_params}")
    
    print("Fine-tuning XGBClassifier...")
    base_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
    random_search_clf = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_distributions,
        n_iter=60,
        scoring='neg_log_loss',
        cv=cv_splitter,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    random_search_clf.fit(X_train_val, y_clf_train_val, sample_weight=train_val_weights)
    best_clf = random_search_clf.best_estimator_
    best_clf_params = random_search_clf.best_params_
    print(f"Best Classifier Params: {best_clf_params}")
    
    # 5. Evaluate on held-out June 2026 test set
    X_test = test_df[features]
    y_reg_test = test_df['Target']
    y_clf_test = (y_reg_test > 0).astype(int)
    
    # Regressor predictions
    y_reg_pred = best_reg.predict(X_test)
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    reg_accuracy = np.mean((y_reg_test > 0) == (y_reg_pred > 0)) * 100.0
    
    # Calculate sigma of residuals on the training-validation data to convert regressor predictions to probabilities
    y_reg_cv_pred = best_reg.predict(X_train_val)
    residuals_cv = y_reg_train_val - y_reg_cv_pred
    sigma_residuals_cv = float(np.std(residuals_cv))
    
    # Probability of home win from regressor (using normal CDF)
    y_reg_prob = norm.cdf(y_reg_pred / sigma_residuals_cv)
    reg_logloss = log_loss(y_clf_test, y_reg_prob)
    
    # Classifier predictions
    y_clf_prob = best_clf.predict_proba(X_test)[:, 1]
    clf_logloss = log_loss(y_clf_test, y_clf_prob)
    clf_accuracy = np.mean(y_clf_test == (y_clf_prob >= 0.5)) * 100.0
    
    print("\n================ JUNE 2026 HELD-OUT TEST METRICS ================")
    print(f"XGBRegressor (objective='reg:squarederror'):")
    print(f"  MAE:                    {reg_mae:.4f}")
    print(f"  Win/Loss Accuracy:      {reg_accuracy:.2f}%")
    print(f"  LogLoss (implied CDF):  {reg_logloss:.4f}")
    print(f"XGBClassifier (objective='binary:logistic'):")
    print(f"  Win/Loss Accuracy:      {clf_accuracy:.2f}%")
    print(f"  LogLoss:                {clf_logloss:.4f}")
    print("================================================================")
    
    # 6. Re-fit on the full historical dataset (2018-present)
    refit_df = df.copy().reset_index(drop=True)
    X_refit = refit_df[features]
    y_reg_refit = refit_df['Target']
    y_clf_refit = (y_reg_refit > 0).astype(int)
    
    max_refit_date = refit_df['Date'].max()
    days_diff_refit = (pd.to_datetime(max_refit_date) - pd.to_datetime(refit_df['Date'])).dt.days
    refit_weights = np.maximum(0.2, np.exp(-0.000551 * days_diff_refit)).values
    
    print(f"\nRe-fitting best estimators on entire historical dataset ({len(refit_df)} matches) using sample weights...")
    best_reg.fit(X_refit, y_reg_refit, sample_weight=refit_weights)
    best_clf.fit(X_refit, y_clf_refit, sample_weight=refit_weights)
    
    # Recalculate sigma of residuals on the re-fitted historical dataset
    y_reg_refit_pred = best_reg.predict(X_refit)
    refit_residuals = y_reg_refit - y_reg_refit_pred
    sigma_residuals_refit = float(np.std(refit_residuals))
    print(f"Re-fitted Residuals Standard Deviation (sigma): {sigma_residuals_refit:.4f}")
    
    # 7. Save both models as a dictionary in wnba_spread_model.pkl
    model_filename = 'wnba_spread_model.pkl'
    print(f"Saving models to {model_filename}...")
    ensemble_dict = {
        'regressor': best_reg,
        'classifier': best_clf
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(ensemble_dict, f)
        
    # Get feature importances
    reg_importances = best_reg.feature_importances_
    reg_feat_imp = sorted(
        {feat: float(imp) for feat, imp in zip(features, reg_importances)}.items(),
        key=lambda item: item[1],
        reverse=True
    )
    clf_importances = best_clf.feature_importances_
    clf_feat_imp = sorted(
        {feat: float(imp) for feat, imp in zip(features, clf_importances)}.items(),
        key=lambda item: item[1],
        reverse=True
    )
    
    # 8. Save metadata to model_metadata.json
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'features': features,
        'best_hyperparameters': {
            'regressor': best_reg_params,
            'classifier': best_clf_params
        },
        'metrics': {
            'test_june_2026': {
                'regressor_mae': float(reg_mae),
                'regressor_logloss': float(reg_logloss),
                'regressor_accuracy': float(reg_accuracy),
                'classifier_logloss': float(clf_logloss),
                'classifier_accuracy': float(clf_accuracy)
            }
        },
        'sigma_residuals': sigma_residuals_refit,
        'feature_importances': {
            'regressor': reg_feat_imp[:20],
            'classifier': clf_feat_imp[:20]
        }
    }
    
    metadata_filename = 'model_metadata.json'
    print(f"Saving metadata to {metadata_filename}...")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nFeature Importances (Top 10 Regressor):")
    for feat, imp in reg_feat_imp[:10]:
        print(f"  {feat}: {imp:.4f}")
        
    print("\nTraining workflow complete!")

if __name__ == '__main__':
    train_model()
