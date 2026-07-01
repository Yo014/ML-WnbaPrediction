import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, log_loss
from scipy.stats import norm

from stacking_models import StackedEnsembleRegressor, StackedEnsembleClassifier

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
    
    # Define Target = HomeScore - AwayScore
    df['Target'] = df['HomeScore'] - df['AwayScore']
    
    # 2. Define the features and feature sets
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
    
    market_features = ['BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder', 'Prob_Home']
    
    baseline_features = [f for f in features if f not in market_features]
    full_features = baseline_features + market_features
    
    print(f"Total features defined: {len(features)}")
    print(f"Baseline features: {len(baseline_features)}, Full features: {len(full_features)}")
    
    # Verify features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
        
    # 3. Prepare data for tuning/training (chronological split)
    # Train/validation before 2026-06-01, test after 2026-06-01
    train_val_mask = df['Date'] < '2026-06-01'
    train_val_df = df[train_val_mask].copy().reset_index(drop=True)
    test_df = df[~train_val_mask].copy().reset_index(drop=True)
    
    if test_df.empty:
        raise ValueError("Held-out test set (Date >= 2026-06-01) is empty.")
        
    # Fill NaNs using the mean of the training-validation set to prevent look-ahead bias and handle scikit-learn estimators
    feature_means = {}
    for col in features:
        mean_val = train_val_df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        feature_means[col] = float(mean_val)
        train_val_df[col] = train_val_df[col].fillna(mean_val)
        test_df[col] = test_df[col].fillna(mean_val)
        df[col] = df[col].fillna(mean_val)
        
    # Calculate sample weights
    max_train_date = train_val_df['Date'].max()
    days_diff = (pd.to_datetime(max_train_date) - pd.to_datetime(train_val_df['Date'])).dt.days
    train_val_weights = np.maximum(0.2, np.exp(-0.000551 * days_diff)).values
    
    print(f"Tuning dataset size (Date < 2026-06-01): {len(train_val_df)} matches")
    print(f"June 2026 test set size (Date >= 2026-06-01): {len(test_df)} matches")
    
    # Generate custom splits
    cv_splitter = WalkForwardSeasonSplitter(train_val_df['Season'])
    cv_splits = list(cv_splitter.split(train_val_df))
    
    # 4. Train models
    print("\nTraining Stage 1 Stacking Models on baseline_features...")
    y_reg_train_val = train_val_df['Target'].values
    y_clf_train_val = (y_reg_train_val > 0).astype(int)
    
    stage1_reg = StackedEnsembleRegressor()
    stage1_reg.fit(train_val_df[baseline_features], y_reg_train_val, cv_splits, sample_weight=train_val_weights)
    
    stage1_clf = StackedEnsembleClassifier()
    stage1_clf.fit(train_val_df[baseline_features], y_clf_train_val, cv_splits, sample_weight=train_val_weights)
    
    print("\nTraining Stage 2 Stacking Models (Residual & Direct Classification) on full_features...")
    y_residual_train_val = y_reg_train_val - train_val_df['ClosingSpread'].values
    
    stage2_reg = StackedEnsembleRegressor()
    stage2_reg.fit(train_val_df[full_features], y_residual_train_val, cv_splits, sample_weight=train_val_weights)
    
    stage2_clf = StackedEnsembleClassifier()
    stage2_clf.fit(train_val_df[full_features], y_clf_train_val, cv_splits, sample_weight=train_val_weights)
    
    print("\nTraining Stage 2 Quantile Regressors (LGBM) on full_features...")
    quantile_10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, random_state=42, verbose=-1)
    quantile_10.fit(train_val_df[full_features], y_reg_train_val, sample_weight=train_val_weights)
    
    quantile_90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, random_state=42, verbose=-1)
    quantile_90.fit(train_val_df[full_features], y_reg_train_val, sample_weight=train_val_weights)
    
    # 5. Evaluate all models on the held-out June 2026 test set
    X_test_baseline = test_df[baseline_features]
    X_test_full = test_df[full_features]
    y_test_reg = test_df['Target'].values
    y_test_clf = (y_test_reg > 0).astype(int)
    
    # Baseline (Market)
    market_margin = -test_df['ClosingSpread'].values
    market_prob = test_df['Prob_Home'].values
    market_mae = mean_absolute_error(y_test_reg, market_margin)
    market_accuracy = np.mean(y_test_clf == (market_prob >= 0.5)) * 100.0
    market_logloss = log_loss(y_test_clf, market_prob)
    
    # Stage 1 Stacking
    stage1_pred_margin = stage1_reg.predict(X_test_baseline)
    stage1_prob_clf = stage1_clf.predict_proba(X_test_baseline)[:, 1]
    stage1_mae = mean_absolute_error(y_test_reg, stage1_pred_margin)
    stage1_clf_accuracy = np.mean(y_test_clf == (stage1_prob_clf >= 0.5)) * 100.0
    stage1_logloss = log_loss(y_test_clf, stage1_prob_clf)
    
    # Stage 2 Stacking & Volatility
    stage2_residual_pred = stage2_reg.predict(X_test_full)
    stage2_predicted_margin = test_df['ClosingSpread'].values + stage2_residual_pred
    
    p10_pred = quantile_10.predict(X_test_full)
    p90_pred = quantile_90.predict(X_test_full)
    
    # Volatility standard deviation: sigma_pred = (p90 - p10) / 2.563
    sigma_pred = (p90_pred - p10_pred) / 2.563
    sigma_pred = np.maximum(sigma_pred, 1e-3)  # Prevent division by zero
    
    # CDF-derived probability: P_CDF = norm.cdf(predicted_margin / sigma_pred)
    P_CDF = norm.cdf(stage2_predicted_margin / sigma_pred)
    
    # Stage 2 Classifier probability
    P_Clf = stage2_clf.predict_proba(X_test_full)[:, 1]
    
    # Blend P_CDF 50/50 with Stage 2 Classifier
    final_win_prob = 0.5 * P_CDF + 0.5 * P_Clf
    
    stage2_mae = mean_absolute_error(y_test_reg, stage2_predicted_margin)
    stage2_clf_accuracy = np.mean(y_test_clf == (final_win_prob >= 0.5)) * 100.0
    stage2_logloss = log_loss(y_test_clf, final_win_prob)
    
    # Print metrics table
    print("\n" + "="*80)
    print(f"{'MODEL EVALUATION SUMMARY (JUNE 2026 TEST SET)':^80}")
    print("="*80)
    print(f"{'Model / Metric':<35} | {'MAE':^12} | {'Accuracy (%)':^15} | {'Log Loss':^12}")
    print("-"*80)
    print(f"{'Market Baseline (Closing Lines)':<35} | {market_mae:^12.4f} | {market_accuracy:^15.2f} | {market_logloss:^12.4f}")
    print(f"{'Stage 1 Stacking (Baseline Features)':<35} | {stage1_mae:^12.4f} | {stage1_clf_accuracy:^15.2f} | {stage1_logloss:^12.4f}")
    print(f"{'Stage 2 Stacking (Two-Stage + Vol)':<35} | {stage2_mae:^12.4f} | {stage2_clf_accuracy:^15.2f} | {stage2_logloss:^12.4f}")
    print("="*80)
    
    # 6. Re-fit all best estimators on the full historical dataset (2018–2026) using sample weights
    refit_df = df.copy().reset_index(drop=True)
    X_refit_baseline = refit_df[baseline_features]
    X_refit_full = refit_df[full_features]
    y_reg_refit = refit_df['Target'].values
    y_clf_refit = (y_reg_refit > 0).astype(int)
    
    max_refit_date = refit_df['Date'].max()
    days_diff_refit = (pd.to_datetime(max_refit_date) - pd.to_datetime(refit_df['Date'])).dt.days
    refit_weights = np.maximum(0.2, np.exp(-0.000551 * days_diff_refit)).values
    
    print(f"\nRe-fitting models on entire historical dataset ({len(refit_df)} matches) using sample weights...")
    cv_splitter_refit = WalkForwardSeasonSplitter(refit_df['Season'])
    cv_splits_refit = list(cv_splitter_refit.split(refit_df))
    
    # Stage 1 Stacking Re-fit
    stage1_reg.fit(X_refit_baseline, y_reg_refit, cv_splits_refit, sample_weight=refit_weights)
    stage1_clf.fit(X_refit_baseline, y_clf_refit, cv_splits_refit, sample_weight=refit_weights)
    
    # Stage 2 Stacking Re-fit
    y_residual_refit = y_reg_refit - refit_df['ClosingSpread'].values
    stage2_reg.fit(X_refit_full, y_residual_refit, cv_splits_refit, sample_weight=refit_weights)
    stage2_clf.fit(X_refit_full, y_clf_refit, cv_splits_refit, sample_weight=refit_weights)
    
    # Quantile Re-fit
    quantile_10.fit(X_refit_full, y_reg_refit, sample_weight=refit_weights)
    quantile_90.fit(X_refit_full, y_reg_refit, sample_weight=refit_weights)
    
    # Calculate traditional standard deviation of residuals on refitted full dataset
    stage2_pred_residual_refit = stage2_reg.predict(X_refit_full)
    refit_residuals = y_reg_refit - (refit_df['ClosingSpread'].values + stage2_pred_residual_refit)
    sigma_residuals_refit = float(np.std(refit_residuals))
    print(f"Re-fitted Residuals Standard Deviation (sigma): {sigma_residuals_refit:.4f}")
    
    # 7. Save the trained models in a dictionary structure inside wnba_spread_model.pkl
    model_filename = 'wnba_spread_model.pkl'
    print(f"Saving models to {model_filename}...")
    model_dict = {
        'stage1_regressor': stage1_reg,
        'stage1_classifier': stage1_clf,
        'stage2_regressor': stage2_reg,
        'stage2_classifier': stage2_clf,
        'quantile_10': quantile_10,
        'quantile_90': quantile_90
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
        
    # Get feature importances from Stage 2 base models (XGBoost)
    try:
        reg_importances = stage2_reg.base_models_[0].feature_importances_
        reg_feat_imp = sorted(
            {feat: float(imp) for feat, imp in zip(full_features, reg_importances)}.items(),
            key=lambda item: item[1],
            reverse=True
        )
    except Exception as e:
        print(f"Could not compute regressor feature importances: {e}")
        reg_feat_imp = []

    try:
        clf_importances = stage2_clf.base_models_[0].feature_importances_
        clf_feat_imp = sorted(
            {feat: float(imp) for feat, imp in zip(full_features, clf_importances)}.items(),
            key=lambda item: item[1],
            reverse=True
        )
    except Exception as e:
        print(f"Could not compute classifier feature importances: {e}")
        clf_feat_imp = []
        
    # 8. Save features list and other metadata to model_metadata.json
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'feature_means': feature_means,
        'features': full_features,
        'baseline_features': baseline_features,
        'full_features': full_features,
        'metrics': {
            'test_june_2026': {
                'market_mae': float(market_mae),
                'market_accuracy': float(market_accuracy),
                'market_logloss': float(market_logloss),
                'stage1_mae': float(stage1_mae),
                'stage1_accuracy': float(stage1_clf_accuracy),
                'stage1_logloss': float(stage1_logloss),
                'stage2_mae': float(stage2_mae),
                'stage2_accuracy': float(stage2_clf_accuracy),
                'stage2_logloss': float(stage2_logloss)
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
        
    print("\nFeature Importances (Top 10 Regressor from Stage 2 base XGBRegressor):")
    for feat, imp in reg_feat_imp[:10]:
        print(f"  {feat}: {imp:.4f}")
        
    print("\nTraining workflow complete!")

if __name__ == '__main__':
    train_model()
