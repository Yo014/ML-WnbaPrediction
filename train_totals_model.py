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
from sklearn.isotonic import IsotonicRegression

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

def get_walk_forward_splits(seasons_series, min_train_seasons=2):
    seasons_arr = np.array(seasons_series)
    unique_seasons = sorted(list(np.unique(seasons_arr)))
    splits = []
    for i in range(min_train_seasons, len(unique_seasons)):
        train_seasons = unique_seasons[:i]
        val_season = unique_seasons[i]
        train_idx = np.where(np.isin(seasons_arr, train_seasons))[0]
        val_idx = np.where(seasons_arr == val_season)[0]
        splits.append((train_idx, val_idx))
    return splits

def train_totals_model():
    # 1. Read the engineered dataset
    data_path = 'ml_ready_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Sort chronologically to make sure it is in order
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Define Target_Total = HomeScore + AwayScore, and decoupled targets
    df['Target_Total'] = df['HomeScore'] + df['AwayScore']
    df['Target_Pace'] = (df['HomePossessions'] + df['AwayPossessions']) / 2.0
    possessions = df['Target_Pace'].replace(0, np.nan)
    df['Target_Home_Eff'] = (100.0 * df['HomeScore'] / possessions).fillna(0.0)
    df['Target_Away_Eff'] = (100.0 * df['AwayScore'] / possessions).fillna(0.0)
    
    # 2. Define the features and feature sets
    features = [
        # Ratings (EMA)
        'Home_Offensive_Rating_EMA_5', 'Home_Defensive_Rating_EMA_5',
        'Home_Offensive_Rating_EMA_10', 'Home_Defensive_Rating_EMA_10',
        'Away_Offensive_Rating_EMA_5', 'Away_Defensive_Rating_EMA_5',
        'Away_Offensive_Rating_EMA_10', 'Away_Defensive_Rating_EMA_10',
        'Home_Net_Rating_EMA_5', 'Away_Net_Rating_EMA_5',
        'Home_Net_Rating_EMA_10', 'Away_Net_Rating_EMA_10',
        
        # Pace (EMA)
        'Home_Pace_EMA_5', 'Home_Pace_EMA_10',
        'Away_Pace_EMA_5', 'Away_Pace_EMA_10',
        
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
        'BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder', 'Prob_Home', 'Market_Disagreement'
    ]
    
    market_features = ['BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder', 'Prob_Home', 'Market_Disagreement']
    baseline_features = [f for f in features if f not in market_features]
    full_features = baseline_features + market_features
    
    print(f"Total features defined: {len(features)}")
    print(f"Baseline features: {len(baseline_features)}, Full features: {len(full_features)}")
    
    # Verify features exist in dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
        
    # 3. Prepare data for tuning/training (chronological split)
    train_val_mask = df['Date'] < '2026-06-01'
    train_val_df = df[train_val_mask].copy().reset_index(drop=True)
    test_df = df[~train_val_mask].copy().reset_index(drop=True)
    
    if test_df.empty:
        raise ValueError("Held-out test set (Date >= 2026-06-01) is empty.")
        
    # Fill NaNs using the mean of the training-validation set to prevent look-ahead bias
    feature_means = {}
    for col in features:
        mean_val = train_val_df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        feature_means[col] = float(mean_val)
        train_val_df[col] = train_val_df[col].fillna(mean_val)
        test_df[col] = test_df[col].fillna(mean_val)
        df[col] = df[col].fillna(mean_val)
        
    print(f"Tuning dataset size (Date < 2026-06-01): {len(train_val_df)} matches")
    print(f"June 2026 test set size (Date >= 2026-06-01): {len(test_df)} matches")
    
    # Generate custom splits
    cv_splitter = WalkForwardSeasonSplitter(train_val_df['Season'])
    cv_splits = list(cv_splitter.split(train_val_df))
    
    # Calculate days diff for sample weights calculation
    max_train_date = train_val_df['Date'].max()
    days_diff = (pd.to_datetime(max_train_date) - pd.to_datetime(train_val_df['Date'])).dt.days
    
    # 4. Baseline Feature Selection
    print("\nPerforming baseline feature selection using a preliminary XGBRegressor for totals...")
    prelim_weights = np.maximum(0.2, np.exp(-0.000551 * days_diff)).values
    prelim_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0)
    prelim_xgb.fit(train_val_df[full_features], train_val_df['Target_Total'].values, sample_weight=prelim_weights)
    importances = prelim_xgb.feature_importances_
    
    selected_features = [feat for feat, imp in zip(full_features, importances) if imp >= 0.002]
    if not selected_features:
        selected_features = list(full_features)
        
    selected_baseline_features = [f for f in selected_features if f in baseline_features]
    selected_full_features = [f for f in selected_features if f in full_features]
    
    print(f"Selected {len(selected_features)} features (out of {len(full_features)}) with importance >= 0.002")
    print(f"Selected Baseline features: {len(selected_baseline_features)}")
    print(f"Selected Full features: {len(selected_full_features)}")
    
    # 5. Dynamic Decay Parameter Optimization (Grid Search)
    lambdas = [0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.0015, 0.002]
    best_log_loss = float('inf')
    best_lambda = None
    best_oof_s1 = None
    best_oof_s2 = None
    best_oof_y_s1 = None
    best_oof_y_s2 = None
    best_s1_total_pred_train_val = None
    
    for lmbda in lambdas:
        print(f"\nEvaluating decay lambda = {lmbda}...")
        
        # Initialize validation arrays for out-of-fold predictions
        oof_stage1_probs = np.zeros(len(train_val_df))
        oof_stage2_probs = np.zeros(len(train_val_df))
        s1_total_pred_oof = np.zeros(len(train_val_df))
        val_indices_seen = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            train_fold_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
            
            # Calculate weights
            max_fold_date = train_fold_df['Date'].max()
            fold_days_diff = (pd.to_datetime(max_fold_date) - pd.to_datetime(train_fold_df['Date'])).dt.days
            w_fold = np.maximum(0.2, np.exp(-lmbda * fold_days_diff)).values
            
            # Nested Walk-Forward Splits
            nested_splits = get_walk_forward_splits(train_fold_df['Season'], min_train_seasons=2)
            if len(nested_splits) == 0:
                n_samples = len(train_fold_df)
                split_idx = int(0.8 * n_samples)
                nested_splits = [(np.arange(split_idx), np.arange(split_idx, n_samples))]
                
            y_tr_reg = train_fold_df['Target_Total'].values
            y_tr_pace = train_fold_df['Target_Pace'].values
            y_tr_home_eff = train_fold_df['Target_Home_Eff'].values
            y_tr_away_eff = train_fold_df['Target_Away_Eff'].values
            
            median_total = np.median(y_tr_reg)
            y_tr_clf_s1 = (y_tr_reg > median_total).astype(int)
            y_tr_clf_s2 = (y_tr_reg > train_fold_df['OverUnder'].values).astype(int)
            
            # Fit Stage 1 Models (Baseline)
            fold_stage1_pace_reg = StackedEnsembleRegressor()
            fold_stage1_pace_reg.fit(train_fold_df[selected_baseline_features], y_tr_pace, nested_splits, sample_weight=w_fold)
            
            fold_stage1_home_eff_reg = StackedEnsembleRegressor()
            fold_stage1_home_eff_reg.fit(train_fold_df[selected_baseline_features], y_tr_home_eff, nested_splits, sample_weight=w_fold)
            
            fold_stage1_away_eff_reg = StackedEnsembleRegressor()
            fold_stage1_away_eff_reg.fit(train_fold_df[selected_baseline_features], y_tr_away_eff, nested_splits, sample_weight=w_fold)
            
            fold_stage1_clf = StackedEnsembleClassifier()
            fold_stage1_clf.fit(train_fold_df[selected_baseline_features], y_tr_clf_s1, nested_splits, sample_weight=w_fold)
            
            # Generate s1_total_pred for train_fold_df (in-sample for Fold 0)
            if fold_idx == 0:
                X_train_baseline = train_fold_df[selected_baseline_features]
                pace_pred_tr = fold_stage1_pace_reg.predict(X_train_baseline)
                home_eff_pred_tr = fold_stage1_home_eff_reg.predict(X_train_baseline)
                away_eff_pred_tr = fold_stage1_away_eff_reg.predict(X_train_baseline)
                s1_total_pred_oof[train_idx] = pace_pred_tr * (home_eff_pred_tr + away_eff_pred_tr) / 100.0
                
            # Assign s1_total_pred to train_fold_df
            train_fold_df['s1_total_pred'] = s1_total_pred_oof[train_idx]
            selected_full_features_s2 = selected_full_features + ['s1_total_pred']
            
            # Fit Stage 2 Models (Residuals)
            y_residual_tr = y_tr_reg - train_fold_df['OverUnder'].values
            
            fold_stage2_reg = StackedEnsembleRegressor()
            fold_stage2_reg.fit(train_fold_df[selected_full_features_s2], y_residual_tr, nested_splits, sample_weight=w_fold)
            
            fold_stage2_clf = StackedEnsembleClassifier()
            fold_stage2_clf.fit(train_fold_df[selected_full_features_s2], y_tr_clf_s2, nested_splits, sample_weight=w_fold)
            
            # Fit Quantile Regressors
            fold_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, random_state=42, verbose=-1)
            fold_q10.fit(train_fold_df[selected_full_features_s2], y_tr_reg, sample_weight=w_fold)
            
            fold_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, random_state=42, verbose=-1)
            fold_q90.fit(train_fold_df[selected_full_features_s2], y_tr_reg, sample_weight=w_fold)
            
            # Predict on val_idx
            X_val_baseline = train_val_df.iloc[val_idx][selected_baseline_features]
            X_val_full = train_val_df.iloc[val_idx][selected_full_features]
            
            # Stage 1 prediction
            pace_pred_val = fold_stage1_pace_reg.predict(X_val_baseline)
            home_eff_pred_val = fold_stage1_home_eff_reg.predict(X_val_baseline)
            away_eff_pred_val = fold_stage1_away_eff_reg.predict(X_val_baseline)
            s1_total_pred_val = pace_pred_val * (home_eff_pred_val + away_eff_pred_val) / 100.0
            s1_total_pred_oof[val_idx] = s1_total_pred_val
            
            s1_prob_clf = fold_stage1_clf.predict_proba(X_val_baseline)[:, 1]
            s1_sigma = 12.0 # baseline total volatility estimate
            s1_prob_cdf = norm.cdf((s1_total_pred_val - median_total) / s1_sigma)
            s1_prob = 0.5 * s1_prob_cdf + 0.5 * s1_prob_clf
            oof_stage1_probs[val_idx] = s1_prob
            
            # Stage 2 prediction
            X_val_full_s2 = X_val_full.copy()
            X_val_full_s2['s1_total_pred'] = s1_total_pred_val
            
            s2_residual_pred = fold_stage2_reg.predict(X_val_full_s2)
            p10_pred = fold_q10.predict(X_val_full_s2)
            p90_pred = fold_q90.predict(X_val_full_s2)
            
            sigma_pred = (p90_pred - p10_pred) / 2.563
            sigma_pred = np.maximum(sigma_pred, 1e-3)
            
            P_CDF = norm.cdf(s2_residual_pred / sigma_pred)
            P_Clf = fold_stage2_clf.predict_proba(X_val_full_s2)[:, 1]
            s2_prob = 0.5 * P_CDF + 0.5 * P_Clf
            oof_stage2_probs[val_idx] = s2_prob
            
            val_indices_seen.extend(val_idx)
            
        val_indices_seen = sorted(list(set(val_indices_seen)))
        y_val_clf_s2 = (train_val_df.iloc[val_indices_seen]['Target_Total'].values > train_val_df.iloc[val_indices_seen]['OverUnder'].values).astype(int)
        
        loss = log_loss(y_val_clf_s2, oof_stage2_probs[val_indices_seen])
        print(f"Lambda = {lmbda} -> Validation Log Loss (Stage 2): {loss:.6f}")
        
        if loss < best_log_loss:
            best_log_loss = loss
            best_lambda = lmbda
            best_s1_total_pred_train_val = s1_total_pred_oof.copy()
            
            # For Stage 1 target we map against baseline median
            median_val_set = train_val_df.iloc[val_indices_seen]['Target_Total'].median()
            y_val_clf_s1 = (train_val_df.iloc[val_indices_seen]['Target_Total'].values > median_val_set).astype(int)
            best_oof_s1 = oof_stage1_probs[val_indices_seen]
            best_oof_s2 = oof_stage2_probs[val_indices_seen]
            best_oof_y_s1 = y_val_clf_s1
            best_oof_y_s2 = y_val_clf_s2
            
    print(f"\nOptimal lambda: {best_lambda} (Validation Log Loss: {best_log_loss:.6f})")
    
    # 6. Fit Isotonic calibrators
    print("\nFitting Isotonic Calibrators on Out-of-Fold blended probabilities...")
    stage1_calibrator = IsotonicRegression(out_of_bounds='clip')
    stage1_calibrator.fit(best_oof_s1, best_oof_y_s1)
    
    stage2_calibrator = IsotonicRegression(out_of_bounds='clip')
    stage2_calibrator.fit(best_oof_s2, best_oof_y_s2)
    
    # 7. Train final models using optimal lambda
    print(f"\nTraining final models on entire tuning dataset with optimal decay = {best_lambda}...")
    train_val_weights = np.maximum(0.2, np.exp(-best_lambda * days_diff)).values
    y_reg_train_val = train_val_df['Target_Total'].values
    y_reg_train_val_pace = train_val_df['Target_Pace'].values
    y_reg_train_val_home_eff = train_val_df['Target_Home_Eff'].values
    y_reg_train_val_away_eff = train_val_df['Target_Away_Eff'].values
    
    median_train_val = np.median(y_reg_train_val)
    y_clf_train_val_s1 = (y_reg_train_val > median_train_val).astype(int)
    y_clf_train_val_s2 = (y_reg_train_val > train_val_df['OverUnder'].values).astype(int)
    
    stage1_pace_reg = StackedEnsembleRegressor()
    stage1_pace_reg.fit(train_val_df[selected_baseline_features], y_reg_train_val_pace, cv_splits, sample_weight=train_val_weights)
    
    stage1_home_eff_reg = StackedEnsembleRegressor()
    stage1_home_eff_reg.fit(train_val_df[selected_baseline_features], y_reg_train_val_home_eff, cv_splits, sample_weight=train_val_weights)
    
    stage1_away_eff_reg = StackedEnsembleRegressor()
    stage1_away_eff_reg.fit(train_val_df[selected_baseline_features], y_reg_train_val_away_eff, cv_splits, sample_weight=train_val_weights)
    
    stage1_clf = StackedEnsembleClassifier()
    stage1_clf.fit(train_val_df[selected_baseline_features], y_clf_train_val_s1, cv_splits, sample_weight=train_val_weights)
    
    # Add s1_total_pred to train_val_df
    train_val_df['s1_total_pred'] = best_s1_total_pred_train_val
    selected_full_features_s2 = selected_full_features + ['s1_total_pred']
    
    y_residual_train_val = y_reg_train_val - train_val_df['OverUnder'].values
    stage2_reg = StackedEnsembleRegressor()
    stage2_reg.fit(train_val_df[selected_full_features_s2], y_residual_train_val, cv_splits, sample_weight=train_val_weights)
    
    stage2_clf = StackedEnsembleClassifier()
    stage2_clf.fit(train_val_df[selected_full_features_s2], y_clf_train_val_s2, cv_splits, sample_weight=train_val_weights)
    
    quantile_10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, random_state=42, verbose=-1)
    quantile_10.fit(train_val_df[selected_full_features_s2], y_reg_train_val, sample_weight=train_val_weights)
    
    quantile_90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, random_state=42, verbose=-1)
    quantile_90.fit(train_val_df[selected_full_features_s2], y_reg_train_val, sample_weight=train_val_weights)
    
    # 8. Evaluate on the held-out June 2026 test set
    X_test_baseline = test_df[selected_baseline_features]
    X_test_full = test_df[selected_full_features]
    y_test_reg = test_df['Target_Total'].values
    y_test_clf_s2 = (y_test_reg > test_df['OverUnder'].values).astype(int)
    
    # Baseline Market (Closing OU)
    market_mae = mean_absolute_error(y_test_reg, test_df['OverUnder'].values)
    # Using 50% baseline for market accuracy
    market_accuracy = 50.0
    market_logloss = 0.693
    
    # Compute s1_total_pred for test set
    pace_pred_test = stage1_pace_reg.predict(X_test_baseline)
    home_eff_pred_test = stage1_home_eff_reg.predict(X_test_baseline)
    away_eff_pred_test = stage1_away_eff_reg.predict(X_test_baseline)
    s1_total_pred_test = pace_pred_test * (home_eff_pred_test + away_eff_pred_test) / 100.0
    
    # Stage 2 predictions
    X_test_full_s2 = X_test_full.copy()
    X_test_full_s2['s1_total_pred'] = s1_total_pred_test
    
    stage2_residual_pred = stage2_reg.predict(X_test_full_s2)
    stage2_predicted_total = test_df['OverUnder'].values + stage2_residual_pred
    p10_pred = quantile_10.predict(X_test_full_s2)
    p90_pred = quantile_90.predict(X_test_full_s2)
    
    sigma_pred = (p90_pred - p10_pred) / 2.563
    sigma_pred = np.maximum(sigma_pred, 1e-3)
    
    P_CDF = norm.cdf(stage2_residual_pred / sigma_pred)
    P_Clf = stage2_clf.predict_proba(X_test_full_s2)[:, 1]
    
    final_over_prob = 0.5 * P_CDF + 0.5 * P_Clf
    stage2_mae = mean_absolute_error(y_test_reg, stage2_predicted_total)
    stage2_clf_accuracy = np.mean(y_test_clf_s2 == (final_over_prob >= 0.5)) * 100.0
    stage2_logloss = log_loss(y_test_clf_s2, final_over_prob)
    
    # Calibrated Stage 2
    stage2_prob_cal = stage2_calibrator.predict(final_over_prob)
    stage2_cal_accuracy = np.mean(y_test_clf_s2 == (stage2_prob_cal >= 0.5)) * 100.0
    stage2_cal_logloss = log_loss(y_test_clf_s2, stage2_prob_cal)
    
    print("\n" + "="*95)
    print(f"{'TOTALS MODEL EVALUATION SUMMARY (JUNE 2026 TEST SET)':^95}")
    print("="*95)
    print(f"{'Model / Metric':<45} | {'MAE':^12} | {'Accuracy (%)':^15} | {'Log Loss':^12}")
    print("-"*95)
    print(f"{'Market Closing OverUnder Line':<45} | {market_mae:^12.4f} | {market_accuracy:^15.2f} | {market_logloss:^12.4f}")
    print(f"{'Stage 2 Totals Stacking (Uncalibrated)':<45} | {stage2_mae:^12.4f} | {stage2_clf_accuracy:^15.2f} | {stage2_logloss:^12.4f}")
    print(f"{'Stage 2 Totals Stacking (Calibrated)':<45} | {stage2_mae:^12.4f} | {stage2_cal_accuracy:^15.2f} | {stage2_cal_logloss:^12.4f}")
    print("="*95)
    
    # 9. Re-fit models on full historical dataset (2018–2026)
    refit_df = df.copy().reset_index(drop=True)
    X_refit_baseline = refit_df[selected_baseline_features]
    X_refit_full = refit_df[selected_full_features]
    
    y_reg_refit = refit_df['Target_Total'].values
    y_reg_refit_pace = refit_df['Target_Pace'].values
    y_reg_refit_home_eff = refit_df['Target_Home_Eff'].values
    y_reg_refit_away_eff = refit_df['Target_Away_Eff'].values
    
    median_refit = np.median(y_reg_refit)
    y_clf_refit_s1 = (y_reg_refit > median_refit).astype(int)
    y_clf_refit_s2 = (y_reg_refit > refit_df['OverUnder'].values).astype(int)
    
    max_refit_date = refit_df['Date'].max()
    days_diff_refit = (pd.to_datetime(max_refit_date) - pd.to_datetime(refit_df['Date'])).dt.days
    refit_weights = np.maximum(0.2, np.exp(-best_lambda * days_diff_refit)).values
    
    print(f"\nRe-fitting totals models on entire historical dataset ({len(refit_df)} matches) using sample weights...")
    cv_splitter_refit = WalkForwardSeasonSplitter(refit_df['Season'])
    cv_splits_refit = list(cv_splitter_refit.split(refit_df))
    
    stage1_pace_reg.fit(X_refit_baseline, y_reg_refit_pace, cv_splits_refit, sample_weight=refit_weights)
    stage1_home_eff_reg.fit(X_refit_baseline, y_reg_refit_home_eff, cv_splits_refit, sample_weight=refit_weights)
    stage1_away_eff_reg.fit(X_refit_baseline, y_reg_refit_away_eff, cv_splits_refit, sample_weight=refit_weights)
    stage1_clf.fit(X_refit_baseline, y_clf_refit_s1, cv_splits_refit, sample_weight=refit_weights)
    
    # Construct s1_total_pred for refit_df
    s1_total_pred_refit = np.concatenate([best_s1_total_pred_train_val, s1_total_pred_test])
    refit_df['s1_total_pred'] = s1_total_pred_refit
    X_refit_full_s2 = refit_df[selected_full_features + ['s1_total_pred']]
    
    y_residual_refit = y_reg_refit - refit_df['OverUnder'].values
    stage2_reg.fit(X_refit_full_s2, y_residual_refit, cv_splits_refit, sample_weight=refit_weights)
    stage2_clf.fit(X_refit_full_s2, y_clf_refit_s2, cv_splits_refit, sample_weight=refit_weights)
    
    quantile_10.fit(X_refit_full_s2, y_reg_refit, sample_weight=refit_weights)
    quantile_90.fit(X_refit_full_s2, y_reg_refit, sample_weight=refit_weights)
    
    # Calculate standard deviation of residuals on refitted full dataset
    stage2_pred_residual_refit = stage2_reg.predict(X_refit_full_s2)
    refit_residuals = y_reg_refit - (refit_df['OverUnder'].values + stage2_pred_residual_refit)
    sigma_residuals_refit = float(np.std(refit_residuals))
    print(f"Re-fitted Totals Residuals Standard Deviation (sigma): {sigma_residuals_refit:.4f}")
    
    # 10. Save the trained models in wnba_total_model.pkl
    model_filename = 'wnba_total_model.pkl'
    print(f"Saving totals models to {model_filename}...")
    model_dict = {
        'stage1_pace_regressor': stage1_pace_reg,
        'stage1_home_eff_regressor': stage1_home_eff_reg,
        'stage1_away_eff_regressor': stage1_away_eff_reg,
        'stage1_classifier': stage1_clf, # trained on Target_Total > median_total
        'stage2_regressor': stage2_reg, # trained on residual: Target_Total - OverUnder
        'stage2_classifier': stage2_clf, # trained on Target_Total > OverUnder
        'quantile_10': quantile_10,
        'quantile_90': quantile_90,
        'stage1_calibrator': stage1_calibrator,
        'stage2_calibrator': stage2_calibrator
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
        
    # 11. Save metadata to total_model_metadata.json
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'feature_means': feature_means,
        'features': selected_full_features + ['s1_total_pred'],
        'baseline_features': selected_baseline_features,
        'full_features': selected_full_features + ['s1_total_pred'],
        'optimal_decay_lambda': float(best_lambda),
        'median_total': float(median_refit),
        'metrics': {
            'test_june_2026': {
                'market_mae': float(market_mae),
                'stage2_mae': float(stage2_mae),
                'stage2_accuracy': float(stage2_clf_accuracy),
                'stage2_logloss': float(stage2_logloss),
                'stage2_calibrated_accuracy': float(stage2_cal_accuracy),
                'stage2_calibrated_logloss': float(stage2_cal_logloss)
            }
        },
        'sigma_residuals': sigma_residuals_refit
    }
    
    metadata_filename = 'total_model_metadata.json'
    print(f"Saving totals metadata to {metadata_filename}...")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nTotals model training workflow complete!")

if __name__ == '__main__':
    train_totals_model()
