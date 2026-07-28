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

from stacking_models import StackedEnsembleRegressor, StackedEnsembleClassifier, FastDistributionRegressor

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
        'Home_Travel_Miles_7d', 'Home_Timezone_Changes_7d', 'Home_Fatigue_Score',
        'Away_Travel_Miles_7d', 'Away_Timezone_Changes_7d', 'Away_Fatigue_Score',
        'Travel_Miles_Diff', 'Fatigue_Score_Diff',
        
        # Talent Floor
        'Home_Talent_Floor', 'Away_Talent_Floor', 'Talent_Floor_Diff',
        
        # Squad Health
        'Home_Missing_Usage_Pct', 'Away_Missing_Usage_Pct',
        'Home_Missing_Net_Rating', 'Away_Missing_Net_Rating',
        'Home_Missing_PIE', 'Away_Missing_PIE',
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
    # Train/validation before 2026-07-01, test after 2026-07-01
    train_val_mask = df['Date'] < '2026-07-01'
    train_val_df = df[train_val_mask].copy().reset_index(drop=True)
    test_df = df[~train_val_mask].copy().reset_index(drop=True)
    
    if test_df.empty:
        raise ValueError("Held-out test set (Date >= 2026-07-01) is empty.")
        
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
        
    print(f"Tuning dataset size (Date < 2026-07-01): {len(train_val_df)} matches")
    print(f"July 2026 test set size (Date >= 2026-07-01): {len(test_df)} matches")
    
    # Generate custom splits
    cv_splitter = WalkForwardSeasonSplitter(train_val_df['Season'])
    cv_splits = list(cv_splitter.split(train_val_df))
    
    # Calculate days diff for sample weights calculation
    max_train_date = train_val_df['Date'].max()
    days_diff = (pd.to_datetime(max_train_date) - pd.to_datetime(train_val_df['Date'])).dt.days
    
    # 4. Baseline Feature Selection
    print("\nPerforming consensus baseline feature selection across XGBoost, LightGBM, and CatBoost...")
    prelim_weights = np.maximum(0.2, np.exp(-0.000551 * days_diff)).values
    
    prelim_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0)
    prelim_xgb.fit(train_val_df[full_features], train_val_df['Target'].values, sample_weight=prelim_weights)
    imp_xgb = prelim_xgb.feature_importances_
    if imp_xgb.sum() > 0:
        imp_xgb = imp_xgb / imp_xgb.sum()

    prelim_lgb = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1)
    prelim_lgb.fit(train_val_df[full_features], train_val_df['Target'].values, sample_weight=prelim_weights)
    imp_lgb = prelim_lgb.feature_importances_
    if imp_lgb.sum() > 0:
        imp_lgb = imp_lgb / imp_lgb.sum()

    from catboost import CatBoostRegressor
    prelim_cat = CatBoostRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, verbose=0)
    prelim_cat.fit(train_val_df[full_features], train_val_df['Target'].values, sample_weight=prelim_weights)
    imp_cat = prelim_cat.feature_importances_
    if imp_cat.sum() > 0:
        imp_cat = imp_cat / imp_cat.sum()

    consensus_importances = (imp_xgb + imp_lgb + imp_cat) / 3.0
    
    selected_features = [feat for feat, imp in zip(full_features, consensus_importances) if imp >= 0.001]
    # Fallback to avoid empty selection
    if not selected_features:
        selected_features = list(full_features)
        
    selected_baseline_features = [f for f in selected_features if f in baseline_features]
    selected_full_features = [f for f in selected_features if f in full_features]
    
    print(f"Selected {len(selected_features)} features (out of {len(full_features)}) with consensus importance >= 0.001")
    print(f"Selected Baseline features: {len(selected_baseline_features)}")
    print(f"Selected Full features: {len(selected_full_features)}")
    
    # 5. Dynamic Decay Parameter Optimization (Grid Search)
    lambdas = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
    best_log_loss = float('inf')
    best_lambda = None
    best_oof_s1 = None
    best_oof_s2 = None
    best_oof_y = None
    
    for lmbda in lambdas:
        print(f"\nEvaluating decay lambda = {lmbda}...")
        
        # Initialize validation arrays for out-of-fold predictions
        oof_stage1_probs = np.zeros(len(train_val_df))
        oof_stage2_probs = np.zeros(len(train_val_df))
        val_indices_seen = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            train_fold_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
            
            # Calculate weights for this training fold relative to its max date
            max_fold_date = train_fold_df['Date'].max()
            fold_days_diff = (pd.to_datetime(max_fold_date) - pd.to_datetime(train_fold_df['Date'])).dt.days
            w_fold = np.maximum(0.2, np.exp(-lmbda * fold_days_diff)).values
            
            # Nested Walk-Forward Splits for the training fold
            nested_splits = get_walk_forward_splits(train_fold_df['Season'], min_train_seasons=2)
            if len(nested_splits) == 0:
                # Fallback to simple split
                n_samples = len(train_fold_df)
                split_idx = int(0.8 * n_samples)
                nested_splits = [(np.arange(split_idx), np.arange(split_idx, n_samples))]
                
            y_tr_reg = train_fold_df['Target'].values
            y_tr_clf = (y_tr_reg > 0).astype(int)
            
            # Fit Stage 1 Models
            fold_stage1_reg = StackedEnsembleRegressor()
            fold_stage1_reg.fit(train_fold_df[selected_baseline_features], y_tr_reg, nested_splits, sample_weight=w_fold)
            
            fold_stage1_clf = StackedEnsembleClassifier()
            fold_stage1_clf.fit(train_fold_df[selected_baseline_features], y_tr_clf, nested_splits, sample_weight=w_fold)
            
            # Fit Stage 2 Models
            y_residual_tr = y_tr_reg - train_fold_df['ClosingSpread'].values
            
            fold_stage2_reg = FastDistributionRegressor()
            fold_stage2_reg.fit(train_fold_df[selected_full_features], y_residual_tr, cv_splits=nested_splits, sample_weight=w_fold)
            
            fold_stage2_clf = StackedEnsembleClassifier()
            fold_stage2_clf.fit(train_fold_df[selected_full_features], y_tr_clf, nested_splits, sample_weight=w_fold)
            
            # Predict on val_idx
            X_val_baseline = train_val_df.iloc[val_idx][selected_baseline_features]
            X_val_full = train_val_df.iloc[val_idx][selected_full_features]
            
            # Stage 1 prediction
            s1_prob = fold_stage1_clf.predict_proba(X_val_baseline)[:, 1]
            oof_stage1_probs[val_idx] = s1_prob
            
            # Stage 2 prediction (blended) using FastDistributionRegressor
            stage2_dist = fold_stage2_reg.pred_dist(X_val_full)
            s2_residual_pred = stage2_dist.loc
            s2_predicted_margin = train_val_df.iloc[val_idx]['ClosingSpread'].values + s2_residual_pred
            
            sigma_pred = stage2_dist.scale
            sigma_pred = np.maximum(sigma_pred, 1e-3)
            
            P_CDF = norm.cdf(s2_predicted_margin / sigma_pred)
            P_Clf = fold_stage2_clf.predict_proba(X_val_full)[:, 1]
            s2_prob = 0.5 * P_CDF + 0.5 * P_Clf
            oof_stage2_probs[val_idx] = s2_prob
            
            val_indices_seen.extend(val_idx)
            
        val_indices_seen = sorted(list(set(val_indices_seen)))
        y_val_clf = (train_val_df.iloc[val_indices_seen]['Target'].values > 0).astype(int)
        
        # Calculate Validation Log Loss of the blended Stage 2 predictions
        loss = log_loss(y_val_clf, oof_stage2_probs[val_indices_seen])
        print(f"Lambda = {lmbda} -> Validation Log Loss: {loss:.6f}")
        
        if loss < best_log_loss:
            best_log_loss = loss
            best_lambda = lmbda
            best_oof_s1 = oof_stage1_probs[val_indices_seen]
            best_oof_s2 = oof_stage2_probs[val_indices_seen]
            best_oof_y = y_val_clf
            
    print(f"\nOptimal lambda: {best_lambda} (Validation Log Loss: {best_log_loss:.6f})")
    
    # 6. Fit Platt/Isotonic calibrators on OOF probabilities
    print("\nFitting Isotonic Calibrators on Out-of-Fold blended probabilities...")
    stage1_calibrator = IsotonicRegression(out_of_bounds='clip')
    stage1_calibrator.fit(best_oof_s1, best_oof_y)
    
    stage2_calibrator = IsotonicRegression(out_of_bounds='clip')
    stage2_calibrator.fit(best_oof_s2, best_oof_y)
    
    # 7. Train final models on train_val_df using the optimal lambda
    print(f"\nTraining final models on entire tuning dataset (Date < 2026-07-01) with optimal decay = {best_lambda}...")
    train_val_weights = np.maximum(0.2, np.exp(-best_lambda * days_diff)).values
    y_reg_train_val = train_val_df['Target'].values
    y_clf_train_val = (y_reg_train_val > 0).astype(int)
    
    stage1_reg = StackedEnsembleRegressor()
    stage1_reg.fit(train_val_df[selected_baseline_features], y_reg_train_val, cv_splits, sample_weight=train_val_weights)
    
    stage1_clf = StackedEnsembleClassifier()
    stage1_clf.fit(train_val_df[selected_baseline_features], y_clf_train_val, cv_splits, sample_weight=train_val_weights)
    
    print("\nTraining Stage 2 FastDistributionRegressor on full_features...")
    y_residual_train_val = y_reg_train_val - train_val_df['ClosingSpread'].values
    
    stage2_reg = FastDistributionRegressor()
    stage2_reg.fit(train_val_df[selected_full_features], y_residual_train_val, cv_splits=cv_splits, sample_weight=train_val_weights)
    
    stage2_clf = StackedEnsembleClassifier()
    stage2_clf.fit(train_val_df[selected_full_features], y_clf_train_val, cv_splits, sample_weight=train_val_weights)
    
    # 8. Evaluate all models on the held-out June 2026 test set
    X_test_baseline = test_df[selected_baseline_features]
    X_test_full = test_df[selected_full_features]
    y_test_reg = test_df['Target'].values
    y_test_clf = (y_test_reg > 0).astype(int)
    
    # Baseline (Market)
    market_margin = -test_df['ClosingSpread'].values
    market_prob = test_df['Prob_Home'].values
    market_mae = mean_absolute_error(y_test_reg, market_margin)
    market_accuracy = np.mean(y_test_clf == (market_prob >= 0.5)) * 100.0
    market_logloss = log_loss(y_test_clf, market_prob)
    
    # Stage 1 Stacking (Uncalibrated)
    stage1_pred_margin = stage1_reg.predict(X_test_baseline)
    stage1_prob_clf = stage1_clf.predict_proba(X_test_baseline)[:, 1]
    stage1_mae = mean_absolute_error(y_test_reg, stage1_pred_margin)
    stage1_clf_accuracy = np.mean(y_test_clf == (stage1_prob_clf >= 0.5)) * 100.0
    stage1_logloss = log_loss(y_test_clf, stage1_prob_clf)
    
    # Stage 1 Stacking (Calibrated)
    stage1_prob_cal = stage1_calibrator.predict(stage1_prob_clf)
    stage1_cal_accuracy = np.mean(y_test_clf == (stage1_prob_cal >= 0.5)) * 100.0
    stage1_cal_logloss = log_loss(y_test_clf, stage1_prob_cal)
    
    # Stage 2 Volatility & prediction using FastDistributionRegressor
    stage2_dist = stage2_reg.pred_dist(X_test_full)
    stage2_residual_pred = stage2_dist.loc
    stage2_predicted_margin = test_df['ClosingSpread'].values + stage2_residual_pred
    
    sigma_pred = stage2_dist.scale
    sigma_pred = np.maximum(sigma_pred, 1e-3)
    
    P_CDF = norm.cdf(stage2_predicted_margin / sigma_pred)
    P_Clf = stage2_clf.predict_proba(X_test_full)[:, 1]
    
    # Blend P_CDF 50/50 with Stage 2 Classifier (Uncalibrated)
    final_win_prob = 0.5 * P_CDF + 0.5 * P_Clf
    stage2_mae = mean_absolute_error(y_test_reg, stage2_predicted_margin)
    stage2_clf_accuracy = np.mean(y_test_clf == (final_win_prob >= 0.5)) * 100.0
    stage2_logloss = log_loss(y_test_clf, final_win_prob)
    
    # Stage 2 Stacking (Calibrated)
    stage2_prob_cal = stage2_calibrator.predict(final_win_prob)
    stage2_cal_accuracy = np.mean(y_test_clf == (stage2_prob_cal >= 0.5)) * 100.0
    stage2_cal_logloss = log_loss(y_test_clf, stage2_prob_cal)
    
    # Print metrics table
    print("\n" + "="*95)
    print(f"{'MODEL EVALUATION SUMMARY (JULY 2026 TEST SET)':^95}")
    print("="*95)
    print(f"{'Model / Metric':<45} | {'MAE':^12} | {'Accuracy (%)':^15} | {'Log Loss':^12}")
    print("-"*95)
    print(f"{'Market Baseline (Closing Lines)':<45} | {market_mae:^12.4f} | {market_accuracy:^15.2f} | {market_logloss:^12.4f}")
    print(f"{'Stage 1 Stacking (Baseline Features, Uncal)':<45} | {stage1_mae:^12.4f} | {stage1_clf_accuracy:^15.2f} | {stage1_logloss:^12.4f}")
    print(f"{'Stage 1 Stacking (Baseline Features, Calibrated)':<45} | {stage1_mae:^12.4f} | {stage1_cal_accuracy:^15.2f} | {stage1_cal_logloss:^12.4f}")
    print(f"{'Stage 2 Stacking (Two-Stage + Vol, Uncal)':<45} | {stage2_mae:^12.4f} | {stage2_clf_accuracy:^15.2f} | {stage2_logloss:^12.4f}")
    print(f"{'Stage 2 Stacking (Two-Stage + Vol, Calibrated)':<45} | {stage2_mae:^12.4f} | {stage2_cal_accuracy:^15.2f} | {stage2_cal_logloss:^12.4f}")
    print("="*95)
    
    # 9. Re-fit all best estimators on the full historical dataset (2018–2026) using sample weights
    refit_df = df.copy().reset_index(drop=True)
    X_refit_baseline = refit_df[selected_baseline_features]
    X_refit_full = refit_df[selected_full_features]
    y_reg_refit = refit_df['Target'].values
    y_clf_refit = (y_reg_refit > 0).astype(int)
    
    max_refit_date = refit_df['Date'].max()
    days_diff_refit = (pd.to_datetime(max_refit_date) - pd.to_datetime(refit_df['Date'])).dt.days
    refit_weights = np.maximum(0.2, np.exp(-best_lambda * days_diff_refit)).values
    
    print(f"\nRe-fitting models on entire historical dataset ({len(refit_df)} matches) using sample weights with optimal lambda = {best_lambda}...")
    cv_splitter_refit = WalkForwardSeasonSplitter(refit_df['Season'])
    cv_splits_refit = list(cv_splitter_refit.split(refit_df))
    
    # Stage 1 Stacking Re-fit
    stage1_reg.fit(X_refit_baseline, y_reg_refit, cv_splits_refit, sample_weight=refit_weights)
    stage1_clf.fit(X_refit_baseline, y_clf_refit, cv_splits_refit, sample_weight=refit_weights)
    
    # Stage 2 Stacking Re-fit
    y_residual_refit = y_reg_refit - refit_df['ClosingSpread'].values
    stage2_reg = FastDistributionRegressor()
    stage2_reg.fit(X_refit_full, y_residual_refit, cv_splits=cv_splits_refit, sample_weight=refit_weights)
    stage2_clf.fit(X_refit_full, y_clf_refit, cv_splits_refit, sample_weight=refit_weights)
    
    # Calculate traditional standard deviation of residuals on refitted full dataset
    stage2_pred_residual_refit = stage2_reg.predict(X_refit_full)
    refit_residuals = y_reg_refit - (refit_df['ClosingSpread'].values + stage2_pred_residual_refit)
    sigma_residuals_refit = float(np.std(refit_residuals))
    print(f"Re-fitted Residuals Standard Deviation (sigma): {sigma_residuals_refit:.4f}")
    
    # 10. Save the trained models in a dictionary structure inside wnba_spread_model.pkl
    model_filename = 'wnba_spread_model.pkl'
    print(f"Saving models to {model_filename}...")
    model_dict = {
        'stage1_regressor': stage1_reg,
        'stage1_classifier': stage1_clf,
        'stage2_regressor': stage2_reg,
        'stage2_classifier': stage2_clf,
        'stage1_calibrator': stage1_calibrator,
        'stage2_calibrator': stage2_calibrator
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
        
    # Get feature importances from Stage 2 base models
    try:
        if hasattr(stage2_reg, 'base_estimator_') and hasattr(stage2_reg.base_estimator_, 'base_models_'):
            reg_importances = stage2_reg.base_estimator_.base_models_[0].feature_importances_
        elif hasattr(stage2_reg, 'feature_importances_'):
            reg_importances = stage2_reg.feature_importances_
            if isinstance(reg_importances, (list, tuple, np.ndarray)) and len(reg_importances) > 0 and isinstance(reg_importances[0], (list, tuple, np.ndarray)):
                reg_importances = reg_importances[0]
        else:
            reg_importances = np.zeros(len(selected_full_features))

        reg_feat_imp = sorted(
            {feat: float(imp) for feat, imp in zip(selected_full_features, reg_importances)}.items(),
            key=lambda item: item[1],
            reverse=True
        )
    except Exception as e:
        print(f"Could not compute regressor feature importances: {e}")
        reg_feat_imp = []

    try:
        clf_importances = stage2_clf.base_models_[0].feature_importances_
        clf_feat_imp = sorted(
            {feat: float(imp) for feat, imp in zip(selected_full_features, clf_importances)}.items(),
            key=lambda item: item[1],
            reverse=True
        )
    except Exception as e:
        print(f"Could not compute classifier feature importances: {e}")
        clf_feat_imp = []
        
    # 11. Save features list and other metadata to model_metadata.json
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'feature_means': feature_means,
        'features': selected_full_features,
        'baseline_features': selected_baseline_features,
        'full_features': selected_full_features,
        'optimal_decay_lambda': float(best_lambda),
        'metrics': {
            'test_june_2026': {
                'market_mae': float(market_mae),
                'market_accuracy': float(market_accuracy),
                'market_logloss': float(market_logloss),
                'stage1_mae': float(stage1_mae),
                'stage1_accuracy': float(stage1_clf_accuracy),
                'stage1_logloss': float(stage1_logloss),
                'stage1_calibrated_accuracy': float(stage1_cal_accuracy),
                'stage1_calibrated_logloss': float(stage1_cal_logloss),
                'stage2_mae': float(stage2_mae),
                'stage2_accuracy': float(stage2_clf_accuracy),
                'stage2_logloss': float(stage2_logloss),
                'stage2_calibrated_accuracy': float(stage2_cal_accuracy),
                'stage2_calibrated_logloss': float(stage2_cal_logloss)
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

