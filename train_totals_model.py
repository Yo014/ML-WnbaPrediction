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
from sklearn.linear_model import LogisticRegression

from stacking_models import StackedEnsembleRegressor, StackedEnsembleClassifier, FastDistributionRegressor, DummyPassThroughModel

class WalkForwardSeasonSplitter:
    def __init__(self, seasons_series):
        self.seasons = np.array(seasons_series)
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return 4
        
    def split(self, X, y=None, groups=None):
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



def train_totals_model():
    # 1. Read the engineered dataset
    data_path = 'ml_ready_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Target definitions
    df['Target_Total'] = df['HomeScore'] + df['AwayScore']
    df['Target_Residual'] = df['Target_Total'] - df['OverUnder']
    
    # 2. Features setup
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
        'Expected_Pace',
        
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
        'Travel_Miles_Diff', 'Fatigue_Score_Diff', 'Combined_Fatigue_Score',
        
        # Expected Scores
        'Home_Expected_Pts', 'Away_Expected_Pts', 'Expected_Game_Total',
        
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
        'Ref_Pts_EMA', 'Ref_Fouls_EMA', 'Ref_HomeWin_EMA', 'Ref_Foul_Impact', 'Ref_FreeThrow_Rate',
        
        # Market Features
        'BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder', 'Prob_Home', 'Market_Disagreement'
    ]
    
    market_features = ['BookieHomeOdds', 'BookieAwayOdds', 'OpeningSpread', 'ClosingSpread', 'OverUnder', 'Prob_Home', 'Market_Disagreement']
    baseline_features = [f for f in features if f not in market_features]
    full_features = list(features)
    
    # Train-Validation (Date < 2026-07-01) and Held-out Test Set (Date >= 2026-07-01)
    train_val_mask = df['Date'] < '2026-07-01'
    train_val_df = df[train_val_mask].copy().reset_index(drop=True)
    test_df = df[~train_val_mask].copy().reset_index(drop=True)
    
    if test_df.empty:
        raise ValueError("Held-out test set (Date >= 2026-07-01) is empty.")
        
    print(f"Train/Val dataset size (Date < 2026-07-01): {len(train_val_df)} matches")
    print(f"Held-out test set size (Date >= 2026-07-01): {len(test_df)} matches")
    
    # Missing value imputation: feature_means computed strictly on train_val_df before imputing test_df and df to prevent test-set leakage
    feature_means = {}
    for col in features:
        mean_val = train_val_df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        feature_means[col] = float(mean_val)
        train_val_df[col] = train_val_df[col].fillna(mean_val)
        test_df[col] = test_df[col].fillna(mean_val)
        df[col] = df[col].fillna(mean_val)
        
    cv_splitter = WalkForwardSeasonSplitter(train_val_df['Season'])
    cv_splits = list(cv_splitter.split(train_val_df))
    
    max_train_date = train_val_df['Date'].max()
    days_diff = (pd.to_datetime(max_train_date) - pd.to_datetime(train_val_df['Date'])).dt.days
    
    # 3. Consensus Feature Selection
    print("\nPerforming baseline feature selection for streamlined totals model...")
    prelim_weights = np.maximum(0.2, np.exp(-0.0005 * days_diff)).values
    prelim_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0)
    prelim_xgb.fit(train_val_df[full_features], train_val_df['Target_Residual'].values, sample_weight=prelim_weights)
    importances = prelim_xgb.feature_importances_
    
    selected_full_features = [feat for feat, imp in zip(full_features, importances) if imp >= 0.001]
    if not selected_full_features:
        selected_full_features = list(full_features)
    selected_baseline_features = [f for f in selected_full_features if f in baseline_features]
    
    print(f"Selected {len(selected_full_features)} full features for streamlined totals model.")
    
    # 4. Streamlined Decay Grid Search
    lambdas = [0.0005, 0.001, 0.0015, 0.002]
    best_log_loss = float('inf')
    best_lambda = None
    best_oof_probs = None
    best_oof_labels = None
    
    for lmbda in lambdas:
        print(f"Evaluating decay lambda = {lmbda}...")
        oof_probs = np.zeros(len(train_val_df))
        val_indices_seen = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            train_fold_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
            val_fold_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)
            
            max_fold_date = train_fold_df['Date'].max()
            fold_days_diff = (pd.to_datetime(max_fold_date) - pd.to_datetime(train_fold_df['Date'])).dt.days
            w_fold = np.maximum(0.2, np.exp(-lmbda * fold_days_diff)).values
            
            y_residual_tr = train_fold_df['Target_Residual'].values
            
            # Fit single residual distribution regressor
            fold_dist_reg = FastDistributionRegressor()
            fold_dist_reg.fit(train_fold_df[selected_full_features], y_residual_tr, sample_weight=w_fold)
            
            # Predict validation distribution
            val_dist = fold_dist_reg.pred_dist(val_fold_df[selected_full_features])
            val_pred_residual = val_dist.mean()
            val_sigma = np.maximum(val_dist.std(), 1e-3)
            
            # Compute CDF Over probability against residual line 0
            P_CDF = norm.cdf(val_pred_residual / val_sigma)
            oof_probs[val_idx] = P_CDF
            val_indices_seen.extend(val_idx)
            
        val_indices_seen = sorted(list(set(val_indices_seen)))
        y_val_clf = (train_val_df.iloc[val_indices_seen]['Target_Total'].values > train_val_df.iloc[val_indices_seen]['OverUnder'].values).astype(int)
        
        loss = log_loss(y_val_clf, oof_probs[val_indices_seen])
        print(f"Lambda = {lmbda} -> Streamlined Validation Log Loss: {loss:.6f}")
        
        if loss < best_log_loss:
            best_log_loss = loss
            best_lambda = lmbda
            best_oof_probs = oof_probs[val_indices_seen]
            best_oof_labels = y_val_clf
            
    print(f"\nOptimal lambda: {best_lambda} (Validation Log Loss: {best_log_loss:.6f})")
    
    # 5. Single Isotonic Calibrator
    print("\nFitting Isotonic Calibrator on Out-of-Fold CDF probabilities...")
    stage2_calibrator = LogisticRegression(C=1.0)
    stage2_calibrator.fit(best_oof_probs.reshape(-1, 1), best_oof_labels)
    
    # 6. Fit Final Production Models
    print(f"\nFitting final Streamlined Totals Distribution Regressors with optimal lambda = {best_lambda}...")
    train_val_weights = np.maximum(0.2, np.exp(-best_lambda * days_diff)).values
    
    # Primary residual distribution model (Stage 2)
    stage2_reg = FastDistributionRegressor()
    stage2_reg.fit(train_val_df[selected_full_features], train_val_df['Target_Residual'].values, sample_weight=train_val_weights)
    
    # Baseline direct total points model (Stage 1 fallback when no market odds exist)
    stage1_reg = FastDistributionRegressor()
    stage1_reg.fit(train_val_df[selected_baseline_features], train_val_df['Target_Total'].values, sample_weight=train_val_weights)
    
    # 7. Evaluate on June 2026 Test Set
    test_dist = stage2_reg.pred_dist(test_df[selected_full_features])
    test_residual_pred = test_dist.mean()
    test_predicted_total = test_df['OverUnder'].values + test_residual_pred
    test_sigma = np.maximum(test_dist.std(), 1e-3)
    
    test_cdf_over = norm.cdf(test_residual_pred / test_sigma)
    test_cal_over = stage2_calibrator.predict_proba(test_cdf_over.reshape(-1, 1))[:, 1]
    
    y_test_reg = test_df['Target_Total'].values
    y_test_clf = (y_test_reg > test_df['OverUnder'].values).astype(int)
    
    market_mae = mean_absolute_error(y_test_reg, test_df['OverUnder'].values)
    model_mae = mean_absolute_error(y_test_reg, test_predicted_total)
    model_acc = np.mean(y_test_clf == (test_cal_over >= 0.5)) * 100.0
    model_logloss = log_loss(y_test_clf, test_cal_over)
    
    print("\n" + "="*90)
    print(f"{'STREAMLINED TOTALS MODEL EVALUATION (JUNE 2026 TEST SET)':^90}")
    print("="*90)
    print(f"{'Model / Metric':<45} | {'MAE':^12} | {'Accuracy (%)':^15} | {'Log Loss':^12}")
    print("-"*90)
    print(f"{'Market Closing OverUnder Line':<45} | {market_mae:^12.4f} | {'50.00':^15} | {'0.6930':^12}")
    print(f"{'Streamlined Totals Residual Engine (Calibrated)':<45} | {model_mae:^12.4f} | {model_acc:^15.2f} | {model_logloss:^12.4f}")
    print("="*90)
    
    # 8. Refit on entire dataset (2018-2026)
    refit_df = df.copy().reset_index(drop=True)
    max_refit_date = refit_df['Date'].max()
    days_diff_refit = (pd.to_datetime(max_refit_date) - pd.to_datetime(refit_df['Date'])).dt.days
    refit_weights = np.maximum(0.2, np.exp(-best_lambda * days_diff_refit)).values
    
    print(f"\nRe-fitting final models on entire historical dataset ({len(refit_df)} matches)...")
    stage2_reg_refit = FastDistributionRegressor()
    stage2_reg_refit.fit(refit_df[selected_full_features], refit_df['Target_Residual'].values, sample_weight=refit_weights)
    
    stage1_reg_refit = FastDistributionRegressor()
    stage1_reg_refit.fit(refit_df[selected_baseline_features], refit_df['Target_Total'].values, sample_weight=refit_weights)
    
    refit_dist = stage2_reg_refit.pred_dist(refit_df[selected_full_features])
    refit_residuals = refit_df['Target_Total'].values - (refit_df['OverUnder'].values + refit_dist.mean())
    sigma_residuals_refit = float(np.std(refit_residuals))
    print(f"Final Streamlined Totals Sigma (Residual Uncertainty): {sigma_residuals_refit:.4f}")
    
    # 9. Save updated model artifact with backwards-compatible dictionary structure
    dummy_model = DummyPassThroughModel()
    model_filename = 'wnba_total_model.pkl'
    print(f"Saving streamlined totals models to {model_filename}...")
    model_dict = {
        'stage2_regressor': stage2_reg_refit,
        'stage2_calibrator': stage2_calibrator,
        'stage1_regressor': stage1_reg_refit,
        'stage1_calibrator': stage2_calibrator,
        'stage1_pace_regressor': dummy_model,
        'stage1_home_eff_regressor': dummy_model,
        'stage1_away_eff_regressor': dummy_model,
        'stage1_classifier': dummy_model,
        'stage2_classifier': dummy_model
    }
    with open(model_filename, 'wb') as f:
        pickle.dump(model_dict, f)
        
    # 10. Save metadata
    median_refit = float(np.median(refit_df['Target_Total'].values))
    metadata = {
        'training_timestamp': datetime.now().isoformat(),
        'feature_means': feature_means,
        'features': selected_full_features,
        'baseline_features': selected_baseline_features,
        'full_features': selected_full_features,
        'optimal_decay_lambda': float(best_lambda),
        'optimal_distribution': 'FastDistributionRegressor',
        'median_total': median_refit,
        'metrics': {
            'test_june_2026': {
                'market_mae': float(market_mae),
                'stage2_mae': float(model_mae),
                'stage2_accuracy': float(model_acc),
                'stage2_logloss': float(model_logloss),
                'stage2_calibrated_accuracy': float(model_acc),
                'stage2_calibrated_logloss': float(model_logloss)
            }
        },
        'sigma_residuals': sigma_residuals_refit
    }
    
    metadata_filename = 'total_model_metadata.json'
    print(f"Saving totals metadata to {metadata_filename}...")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("\nStreamlined totals model training complete!")

if __name__ == '__main__':
    train_totals_model()
