import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from stacking_models import StackedEnsembleRegressor, StackedEnsembleClassifier
from ngboost import NGBRegressor
from ngboost.distns import Laplace

print("Loading data...", flush=True)
df = pd.read_csv('ml_ready_data.csv').sort_values(by='Date').reset_index(drop=True)
df['Target_Total'] = df['HomeScore'] + df['AwayScore']
df['Target_Pace'] = (df['HomePossessions'] + df['AwayPossessions']) / 2.0
possessions = df['Target_Pace'].replace(0, np.nan)
df['Target_Home_Eff'] = (100.0 * df['HomeScore'] / possessions).fillna(0.0)
df['Target_Away_Eff'] = (100.0 * df['AwayScore'] / possessions).fillna(0.0)

train_val_df = df[df['Date'] < '2026-07-01'].copy().reset_index(drop=True)
numeric_cols = train_val_df.select_dtypes(include=[np.number]).columns
features = [col for col in numeric_cols if col not in ['HomeScore', 'AwayScore', 'Target_Total', 'Target_Pace', 'Target_Home_Eff', 'Target_Away_Eff', 'Season']]
for col in features:
    col_mean = train_val_df[col].mean()
    train_val_df[col] = train_val_df[col].fillna(0.0 if np.isnan(col_mean) else col_mean).fillna(0.0)

w_fold = np.ones(len(train_val_df))
nested_splits = [(np.arange(1000), np.arange(1000, len(train_val_df)))]

print("Test 1: StackedEnsembleRegressor fit...", flush=True)
s_reg = StackedEnsembleRegressor()
s_reg.fit(train_val_df[features], train_val_df['Target_Total'], nested_splits, sample_weight=w_fold)
print("Test 1 PASSED!", flush=True)

print("Test 2: StackedEnsembleClassifier fit...", flush=True)
s_clf = StackedEnsembleClassifier()
s_clf.fit(train_val_df[features], (train_val_df['Target_Total'] > 160).astype(int), nested_splits, sample_weight=w_fold)
print("Test 2 PASSED!", flush=True)

print("Test 3: NGBRegressor(Dist=Laplace) fit...", flush=True)
ngb = NGBRegressor(Dist=Laplace, random_state=42, verbose=False)
ngb.fit(train_val_df[features], train_val_df['Target_Total'] - 160, sample_weight=w_fold)
print("Test 3 PASSED!", flush=True)

print("Test 4: NGBRegressor pred_dist...", flush=True)
dist = ngb.pred_dist(train_val_df[features])
print("Test 4 PASSED! mean shape:", dist.mean().shape, flush=True)

print("ALL TESTS PASSED SUCCESSFULLY!", flush=True)
