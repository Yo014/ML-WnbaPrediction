import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge, LogisticRegression, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Stacked Ensemble Regressor that trains base estimators
    and fits a meta-estimator using walk-forward out-of-fold (OOF) predictions.

    Base Estimators:
    1. XGBoost (XGBRegressor)
    2. LightGBM (LGBMRegressor)
    3. CatBoost (CatBoostRegressor)
    4. Ridge

    Meta-Estimator:
    Ridge
    """
    _estimator_type = "regressor"

    def __init__(self, xgb_params=None, lgbm_params=None, cat_params=None, ridge_params=None, meta_params=None):
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.cat_params = cat_params
        self.ridge_params = ridge_params
        self.meta_params = meta_params
        
    def _initialize_models(self):
        # Sensible default parameters to keep training quiet and prevent overfitting
        default_xgb = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03, "verbosity": 0}
        default_lgbm = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03, "verbose": -1}
        default_cat = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03, "verbose": 0}
        
        xgb_params = {**default_xgb, **(self.xgb_params or {})}
        lgbm_params = {**default_lgbm, **(self.lgbm_params or {})}
        cat_params = {**default_cat, **(self.cat_params or {})}
        
        ridge_params = self.ridge_params or {}
        
        default_meta = {"cv": 5, "max_iter": 2000}
        meta_params = {**default_meta, **(self.meta_params or {})}
        
        self.xgb_ = XGBRegressor(**xgb_params)
        self.lgbm_ = LGBMRegressor(**lgbm_params)
        self.cat_ = CatBoostRegressor(**cat_params)
        self.ridge_ = make_pipeline(StandardScaler(), Ridge(**ridge_params))
        
        self.base_models_ = [self.xgb_, self.lgbm_, self.cat_, self.ridge_]
        self.meta_estimator_ = LassoCV(**meta_params)

    def fit(self, X, y, cv_splits, sample_weight=None):
        """
        Fit the stacked ensemble using custom walk-forward CV splits.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target values.
        cv_splits : list of tuples
            List of (train_idx, val_idx) tuples representing the custom walk-forward folds.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
        """
        if not cv_splits:
            raise ValueError("cv_splits must contain at least one split tuple (train_idx, val_idx).")

        # Convert to DataFrame if not already
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_df = X.copy()
        else:
            cols = [f"col_{i}" for i in range(np.asarray(X).shape[1])]
            self.feature_names_in_ = cols
            X_df = pd.DataFrame(X, columns=cols)
            
        y_arr = np.asarray(y)
        w_arr = np.asarray(sample_weight) if sample_weight is not None else None
        
        # Initialize base models and meta-estimator
        self._initialize_models()
        
        oof_preds_list = []
        oof_targets_list = []
        oof_weights_list = []
        
        # 1. Walk-forward out-of-fold fitting
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, y_train = X_df.iloc[train_idx], y_arr[train_idx]
            X_val, y_val = X_df.iloc[val_idx], y_arr[val_idx]
            
            w_train = w_arr[train_idx] if w_arr is not None else None
            w_val = w_arr[val_idx] if w_arr is not None else None
            
            # Clone base models for this fold to avoid target leakage/contamination
            cloned_models = [clone(model) for model in self.base_models_]
            
            fold_preds = []
            for model in cloned_models:
                if w_train is not None:
                    if hasattr(model, "steps") and hasattr(model, "named_steps"):
                        fit_params = {f"{model.steps[-1][0]}__sample_weight": w_train}
                        model.fit(X_train, y_train, **fit_params)
                    else:
                        model.fit(X_train, y_train, sample_weight=w_train)
                else:
                    model.fit(X_train, y_train)
                
                pred = model.predict(X_val)
                fold_preds.append(pred)
            
            # Stack predictions to get shape (len(val_idx), n_base_models)
            fold_preds_stacked = np.column_stack(fold_preds)
            oof_preds_list.append(fold_preds_stacked)
            oof_targets_list.append(y_val)
            if w_val is not None:
                oof_weights_list.append(w_val)
                
        # 2. Collect OOF predictions
        X_meta = np.vstack(oof_preds_list)
        y_meta = np.concatenate(oof_targets_list)
        w_meta = np.concatenate(oof_weights_list) if w_arr is not None else None
        
        # 3. Train meta-estimator on OOF predictions
        if w_meta is not None:
            self.meta_estimator_.fit(X_meta, y_meta, sample_weight=w_meta)
        else:
            self.meta_estimator_.fit(X_meta, y_meta)
            
        # 4. Fit final base models on the entire dataset
        for model in self.base_models_:
            if w_arr is not None:
                if hasattr(model, "steps") and hasattr(model, "named_steps"):
                    fit_params = {f"{model.steps[-1][0]}__sample_weight": w_arr}
                    model.fit(X_df, y_arr, **fit_params)
                else:
                    model.fit(X_df, y_arr, sample_weight=w_arr)
            else:
                model.fit(X_df, y_arr)
                
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict regression targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted continuous targets.
        """
        check_is_fitted(self, attributes=["is_fitted_"])
        
        # Convert input to DataFrame using fitted feature names
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Get base model predictions
        preds = []
        for model in self.base_models_:
            preds.append(model.predict(X_df))
            
        X_meta_test = np.column_stack(preds)
        return self.meta_estimator_.predict(X_meta_test)


class StackedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible Stacked Ensemble Classifier that trains base estimators
    and fits a meta-estimator using walk-forward out-of-fold (OOF) predictions.

    Base Estimators:
    1. XGBoost (XGBClassifier)
    2. LightGBM (LGBMClassifier)
    3. CatBoost (CatBoostClassifier)
    4. LogisticRegression

    Meta-Estimator:
    LogisticRegression
    """
    _estimator_type = "classifier"

    def __init__(self, xgb_params=None, lgbm_params=None, cat_params=None, lr_params=None, meta_params=None):
        self.xgb_params = xgb_params
        self.lgbm_params = lgbm_params
        self.cat_params = cat_params
        self.lr_params = lr_params
        self.meta_params = meta_params
        
    def _initialize_models(self):
        # Sensible default parameters to keep training quiet and prevent overfitting
        default_xgb = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03, "verbosity": 0}
        default_lgbm = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03, "verbose": -1}
        default_cat = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.03, "verbose": 0}
        default_lr = {"max_iter": 2000}
        
        xgb_params = {**default_xgb, **(self.xgb_params or {})}
        lgbm_params = {**default_lgbm, **(self.lgbm_params or {})}
        cat_params = {**default_cat, **(self.cat_params or {})}
        
        lr_params = {**default_lr, **(self.lr_params or {})}
        
        # Classifier's meta-estimator: L1 regularization, liblinear solver, C=1.0
        default_meta = {"penalty": "l1", "solver": "liblinear", "C": 1.0, "max_iter": 2000}
        meta_params = {**default_meta, **(self.meta_params or {})}
        
        self.xgb_ = XGBClassifier(**xgb_params)
        self.lgbm_ = LGBMClassifier(**lgbm_params)
        self.cat_ = CatBoostClassifier(**cat_params)
        self.lr_ = make_pipeline(StandardScaler(), LogisticRegression(**lr_params))
        
        self.base_models_ = [self.xgb_, self.lgbm_, self.cat_, self.lr_]
        self.meta_estimator_ = LogisticRegression(**meta_params)

    def fit(self, X, y, cv_splits, sample_weight=None):
        """
        Fit the stacked ensemble using custom walk-forward CV splits.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector.
        y : array-like of shape (n_samples,)
            Target class labels.
        cv_splits : list of tuples
            List of (train_idx, val_idx) tuples representing the custom walk-forward folds.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
        """
        if not cv_splits:
            raise ValueError("cv_splits must contain at least one split tuple (train_idx, val_idx).")

        # Convert to DataFrame if not already
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_df = X.copy()
        else:
            cols = [f"col_{i}" for i in range(np.asarray(X).shape[1])]
            self.feature_names_in_ = cols
            X_df = pd.DataFrame(X, columns=cols)
            
        y_arr = np.asarray(y)
        w_arr = np.asarray(sample_weight) if sample_weight is not None else None
        
        self.classes_ = np.unique(y_arr)
        
        # Initialize base models and meta-estimator
        self._initialize_models()
        
        oof_preds_list = []
        oof_targets_list = []
        oof_weights_list = []
        
        # 1. Walk-forward out-of-fold fitting
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_train, y_train = X_df.iloc[train_idx], y_arr[train_idx]
            X_val, y_val = X_df.iloc[val_idx], y_arr[val_idx]
            
            w_train = w_arr[train_idx] if w_arr is not None else None
            w_val = w_arr[val_idx] if w_arr is not None else None
            
            # Clone base models for this fold to avoid target leakage/contamination
            cloned_models = [clone(model) for model in self.base_models_]
            
            fold_preds = []
            for model in cloned_models:
                if w_train is not None:
                    if hasattr(model, "steps") and hasattr(model, "named_steps"):
                        fit_params = {f"{model.steps[-1][0]}__sample_weight": w_train}
                        model.fit(X_train, y_train, **fit_params)
                    else:
                        model.fit(X_train, y_train, sample_weight=w_train)
                else:
                    model.fit(X_train, y_train)
                
                # Extract classification predictions.
                # Probability of positive class (class 1) is used for binary stacking
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_val)
                    if prob.ndim == 2 and prob.shape[1] == 2:
                        pred = prob[:, 1]
                    else:
                        pred = model.predict(X_val)
                else:
                    pred = model.predict(X_val)
                fold_preds.append(pred)
            
            # Stack predictions to get shape (len(val_idx), n_base_models)
            fold_preds_stacked = np.column_stack(fold_preds)
            oof_preds_list.append(fold_preds_stacked)
            oof_targets_list.append(y_val)
            if w_val is not None:
                oof_weights_list.append(w_val)
                
        # 2. Collect OOF predictions
        X_meta = np.vstack(oof_preds_list)
        y_meta = np.concatenate(oof_targets_list)
        w_meta = np.concatenate(oof_weights_list) if w_arr is not None else None
        
        # 3. Train meta-estimator on OOF predictions
        if w_meta is not None:
            self.meta_estimator_.fit(X_meta, y_meta, sample_weight=w_meta)
        else:
            self.meta_estimator_.fit(X_meta, y_meta)
            
        # 4. Fit final base models on the entire dataset
        for model in self.base_models_:
            if w_arr is not None:
                if hasattr(model, "steps") and hasattr(model, "named_steps"):
                    fit_params = {f"{model.steps[-1][0]}__sample_weight": w_arr}
                    model.fit(X_df, y_arr, **fit_params)
                else:
                    model.fit(X_df, y_arr, sample_weight=w_arr)
            else:
                model.fit(X_df, y_arr)
                
        self.is_fitted_ = True
        return self

    def _get_meta_features(self, X_df):
        preds = []
        for model in self.base_models_:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_df)
                if prob.ndim == 2 and prob.shape[1] == 2:
                    pred = prob[:, 1]
                else:
                    pred = model.predict(X_df)
            else:
                pred = model.predict(X_df)
            preds.append(pred)
        return np.column_stack(preds)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, attributes=["is_fitted_"])
        
        # Convert input to DataFrame using fitted feature names
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            
        X_meta_test = self._get_meta_features(X_df)
        return self.meta_estimator_.predict(X_meta_test)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        check_is_fitted(self, attributes=["is_fitted_"])
        
        # Convert input to DataFrame using fitted feature names
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
            
        X_meta_test = self._get_meta_features(X_df)
        return self.meta_estimator_.predict_proba(X_meta_test)


# --- Unit Tests & Validation Checks ---
if __name__ == "__main__":
    print("Running StackedEnsemble tests...")
    
    from sklearn.datasets import make_regression, make_classification
    
    # 1. Test StackedEnsembleRegressor
    print("\n--- Testing StackedEnsembleRegressor ---")
    X_reg, y_reg = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Define custom walk-forward splits
    cv_splits_reg = [
        (np.arange(0, 50), np.arange(50, 65)),
        (np.arange(0, 65), np.arange(65, 80)),
        (np.arange(0, 80), np.arange(80, 100))
    ]
    
    # Test fitting and predicting without weights
    print("Fitting regressor without sample weights...")
    reg = StackedEnsembleRegressor()
    reg.fit(X_reg, y_reg, cv_splits_reg)
    preds_reg = reg.predict(X_reg)
    print(f"Predictions shape: {preds_reg.shape}")
    assert preds_reg.shape == (100,), f"Expected shape (100,), got {preds_reg.shape}"
    print("Regressor fit and predict without weights: SUCCESS")
    
    # Test fitting and predicting with sample weights
    print("Fitting regressor with sample weights...")
    w_reg = np.random.rand(100)
    reg_w = StackedEnsembleRegressor()
    reg_w.fit(X_reg, y_reg, cv_splits_reg, sample_weight=w_reg)
    preds_reg_w = reg_w.predict(X_reg)
    print(f"Weighted predictions shape: {preds_reg_w.shape}")
    assert preds_reg_w.shape == (100,), f"Expected shape (100,), got {preds_reg_w.shape}"
    print("Regressor fit and predict with weights: SUCCESS")
    
    # 2. Test StackedEnsembleClassifier
    print("\n--- Testing StackedEnsembleClassifier ---")
    X_clf, y_clf = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    
    # Define custom walk-forward splits
    cv_splits_clf = [
        (np.arange(0, 50), np.arange(50, 65)),
        (np.arange(0, 65), np.arange(65, 80)),
        (np.arange(0, 80), np.arange(80, 100))
    ]
    
    # Test fitting and predicting without weights
    print("Fitting classifier without sample weights...")
    clf = StackedEnsembleClassifier()
    clf.fit(X_clf, y_clf, cv_splits_clf)
    preds_clf = clf.predict(X_clf)
    probs_clf = clf.predict_proba(X_clf)
    print(f"Predictions shape: {preds_clf.shape}")
    print(f"Probabilities shape: {probs_clf.shape}")
    assert preds_clf.shape == (100,), f"Expected shape (100,), got {preds_clf.shape}"
    assert probs_clf.shape == (100, 2), f"Expected shape (100, 2), got {probs_clf.shape}"
    print("Classifier fit and predict without weights: SUCCESS")
    
    # Test fitting and predicting with sample weights
    print("Fitting classifier with sample weights...")
    w_clf = np.random.rand(100)
    clf_w = StackedEnsembleClassifier()
    clf_w.fit(X_clf, y_clf, cv_splits_clf, sample_weight=w_clf)
    preds_clf_w = clf_w.predict(X_clf)
    probs_clf_w = clf_w.predict_proba(X_clf)
    print(f"Weighted predictions shape: {preds_clf_w.shape}")
    print(f"Weighted probabilities shape: {probs_clf_w.shape}")
    assert preds_clf_w.shape == (100,), f"Expected shape (100,), got {preds_clf_w.shape}"
    assert probs_clf_w.shape == (100, 2), f"Expected shape (100, 2), got {probs_clf_w.shape}"
    print("Classifier fit and predict with weights: SUCCESS")
    
    print("\nAll unit tests passed successfully!")
