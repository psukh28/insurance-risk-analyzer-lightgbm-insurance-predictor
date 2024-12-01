import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb

class LGBMModel(BaseEstimator, RegressorMixin):
    """LightGBM model wrapper for insurance premium prediction"""
    
    def __init__(self, n_estimators=1000, learning_rate=0.05):
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': learning_rate,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.n_estimators = n_estimators
        self.model = None
        
    def fit(self, X, y):
        """Train the model with early stopping"""
        train_data = lgb.Dataset(X, label=y)
        valid_data = lgb.Dataset(X, label=y, reference=train_data)
        
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[lgb.early_stopping(50)]
        )
        
        return self
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X)
