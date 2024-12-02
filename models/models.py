from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb

@dataclass
class LGBMParameters:
    """LightGBM model parameters with defaults optimized for insurance premium prediction."""
    objective: str = 'regression'
    metric: str = 'rmse'
    boosting_type: str = 'gbdt'
    num_leaves: int = 31
    learning_rate: float = 0.05
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    verbose: int = -1
    random_state: int = 42

class LGBMModel(BaseEstimator, RegressorMixin):
    """LightGBM model wrapper for insurance premium prediction.
    
    Args:
        n_estimators: Maximum number of boosting rounds
        learning_rate: Step size shrinkage to prevent overfitting
        params: Optional custom parameters to override defaults
    """
    
    def __init__(
        self, 
        n_estimators: int = 1000, 
        learning_rate: float = 0.05,
        params: Optional[Dict[str, Any]] = None
    ):
        self.params = LGBMParameters(learning_rate=learning_rate).__dict__
        if params:
            self.params.update(params)
        self.n_estimators = n_estimators
        self.model: Optional[lgb.Booster] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LGBMModel':
        """Train the model with early stopping.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Trained model instance
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
            
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
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")
        return self.model.predict(X)
