import numpy as np
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Logarithmic Error
    Note: This function expects the input values to be in their original scale (not log-transformed)
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        RMSLE score
    """
    # Ensure inputs are non-negative
    y_true = np.clip(y_true, a_min=0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    
    # Add small constant to prevent log(0)
    y_true = np.maximum(y_true, 1e-6)
    y_pred = np.maximum(y_pred, 1e-6)
    
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def rmsle_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Scorer function for sklearn's cross_val_score
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: Negative RMSLE score (negative for sklearn compatibility)
    """
    return -rmsle(y_true, y_pred)
