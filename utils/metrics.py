from typing import Union, Sequence
import numpy as np
from sklearn.metrics import mean_squared_log_error

def rmsle(
    true_values: Union[Sequence, np.ndarray], 
    predicted_values: Union[Sequence, np.ndarray]
) -> float:
    """Calculate Root Mean Squared Logarithmic Error.
    
    Args:
        true_values: Ground truth target values
        predicted_values: Model predictions
        
    Returns:
        RMSLE score
        
    Raises:
        ValueError: If inputs contain negative values before clipping
    """
    true_values = np.asarray(true_values)
    predicted_values = np.asarray(predicted_values)
    
    if (true_values < 0).any() or (predicted_values < 0).any():
        raise ValueError("Input arrays contain negative values")
    
    # Clip and add small constant to prevent log(0)
    eps = 1e-6
    true_values = np.clip(true_values, eps, None)
    predicted_values = np.clip(predicted_values, eps, None)
    
    return np.sqrt(mean_squared_log_error(true_values, predicted_values))

def rmsle_scorer(
    true_values: Union[Sequence, np.ndarray], 
    predicted_values: Union[Sequence, np.ndarray]
) -> float:
    """Scorer function for sklearn's cross_val_score.
    
    Args:
        true_values: Ground truth target values
        predicted_values: Model predictions
        
    Returns:
        Negative RMSLE score (negative for sklearn compatibility)
    """
    return -rmsle(true_values, predicted_values)
