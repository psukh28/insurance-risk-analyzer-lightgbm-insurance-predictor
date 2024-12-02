import sys
from pathlib import Path
import argparse
from typing import Tuple, List
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import logging

from features.feature_engineering import FeatureEngineer
from models.models import LGBMModel
from utils.metrics import rmsle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare training and test datasets.
    
    Args:
        data_dir: Directory containing data files
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If data files are not found
    """
    try:
        train_df = pd.read_csv(data_dir / 'train-2.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        
        # Drop ID columns
        train_df.drop('id', axis=1, inplace=True)
        test_df.drop('id', axis=1, inplace=True)
        
        return train_df, test_df
    except FileNotFoundError as e:
        logger.error(f"Failed to load data: {e}")
        raise

def train_model(
    X: pd.DataFrame, 
    y: pd.Series,
    feature_engineer: FeatureEngineer,
    model: LGBMModel,
    n_splits: int = 5
) -> Tuple[List[float], LGBMModel]:
    """Train model using k-fold cross-validation.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_engineer: Feature engineering transformer
        model: Model instance
        n_splits: Number of CV folds
        
    Returns:
        Tuple of (cv_scores, trained_model)
    """
    cv_scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logger.info(f"\nTraining Fold {fold}")
        
        # Split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Transform features
        X_train_processed = feature_engineer.fit_transform(X_train)
        X_val_processed = feature_engineer.transform(X_val)
        
        # Train and validate
        model.fit(X_train_processed, y_train)
        val_pred = model.predict(X_val_processed)
        
        # Score fold
        fold_score = rmsle(y_val, val_pred)
        cv_scores.append(fold_score)
        logger.info(f"Fold {fold} RMSLE: {fold_score:.4f}")
    
    logger.info(f"\nMean CV RMSLE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    return cv_scores, model

def save_submission(
    predictions: np.ndarray,
    data_dir: Path,
    submission_name: str = None
) -> None:
    """Save model predictions to a submission file.
    
    Args:
        predictions: Model predictions
        data_dir: Directory containing original data
        submission_name: Optional custom name for submission file
    """
    submissions_dir = Path(__file__).parent.parent / 'submissions'
    submissions_dir.mkdir(exist_ok=True)
    
    if submission_name:
        filename = f"{submission_name}.csv"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"
    
    submission_path = submissions_dir / filename
    
    submission = pd.DataFrame({
        'id': pd.read_csv(data_dir / 'test.csv')['id'],
        'Premium Amount': predictions
    })
    
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to: {submission_path}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train insurance premium prediction model')
    parser.add_argument('--save-submission', action='store_true',
                      help='Save the submission file')
    parser.add_argument('--submission-name', type=str,
                      help='Name for the submission file (without .csv extension)')
    return parser.parse_args()

def main() -> None:
    """Main training pipeline."""
    try:
        args = parse_args()
        logger.info("Starting training pipeline...")
        
        # Setup paths
        data_dir = Path('/Users/pranavsukumaran/Desktop/kagglecomp/src/data')
        
        # Load and prepare data
        train_df, test_df = load_data(data_dir)
        y = train_df['Premium Amount']
        X = train_df.drop('Premium Amount', axis=1)
        
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Initialize model and feature engineering
        feature_engineer = FeatureEngineer()
        model = LGBMModel()
        
        # Train model
        _, trained_model = train_model(X, y, feature_engineer, model)
        
        # Generate predictions
        logger.info("\nGenerating predictions...")
        X_test_processed = feature_engineer.transform(test_df)
        predictions = trained_model.predict(X_test_processed)
        
        # Save submission if requested
        if args.save_submission:
            save_submission(predictions, data_dir, args.submission_name)
        else:
            logger.info("Submission file not saved (use --save-submission to save)")
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
