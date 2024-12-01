import sys
from pathlib import Path
import argparse

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import logging
from datetime import datetime

from features.feature_engineering import FeatureEngineer
from models.models import LGBMModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def parse_args():
    parser = argparse.ArgumentParser(description='Train insurance premium prediction model')
    parser.add_argument('--save-submission', action='store_true',
                      help='Save the submission file')
    parser.add_argument('--submission-name', type=str,
                      help='Name for the submission file (without .csv extension)')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info("Starting training pipeline...")
    
    # Load data
    logger.info("Loading data...")
    data_dir = Path('/Users/pranavsukumaran/Desktop/kagglecomp/src/data')
    train = pd.read_csv(data_dir / 'train-2.csv')
    test = pd.read_csv(data_dir / 'test.csv')
    
    # Drop ID column
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)
    
    # Split features and target
    y = train['Premium Amount']
    X = train.drop('Premium Amount', axis=1)
    
    logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
    
    # Initialize models and transformers
    feature_engineer = FeatureEngineer()
    model = LGBMModel()
    
    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # Train and validate
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logger.info(f"\nTraining Fold {fold}")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Transform features
        X_train_processed = feature_engineer.fit_transform(X_train_fold)
        X_val_processed = feature_engineer.transform(X_val_fold)
        
        # Train model
        model.fit(X_train_processed, y_train_fold)
        
        # Validate
        val_pred = model.predict(X_val_processed)
        fold_score = rmsle(y_val_fold, val_pred)
        cv_scores.append(fold_score)
        logger.info(f"Fold {fold} RMSLE: {fold_score:.4f}")
    
    logger.info(f"\nMean CV RMSLE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    
    # Generate predictions
    logger.info("\nGenerating predictions...")
    X_test_processed = feature_engineer.transform(test)
    test_pred = model.predict(X_test_processed)
    
    if args.save_submission:
        # Create submissions directory if it doesn't exist
        submissions_dir = Path(__file__).parent.parent / 'submissions'
        submissions_dir.mkdir(exist_ok=True)
        
        # Generate submission filename
        if args.submission_name:
            filename = f"{args.submission_name}.csv"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.csv"
        
        submission_path = submissions_dir / filename
        
        # Save submission
        submission = pd.DataFrame({
            'id': pd.read_csv(data_dir / 'test.csv')['id'],
            'Premium Amount': test_pred
        })
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to: {submission_path}")
    else:
        logger.info("Submission file not saved (use --save-submission to save)")

if __name__ == "__main__":
    main()
