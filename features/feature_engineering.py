from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Constants
CATEGORICAL_COLUMNS = [
    'Gender', 'Marital Status', 'Education Level', 'Occupation',
    'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status',
    'Exercise Frequency', 'Property Type'
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Transform raw insurance data into model-ready features.
    
    This transformer handles:
    - Categorical variable encoding
    - Missing value imputation
    - Feature creation from domain knowledge
    - Date feature extraction
    """
    
    def __init__(self, categorical_columns: List[str] = CATEGORICAL_COLUMNS):
        self.categorical_columns = categorical_columns
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineer':
        """Fit the transformer (no-op as this is a stateless transformer)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data with feature engineering.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature matrix
        """
        X = X.copy()
        
        # Existing transformations
        for col in self.categorical_columns:
            X[col] = X[col].fillna('None').astype('category')
        
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # New feature combinations
        X['Income_Per_Dependent'] = X['Annual Income'] / (X['Number of Dependents'] + 1)
        X['Claims_Per_Year'] = X['Previous Claims'] / X['Insurance Duration']
        
        # Additional features
        X['Age_Group'] = pd.qcut(X['Age'], q=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])
        X['Income_Group'] = pd.qcut(X['Annual Income'], q=5, labels=['Low', 'Below Average', 'Average', 'Above Average', 'High'])
        X['Risk_Score'] = (
            (100 - X['Health Score']) * 0.3 +
            (X['Previous Claims'] * 20) +
            (X['Age'] * 0.5) +
            (X['Vehicle Age'] * 2)
        )
        
        # Interaction features
        X['Health_Age_Interaction'] = X['Health Score'] * (100 - X['Age']) / 100
        X['Claims_Duration_Ratio'] = X['Previous Claims'] / (X['Insurance Duration'] + 1)
        X['Income_Credit_Score'] = X['Annual Income'] * X['Credit Score'] / 100000
        
        # Binary flags
        X['Is_Smoker'] = (X['Smoking Status'] == 'Smoker').astype(int)
        X['Is_Urban'] = (X['Location'] == 'Urban').astype(int)
        X['Has_Dependents'] = (X['Number of Dependents'] > 0).astype(int)
        
        policy_start = pd.to_datetime(X['Policy Start Date'])
        X['Days_Since_Start'] = (pd.Timestamp.now() - policy_start).dt.days
        
        # Drop original date columns
        X = X.drop(['Policy Start Date', 'Policy_Start_Date'], axis=1, errors='ignore')
        
        return X