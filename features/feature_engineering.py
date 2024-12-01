import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Simple feature engineering transformer"""
    
    def __init__(self):
        self.cat_columns = [
            'Gender', 'Marital Status', 'Education Level', 'Occupation',
            'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status',
            'Exercise Frequency', 'Property Type'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Handle categorical columns
        for col in self.cat_columns:
            X[col] = X[col].fillna('None').astype('category')
        
        # Handle numeric columns
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].mean())
        
        # Simple feature combinations
        X['Income_Per_Dependent'] = X['Annual Income'] / (X['Number of Dependents'] + 1)
        X['Claims_Per_Year'] = X['Previous Claims'] / X['Insurance Duration']
        
        # Convert date to numeric
        X['Policy_Start_Date'] = pd.to_datetime(X['Policy Start Date'])
        X['Days_Since_Start'] = (pd.Timestamp.now() - X['Policy_Start_Date']).dt.days
        
        # Drop original date column
        X = X.drop(['Policy Start Date', 'Policy_Start_Date'], axis=1, errors='ignore')
        
        return X
