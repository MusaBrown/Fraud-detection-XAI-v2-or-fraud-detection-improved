"""
Preprocessing pipeline for fraud detection datasets.
Handles categorical encoding, scaling, missing values, and class imbalance.
"""
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessor for fraud detection data.
    Handles mixed categorical/numerical features with memory efficiency.
    """
    
    def __init__(
        self,
        categorical_threshold: int = 20,
        max_categories: int = 100,
        handle_missing: str = 'median',
        scale_features: bool = True,
        reduce_memory: bool = True
    ):
        self.categorical_threshold = categorical_threshold
        self.max_categories = max_categories
        self.handle_missing = handle_missing
        self.scale_features = scale_features
        self.reduce_memory = reduce_memory
        
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names: List[str] = []
        
        self._imputers: Dict[str, SimpleImputer] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._onehot_encoders: Dict[str, OneHotEncoder] = {}
        self._category_mappings: Dict[str, Dict] = {}
        
    def fit(self, df: pd.DataFrame, target_col: str = 'isFraud') -> 'FraudDataPreprocessor':
        """Fit preprocessor on training data."""
        logger.info("Fitting preprocessor...")
        
        # Identify feature types
        self._identify_feature_types(df, target_col)
        
        # Fit numeric transformers
        for col in self.numeric_features:
            # Imputer
            imputer = SimpleImputer(strategy=self.handle_missing)
            imputer.fit(df[[col]])
            self._imputers[col] = imputer
            
            # Scaler
            if self.scale_features:
                scaler = StandardScaler()
                scaler.fit(df[[col]])
                self._scalers[col] = scaler
        
        # Fit categorical transformers
        for col in self.categorical_features:
            # Handle missing values as a category
            df[col] = df[col].fillna('MISSING')
            
            # Limit categories
            value_counts = df[col].value_counts()
            top_categories = value_counts.head(self.max_categories).index.tolist()
            self._category_mappings[col] = {cat: i for i, cat in enumerate(top_categories)}
            
            # Label encoder for high cardinality
            le = LabelEncoder()
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'OTHER')
            le.fit(df[col])
            self._label_encoders[col] = le
        
        logger.info(f"Fitted preprocessor: {len(self.numeric_features)} numeric, "
                   f"{len(self.categorical_features)} categorical features")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        df = df.copy()
        
        # Transform numeric features
        for col in self.numeric_features:
            if col in df.columns:
                # Impute
                if col in self._imputers:
                    df[col] = self._imputers[col].transform(df[[col]])
                
                # Scale
                if self.scale_features and col in self._scalers:
                    df[col] = self._scalers[col].transform(df[[col]])
        
        # Transform categorical features
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('MISSING')
                df[col] = df[col].apply(
                    lambda x: x if x in self._category_mappings[col] else 'OTHER'
                )
                if col in self._label_encoders:
                    df[col] = self._label_encoders[col].transform(df[col])
        
        # Reduce memory usage
        if self.reduce_memory:
            df = self._reduce_memory(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'isFraud') -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, target_col).transform(df)
    
    def _identify_feature_types(self, df: pd.DataFrame, target_col: str):
        """Identify numeric and categorical features."""
        exclude_cols = [target_col, 'TransactionID', 'TransactionDT']
        
        for col in df.columns:
            if col in exclude_cols:
                continue
                
            if df[col].dtype in ['object', 'category']:
                self.categorical_features.append(col)
            elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                n_unique = df[col].nunique()
                if n_unique <= self.categorical_threshold:
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
        
        self.feature_names = self.numeric_features + self.categorical_features
    
    def _reduce_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage by downcasting numeric types."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df
    
    def save(self, path: str):
        """Save preprocessor state."""
        joblib.dump({
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'imputers': self._imputers,
            'scalers': self._scalers,
            'label_encoders': self._label_encoders,
            'category_mappings': self._category_mappings,
            'params': {
                'categorical_threshold': self.categorical_threshold,
                'max_categories': self.max_categories,
                'scale_features': self.scale_features
            }
        }, path)
        logger.info(f"Saved preprocessor to {path}")
    
    def load(self, path: str):
        """Load preprocessor state."""
        state = joblib.load(path)
        self.numeric_features = state['numeric_features']
        self.categorical_features = state['categorical_features']
        self.feature_names = state['feature_names']
        self._imputers = state['imputers']
        self._scalers = state['scalers']
        self._label_encoders = state['label_encoders']
        self._category_mappings = state['category_mappings']
        
        params = state['params']
        self.categorical_threshold = params['categorical_threshold']
        self.max_categories = params['max_categories']
        self.scale_features = params['scale_features']
        
        logger.info(f"Loaded preprocessor from {path}")
        return self


class ClassBalancer:
    """
    Handle class imbalance for fraud detection.
    Supports multiple strategies without data leakage.
    """
    
    def __init__(self, method: str = 'class_weight', sampling_ratio: float = 0.1):
        self.method = method
        self.sampling_ratio = sampling_ratio
        self.class_weights: Optional[Dict] = None
        
    def compute_class_weights(self, y: np.ndarray) -> Dict:
        """Compute balanced class weights."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        self.class_weights = {c: w for c, w in zip(classes, weights)}
        
        logger.info(f"Computed class weights: {self.class_weights}")
        return self.class_weights
    
    def undersample_majority(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Undersample majority class."""
        df = X.copy()
        df['target'] = y
        
        fraud_df = df[df['target'] == 1]
        legit_df = df[df['target'] == 0]
        
        # Sample majority class
        n_fraud = len(fraud_df)
        n_legit_target = int(n_fraud / self.sampling_ratio)
        
        legit_sampled = legit_df.sample(n=min(n_legit_target, len(legit_df)), random_state=random_state)
        
        balanced_df = pd.concat([fraud_df, legit_sampled])
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        y_balanced = balanced_df['target']
        X_balanced = balanced_df.drop('target', axis=1)
        
        logger.info(f"Undersampled: {len(X_balanced)} samples, fraud rate: {y_balanced.mean():.4f}")
        
        return X_balanced, y_balanced


def create_weighted_sampler(y: np.ndarray) -> Optional['WeightedRandomSampler']:
    """Create PyTorch weighted sampler for imbalanced data."""
    try:
        from torch.utils.data import WeightedRandomSampler
        import torch
        
        class_counts = np.bincount(y.astype(int))
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[y.astype(int)]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(y),
            replacement=True
        )
        return sampler
    except ImportError:
        logger.warning("PyTorch not available, skipping sampler creation")
        return None
