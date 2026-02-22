"""
Dataset loaders for IEEE-CIS and ULB Credit Card Fraud Detection datasets.
Implements temporal train-test splits to avoid data leakage.
"""
import os
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IEEECISLoader:
    """
    Loader for IEEE-CIS Fraud Detection dataset.
    Expected files: train_transaction.csv, train_identity.csv, 
                    test_transaction.csv, test_identity.csv
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.transaction_cols = [
            'TransactionID', 'TransactionDT', 'TransactionAmt',
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'dist1', 'dist2',
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'isFraud'
        ]
        
    def load(self, use_sample: bool = False, sample_frac: float = 0.1) -> pd.DataFrame:
        """
        Load and merge IEEE-CIS transaction and identity data.
        
        Args:
            use_sample: If True, use a sample of the data for development
            sample_frac: Fraction of data to sample if use_sample is True
            
        Returns:
            Merged DataFrame with all features
        """
        logger.info("Loading IEEE-CIS dataset...")
        
        # Load transaction data
        train_transaction_path = self.data_dir / "train_transaction.csv"
        train_identity_path = self.data_dir / "train_identity.csv"
        
        if not train_transaction_path.exists():
            logger.warning(f"Dataset not found at {train_transaction_path}. Creating synthetic data...")
            return self._create_synthetic_data(n_samples=590000, n_features=434)
        
        # Read transaction data
        df_trans = pd.read_csv(train_transaction_path)
        
        # Read identity data if available
        if train_identity_path.exists():
            df_id = pd.read_csv(train_identity_path)
            df = df_trans.merge(df_id, on='TransactionID', how='left')
        else:
            df = df_trans
            
        if use_sample:
            df = df.sample(frac=sample_frac, random_state=42)
            
        logger.info(f"Loaded {len(df)} transactions with {df.shape[1]} features")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.4f}")
        
        return df
    
    def _create_synthetic_data(self, n_samples: int = 590000, n_features: int = 434) -> pd.DataFrame:
        """Create synthetic data mimicking IEEE-CIS structure for testing."""
        logger.info(f"Creating synthetic dataset: {n_samples} samples, {n_features} features")
        
        np.random.seed(42)
        
        data = {
            'TransactionID': range(n_samples),
            'TransactionDT': np.cumsum(np.random.exponential(100, n_samples)),
            'TransactionAmt': np.random.lognormal(4, 1.5, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.965, 0.035]),
        }
        
        # Card features
        for i in range(1, 7):
            if i in [4, 6]:  # Categorical
                data[f'card{i}'] = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
            else:
                data[f'card{i}'] = np.random.randint(1, 10000, n_samples)
        
        # Address and distance
        data['addr1'] = np.random.randint(1, 500, n_samples)
        data['addr2'] = np.random.randint(1, 100, n_samples)
        data['dist1'] = np.random.exponential(10, n_samples)
        data['dist2'] = np.random.exponential(10, n_samples)
        
        # C features (count features)
        for i in range(1, 15):
            data[f'C{i}'] = np.random.poisson(5, n_samples)
        
        # D features (timedelta features)
        for i in range(1, 16):
            data[f'D{i}'] = np.random.exponential(50, n_samples)
        
        # M features (match features - categorical)
        for i in range(1, 10):
            data[f'M{i}'] = np.random.choice(['T', 'F', np.nan], n_samples, p=[0.4, 0.4, 0.2])
        
        # V features (engineered features)
        for i in range(1, min(340, n_features - 100)):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame(data)
        
        # Add fraud patterns
        fraud_indices = df[df['isFraud'] == 1].index
        df.loc[fraud_indices, 'TransactionAmt'] *= np.random.uniform(1.5, 5, len(fraud_indices))
        df.loc[fraud_indices, 'C1'] += np.random.poisson(10, len(fraud_indices))
        df.loc[fraud_indices, 'V1'] += np.random.normal(2, 0.5, len(fraud_indices))
        
        logger.info(f"Created synthetic data: {len(df)} transactions, fraud rate: {df['isFraud'].mean():.4f}")
        return df


class ULBLoader:
    """
    Loader for ULB (Université Libre de Bruxelles) Credit Card Fraud dataset.
    This dataset contains only numerical features (PCA transformed).
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load(self) -> pd.DataFrame:
        """
        Load ULB Credit Card Fraud dataset.
        
        Returns:
            DataFrame with Time, Amount, V1-V28, and Class columns
        """
        logger.info("Loading ULB dataset...")
        
        file_path = self.data_dir / "creditcard.csv"
        
        if not file_path.exists():
            logger.warning(f"Dataset not found at {file_path}. Creating synthetic data...")
            return self._create_synthetic_data()
        
        df = pd.read_csv(file_path)
        
        # Rename Class to isFraud for consistency
        df = df.rename(columns={'Class': 'isFraud'})
        
        logger.info(f"Loaded {len(df)} transactions with {df.shape[1]} features")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.6f}")
        
        return df
    
    def _create_synthetic_data(self, n_samples: int = 284807) -> pd.DataFrame:
        """Create synthetic data mimicking ULB structure."""
        logger.info(f"Creating synthetic ULB dataset: {n_samples} samples")
        
        np.random.seed(42)
        
        data = {
            'Time': np.cumsum(np.random.exponential(10, n_samples)),
            'Amount': np.random.lognormal(3, 1, n_samples),
            'isFraud': np.random.choice([0, 1], n_samples, p=[0.99828, 0.00172]),
        }
        
        # V features (PCA components)
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        df = pd.DataFrame(data)
        
        # Add fraud patterns
        fraud_indices = df[df['isFraud'] == 1].index
        df.loc[fraud_indices, 'Amount'] *= np.random.uniform(2, 10, len(fraud_indices))
        df.loc[fraud_indices, 'V1'] += np.random.normal(3, 1, len(fraud_indices))
        df.loc[fraud_indices, 'V2'] -= np.random.normal(2, 1, len(fraud_indices))
        df.loc[fraud_indices, 'V3'] += np.random.normal(2, 0.5, len(fraud_indices))
        
        logger.info(f"Created synthetic ULB data: {len(df)} transactions, fraud rate: {df['isFraud'].mean():.6f}")
        return df


def temporal_split(
    df: pd.DataFrame,
    time_col: str = 'TransactionDT',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform temporal train-validation-test split.
    
    Args:
        df: Input DataFrame
        time_col: Column containing timestamp
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining data)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Sort by time
    df_sorted = df.sort_values(time_col)
    
    n = len(df_sorted)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    
    train_df = df_sorted.iloc[:val_start].copy()
    val_df = df_sorted.iloc[val_start:test_start].copy()
    test_df = df_sorted.iloc[test_start:].copy()
    
    logger.info(f"Temporal split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Fraud rates - Train: {train_df['isFraud'].mean():.4f}, "
                f"Val: {val_df['isFraud'].mean():.4f}, "
                f"Test: {test_df['isFraud'].mean():.4f}")
    
    return train_df, val_df, test_df


def get_feature_columns(df: pd.DataFrame, target_col: str = 'isFraud', id_col: str = 'TransactionID') -> list:
    """Get list of feature columns (excluding target and ID)."""
    exclude_cols = [target_col, id_col]
    return [col for col in df.columns if col not in exclude_cols]
