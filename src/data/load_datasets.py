"""
Dataset loader for ULB Credit Card Fraud Detection dataset.

This project ONLY uses the real ULB Credit Card Fraud dataset.
NO synthetic data is used - all experiments run on real data only.

Dataset source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download: Use download_ulb_data.py or download manually to data/raw/creditcard.csv

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


class ULBLoader:
    """
    Loader for ULB (Université Libre de Bruxelles) Credit Card Fraud dataset.
    This dataset contains only numerical features (PCA transformed).
    
    IMPORTANT: This loader ONLY works with real data. No synthetic fallback.
    Download the dataset from Kaggle before use.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load(self) -> pd.DataFrame:
        """
        Load ULB Credit Card Fraud dataset.
        
        Returns:
            DataFrame with Time, Amount, V1-V28, and Class columns
            
        Raises:
            FileNotFoundError: If creditcard.csv is not found in data_dir
        """
        logger.info("Loading ULB dataset...")
        
        file_path = self.data_dir / "creditcard.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"ULB Credit Card Fraud dataset not found at: {file_path}\n\n"
                "This project ONLY uses real ULB data. No synthetic data is used.\n\n"
                "To download the dataset:\n"
                "1. Run: python download_ulb_data.py\n"
                "2. Or download manually from:\n"
                "   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "   and place creditcard.csv in data/raw/"
            )
        
        df = pd.read_csv(file_path)
        
        # Rename Class to isFraud for consistency
        df = df.rename(columns={'Class': 'isFraud'})
        
        logger.info(f"Loaded {len(df)} transactions with {df.shape[1]} features")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.6f}")
        
        return df


def temporal_split(
    df: pd.DataFrame,
    time_col: str = 'Time',
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform temporal train-validation-test split.
    
    Args:
        df: Input DataFrame
        time_col: Column containing timestamp (default 'Time' for ULB dataset)
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
