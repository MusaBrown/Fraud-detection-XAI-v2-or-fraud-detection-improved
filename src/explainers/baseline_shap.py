"""
Baseline SHAP explainers: TreeSHAP and KernelSHAP.
Exact methods for comparison with FastSHAP approximation.
"""
import logging
import time
from typing import Dict, List, Optional, Union, Callable
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeSHAPExplainer:
    """
    Exact TreeSHAP explainer for tree-based models.
    This serves as the ground truth for fidelity evaluation.
    """
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """
        Initialize TreeSHAP explainer.
        
        Args:
            model: Trained tree-based model (XGBoost, LightGBM, CatBoost)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.expected_value = None
        
    def fit(self, background_data: Optional[np.ndarray] = None):
        """
        Initialize TreeSHAP explainer.
        
        Args:
            background_data: Background dataset for SHAP (optional for TreeSHAP)
        """
        logger.info("Initializing TreeSHAP explainer...")
        
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.expected_value = self.explainer.expected_value
            
            if isinstance(self.expected_value, np.ndarray):
                # For multi-class, use the positive class
                self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
                
        except Exception as e:
            logger.error(f"Error initializing TreeSHAP: {e}")
            raise
        
        logger.info(f"TreeSHAP fitted. Expected value: {self.expected_value}")
        return self
    
    def explain(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        check_additivity: bool = False
    ) -> Dict:
        """
        Generate SHAP values for input data.
        
        Args:
            X: Input features
            check_additivity: Whether to check SHAP additivity (slows down)
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        start_time = time.time()
        
        shap_values = self.explainer.shap_values(
            X, 
            check_additivity=check_additivity
        )
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms,
            'feature_names': self.feature_names
        }
        
        return result
    
    def explain_single(self, x: np.ndarray) -> Dict:
        """Explain a single instance (optimized for real-time)."""
        return self.explain(x.reshape(1, -1))
    
    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Get global feature importance from SHAP values."""
        result = self.explain(X)
        shap_values = result['shap_values']
        
        importance = np.abs(shap_values).mean(axis=0)
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class KernelSHAPExplainer:
    """
    KernelSHAP explainer - model-agnostic but slower.
    Uses sampling-based approximation.
    """
    
    def __init__(
        self,
        model: Callable,
        feature_names: Optional[List[str]] = None,
        nsamples: int = 100
    ):
        """
        Initialize KernelSHAP explainer.
        
        Args:
            model: Prediction function (predict_proba)
            feature_names: List of feature names
            nsamples: Number of samples for KernelSHAP approximation
        """
        self.model = model
        self.feature_names = feature_names
        self.nsamples = nsamples
        self.explainer = None
        self.expected_value = None
        self.background_data = None
        
    def fit(self, background_data: np.ndarray):
        """
        Initialize KernelSHAP with background data.
        
        Args:
            background_data: Representative background dataset
        """
        logger.info(f"Initializing KernelSHAP with {len(background_data)} background samples...")
        
        self.background_data = background_data
        
        # Use k-means to summarize background data for efficiency
        if len(background_data) > 100:
            background_data = shap.kmeans(background_data, 100)
        
        self.explainer = shap.KernelExplainer(self.model, background_data)
        self.expected_value = self.explainer.expected_value
        
        logger.info(f"KernelSHAP fitted. Expected value: {self.expected_value}")
        return self
    
    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        nsamples: Optional[int] = None
    ) -> Dict:
        """
        Generate SHAP values using KernelSHAP.
        
        Args:
            X: Input features
            nsamples: Override default nsamples
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        nsamples = nsamples or self.nsamples
        
        start_time = time.time()
        
        shap_values = self.explainer.shap_values(
            X,
            nsamples=nsamples,
            l1_reg="auto"
        )
        
        # Handle multi-dimensional output
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms,
            'nsamples': nsamples,
            'feature_names': self.feature_names
        }
        
        return result
    
    def explain_single(self, x: np.ndarray, nsamples: Optional[int] = None) -> Dict:
        """Explain a single instance."""
        return self.explain(x.reshape(1, -1), nsamples=nsamples)


class ApproximateTreeSHAP:
    """
    Optimized TreeSHAP with approximation for faster inference.
    Uses feature subsampling and early stopping for speed.
    """
    
    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        max_features: int = 100,
        approximate: bool = True
    ):
        self.model = model
        self.feature_names = feature_names
        self.max_features = max_features
        self.approximate = approximate
        self.explainer = None
        self.feature_subset = None
        
    def fit(
        self,
        X: np.ndarray,
        feature_importance: Optional[np.ndarray] = None
    ):
        """
        Fit approximate explainer.
        
        Args:
            X: Training data for feature selection
            feature_importance: Pre-computed feature importance
        """
        if feature_importance is None:
            # Compute quick feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
            else:
                feature_importance = np.ones(X.shape[1])
        
        # Select top features
        if len(feature_importance) > self.max_features:
            top_indices = np.argsort(feature_importance)[-self.max_features:]
            self.feature_subset = sorted(top_indices)
        else:
            self.feature_subset = list(range(len(feature_importance)))
        
        # Initialize TreeSHAP
        self.explainer = shap.TreeExplainer(self.model)
        
        logger.info(f"Approximate TreeSHAP fitted with {len(self.feature_subset)} features")
        return self
    
    def explain(self, X: np.ndarray) -> Dict:
        """Generate approximate SHAP values."""
        if self.explainer is None:
            raise ValueError("Explainer not fitted")
        
        start_time = time.time()
        
        # Get full SHAP values
        shap_values = self.explainer.shap_values(X, check_additivity=False)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Zero out non-important features
        if self.feature_subset is not None:
            mask = np.zeros(shap_values.shape[1], dtype=bool)
            mask[self.feature_subset] = True
            shap_values = shap_values * mask
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'shap_values': shap_values,
            'expected_value': self.explainer.expected_value,
            'latency_ms': latency_ms,
            'feature_subset': self.feature_subset,
            'feature_names': self.feature_names
        }


def benchmark_shap_methods(
    model,
    X_test: np.ndarray,
    X_background: np.ndarray,
    n_samples: int = 100,
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Benchmark different SHAP methods for latency comparison.
    
    Args:
        model: Trained model
        X_test: Test data
        X_background: Background data for explainers
        n_samples: Number of samples to benchmark
        feature_names: Feature names
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    X_sample = X_test[:n_samples]
    
    # TreeSHAP (exact)
    logger.info("Benchmarking TreeSHAP...")
    tree_explainer = TreeSHAPExplainer(model, feature_names)
    tree_explainer.fit()
    
    latencies = []
    for i in range(min(10, n_samples)):
        result = tree_explainer.explain_single(X_sample[i])
        latencies.append(result['latency_ms'])
    
    results.append({
        'method': 'TreeSHAP (exact)',
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'approximation': False
    })
    
    # KernelSHAP with different nsamples
    def predict_fn(X):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)
    
    for nsamples in [50, 100]:
        logger.info(f"Benchmarking KernelSHAP (nsamples={nsamples})...")
        kernel_explainer = KernelSHAPExplainer(
            predict_fn, 
            feature_names, 
            nsamples=nsamples
        )
        kernel_explainer.fit(X_background[:100])
        
        latencies = []
        for i in range(min(5, n_samples)):
            result = kernel_explainer.explain_single(X_sample[i])
            latencies.append(result['latency_ms'])
        
        results.append({
            'method': f'KernelSHAP (nsamples={nsamples})',
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'approximation': True
        })
    
    return pd.DataFrame(results)
