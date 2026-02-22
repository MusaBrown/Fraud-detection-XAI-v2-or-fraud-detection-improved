"""
Optimized LIME explainer for fraud detection.
Includes hyperparameter tuning for faster, more stable explanations.
"""
import logging
import time
from typing import Dict, List, Optional, Union, Callable, Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedLIME:
    """
    Optimized LIME explainer with tuned hyperparameters for speed.
    """
    
    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        mode: str = 'classification',
        kernel_width: Optional[float] = None,
        n_samples: int = 500,
        discretize_continuous: bool = True,
        sample_around_instance: bool = True
    ):
        """
        Initialize optimized LIME explainer.
        
        Args:
            training_data: Background training data
            feature_names: Feature names
            categorical_features: Indices of categorical features
            mode: 'classification' or 'regression'
            kernel_width: Kernel width for distance (None = auto)
            n_samples: Number of perturbed samples
            discretize_continuous: Whether to discretize continuous features
            sample_around_instance: Sample around instance vs global
        """
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.mode = mode
        self.n_samples = n_samples
        self.training_data = training_data
        
        # Auto-compute kernel width if not provided
        if kernel_width is None:
            # Standard LIME formula: sqrt(n_features) * 0.75
            kernel_width = np.sqrt(training_data.shape[1]) * 0.75
        
        self.kernel_width = kernel_width
        
        # Initialize LIME explainer
        logger.info(f"Initializing LIME with kernel_width={kernel_width:.2f}, n_samples={n_samples}")
        
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=['legitimate', 'fraud'],
            categorical_features=categorical_features,
            mode=mode,
            kernel_width=kernel_width,
            discretize_continuous=discretize_continuous,
            sample_around_instance=sample_around_instance,
            verbose=False
        )
        
    def explain(
        self,
        instance: np.ndarray,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: Optional[int] = None
    ) -> Dict:
        """
        Generate LIME explanation.
        
        Args:
            instance: Single instance to explain
            predict_fn: Prediction function (predict_proba)
            num_features: Number of features to include
            num_samples: Override default n_samples
            
        Returns:
            Dictionary with explanation and latency
        """
        num_samples = num_samples or self.n_samples
        
        start_time = time.time()
        
        # Get explanation
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert to standardized format
        feature_weights = explanation.as_list()
        
        # Create SHAP-like array
        shap_like = np.zeros(len(instance))
        
        for feature_desc, weight in feature_weights:
            # Parse feature index from description
            try:
                if ' < ' in feature_desc or ' > ' in feature_desc or ' <= ' in feature_desc:
                    # Extract feature name
                    feature_name = feature_desc.split(' ')[0]
                    if self.feature_names and feature_name in self.feature_names:
                        idx = self.feature_names.index(feature_name)
                        shap_like[idx] = weight
            except:
                pass
        
        return {
            'lime_explanation': explanation,
            'feature_weights': feature_weights,
            'shap_like_values': shap_like,
            'latency_ms': latency_ms,
            'intercept': explanation.intercept[1] if self.mode == 'classification' else explanation.intercept,
            'score': explanation.score,
            'local_pred': explanation.local_pred
        }
    
    def explain_batch(
        self,
        X: np.ndarray,
        predict_fn: Callable,
        num_features: int = 10
    ) -> List[Dict]:
        """Explain multiple instances."""
        results = []
        for i in range(len(X)):
            result = self.explain(X[i], predict_fn, num_features)
            results.append(result)
        return results


class LIMETuner:
    """
    Hyperparameter tuner for LIME to optimize speed/fidelity trade-off.
    """
    
    def __init__(
        self,
        training_data: np.ndarray,
        validation_data: np.ndarray,
        true_shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None
    ):
        self.training_data = training_data
        self.validation_data = validation_data
        self.true_shap_values = true_shap_values
        self.feature_names = feature_names
        
    def tune(
        self,
        predict_fn: Callable,
        kernel_widths: List[float] = None,
        n_samples_options: List[int] = None
    ) -> pd.DataFrame:
        """
        Tune LIME hyperparameters.
        
        Args:
            predict_fn: Prediction function
            kernel_widths: List of kernel widths to try
            n_samples_options: List of n_samples to try
            
        Returns:
            DataFrame with tuning results
        """
        if kernel_widths is None:
            n_features = self.training_data.shape[1]
            base = np.sqrt(n_features) * 0.75
            kernel_widths = [base * 0.5, base, base * 1.5, base * 2]
        
        if n_samples_options is None:
            n_samples_options = [200, 500, 1000]
        
        results = []
        n_val = min(len(self.validation_data), 100)
        
        for kernel_width in kernel_widths:
            for n_samples in n_samples_options:
                logger.info(f"Testing kernel_width={kernel_width:.2f}, n_samples={n_samples}")
                
                try:
                    lime = OptimizedLIME(
                        training_data=self.training_data,
                        feature_names=self.feature_names,
                        kernel_width=kernel_width,
                        n_samples=n_samples
                    )
                    
                    # Measure latency and fidelity
                    latencies = []
                    fidelity_scores = []
                    
                    for i in range(n_val):
                        result = lime.explain(
                            self.validation_data[i],
                            predict_fn,
                            num_samples=n_samples
                        )
                        latencies.append(result['latency_ms'])
                        
                        # Compute fidelity (correlation with SHAP)
                        lime_values = result['shap_like_values']
                        true_values = self.true_shap_values[i]
                        
                        if np.std(lime_values) > 0 and np.std(true_values) > 0:
                            from scipy.stats import pearsonr
                            r, _ = pearsonr(lime_values, true_values)
                            fidelity_scores.append(abs(r))
                    
                    results.append({
                        'kernel_width': kernel_width,
                        'n_samples': n_samples,
                        'mean_latency_ms': np.mean(latencies),
                        'p95_latency_ms': np.percentile(latencies, 95),
                        'mean_fidelity': np.mean(fidelity_scores),
                        'config_score': np.mean(fidelity_scores) / (np.mean(latencies) / 1000 + 1)
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing config: {e}")
        
        return pd.DataFrame(results)
    
    def get_optimal_config(self, results_df: pd.DataFrame) -> Dict:
        """Get optimal configuration from tuning results."""
        best_idx = results_df['config_score'].idxmax()
        best = results_df.loc[best_idx]
        
        return {
            'kernel_width': best['kernel_width'],
            'n_samples': int(best['n_samples']),
            'expected_latency_ms': best['mean_latency_ms'],
            'expected_fidelity': best['mean_fidelity']
        }


def benchmark_lime(
    training_data: np.ndarray,
    test_data: np.ndarray,
    true_shap_values: np.ndarray,
    predict_fn: Callable,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Benchmark LIME performance.
    
    Args:
        training_data: Background data
        test_data: Test instances
        true_shap_values: Ground truth SHAP values
        predict_fn: Prediction function
        feature_names: Feature names
        
    Returns:
        Benchmark results
    """
    logger.info("Benchmarking LIME...")
    
    lime = OptimizedLIME(
        training_data=training_data,
        feature_names=feature_names,
        n_samples=500
    )
    
    n_test = min(len(test_data), 50)
    latencies = []
    fidelity_scores = []
    
    for i in range(n_test):
        result = lime.explain(test_data[i], predict_fn)
        latencies.append(result['latency_ms'])
        
        # Fidelity
        lime_values = result['shap_like_values']
        true_values = true_shap_values[i]
        
        if np.std(lime_values) > 0 and np.std(true_values) > 0:
            from scipy.stats import pearsonr
            r, _ = pearsonr(lime_values, true_values)
            fidelity_scores.append(abs(r))
    
    return {
        'method': 'LIME',
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'mean_fidelity': np.mean(fidelity_scores) if fidelity_scores else 0,
        'std_fidelity': np.std(fidelity_scores) if fidelity_scores else 0
    }
