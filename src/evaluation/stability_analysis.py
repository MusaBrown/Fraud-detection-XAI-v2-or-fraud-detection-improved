"""
Stability analysis module for explanation methods.
Evaluates explanation consistency under perturbations and model retraining.
"""
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StabilityMetrics:
    """Metrics for explanation stability."""
    mean_cv: float  # Coefficient of variation
    max_cv: float
    perturbation_consistency: float
    rank_correlation_mean: float
    rank_correlation_std: float
    sign_consistency: float


class StabilityAnalyzer:
    """
    Analyze stability of explanations under various perturbations.
    """
    
    def __init__(self, explainer_fn: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize analyzer.
        
        Args:
            explainer_fn: Function that takes features and returns SHAP values
        """
        self.explainer_fn = explainer_fn
        
    def analyze_input_perturbation(
        self,
        X: np.ndarray,
        n_perturbations: int = 10,
        noise_level: float = 0.01
    ) -> StabilityMetrics:
        """
        Analyze stability under small input perturbations.
        
        Args:
            X: Input samples (n_samples, n_features)
            n_perturbations: Number of perturbations per sample
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            StabilityMetrics
        """
        logger.info(f"Analyzing input perturbation stability on {len(X)} samples...")
        
        all_cv = []
        all_consistency = []
        
        for i in range(len(X)):
            x_base = X[i]
            explanations = []
            
            # Generate perturbed versions
            for _ in range(n_perturbations):
                x_perturbed = x_base + np.random.normal(0, noise_level, size=x_base.shape)
                exp = self.explainer_fn(x_perturbed.reshape(1, -1))
                explanations.append(exp.flatten())
            
            explanations = np.array(explanations)
            
            # Coefficient of variation per feature
            for j in range(explanations.shape[1]):
                feature_vals = explanations[:, j]
                mean_val = np.mean(feature_vals)
                if abs(mean_val) > 1e-10:
                    cv = np.std(feature_vals) / abs(mean_val)
                    all_cv.append(cv)
            
            # Pairwise consistency
            for j in range(len(explanations)):
                for k in range(j + 1, len(explanations)):
                    r, _ = pearsonr(explanations[j], explanations[k])
                    all_consistency.append(r)
        
        return StabilityMetrics(
            mean_cv=np.mean(all_cv) if all_cv else 0,
            max_cv=np.max(all_cv) if all_cv else 0,
            perturbation_consistency=np.mean(all_consistency) if all_consistency else 0,
            rank_correlation_mean=np.mean(all_consistency) if all_consistency else 0,
            rank_correlation_std=np.std(all_consistency) if all_consistency else 0,
            sign_consistency=self._compute_sign_consistency(np.array(explanations))
        )
    
    def analyze_feature_subset_stability(
        self,
        X: np.ndarray,
        n_subsets: int = 5,
        subset_ratio: float = 0.9
    ) -> Dict:
        """
        Analyze stability when using different feature subsets.
        
        Args:
            X: Input samples
            n_subsets: Number of random subsets to test
            subset_ratio: Ratio of features to keep
            
        Returns:
            Stability metrics
        """
        logger.info("Analyzing feature subset stability...")
        
        n_features = X.shape[1]
        n_keep = int(n_features * subset_ratio)
        
        all_explanations = []
        
        for _ in range(n_subsets):
            # Random feature subset
            subset_idx = np.random.choice(n_features, n_keep, replace=False)
            X_subset = X[:, subset_idx]
            
            # Get explanation (may need to modify explainer for this)
            exp = self.explainer_fn(X)
            all_explanations.append(exp)
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(all_explanations)):
            for j in range(i + 1, len(all_explanations)):
                flat_i = all_explanations[i].flatten()
                flat_j = all_explanations[j].flatten()
                r, _ = pearsonr(flat_i, flat_j)
                correlations.append(r)
        
        return {
            'mean_correlation': np.mean(correlations),
            'min_correlation': np.min(correlations),
            'std_correlation': np.std(correlations)
        }
    
    def _compute_sign_consistency(self, explanations: np.ndarray) -> float:
        """Compute sign consistency across explanations."""
        # For each feature, check if sign is consistent
        consistent_signs = 0
        total = 0
        
        for j in range(explanations.shape[1]):
            signs = np.sign(explanations[:, j])
            # Check if all signs are the same
            if np.all(signs == signs[0]) or np.all(signs[signs != 0] == signs[signs != 0][0]):
                consistent_signs += 1
            total += 1
        
        return consistent_signs / total if total > 0 else 0
    
    def analyze_temporal_stability(
        self,
        X: np.ndarray,
        window_size: int = 100
    ) -> pd.DataFrame:
        """
        Analyze explanation stability over time (temporal consistency).
        
        Args:
            X: Time-series of samples
            window_size: Size of sliding window
            
        Returns:
            DataFrame with temporal stability metrics
        """
        logger.info("Analyzing temporal stability...")
        
        results = []
        
        for start in range(0, len(X) - window_size, window_size // 2):
            window = X[start:start + window_size]
            
            # Get explanations for window
            explanations = self.explainer_fn(window)
            
            # Compute global feature importance for window
            global_imp = np.abs(explanations).mean(axis=0)
            
            results.append({
                'window_start': start,
                'window_end': start + window_size,
                'mean_importance': global_imp.mean(),
                'std_importance': global_imp.std()
            })
        
        df = pd.DataFrame(results)
        
        # Compute stability as correlation between consecutive windows
        if len(df) > 1:
            df['importance_stability'] = df['mean_importance'].rolling(2).apply(
                lambda x: 1 - abs(x.iloc[1] - x.iloc[0]) / (abs(x.iloc[0]) + 1e-10)
            )
        
        return df


class ModelRetrainingStability:
    """
    Analyze explanation stability across model retraining events.
    """
    
    def __init__(self):
        self.model_snapshots: List[Tuple[Any, Any]] = []  # (model, explainer)
        
    def add_model_snapshot(self, model, explainer):
        """Add a model snapshot after retraining."""
        self.model_snapshots.append((model, explainer))
        
    def compute_explanation_drift(
        self,
        X_test: np.ndarray,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Compute explanation drift across model versions.
        
        Args:
            X_test: Test samples
            top_k: Number of top features to compare
            
        Returns:
            DataFrame with drift metrics
        """
        if len(self.model_snapshots) < 2:
            logger.warning("Need at least 2 model snapshots to compute drift")
            return pd.DataFrame()
        
        results = []
        
        for i in range(len(self.model_snapshots) - 1):
            _, explainer_a = self.model_snapshots[i]
            _, explainer_b = self.model_snapshots[i + 1]
            
            # Get explanations
            exp_a = explainer_a(X_test)
            exp_b = explainer_b(X_test)
            
            # Global importance
            global_a = np.abs(exp_a).mean(axis=0)
            global_b = np.abs(exp_b).mean(axis=0)
            
            # Rank correlation
            rank_a = np.argsort(np.argsort(-global_a))
            rank_b = np.argsort(np.argsort(-global_b))
            
            rank_corr, _ = spearmanr(rank_a, rank_b)
            
            # Top-k overlap
            top_a = set(np.argsort(-global_a)[:top_k])
            top_b = set(np.argsort(-global_b)[:top_k])
            top_k_overlap = len(top_a & top_b) / top_k
            
            results.append({
                'from_version': i,
                'to_version': i + 1,
                'rank_correlation': rank_corr,
                'top_k_overlap': top_k_overlap,
                'mean_abs_diff': np.mean(np.abs(global_a - global_b))
            })
        
        return pd.DataFrame(results)
    
    def plot_explanation_drift(
        self,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot how explanations change across model versions."""
        n_snapshots = len(self.model_snapshots)
        n_features = X_test.shape[1]
        
        # Collect importance across versions
        importance_history = np.zeros((n_snapshots, n_features))
        
        for i, (_, explainer) in enumerate(self.model_snapshots):
            exp = explainer(X_test)
            importance_history[i] = np.abs(exp).mean(axis=0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.imshow(
            importance_history.T,
            aspect='auto',
            cmap='YlOrRd',
            interpolation='nearest'
        )
        
        ax.set_xlabel('Model Version')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance Evolution Across Model Retraining')
        
        if feature_names:
            ax.set_yticks(range(len(feature_names)))
            ax.set_yticklabels(feature_names, fontsize=6)
        
        plt.colorbar(im, ax=ax, label='Mean |SHAP Value|')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved drift plot to {save_path}")
        
        return fig


def compute_jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0


def compute_explanation_stability_score(
    explanations: np.ndarray,
    method: str = 'correlation'
) -> float:
    """
    Compute overall stability score for explanations.
    
    Args:
        explanations: Array of explanations (n_samples, n_features)
        method: 'correlation' or 'variance'
        
    Returns:
        Stability score [0, 1]
    """
    if method == 'correlation':
        # Average pairwise correlation
        correlations = []
        for i in range(len(explanations)):
            for j in range(i + 1, len(explanations)):
                if np.std(explanations[i]) > 0 and np.std(explanations[j]) > 0:
                    r, _ = pearsonr(explanations[i], explanations[j])
                    correlations.append(abs(r))
        return np.mean(correlations) if correlations else 0
    
    elif method == 'variance':
        # Inverse of normalized variance
        variances = np.var(explanations, axis=0)
        means = np.abs(np.mean(explanations, axis=0))
        cvs = variances / (means + 1e-10)
        return 1 / (1 + np.mean(cvs))
    
    return 0.0
