"""
Unified metrics framework for evaluating explanation methods.
Includes: Fidelity, Stability, and Efficiency metrics.
"""
import logging
import time
import psutil
import os
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FIDELITY METRICS - How well explanations match ground truth
# =============================================================================

@dataclass
class FidelityMetrics:
    """
    Fidelity metrics measure how well approximate explanations match ground truth.
    Higher values indicate better agreement with exact methods (e.g., TreeSHAP).
    """
    # Correlation metrics with ground truth
    pearson_r: float = 0.0
    pearson_pvalue: float = 0.0
    spearman_r: float = 0.0
    spearman_pvalue: float = 0.0
    kendall_tau: float = 0.0
    
    # Per-instance fidelity
    pearson_per_instance_mean: float = 0.0
    pearson_per_instance_std: float = 0.0
    
    # Top-k rank preservation
    top5_rank_correlation: float = 0.0
    top10_rank_correlation: float = 0.0
    
    # Magnitude accuracy
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    
    # Sign agreement (direction of feature impact)
    sign_agreement: float = 0.0
    
    # Global feature importance rank correlation
    global_rank_correlation: float = 0.0
    
    # Composite fidelity score [0, 1]
    fidelity_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'pearson_r': self.pearson_r,
            'spearman_r': self.spearman_r,
            'kendall_tau': self.kendall_tau,
            'top5_rank_correlation': self.top5_rank_correlation,
            'top10_rank_correlation': self.top10_rank_correlation,
            'sign_agreement': self.sign_agreement,
            'mse': self.mse,
            'mae': self.mae,
            'rmse': self.rmse,
            'fidelity_score': self.fidelity_score
        }


# =============================================================================
# STABILITY METRICS - Consistency under perturbations
# =============================================================================

@dataclass
class StabilityMetrics:
    """
    Stability metrics measure explanation consistency under small input changes.
    Higher values indicate more robust explanations.
    """
    # Coefficient of variation (lower is more stable)
    mean_cv: float = 0.0
    max_cv: float = 0.0
    
    # Pairwise correlation under perturbation
    perturbation_consistency: float = 0.0
    rank_correlation_mean: float = 0.0
    rank_correlation_std: float = 0.0
    
    # Sign consistency
    sign_consistency: float = 0.0
    
    # Jaccard similarity for top-k features
    top5_jaccard: float = 0.0
    top10_jaccard: float = 0.0
    
    # Relative standard deviation
    rsd_mean: float = 0.0
    rsd_max: float = 0.0
    
    # Composite stability score [0, 1]
    stability_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'mean_cv': self.mean_cv,
            'max_cv': self.max_cv,
            'perturbation_consistency': self.perturbation_consistency,
            'sign_consistency': self.sign_consistency,
            'top5_jaccard': self.top5_jaccard,
            'top10_jaccard': self.top10_jaccard,
            'rsd_mean': self.rsd_mean,
            'stability_score': self.stability_score
        }


# =============================================================================
# EFFICIENCY METRICS - Computational performance
# =============================================================================

@dataclass
class EfficiencyMetrics:
    """
    Efficiency metrics measure computational performance.
    Lower latency and higher throughput indicate better efficiency.
    """
    # Latency metrics (ms)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput metrics
    throughput_tps: float = 0.0  # Transactions per second
    throughput_per_second: float = 0.0  # Explanations per second
    
    # Computational cost
    mean_flops: float = 0.0  # Estimated FLOPs per explanation
    memory_mb: float = 0.0  # Peak memory usage
    
    # Training efficiency (for surrogate models like FastSHAP)
    training_time_sec: float = 0.0
    convergence_epochs: int = 0
    
    # Model size (for neural surrogates)
    model_size_mb: float = 0.0
    
    # Composite efficiency score [0, 1] - higher is more efficient
    efficiency_score: float = 0.0
    
    # Real-time compliance
    meets_p50_50ms: bool = False
    meets_p95_50ms: bool = False
    meets_p99_100ms: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'mean_latency_ms': self.mean_latency_ms,
            'p50_latency_ms': self.p50_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'throughput_tps': self.throughput_tps,
            'memory_mb': self.memory_mb,
            'training_time_sec': self.training_time_sec,
            'model_size_mb': self.model_size_mb,
            'efficiency_score': self.efficiency_score,
            'meets_p95_50ms': self.meets_p95_50ms,
            'meets_p99_100ms': self.meets_p99_100ms
        }


# =============================================================================
# UNIFIED EXPLANATION METRICS
# =============================================================================

@dataclass
class ExplanationMetrics:
    """
    Unified container for all explanation quality metrics.
    Combines Fidelity, Stability, and Efficiency into a single evaluation.
    """
    method_name: str = ""
    
    # Three core metric categories
    fidelity: FidelityMetrics = field(default_factory=FidelityMetrics)
    stability: StabilityMetrics = field(default_factory=StabilityMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    
    # Overall composite score
    overall_score: float = 0.0
    
    # Metadata
    n_samples_evaluated: int = 0
    evaluation_time_sec: float = 0.0
    
    def get_summary(self) -> Dict:
        """Get summary dictionary of all metrics."""
        return {
            'method': self.method_name,
            'fidelity_score': self.fidelity.fidelity_score,
            'stability_score': self.stability.stability_score,
            'efficiency_score': self.efficiency.efficiency_score,
            'overall_score': self.overall_score,
            'p95_latency_ms': self.efficiency.p95_latency_ms,
            'pearson_r': self.fidelity.pearson_r,
            'throughput_tps': self.efficiency.throughput_tps
        }
    
    def to_dataframe_row(self) -> pd.Series:
        """Convert to pandas Series for comparison tables."""
        data = {
            'Method': self.method_name,
            # Fidelity
            'Fidelity Score': f"{self.fidelity.fidelity_score:.4f}",
            'Pearson r': f"{self.fidelity.pearson_r:.4f}",
            'Spearman r': f"{self.fidelity.spearman_r:.4f}",
            'Sign Agreement': f"{self.fidelity.sign_agreement:.4f}",
            # Stability
            'Stability Score': f"{self.stability.stability_score:.4f}",
            'Perturbation Consistency': f"{self.stability.perturbation_consistency:.4f}",
            'Sign Consistency': f"{self.stability.sign_consistency:.4f}",
            # Efficiency
            'Efficiency Score': f"{self.efficiency.efficiency_score:.4f}",
            'P50 Latency (ms)': f"{self.efficiency.p50_latency_ms:.2f}",
            'P95 Latency (ms)': f"{self.efficiency.p95_latency_ms:.2f}",
            'Throughput (TPS)': f"{self.efficiency.throughput_tps:.1f}",
            # Overall
            'Overall Score': f"{self.overall_score:.4f}"
        }
        return pd.Series(data)


class UnifiedExplainerEvaluator:
    """
    Unified evaluator that computes Fidelity, Stability, and Efficiency metrics.
    
    Usage:
        evaluator = UnifiedExplainerEvaluator(ground_truth_explainer=treeshap)
        metrics = evaluator.evaluate_method(
            method_name="FastSHAP",
            explain_fn=fastshap_fn,
            X_test=X_test,
            ground_truth_values=exact_shap_values
        )
        print(f"Fidelity: {metrics.fidelity.fidelity_score:.4f}")
        print(f"Stability: {metrics.stability.stability_score:.4f}")
        print(f"Efficiency: {metrics.efficiency.efficiency_score:.4f}")
    """
    
    def __init__(
        self,
        ground_truth_explainer: Optional[Callable] = None,
        noise_level: float = 0.01,
        n_perturbations: int = 10
    ):
        """
        Initialize unified evaluator.
        
        Args:
            ground_truth_explainer: Exact explainer for fidelity ground truth
            noise_level: Noise level for stability perturbations
            n_perturbations: Number of perturbations for stability testing
        """
        self.ground_truth = ground_truth_explainer
        self.noise_level = noise_level
        self.n_perturbations = n_perturbations
        self.results: List[ExplanationMetrics] = []
        
    def evaluate_method(
        self,
        method_name: str,
        explain_fn: Callable[[np.ndarray], np.ndarray],
        X_test: np.ndarray,
        ground_truth_values: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None
    ) -> ExplanationMetrics:
        """
        Comprehensive evaluation of an explanation method.
        
        Computes all three metrics:
        - Fidelity: vs ground truth (if provided)
        - Stability: under input perturbations
        - Efficiency: latency and throughput
        
        Args:
            method_name: Name of the explanation method
            explain_fn: Function that takes X and returns SHAP-like values
            X_test: Test data
            ground_truth_values: Ground truth SHAP values for fidelity
            n_samples: Number of samples to evaluate (None = all)
            
        Returns:
            ExplanationMetrics with fidelity, stability, and efficiency
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {method_name}")
        logger.info(f"{'='*60}")
        
        eval_start = time.time()
        
        if n_samples and n_samples < len(X_test):
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            X_eval = X_test[indices]
            if ground_truth_values is not None:
                ground_truth_values = ground_truth_values[indices]
        else:
            X_eval = X_test
            
        n_eval = len(X_eval)
        
        # 1. COMPUTE EFFICIENCY METRICS (latency, throughput)
        logger.info("  [1/3] Computing Efficiency metrics...")
        efficiency = self._compute_efficiency(explain_fn, X_eval)
        
        # 2. COMPUTE FIDELITY METRICS (vs ground truth)
        logger.info("  [2/3] Computing Fidelity metrics...")
        fidelity = self._compute_fidelity(explain_fn, X_eval, ground_truth_values)
        
        # 3. COMPUTE STABILITY METRICS (perturbation robustness)
        logger.info("  [3/3] Computing Stability metrics...")
        stability = self._compute_stability(explain_fn, X_eval)
        
        # 4. COMPUTE OVERALL SCORE
        overall = self._compute_overall_score(fidelity, stability, efficiency)
        
        eval_time = time.time() - eval_start
        
        metrics = ExplanationMetrics(
            method_name=method_name,
            fidelity=fidelity,
            stability=stability,
            efficiency=efficiency,
            overall_score=overall,
            n_samples_evaluated=n_eval,
            evaluation_time_sec=eval_time
        )
        
        self.results.append(metrics)
        
        logger.info(f"\n  Evaluation Complete:")
        logger.info(f"    Fidelity Score:  {fidelity.fidelity_score:.4f}")
        logger.info(f"    Stability Score: {stability.stability_score:.4f}")
        logger.info(f"    Efficiency Score: {efficiency.efficiency_score:.4f}")
        logger.info(f"    Overall Score:   {overall:.4f}")
        logger.info(f"    Time: {eval_time:.2f}s")
        
        return metrics
    
    def _compute_efficiency(
        self,
        explain_fn: Callable,
        X: np.ndarray
    ) -> EfficiencyMetrics:
        """Compute efficiency metrics (latency, throughput)."""
        latencies = []
        
        # Warmup
        for _ in range(3):
            explain_fn(X[:1])
        
        # Measure latency
        for i in range(len(X)):
            start = time.time()
            explain_fn(X[i:i+1])
            latencies.append((time.time() - start) * 1000)
        
        latencies = np.array(latencies)
        
        # Compute efficiency score (normalized) [0, 1]
        # P95 latency: 10ms = 1.0, 50ms = 0.8, 100ms = 0.5, 200ms = 0.0
        p95 = float(np.percentile(latencies, 95))
        efficiency_score = max(0.0, min(1.0, 1.0 - (p95 - 10) / 190))
        
        # Throughput
        total_time_sec = np.sum(latencies) / 1000
        throughput = len(X) / total_time_sec if total_time_sec > 0 else 0
        
        return EfficiencyMetrics(
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=p95,
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_tps=throughput,
            throughput_per_second=throughput,
            efficiency_score=efficiency_score,
            meets_p50_50ms=p95 <= 50,
            meets_p95_50ms=p95 <= 50,
            meets_p99_100ms=np.percentile(latencies, 99) <= 100
        )
    
    def _compute_fidelity(
        self,
        explain_fn: Callable,
        X: np.ndarray,
        ground_truth: Optional[np.ndarray]
    ) -> FidelityMetrics:
        """Compute fidelity metrics vs ground truth."""
        if ground_truth is None:
            logger.warning("    No ground truth provided - fidelity metrics will be zero")
            return FidelityMetrics()
        
        # Get approximate explanations
        approx_values = []
        for i in range(len(X)):
            exp = explain_fn(X[i:i+1])
            if isinstance(exp, dict):
                exp = exp.get('shap_values', exp)
            approx_values.append(exp.flatten() if hasattr(exp, 'flatten') else exp[0])
        approx_values = np.array(approx_values)
        
        # Flatten for global correlation
        approx_flat = approx_values.flatten()
        truth_flat = ground_truth.flatten()
        
        # Correlations
        pearson_r, pearson_p = pearsonr(approx_flat, truth_flat)
        spearman_r, spearman_p = spearmanr(approx_flat, truth_flat)
        kendall_tau, _ = kendalltau(approx_flat, truth_flat)
        
        # Per-instance correlation
        per_instance = []
        for i in range(len(approx_values)):
            if np.std(approx_values[i]) > 1e-10 and np.std(ground_truth[i]) > 1e-10:
                r, _ = pearsonr(approx_values[i], ground_truth[i])
                per_instance.append(r)
        
        # Top-k rank correlation
        top5_corr = self._compute_topk_correlation(approx_values, ground_truth, k=5)
        top10_corr = self._compute_topk_correlation(approx_values, ground_truth, k=10)
        
        # Magnitude errors
        mse = mean_squared_error(truth_flat, approx_flat)
        mae = mean_absolute_error(truth_flat, approx_flat)
        rmse = np.sqrt(mse)
        
        # Sign agreement
        sign_agree = np.mean(np.sign(approx_flat) == np.sign(truth_flat))
        
        # Global rank correlation
        global_approx = np.abs(approx_values).mean(axis=0)
        global_truth = np.abs(ground_truth).mean(axis=0)
        global_rank_corr, _ = spearmanr(global_approx, global_truth)
        
        # Composite fidelity score
        fidelity_score = (
            0.3 * max(0, pearson_r) +
            0.2 * top5_corr +
            0.2 * sign_agree +
            0.15 * top10_corr +
            0.15 * max(0, global_rank_corr)
        )
        
        return FidelityMetrics(
            pearson_r=pearson_r,
            pearson_pvalue=pearson_p,
            spearman_r=spearman_r,
            spearman_pvalue=spearman_p,
            kendall_tau=kendall_tau,
            pearson_per_instance_mean=np.mean(per_instance) if per_instance else 0,
            pearson_per_instance_std=np.std(per_instance) if per_instance else 0,
            top5_rank_correlation=top5_corr,
            top10_rank_correlation=top10_corr,
            mse=mse,
            mae=mae,
            rmse=rmse,
            sign_agreement=sign_agree,
            global_rank_correlation=global_rank_corr,
            fidelity_score=fidelity_score
        )
    
    def _compute_stability(
        self,
        explain_fn: Callable,
        X: np.ndarray,
        n_test_samples: int = 50
    ) -> StabilityMetrics:
        """Compute stability metrics under input perturbations."""
        if len(X) > n_test_samples:
            indices = np.random.choice(len(X), n_test_samples, replace=False)
            X_test = X[indices]
        else:
            X_test = X
        
        all_cv = []
        all_jaccard5 = []
        all_jaccard10 = []
        all_consistency = []
        sign_consistent = []
        
        for i in range(len(X_test)):
            x_base = X_test[i]
            explanations = []
            
            # Generate perturbed explanations
            for _ in range(self.n_perturbations):
                x_pert = x_base + np.random.normal(0, self.noise_level, size=x_base.shape)
                exp = explain_fn(x_pert.reshape(1, -1))
                if isinstance(exp, dict):
                    exp = exp.get('shap_values', exp)
                explanations.append(exp.flatten() if hasattr(exp, 'flatten') else exp[0])
            
            explanations = np.array(explanations)
            
            # Coefficient of variation per feature
            for j in range(explanations.shape[1]):
                mean_val = np.mean(explanations[:, j])
                if abs(mean_val) > 1e-10:
                    cv = np.std(explanations[:, j]) / abs(mean_val)
                    all_cv.append(cv)
            
            # Pairwise consistency
            for j in range(len(explanations)):
                for k in range(j + 1, len(explanations)):
                    r, _ = pearsonr(explanations[j], explanations[k])
                    all_consistency.append(r)
            
            # Top-k Jaccard similarity
            top5_base = set(np.argsort(-np.abs(explanations[0]))[:5])
            top10_base = set(np.argsort(-np.abs(explanations[0]))[:10])
            
            for j in range(1, len(explanations)):
                top5_curr = set(np.argsort(-np.abs(explanations[j]))[:5])
                top10_curr = set(np.argsort(-np.abs(explanations[j]))[:10])
                
                jacc5 = len(top5_base & top5_curr) / len(top5_base | top5_curr) if (top5_base | top5_curr) else 0
                jacc10 = len(top10_base & top10_curr) / len(top10_base | top10_curr) if (top10_base | top10_curr) else 0
                
                all_jaccard5.append(jacc5)
                all_jaccard10.append(jacc10)
            
            # Sign consistency
            for j in range(explanations.shape[1]):
                signs = np.sign(explanations[:, j])
                if np.all(signs == signs[0]) or np.all(signs[signs != 0] == signs[signs != 0][0] if any(signs != 0) else True):
                    sign_consistent.append(1)
                else:
                    sign_consistent.append(0)
        
        mean_consistency = np.mean(all_consistency) if all_consistency else 0
        mean_jacc5 = np.mean(all_jaccard5) if all_jaccard5 else 0
        mean_jacc10 = np.mean(all_jaccard10) if all_jaccard10 else 0
        sign_cons = np.mean(sign_consistent) if sign_consistent else 0
        mean_cv = np.mean(all_cv) if all_cv else 0
        
        # Composite stability score
        stability_score = (
            0.4 * max(0, mean_consistency) +
            0.3 * sign_cons +
            0.2 * mean_jacc5 +
            0.1 * (1 - min(mean_cv, 1))  # Lower CV is better
        )
        
        return StabilityMetrics(
            mean_cv=mean_cv,
            max_cv=np.max(all_cv) if all_cv else 0,
            perturbation_consistency=mean_consistency,
            rank_correlation_mean=mean_consistency,
            rank_correlation_std=np.std(all_consistency) if all_consistency else 0,
            sign_consistency=sign_cons,
            top5_jaccard=mean_jacc5,
            top10_jaccard=mean_jacc10,
            rsd_mean=mean_cv,
            stability_score=stability_score
        )
    
    def _compute_topk_correlation(
        self,
        approx: np.ndarray,
        truth: np.ndarray,
        k: int = 10
    ) -> float:
        """Compute average top-k rank correlation."""
        correlations = []
        for i in range(len(approx)):
            top_truth = set(np.argsort(np.abs(truth[i]))[-k:])
            top_approx = set(np.argsort(np.abs(approx[i]))[-k:])
            
            truth_vec = np.zeros(len(truth[i]))
            approx_vec = np.zeros(len(approx[i]))
            truth_vec[list(top_truth)] = 1
            approx_vec[list(top_approx)] = 1
            
            if np.std(truth_vec) > 0 and np.std(approx_vec) > 0:
                rho, _ = spearmanr(truth_vec, approx_vec)
                correlations.append(rho)
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_overall_score(
        self,
        fidelity: FidelityMetrics,
        stability: StabilityMetrics,
        efficiency: EfficiencyMetrics
    ) -> float:
        """Compute weighted overall score."""
        # Weights: Fidelity = 0.4, Stability = 0.3, Efficiency = 0.3
        return (
            0.4 * fidelity.fidelity_score +
            0.3 * stability.stability_score +
            0.3 * efficiency.efficiency_score
        )
    
    def get_comparison_df(self) -> pd.DataFrame:
        """Get comparison DataFrame of all evaluated methods."""
        rows = [r.to_dataframe_row() for r in self.results]
        return pd.DataFrame(rows)
    
    def print_report(self):
        """Print formatted evaluation report."""
        print("\n" + "="*80)
        print("UNIFIED EXPLANATION EVALUATION REPORT")
        print("Metrics: Fidelity | Stability | Efficiency")
        print("="*80)
        
        for m in self.results:
            print(f"\n{m.method_name}")
            print("-"*60)
            print(f"  Fidelity:    {m.fidelity.fidelity_score:.4f}  "
                  f"(Pearson={m.fidelity.pearson_r:.4f}, SignAgree={m.fidelity.sign_agreement:.4f})")
            print(f"  Stability:   {m.stability.stability_score:.4f}  "
                  f"(Perturb={m.stability.perturbation_consistency:.4f}, SignCons={m.stability.sign_consistency:.4f})")
            print(f"  Efficiency:  {m.efficiency.efficiency_score:.4f}  "
                  f"(P95={m.efficiency.p95_latency_ms:.1f}ms, TPS={m.efficiency.throughput_tps:.1f})")
            print(f"  " + "-"*41)
            print(f"  OVERALL:     {m.overall_score:.4f}")
        
        print("\n" + "="*80)


# Legacy class for backwards compatibility
class ExplanationEvaluator:
    """
    Legacy evaluator - kept for backwards compatibility.
    Use UnifiedExplainerEvaluator for new code.
    """
    
    def __init__(self, ground_truth_explainer: Optional[Callable] = None):
        self.ground_truth = ground_truth_explainer
        self.results: List[Dict] = []
        
    def compute_fidelity(
        self,
        approximate_values: np.ndarray,
        ground_truth_values: np.ndarray,
        latencies: Optional[List[float]] = None
    ) -> ExplanationMetrics:
        """Legacy method - computes basic metrics."""
        # Create temporary unified evaluator
        unified = UnifiedExplainerEvaluator(self.ground_truth)
        
        # Compute using new logic
        efficiency = unified._compute_efficiency(lambda x: approximate_values[0:1], approximate_values[:1])
        fidelity = unified._compute_fidelity(lambda x: approximate_values[0:1], approximate_values[:1], ground_truth_values)
        stability = unified._compute_stability(lambda x: approximate_values[0:1], approximate_values[:1], n_test_samples=1)
        
        return ExplanationMetrics(
            method_name="legacy",
            fidelity=fidelity,
            stability=stability,
            efficiency=efficiency
        )
    
    def _compute_topk_rank_correlation(
        self,
        approx: np.ndarray,
        truth: np.ndarray,
        k: int = 10
    ) -> float:
        """Compute rank correlation for top-k features per instance."""
        correlations = []
        
        for i in range(len(approx)):
            # Get top-k indices
            top_truth = set(np.argsort(np.abs(truth[i]))[-k:])
            top_approx = set(np.argsort(np.abs(approx[i]))[-k:])
            
            # Create binary relevance vectors
            truth_vec = np.zeros(len(truth[i]))
            approx_vec = np.zeros(len(approx[i]))
            truth_vec[list(top_truth)] = 1
            approx_vec[list(top_approx)] = 1
            
            # Spearman correlation
            if np.std(truth_vec) > 0 and np.std(approx_vec) > 0:
                rho, _ = spearmanr(truth_vec, approx_vec)
                correlations.append(rho)
        
        return np.mean(correlations) if correlations else 0.0
    
    def evaluate_method(
        self,
        method_name: str,
        explain_fn: Callable,
        X_test: np.ndarray,
        ground_truth_values: np.ndarray,
        n_samples: Optional[int] = None
    ) -> ExplanationMetrics:
        """
        Evaluate an explanation method.
        
        Args:
            method_name: Name of the method
            explain_fn: Function that takes X and returns SHAP-like values
            X_test: Test data
            ground_truth_values: Ground truth SHAP values
            n_samples: Number of samples to evaluate (None = all)
            
        Returns:
            ExplanationMetrics
        """
        logger.info(f"Evaluating {method_name}...")
        
        if n_samples:
            indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
            X_test = X_test[indices]
            ground_truth_values = ground_truth_values[indices]
        
        # Generate explanations
        approx_values = []
        latencies = []
        
        for i in range(len(X_test)):
            import time
            start = time.time()
            result = explain_fn(X_test[i:i+1])
            latency = (time.time() - start) * 1000
            
            approx_values.append(result['shap_values'][0])
            latencies.append(latency)
        
        approx_values = np.array(approx_values)
        
        # Compute metrics
        metrics = self.compute_fidelity(approx_values, ground_truth_values, latencies)
        
        # Store result
        self.results.append({
            'method': method_name,
            'metrics': metrics
        })
        
        return metrics
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Generate comparison DataFrame of all evaluated methods.
        
        Returns:
            DataFrame with comparison results
        """
        rows = []
        for result in self.results:
            m = result['metrics']
            rows.append({
                'Method': result['method'],
                'Mean Latency (ms)': f"{m.mean_latency_ms:.2f}",
                'P95 Latency (ms)': f"{m.p95_latency_ms:.2f}",
                'Pearson r': f"{m.pearson_r:.4f}",
                'Spearman r': f"{m.spearman_r:.4f}",
                'Top-5 Rank Corr': f"{m.top5_rank_correlation:.4f}",
                'Top-10 Rank Corr': f"{m.top10_rank_correlation:.4f}",
                'MSE': f"{m.mse:.6f}",
                'Sign Agreement': f"{m.sign_agreement:.4f}"
            })
        
        return pd.DataFrame(rows)
    
    def get_summary_report(self) -> str:
        """Generate text summary report."""
        lines = [
            "=" * 80,
            "EXPLANATION METHOD EVALUATION REPORT",
            "=" * 80,
            ""
        ]
        
        for result in self.results:
            m = result['metrics']
            lines.extend([
                f"Method: {result['method']}",
                "-" * 40,
                f"  Latency (ms):     Mean={m.mean_latency_ms:.2f}, P95={m.p95_latency_ms:.2f}, P99={m.p99_latency_ms:.2f}",
                f"  Fidelity:         Pearson={m.pearson_r:.4f}, Spearman={m.spearman_r:.4f}",
                f"  Rank Correlation: Top-5={m.top5_rank_correlation:.4f}, Top-10={m.top10_rank_correlation:.4f}",
                f"  Sign Agreement:   {m.sign_agreement:.4f}",
                f"  Error:            RMSE={m.rmse:.6f}, MAE={m.mae:.6f}",
                ""
            ])
        
        return "\n".join(lines)


def compute_stability_metrics(
    explanations: np.ndarray,
    perturbation_level: float = 0.01
) -> Dict[str, float]:
    """
    Compute explanation stability under input perturbations.
    
    Args:
        explanations: Array of explanations for perturbed inputs
        perturbation_level: Level of input perturbation
        
    Returns:
        Stability metrics dict with mean_cv, max_cv, mean_pairwise_correlation
    """
    # Coefficient of variation for each feature
    cv_per_feature = []
    for j in range(explanations.shape[1]):
        values = explanations[:, j]
        if np.mean(values) != 0:
            cv = np.std(values) / abs(np.mean(values))
            cv_per_feature.append(cv)
    
    # Pairwise consistency
    pairwise_corr = []
    for i in range(len(explanations)):
        for j in range(i+1, len(explanations)):
            if np.std(explanations[i]) > 0 and np.std(explanations[j]) > 0:
                r, _ = pearsonr(explanations[i], explanations[j])
                pairwise_corr.append(r)
    
    return {
        'mean_cv': np.mean(cv_per_feature),
        'max_cv': np.max(cv_per_feature),
        'mean_pairwise_correlation': np.mean(pairwise_corr) if pairwise_corr else 0,
        'min_pairwise_correlation': np.min(pairwise_corr) if pairwise_corr else 0
    }


def compute_all_metrics(
    method_name: str,
    explain_fn: Callable,
    X_test: np.ndarray,
    ground_truth_values: Optional[np.ndarray] = None,
    n_samples: Optional[int] = None
) -> ExplanationMetrics:
    """
    Convenience function to compute all three metrics (Fidelity, Stability, Efficiency)
    for an explanation method.
    
    Args:
        method_name: Name of the method
        explain_fn: Explanation function
        X_test: Test data
        ground_truth_values: Ground truth for fidelity comparison
        n_samples: Number of samples to evaluate
        
    Returns:
        ExplanationMetrics with fidelity, stability, and efficiency
        
    Example:
        >>> metrics = compute_all_metrics(
        ...     "FastSHAP",
        ...     fastshap_explainer,
        ...     X_test,
        ...     exact_shap_values
        ... )
        >>> print(f"Fidelity: {metrics.fidelity.fidelity_score:.4f}")
        >>> print(f"Stability: {metrics.stability.stability_score:.4f}")
        >>> print(f"Efficiency: {metrics.efficiency.efficiency_score:.4f}")
    """
    evaluator = UnifiedExplainerEvaluator()
    return evaluator.evaluate_method(
        method_name=method_name,
        explain_fn=explain_fn,
        X_test=X_test,
        ground_truth_values=ground_truth_values,
        n_samples=n_samples
    )


def compute_consistency_across_methods(
    explanations_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compute consistency of explanations across different methods.
    
    Args:
        explanations_dict: Dict mapping method name to SHAP values
        
    Returns:
        DataFrame with pairwise consistency matrix
    """
    methods = list(explanations_dict.keys())
    n_methods = len(methods)
    
    consistency_matrix = np.zeros((n_methods, n_methods))
    
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i == j:
                consistency_matrix[i, j] = 1.0
            elif i < j:
                # Compute correlation
                flat1 = explanations_dict[method1].flatten()
                flat2 = explanations_dict[method2].flatten()
                
                if np.std(flat1) > 0 and np.std(flat2) > 0:
                    r, _ = pearsonr(flat1, flat2)
                    consistency_matrix[i, j] = r
                    consistency_matrix[j, i] = r
    
    return pd.DataFrame(
        consistency_matrix,
        index=methods,
        columns=methods
    )
