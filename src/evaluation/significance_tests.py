"""
Statistical Significance Tests for XAI Method Comparison
=========================================================
Implements t-tests, Wilcoxon signed-rank tests, and effect size calculations
for comparing FastSHAP vs baseline methods.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def paired_t_test(
    method1_scores: np.ndarray,
    method2_scores: np.ndarray,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2"
) -> Dict:
    """
    Perform paired t-test for comparing two methods.
    
    Args:
        method1_scores: Scores from method 1 (e.g., fidelity values)
        method2_scores: Scores from method 2
        method1_name: Name of method 1
        method2_name: Name of method 2
        
    Returns:
        Dictionary with test statistics and interpretation
    """
    # Check normality assumption
    _, p_normal1 = stats.shapiro(method1_scores)
    _, p_normal2 = stats.shapiro(method2_scores)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(method1_scores) - np.mean(method2_scores)
    pooled_std = np.sqrt((np.std(method1_scores, ddof=1) ** 2 + 
                          np.std(method2_scores, ddof=1) ** 2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    result = {
        "test": "paired_t_test",
        "method1": method1_name,
        "method2": method2_name,
        "method1_mean": float(np.mean(method1_scores)),
        "method2_mean": float(np.mean(method2_scores)),
        "mean_difference": float(mean_diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "cohens_d": float(cohens_d),
        "effect_size": effect_size,
        "normality_p": {
            method1_name: float(p_normal1),
            method2_name: float(p_normal2)
        },
        "n_samples": len(method1_scores)
    }
    
    logger.info(f"Paired t-test: {method1_name} vs {method2_name}")
    logger.info(f"  Mean difference: {mean_diff:.6f} (p={p_value:.4f})")
    logger.info(f"  Effect size: {effect_size} (Cohen's d={cohens_d:.3f})")
    
    return result


def wilcoxon_signed_rank_test(
    method1_scores: np.ndarray,
    method2_scores: np.ndarray,
    method1_name: str = "Method 1",
    method2_name: str = "Method 2"
) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    Use when normality assumption is violated.
    """
    statistic, p_value = stats.wilcoxon(method1_scores, method2_scores)
    
    # Effect size (rank-biserial correlation)
    n = len(method1_scores)
    z_score = stats.norm.ppf(1 - p_value / 2) if p_value > 0 else 0
    r = z_score / np.sqrt(n) if n > 0 else 0
    
    result = {
        "test": "wilcoxon_signed_rank",
        "method1": method1_name,
        "method2": method2_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "effect_size_r": float(r),
        "n_samples": n
    }
    
    logger.info(f"Wilcoxon test: {method1_name} vs {method2_name}")
    logger.info(f"  p-value: {p_value:.4f}, effect size r={r:.3f}")
    
    return result


def compare_xai_methods(
    fastshap_fidelities: List[float],
    treeshap_fidelities: List[float],
    fastshap_latencies: List[float],
    treeshap_latencies: List[float]
) -> Dict:
    """
    Comprehensive comparison of FastSHAP vs TreeSHAP.
    
    Returns:
        Dictionary with all statistical tests
    """
    results = {
        "fidelity_comparison": {},
        "latency_comparison": {},
        "summary": {}
    }
    
    # Convert to numpy arrays
    fs_fid = np.array(fastshap_fidelities)
    ts_fid = np.array(treeshap_fidelities)
    fs_lat = np.array(fastshap_latencies)
    ts_lat = np.array(treeshap_latencies)
    
    # Fidelity comparison (expecting no significant difference - both are accurate)
    logger.info("\n" + "="*60)
    logger.info("FIDELITY COMPARISON")
    logger.info("="*60)
    
    results["fidelity_comparison"]["t_test"] = paired_t_test(
        fs_fid, ts_fid, "FastSHAP", "TreeSHAP"
    )
    results["fidelity_comparison"]["wilcoxon"] = wilcoxon_signed_rank_test(
        fs_fid, ts_fid, "FastSHAP", "TreeSHAP"
    )
    
    # Latency comparison (expecting significant difference - FastSHAP is faster)
    logger.info("\n" + "="*60)
    logger.info("LATENCY COMPARISON")
    logger.info("="*60)
    
    results["latency_comparison"]["t_test"] = paired_t_test(
        fs_lat, ts_lat, "FastSHAP", "TreeSHAP"
    )
    results["latency_comparison"]["wilcoxon"] = wilcoxon_signed_rank_test(
        fs_lat, ts_lat, "FastSHAP", "TreeSHAP"
    )
    
    # Summary interpretation
    fid_p = results["fidelity_comparison"]["t_test"]["p_value"]
    lat_p = results["latency_comparison"]["t_test"]["p_value"]
    speedup = np.mean(ts_lat) / np.mean(fs_lat)
    
    results["summary"] = {
        "fidelity_equivalence": fid_p > 0.05,
        "fidelity_p_value": float(fid_p),
        "latency_significantly_lower": lat_p < 0.05 and np.mean(fs_lat) < np.mean(ts_lat),
        "latency_p_value": float(lat_p),
        "speedup_factor": float(speedup),
        "interpretation": (
            f"FastSHAP achieves equivalent fidelity to TreeSHAP (p={fid_p:.4f}) "
            f"with {speedup:.1f}× speedup (p={lat_p:.4e})"
        )
    }
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(results["summary"]["interpretation"])
    
    return results


def cliff_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """
    Calculate Cliff's Delta (effect size for non-parametric tests).
    
    Returns:
        (delta, interpretation)
        delta: -1 to 1 (0 = no difference, positive = x > y)
        interpretation: negligible, small, medium, large
    """
    x, y = np.array(x), np.array(y)
    
    # Count comparisons
    x_greater = np.sum(x[:, None] > y[None, :])
    y_greater = np.sum(y[:, None] > x[None, :])
    total = len(x) * len(y)
    
    delta = (x_greater - y_greater) / total
    
    if abs(delta) < 0.147:
        interpretation = "negligible"
    elif abs(delta) < 0.33:
        interpretation = "small"
    elif abs(delta) < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return float(delta), interpretation


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Simulate fidelity scores (both methods are accurate)
    fastshap_fid = np.random.normal(0.95, 0.01, 100)
    treeshap_fid = np.random.normal(0.96, 0.01, 100)
    
    # Simulate latencies (FastSHAP is much faster)
    fastshap_lat = np.random.normal(0.6, 0.05, 100)
    treeshap_lat = np.random.normal(5.5, 0.5, 100)
    
    results = compare_xai_methods(
        fastshap_fid.tolist(),
        treeshap_fid.tolist(),
        fastshap_lat.tolist(),
        treeshap_lat.tolist()
    )
    
    print("\nResults:", results["summary"]["interpretation"])
