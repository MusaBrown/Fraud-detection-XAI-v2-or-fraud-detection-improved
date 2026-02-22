"""
Demo: Unified Metrics Framework (Fidelity, Stability, Efficiency)

This demo shows how to use the new unified evaluation framework to assess
explanation methods across three key dimensions:
1. FIDELITY: How well explanations match ground truth
2. STABILITY: Consistency under small input perturbations  
3. EFFICIENCY: Computational performance (latency, throughput)
"""
import sys
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import directly to avoid package dependencies
sys.path.insert(0, 'src/explainers')
from fidelity_metrics import (
    UnifiedExplainerEvaluator,
    compute_all_metrics,
    ExplanationMetrics,
    FidelityMetrics,
    StabilityMetrics,
    EfficiencyMetrics
)


def create_dummy_explainer(latency_ms=10, noise_level=0.1):
    """
    Create a dummy explainer for demonstration.
    Simulates different explanation methods with varying characteristics.
    """
    def explainer(X):
        # Simulate latency
        import time
        time.sleep(latency_ms / 1000)
        
        # Generate dummy SHAP values with some noise
        shap_values = np.random.randn(X.shape[0], X.shape[1]) * (1 + noise_level)
        return shap_values
    
    return explainer


def main():
    print("="*80)
    print("UNIFIED METRICS FRAMEWORK DEMO")
    print("Fidelity | Stability | Efficiency")
    print("="*80)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X_test = np.random.randn(n_samples, n_features)
    
    # Create synthetic ground truth (simulating exact TreeSHAP)
    ground_truth = np.random.randn(n_samples, n_features)
    
    # Define explanation methods to evaluate
    methods = {
        "FastSHAP (fast, accurate)": create_dummy_explainer(latency_ms=5, noise_level=0.05),
        "TreeSHAP (slow, exact)": create_dummy_explainer(latency_ms=50, noise_level=0.0),
        "LIME (slow, noisy)": create_dummy_explainer(latency_ms=100, noise_level=0.2),
        "KernelSHAP (medium)": create_dummy_explainer(latency_ms=30, noise_level=0.1),
    }
    
    # Initialize unified evaluator
    evaluator = UnifiedExplainerEvaluator(
        ground_truth_explainer=None,  # We provide ground truth directly
        noise_level=0.01,             # For stability testing
        n_perturbations=5             # Number of perturbations per sample
    )
    
    # Evaluate each method
    print("\nEvaluating methods...\n")
    
    for method_name, explain_fn in methods.items():
        metrics = evaluator.evaluate_method(
            method_name=method_name,
            explain_fn=explain_fn,
            X_test=X_test,
            ground_truth_values=ground_truth,
            n_samples=50  # Use subset for demo speed
        )
    
    # Print unified report
    evaluator.print_report()
    
    # Get comparison DataFrame
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    comparison_df = evaluator.get_comparison_df()
    print(comparison_df.to_string(index=False))
    
    # Example: Accessing individual metrics
    print("\n" + "="*80)
    print("DETAILED METRICS EXAMPLE (FastSHAP)")
    print("="*80)
    
    fastshap_metrics = evaluator.results[0]
    
    print(f"\nFIDELITY METRICS:")
    print(f"  - Pearson r: {fastshap_metrics.fidelity.pearson_r:.4f}")
    print(f"  - Spearman r: {fastshap_metrics.fidelity.spearman_r:.4f}")
    print(f"  - Sign Agreement: {fastshap_metrics.fidelity.sign_agreement:.4f}")
    print(f"  - Top-5 Rank Correlation: {fastshap_metrics.fidelity.top5_rank_correlation:.4f}")
    print(f"  - Fidelity Score: {fastshap_metrics.fidelity.fidelity_score:.4f}")
    
    print(f"\nSTABILITY METRICS:")
    print(f"  - Perturbation Consistency: {fastshap_metrics.stability.perturbation_consistency:.4f}")
    print(f"  - Sign Consistency: {fastshap_metrics.stability.sign_consistency:.4f}")
    print(f"  - Top-5 Jaccard: {fastshap_metrics.stability.top5_jaccard:.4f}")
    print(f"  - Mean CV: {fastshap_metrics.stability.mean_cv:.4f}")
    print(f"  - Stability Score: {fastshap_metrics.stability.stability_score:.4f}")
    
    print(f"\nEFFICIENCY METRICS:")
    print(f"  - P50 Latency: {fastshap_metrics.efficiency.p50_latency_ms:.2f} ms")
    print(f"  - P95 Latency: {fastshap_metrics.efficiency.p95_latency_ms:.2f} ms")
    print(f"  - P99 Latency: {fastshap_metrics.efficiency.p99_latency_ms:.2f} ms")
    print(f"  - Throughput: {fastshap_metrics.efficiency.throughput_tps:.1f} TPS")
    print(f"  - Efficiency Score: {fastshap_metrics.efficiency.efficiency_score:.4f}")
    
    print(f"\nOVERALL SCORE: {fastshap_metrics.overall_score:.4f}")
    
    # Alternative: Using convenience function
    print("\n" + "="*80)
    print("ALTERNATIVE: Using compute_all_metrics() convenience function")
    print("="*80)
    
    metrics = compute_all_metrics(
        method_name="MyExplainer",
        explain_fn=create_dummy_explainer(latency_ms=20, noise_level=0.1),
        X_test=X_test[:20],
        ground_truth_values=ground_truth[:20]
    )
    
    print(f"\nFidelity: {metrics.fidelity.fidelity_score:.4f}")
    print(f"Stability: {metrics.stability.stability_score:.4f}")
    print(f"Efficiency: {metrics.efficiency.efficiency_score:.4f}")
    print(f"Overall: {metrics.overall_score:.4f}")
    
    print("\n" + "="*80)
    print("Demo complete!")
    print("="*80)


if __name__ == "__main__":
    main()
