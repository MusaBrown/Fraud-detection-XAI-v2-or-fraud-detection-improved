# Unified Metrics Framework Guide

## Overview

The unified metrics framework evaluates explanation methods across **three key dimensions**:

| Metric | Description | Key Indicators |
|--------|-------------|----------------|
| **FIDELITY** | How well explanations match ground truth | Pearson r, Spearman r, Sign Agreement, Top-k Rank Correlation |
| **STABILITY** | Consistency under small input perturbations | Perturbation Consistency, Sign Consistency, Jaccard Similarity |
| **EFFICIENCY** | Computational performance | P50/P95/P99 Latency, Throughput (TPS), Memory Usage |

---

## Quick Start

### Basic Usage

```python
from src.explainers.fidelity_metrics import UnifiedExplainerEvaluator

# Initialize evaluator
evaluator = UnifiedExplainerEvaluator(
    noise_level=0.01,        # Noise for stability testing
    n_perturbations=10       # Perturbations per sample
)

# Evaluate your explainer
metrics = evaluator.evaluate_method(
    method_name="FastSHAP",
    explain_fn=your_explainer_function,
    X_test=X_test,
    ground_truth_values=exact_shap_values,  # For fidelity
    n_samples=100
)

# Access scores
print(f"Fidelity:  {metrics.fidelity.fidelity_score:.4f}")
print(f"Stability: {metrics.stability.stability_score:.4f}")
print(f"Efficiency: {metrics.efficiency.efficiency_score:.4f}")
print(f"Overall:   {metrics.overall_score:.4f}")
```

### Convenience Function

```python
from src.explainers.fidelity_metrics import compute_all_metrics

metrics = compute_all_metrics(
    method_name="MyExplainer",
    explain_fn=explainer,
    X_test=X_test,
    ground_truth_values=ground_truth
)
```

---

## Detailed Metrics

### Fidelity Metrics (`metrics.fidelity`)

| Metric | Range | Description |
|--------|-------|-------------|
| `pearson_r` | [-1, 1] | Linear correlation with ground truth |
| `spearman_r` | [-1, 1] | Rank correlation with ground truth |
| `sign_agreement` | [0, 1] | % of signs matching ground truth |
| `top5_rank_correlation` | [-1, 1] | Rank correlation for top-5 features |
| `top10_rank_correlation` | [-1, 1] | Rank correlation for top-10 features |
| `mse` | [0, ∞) | Mean squared error |
| `fidelity_score` | [0, 1] | **Composite fidelity score** |

**Composite Score Weights:**
- Pearson r: 30%
- Top-5 Rank: 20%
- Sign Agreement: 20%
- Top-10 Rank: 15%
- Global Rank: 15%

### Stability Metrics (`metrics.stability`)

| Metric | Range | Description |
|--------|-------|-------------|
| `perturbation_consistency` | [-1, 1] | Mean pairwise correlation under perturbation |
| `sign_consistency` | [0, 1] | % of features with consistent sign |
| `top5_jaccard` | [0, 1] | Jaccard similarity for top-5 features |
| `top10_jaccard` | [0, 1] | Jaccard similarity for top-10 features |
| `mean_cv` | [0, ∞) | Mean coefficient of variation |
| `stability_score` | [0, 1] | **Composite stability score** |

**Composite Score Weights:**
- Perturbation Consistency: 40%
- Sign Consistency: 30%
- Top-5 Jaccard: 20%
- Inverse CV: 10%

### Efficiency Metrics (`metrics.efficiency`)

| Metric | Unit | Description |
|--------|------|-------------|
| `p50_latency_ms` | ms | Median latency |
| `p95_latency_ms` | ms | 95th percentile latency |
| `p99_latency_ms` | ms | 99th percentile latency |
| `throughput_tps` | TPS | Transactions per second |
| `memory_mb` | MB | Peak memory usage |
| `efficiency_score` | [0, 1] | **Composite efficiency score** |

**Composite Score Formula:**
```
efficiency_score = max(0, min(1, 1 - (P95_latency - 10ms) / 190))
```
- 10ms → 1.0 (perfect)
- 50ms → 0.79
- 100ms → 0.53
- 200ms → 0.0

---

## Overall Score

The overall score combines all three metrics:

```
Overall = 0.4 × Fidelity + 0.3 × Stability + 0.3 × Efficiency
```

---

## Comparing Multiple Methods

```python
# Evaluate multiple methods
evaluator = UnifiedExplainerEvaluator()

for name, explainer in methods.items():
    evaluator.evaluate_method(
        method_name=name,
        explain_fn=explainer,
        X_test=X_test,
        ground_truth_values=ground_truth
    )

# Print report
evaluator.print_report()

# Get comparison DataFrame
df = evaluator.get_comparison_df()
print(df)
```

---

## Example Output

```
================================================================================
UNIFIED EXPLANATION EVALUATION REPORT
Metrics: Fidelity | Stability | Efficiency
================================================================================

FastSHAP
------------------------------------------------------------
  Fidelity:    0.9532  (Pearson=0.9621, SignAgree=0.9812)
  Stability:   0.8912  (Perturb=0.9234, SignCons=0.9543)
  Efficiency:  1.0000  (P95=8.2ms, TPS=125.4)
  -----------------------------------------
  OVERALL:     0.9487

TreeSHAP
------------------------------------------------------------
  Fidelity:    1.0000  (Pearson=1.0000, SignAgree=1.0000)
  Stability:   0.9856  (Perturb=0.9912, SignCons=0.9934)
  Efficiency:  0.7567  (P95=55.3ms, TPS=18.2)
  -----------------------------------------
  OVERALL:     0.9156
```

---

## Integration with Existing Code

The new framework is **backward compatible**. Existing code using `ExplanationEvaluator` continues to work:

```python
from src.explainers import ExplanationEvaluator  # Still works!

# Legacy usage
evaluator = ExplanationEvaluator(ground_truth_explainer=treeshap)
metrics = evaluator.compute_fidelity(approx, truth, latencies)
```

For new code, use `UnifiedExplainerEvaluator`:

```python
from src.explainers import UnifiedExplainerEvaluator  # Recommended

evaluator = UnifiedExplainerEvaluator()
metrics = evaluator.evaluate_method(...)
```

---

## Run the Demo

```bash
python demo_unified_metrics.py
```

This demonstrates the framework with dummy explainers simulating different characteristics.
