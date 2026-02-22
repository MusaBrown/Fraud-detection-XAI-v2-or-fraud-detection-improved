"""XAI explanation modules."""
from .baseline_shap import TreeSHAPExplainer
from .fastshap_implementation import FastSHAPExplainer, FastSHAPSurrogate
from .lime_optimized import OptimizedLIME
from .fidelity_metrics import (
    ExplanationEvaluator,           # Legacy evaluator (backward compat)
    UnifiedExplainerEvaluator,      # New unified evaluator
    ExplanationMetrics,             # Unified metrics container
    FidelityMetrics,                # Fidelity metrics
    StabilityMetrics,               # Stability metrics
    EfficiencyMetrics,              # Efficiency metrics
    compute_all_metrics             # Convenience function
)

__all__ = [
    # Explainers
    'TreeSHAPExplainer',
    'FastSHAPExplainer',
    'FastSHAPSurrogate', 
    'OptimizedLIME',
    # Metrics (Legacy)
    'ExplanationEvaluator',
    # Metrics (New Unified)
    'UnifiedExplainerEvaluator',
    'ExplanationMetrics',
    'FidelityMetrics',
    'StabilityMetrics',
    'EfficiencyMetrics',
    'compute_all_metrics'
]
