"""
Minimal demo of the Real-Time XAI Framework using only standard libraries.
"""
import numpy as np
import pandas as pd
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalFraudDetector:
    """Minimal fraud detector using RandomForest."""
    
    def __init__(self, n_estimators=50, max_depth=8):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = None
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X):
        return self.model.predict(X)


class MinimalSHAPExplainer:
    """Minimal SHAP explainer using tree-based approach."""
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.expected_value = None
        
    def fit(self, X_background):
        """Compute expected value from background data."""
        self.expected_value = np.mean(self.model.predict_proba(X_background)[:, 1])
        return self
    
    def explain(self, X):
        """
        Approximate SHAP values using feature importance weighted by deviation.
        This is a simplified approximation for demonstration.
        """
        start = time.time()
        
        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, n_features))
        
        # Get base prediction
        base_pred = self.model.predict_proba(X)[:, 1]
        
        # Approximate SHAP by marginal contributions
        for i in range(n_features):
            X_perturbed = X.copy()
            # Shuffle feature to estimate marginal contribution
            np.random.shuffle(X_perturbed[:, i])
            perturbed_pred = self.model.predict_proba(X_perturbed)[:, 1]
            shap_values[:, i] = base_pred - perturbed_pred
        
        latency_ms = (time.time() - start) * 1000
        
        return {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms
        }
    
    def explain_single(self, x):
        return self.explain(x.reshape(1, -1))


class FastApproxExplainer:
    """
    Fast approximation explainer - simulates FastSHAP concept.
    Uses feature importance for instant explanations.
    """
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.feature_importance = None
        self.expected_value = None
        
    def fit(self, X_background):
        """Pre-compute feature importance for fast inference."""
        self.expected_value = np.mean(self.model.predict_proba(X_background)[:, 1])
        self.feature_importance = self.model.feature_importances_
        return self
    
    def explain(self, X):
        """
        Generate fast explanations using pre-computed importance.
        Scaled by prediction confidence.
        """
        start = time.time()
        
        n_samples, n_features = X.shape
        
        # Get predictions
        preds = self.model.predict_proba(X)[:, 1]
        
        # Scale importance by prediction deviation from mean
        deviations = preds - self.expected_value
        
        # Fast approximation: broadcast importance scaled by deviation
        shap_values = self.feature_importance.reshape(1, -1) * deviations.reshape(-1, 1)
        
        latency_ms = (time.time() - start) * 1000
        
        return {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms
        }
    
    def explain_single(self, x):
        return self.explain(x.reshape(1, -1))


def generate_synthetic_data(n_samples=5000, n_features=20, fraud_rate=0.035):
    """Generate synthetic fraud detection data."""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create fraud labels with some patterns
    # Higher values in first few features indicate fraud
    fraud_scores = (
        2 * X[:, 0] + 
        1.5 * X[:, 1] + 
        1 * X[:, 2] + 
        0.5 * np.random.randn(n_samples)
    )
    
    # Convert to binary labels
    threshold = np.percentile(fraud_scores, 100 * (1 - fraud_rate))
    y = (fraud_scores > threshold).astype(int)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return X, y, feature_names


def benchmark_explainers(explainers, X_test, n_samples=100):
    """Benchmark multiple explainers."""
    results = []
    
    for name, explainer in explainers.items():
        logger.info(f"Benchmarking {name}...")
        
        latencies = []
        for i in range(min(n_samples, len(X_test))):
            result = explainer.explain_single(X_test[i])
            latencies.append(result['latency_ms'])
        
        results.append({
            'method': name,
            'mean_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'min_ms': np.min(latencies),
            'max_ms': np.max(latencies),
            'throughput_tps': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        })
    
    return pd.DataFrame(results)


def compute_fidelity(approx_shap, true_shap):
    """Compute fidelity metrics."""
    from scipy.stats import pearsonr
    
    flat_approx = approx_shap.flatten()
    flat_true = true_shap.flatten()
    
    if np.std(flat_approx) > 0 and np.std(flat_true) > 0:
        r, _ = pearsonr(flat_approx, flat_true)
        return r
    return 0


def main():
    logger.info("=" * 80)
    logger.info("Real-Time XAI Framework - Minimal Demo")
    logger.info("=" * 80)
    
    # 1. Generate data
    logger.info("\n[1/5] Generating synthetic data...")
    X, y, feature_names = generate_synthetic_data(n_samples=5000, n_features=20)
    
    # Temporal split (simulate)
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Fraud rate: {y.mean():.4f}")
    
    # 2. Train model
    logger.info("\n[2/5] Training Random Forest model...")
    model = MinimalFraudDetector(n_estimators=50, max_depth=8)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"Test AUC-ROC: {auc:.4f}")
    logger.info(f"Test F1: {f1:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    
    # 3. Initialize explainers
    logger.info("\n[3/5] Initializing explainers...")
    
    # Full SHAP (slower)
    full_shap = MinimalSHAPExplainer(model.model, feature_names)
    full_shap.fit(X_train[:500])
    
    # Fast approximation
    fast_shap = FastApproxExplainer(model.model, feature_names)
    fast_shap.fit(X_train[:500])
    
    # 4. Benchmark latency
    logger.info("\n[4/5] Benchmarking latency...")
    explainers = {
        'Full SHAP (approx)': full_shap,
        'Fast Approximation': fast_shap
    }
    
    bench_df = benchmark_explainers(explainers, X_test, n_samples=100)
    
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(bench_df.to_string(index=False))
    
    # 5. Evaluate fidelity
    logger.info("\n[5/5] Evaluating fidelity...")
    
    n_eval = 50
    true_results = full_shap.explain(X_test[:n_eval])
    fast_results = fast_shap.explain(X_test[:n_eval])
    
    fidelity = compute_fidelity(fast_results['shap_values'], true_results['shap_values'])
    
    print("\n" + "=" * 80)
    print("FIDELITY EVALUATION")
    print("=" * 80)
    print(f"Fast Approximation vs Full SHAP:")
    print(f"  Pearson correlation: {fidelity:.4f}")
    
    # Summary
    fast_row = bench_df[bench_df['method'] == 'Fast Approximation'].iloc[0]
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model Performance:")
    print(f"  AUC-ROC: {auc:.4f} (target: >0.98: {'PASS' if auc > 0.98 else 'FAIL'})")
    print(f"  F1 Score: {f1:.4f} (target: >0.95: {'PASS' if f1 > 0.95 else 'FAIL'})")
    print(f"\nFast Approximation Explainer:")
    print(f"  P95 Latency: {fast_row['p95_ms']:.2f} ms (target: <50ms: {'PASS' if fast_row['p95_ms'] < 50 else 'FAIL'})")
    print(f"  P99 Latency: {fast_row['p99_ms']:.2f} ms (target: <100ms: {'PASS' if fast_row['p99_ms'] < 100 else 'FAIL'})")
    print(f"  Throughput: {fast_row['throughput_tps']:.0f} TPS")
    print(f"  Fidelity: {fidelity:.4f}")
    
    # Show example explanation
    print("\n" + "=" * 80)
    print("EXAMPLE EXPLANATION (First Test Sample)")
    print("=" * 80)
    
    example_idx = 0
    x_example = X_test[example_idx]
    y_true = y_test[example_idx]
    y_prob_example = model.predict_proba(x_example.reshape(1, -1))[0]
    
    fast_exp = fast_shap.explain_single(x_example)
    shap_values = fast_exp['shap_values'][0]
    
    # Top features
    top_k = 5
    top_indices = np.argsort(np.abs(shap_values))[-top_k:][::-1]
    
    print(f"True Label: {'Fraud' if y_true else 'Legitimate'}")
    print(f"Predicted Probability: {y_prob_example:.4f}")
    print(f"Prediction: {'Fraud' if y_prob_example > 0.5 else 'Legitimate'}")
    print(f"Explanation Latency: {fast_exp['latency_ms']:.2f} ms")
    print(f"\nTop {top_k} Features:")
    for rank, idx in enumerate(top_indices, 1):
        direction = "increases" if shap_values[idx] > 0 else "decreases"
        print(f"  {rank}. {feature_names[idx]}: {shap_values[idx]:+.4f} ({direction} fraud risk)")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
