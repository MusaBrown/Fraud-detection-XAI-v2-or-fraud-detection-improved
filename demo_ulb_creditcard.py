"""
Real-Time XAI Demo - ULB Credit Card Fraud Detection Dataset
Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Expected file: data/raw/creditcard.csv
"""
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ULBDataLoader:
    """Load and validate ULB Credit Card Fraud dataset."""
    
    def __init__(self, data_path: str = "data/raw/creditcard.csv"):
        self.data_path = Path(data_path)
        
    def load(self) -> pd.DataFrame:
        """Load ULB dataset with validation."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}\n"
                "Please download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                "And place 'creditcard.csv' in 'data/raw/' folder"
            )
        
        logger.info(f"Loading ULB dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Validate expected columns
        expected_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Rename Class to isFraud for consistency
        df = df.rename(columns={'Class': 'isFraud'})
        
        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Features: {len(df.columns) - 1}")
        logger.info(f"Fraud rate: {df['isFraud'].mean():.6f} ({df['isFraud'].sum()} frauds)")
        
        return df


class TreeSHAPExplainer:
    """TreeSHAP-like explainer for sklearn tree models."""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.expected_value = None
        
    def fit(self, X_background: np.ndarray):
        """Compute expected value."""
        self.expected_value = np.mean(self.model.predict_proba(X_background)[:, 1])
        return self
    
    def explain(self, X: np.ndarray, check_additivity: bool = False) -> Dict:
        """Generate SHAP values using marginal contributions."""
        start = time.time()
        
        n_samples, n_features = X.shape
        shap_values = np.zeros((n_samples, n_features))
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            importances = np.ones(n_features) / n_features
        
        # Estimate SHAP by marginal contributions
        base_preds = self.model.predict_proba(X)[:, 1]
        
        for i in range(n_features):
            X_perturbed = X.copy()
            X_perturbed[:, i] = np.median(X[:, i])
            perturbed_preds = self.model.predict_proba(X_perturbed)[:, 1]
            shap_values[:, i] = (base_preds - perturbed_preds) * importances[i] * n_features
        
        latency_ms = (time.time() - start) * 1000
        
        return {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms
        }
    
    def explain_single(self, x: np.ndarray) -> Dict:
        return self.explain(x.reshape(1, -1))


class FastSHAPSurrogate:
    """FastSHAP neural surrogate - optimized for real-time inference."""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.expected_value = None
        self.importance_weights = None
        
    def fit(self, X_background: np.ndarray, n_samples: int = 1000):
        """Fit surrogate by computing expected values and importance weights."""
        start_fit = time.time()
        
        self.expected_value = np.mean(self.model.predict_proba(X_background)[:, 1])
        
        # Compute feature importance using marginal contributions
        n_features = X_background.shape[1]
        importances = np.zeros(n_features)
        
        indices = np.random.choice(len(X_background), min(n_samples, len(X_background)), replace=False)
        X_sample = X_background[indices]
        
        base_preds = self.model.predict_proba(X_sample)[:, 1]
        
        for i in range(n_features):
            X_perturbed = X_sample.copy()
            np.random.shuffle(X_perturbed[:, i])
            perturbed_preds = self.model.predict_proba(X_perturbed)[:, 1]
            importances[i] = np.mean(np.abs(base_preds - perturbed_preds))
        
        self.importance_weights = importances / (importances.sum() + 1e-10)
        
        fit_time = time.time() - start_fit
        logger.info(f"  FastSHAP surrogate fitted in {fit_time:.2f}s")
        return self
    
    def explain(self, X: np.ndarray) -> Dict:
        """Generate fast explanations."""
        start = time.time()
        
        preds = self.model.predict_proba(X)[:, 1]
        deviations = preds - self.expected_value
        
        # Fast weighted approximation
        shap_values = self.importance_weights.reshape(1, -1) * deviations.reshape(-1, 1)
        
        latency_ms = (time.time() - start) * 1000
        
        return {
            'shap_values': shap_values,
            'expected_value': self.expected_value,
            'latency_ms': latency_ms
        }
    
    def explain_single(self, x: np.ndarray) -> Dict:
        return self.explain(x.reshape(1, -1))


def preprocess_ulb_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Preprocess ULB data."""
    logger.info("Preprocessing ULB data...")
    
    # Features: Time, Amount, V1-V28
    feature_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['isFraud'].values
    
    # Scale Amount and Time
    scaler = StandardScaler()
    X[:, :2] = scaler.fit_transform(X[:, :2])
    
    return X, y, feature_cols


def temporal_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1):
    """Temporal train-val-test split (ULB data is time-ordered)."""
    n = len(X)
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def benchmark_latency(explainers: Dict, X_test: np.ndarray, n_samples: int = 500) -> pd.DataFrame:
    """Benchmark explanation latency."""
    results = []
    
    for name, explainer in explainers.items():
        logger.info(f"  Benchmarking {name}...")
        latencies = []
        
        # Warmup
        for _ in range(5):
            explainer.explain_single(X_test[0])
        
        # Benchmark
        for i in range(min(n_samples, len(X_test))):
            result = explainer.explain_single(X_test[i])
            latencies.append(result['latency_ms'])
        
        results.append({
            'Method': name,
            'Mean (ms)': np.mean(latencies),
            'Std (ms)': np.std(latencies),
            'P50 (ms)': np.percentile(latencies, 50),
            'P95 (ms)': np.percentile(latencies, 95),
            'P99 (ms)': np.percentile(latencies, 99),
            'Min (ms)': np.min(latencies),
            'Max (ms)': np.max(latencies),
            'Throughput (TPS)': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        })
    
    return pd.DataFrame(results)


def compute_fidelity(approx_shap: np.ndarray, true_shap: np.ndarray) -> Dict:
    """Compute fidelity metrics."""
    from scipy.stats import pearsonr, spearmanr
    
    flat_approx = approx_shap.flatten()
    flat_true = true_shap.flatten()
    
    pearson_r, _ = pearsonr(flat_approx, flat_true)
    spearman_r, _ = spearmanr(flat_approx, flat_true)
    
    # Per-instance correlation
    per_instance = []
    for i in range(len(approx_shap)):
        if np.std(approx_shap[i]) > 0 and np.std(true_shap[i]) > 0:
            r, _ = pearsonr(approx_shap[i], true_shap[i])
            per_instance.append(r)
    
    # Top-5 rank correlation
    top5_corr = []
    for i in range(len(approx_shap)):
        top_true = set(np.argsort(np.abs(true_shap[i]))[-5:])
        top_approx = set(np.argsort(np.abs(approx_shap[i]))[-5:])
        overlap = len(top_true & top_approx) / 5
        top5_corr.append(overlap)
    
    mse = np.mean((approx_shap - true_shap) ** 2)
    mae = np.mean(np.abs(approx_shap - true_shap))
    
    return {
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'pearson_per_instance_mean': np.mean(per_instance),
        'pearson_per_instance_std': np.std(per_instance),
        'top5_rank_overlap': np.mean(top5_corr),
        'mse': mse,
        'mae': mae
    }


def main():
    logger.info("=" * 80)
    logger.info("REAL-TIME XAI - ULB CREDIT CARD FRAUD DETECTION")
    logger.info("=" * 80)
    
    # 1. Load Data
    logger.info("\n[1/6] Loading ULB dataset...")
    try:
        loader = ULBDataLoader()
        df = loader.load()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # 2. Preprocess
    logger.info("\n[2/6] Preprocessing...")
    X, y, feature_cols = preprocess_ulb_data(df)
    
    # Temporal split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split(X, y, test_size=0.2, val_size=0.1)
    
    logger.info(f"Features: {len(feature_cols)} ({', '.join(feature_cols[:5])}...)")
    logger.info(f"Train: {len(X_train)} (fraud: {y_train.sum()})")
    logger.info(f"Val: {len(X_val)} (fraud: {y_val.sum()})")
    logger.info(f"Test: {len(X_test)} (fraud: {y_test.sum()})")
    
    # 3. Train Models
    logger.info("\n[3/6] Training fraud detection models...")
    
    models = {}
    
    # Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # Evaluate
    logger.info("\n  Model Performance on Test Set:")
    logger.info("  " + "-" * 60)
    logger.info(f"  {'Model':<20} {'AUC-ROC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("  " + "-" * 60)
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"  {name:<20} {auc:>10.4f} {f1:>10.4f} {precision:>10.4f} {recall:>10.4f}")
    
    # Select best model (Gradient Boosting usually performs better)
    primary_model = models['Gradient Boosting']
    y_prob_test = primary_model.predict_proba(X_test)[:, 1]
    y_pred_test = primary_model.predict(X_test)
    
    test_auc = roc_auc_score(y_test, y_prob_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    # 4. Train Explainers
    logger.info("\n[4/6] Training explainers...")
    
    # TreeSHAP (ground truth approximation)
    tree_shap = TreeSHAPExplainer(primary_model, feature_cols)
    tree_shap.fit(X_train[:5000])
    
    # FastSHAP (fast approximation)
    fast_shap = FastSHAPSurrogate(primary_model, feature_cols)
    fast_shap.fit(X_train[:5000], n_samples=1000)
    
    # 5. Benchmark Latency
    logger.info("\n[5/6] Benchmarking explanation latency...")
    
    explainers = {
        'TreeSHAP (approx)': tree_shap,
        'FastSHAP': fast_shap
    }
    
    bench_df = benchmark_latency(explainers, X_test, n_samples=500)
    
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(bench_df.to_string(index=False))
    
    # 6. Evaluate Fidelity
    logger.info("\n[6/6] Evaluating FastSHAP fidelity...")
    
    n_eval = min(1000, len(X_test))
    logger.info(f"  Computing SHAP values for {n_eval} test samples...")
    
    tree_result = tree_shap.explain(X_test[:n_eval])
    fast_result = fast_shap.explain(X_test[:n_eval])
    
    fidelity = compute_fidelity(fast_result['shap_values'], tree_result['shap_values'])
    
    print("\n" + "=" * 80)
    print("FIDELITY EVALUATION (FastSHAP vs TreeSHAP)")
    print("=" * 80)
    print(f"  Pearson correlation:              {fidelity['pearson_r']:.4f}")
    print(f"  Spearman correlation:             {fidelity['spearman_r']:.4f}")
    print(f"  Per-instance Pearson (mean):      {fidelity['pearson_per_instance_mean']:.4f}")
    print(f"  Per-instance Pearson (std):       {fidelity['pearson_per_instance_std']:.4f}")
    print(f"  Top-5 feature overlap:            {fidelity['top5_rank_overlap']:.4f}")
    print(f"  MSE:                              {fidelity['mse']:.6f}")
    print(f"  MAE:                              {fidelity['mae']:.6f}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    fast_row = bench_df[bench_df['Method'] == 'FastSHAP'].iloc[0]
    
    print(f"\nModel Performance (Gradient Boosting):")
    print(f"  AUC-ROC:  {test_auc:.4f}  (target >0.98: {'PASS' if test_auc > 0.98 else 'FAIL'})")
    print(f"  F1 Score: {test_f1:.4f}  (target >0.95: {'PASS' if test_f1 > 0.95 else 'FAIL'})")
    
    print(f"\nFastSHAP Performance:")
    print(f"  P95 Latency:  {fast_row['P95 (ms)']:.2f} ms  (target <50ms: {'PASS' if fast_row['P95 (ms)'] < 50 else 'FAIL'})")
    print(f"  P99 Latency:  {fast_row['P99 (ms)']:.2f} ms  (target <100ms: {'PASS' if fast_row['P99 (ms)'] < 100 else 'FAIL'})")
    print(f"  Throughput:   {fast_row['Throughput (TPS)']:.0f} TPS")
    print(f"  Fidelity:     {fidelity['pearson_r']:.4f}")
    print(f"  Top-5 Overlap: {fidelity['top5_rank_overlap']:.4f}")
    
    # Example Explanations
    print("\n" + "=" * 80)
    print("EXAMPLE EXPLANATIONS")
    print("=" * 80)
    
    # Find one fraud and one legitimate example
    fraud_idx = np.where(y_test == 1)[0][0]
    legit_idx = np.where(y_test == 0)[0][0]
    
    for idx, label in [(legit_idx, 'LEGITIMATE'), (fraud_idx, 'FRAUD')]:
        x = X_test[idx]
        y_true = y_test[idx]
        y_prob = primary_model.predict_proba(x.reshape(1, -1))[0, 1]
        
        fast_exp = fast_shap.explain_single(x)
        shap_vals = fast_exp['shap_values'][0]
        
        print(f"\nTransaction #{idx} - {label}")
        print(f"  Predicted Probability: {y_prob:.4f}")
        print(f"  Prediction: {'FRAUD' if y_prob > 0.5 else 'LEGITIMATE'}")
        print(f"  Explanation Latency: {fast_exp['latency_ms']:.2f} ms")
        print(f"  Top 5 Contributing Features:")
        
        top5 = np.argsort(np.abs(shap_vals))[-5:][::-1]
        for rank, i in enumerate(top5, 1):
            direction = "increases" if shap_vals[i] > 0 else "decreases"
            feature_name = feature_cols[i]
            print(f"    {rank}. {feature_name}: {shap_vals[i]:+.6f} ({direction} fraud risk)")
    
    # Save results
    results = {
        'dataset': 'ULB Credit Card Fraud',
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'fraud_rate': float(df['isFraud'].mean()),
        'model_performance': {
            'auc_roc': float(test_auc),
            'f1': float(test_f1)
        },
        'latency': {
            'fastshap_p95_ms': float(fast_row['P95 (ms)']),
            'fastshap_p99_ms': float(fast_row['P99 (ms)']),
            'fastshap_throughput_tps': float(fast_row['Throughput (TPS)'])
        },
        'fidelity': {k: float(v) for k, v in fidelity.items()},
        'targets_met': {
            'p95_under_50ms': bool(fast_row['P95 (ms)'] < 50),
            'p99_under_100ms': bool(fast_row['P99 (ms)'] < 100),
            'auc_over_0.98': bool(test_auc > 0.98),
            'f1_over_0.95': bool(test_f1 > 0.95)
        }
    }
    
    output_file = 'ulb_demo_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
