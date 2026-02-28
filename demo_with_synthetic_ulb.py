"""
Complete demo using realistic synthetic IEEE-CIS-like data.
This demonstrates all framework features without requiring Kaggle download.
"""
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class ULBSyntheticGenerator:
    """Generate realistic synthetic ULB-like fraud data."""
    
    def __init__(self, n_samples=50000, fraud_rate=0.035, random_state=42):
        self.n_samples = n_samples
        self.fraud_rate = fraud_rate
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self) -> pd.DataFrame:
        """Generate synthetic ULB data with realistic fraud patterns."""
        logger.info(f"Generating {self.n_samples} synthetic transactions...")
        
        n_fraud = int(self.n_samples * self.fraud_rate)
        n_legit = self.n_samples - n_fraud
        
        # Generate base features
        data = {
            'TransactionID': range(self.n_samples),
            'TransactionDT': np.cumsum(np.random.exponential(100, self.n_samples)),
            'TransactionAmt': np.random.lognormal(4, 1.2, self.n_samples),
            'card1': np.random.randint(1000, 20000, self.n_samples),
            'card2': np.random.choice([np.nan] + list(range(100, 1000)), self.n_samples, p=[0.1] + [0.9/900]*900),
            'card3': np.random.choice([150, 185, 100], self.n_samples, p=[0.5, 0.4, 0.1]),
            'card5': np.random.choice([100, 150, 200, 250], self.n_samples),
            'addr1': np.random.choice([np.nan] + list(range(100, 500)), self.n_samples, p=[0.15] + [0.85/400]*400),
            'addr2': np.random.choice([np.nan] + list(range(1, 100)), self.n_samples, p=[0.3] + [0.7/99]*99),
            'dist1': np.random.exponential(10, self.n_samples),
            'dist2': np.random.exponential(10, self.n_samples),
        }
        
        # C features (count features)
        for i in range(1, 15):
            data[f'C{i}'] = np.random.poisson(3 + i*0.2, self.n_samples)
        
        # D features (timedelta features)
        for i in range(1, 16):
            data[f'D{i}'] = np.random.exponential(50 + i*5, self.n_samples)
        
        # M features (categorical match)
        for i in range(1, 10):
            data[f'M{i}'] = np.random.choice(['T', 'F', np.nan], self.n_samples, p=[0.4, 0.4, 0.2])
        
        # V features (engineered features)
        for i in range(1, 50):
            data[f'V{i}'] = np.random.normal(0, 1, self.n_samples)
        
        df = pd.DataFrame(data)
        
        # Create realistic fraud patterns
        fraud_indices = np.random.choice(self.n_samples, n_fraud, replace=False)
        df['isFraud'] = 0
        df.loc[fraud_indices, 'isFraud'] = 1
        
        # Fraud patterns (make fraud detectable)
        # Pattern 1: High amount transactions
        df.loc[fraud_indices, 'TransactionAmt'] *= np.random.uniform(2, 8, len(fraud_indices))
        
        # Pattern 2: Unusual card activity
        df.loc[fraud_indices, 'C1'] += np.random.poisson(15, len(fraud_indices))
        df.loc[fraud_indices, 'C2'] += np.random.poisson(10, len(fraud_indices))
        
        # Pattern 3: Geographic anomalies
        df.loc[fraud_indices, 'dist1'] = np.random.exponential(50, len(fraud_indices))
        df.loc[fraud_indices, 'addr1'] = np.random.choice(range(400, 600), len(fraud_indices))
        
        # Pattern 4: Time-based features
        df.loc[fraud_indices, 'D1'] = np.random.exponential(200, len(fraud_indices))
        df.loc[fraud_indices, 'D2'] = np.random.exponential(100, len(fraud_indices))
        
        # Pattern 5: V features (PCA-like components)
        for i in range(1, 20):
            df.loc[fraud_indices, f'V{i}'] += np.random.normal(2, 0.5, len(fraud_indices))
        
        logger.info(f"Generated data: {len(df)} transactions, fraud rate: {df['isFraud'].mean():.4f}")
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
        """Generate SHAP values using feature contributions."""
        start = time.time()
        
        # For tree models, use feature importance weighted approach
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
            # Replace with median to estimate contribution
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
    """
    FastSHAP neural surrogate - simplified version using linear approximation.
    In production, this would be a trained neural network.
    """
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.expected_value = None
        self.importance_weights = None
        
    def fit(self, X_background: np.ndarray, n_samples: int = 1000):
        """Fit surrogate by computing expected values and weights."""
        start_fit = time.time()
        
        self.expected_value = np.mean(self.model.predict_proba(X_background)[:, 1])
        
        # Compute feature importance using marginal contributions
        n_features = X_background.shape[1]
        importances = np.zeros(n_features)
        
        # Sample subset for efficiency
        indices = np.random.choice(len(X_background), min(n_samples, len(X_background)), replace=False)
        X_sample = X_background[indices]
        
        base_preds = self.model.predict_proba(X_sample)[:, 1]
        
        for i in range(n_features):
            X_perturbed = X_sample.copy()
            np.random.shuffle(X_perturbed[:, i])
            perturbed_preds = self.model.predict_proba(X_perturbed)[:, 1]
            importances[i] = np.mean(np.abs(base_preds - perturbed_preds))
        
        # Normalize
        self.importance_weights = importances / (importances.sum() + 1e-10)
        
        fit_time = time.time() - start_fit
        logger.info(f"FastSHAP surrogate fitted in {fit_time:.2f}s")
        return self
    
    def explain(self, X: np.ndarray) -> Dict:
        """Generate fast explanations."""
        start = time.time()
        
        n_samples = X.shape[0]
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


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Preprocess ULB data."""
    df = df.copy()
    
    # Handle missing values
    for col in df.columns:
        if col in ['TransactionID', 'isFraud']:
            continue
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('MISSING')
            # Simple label encoding
            df[col] = pd.Categorical(df[col]).codes
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Select features
    exclude = ['TransactionID', 'isFraud', 'TransactionDT']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df, feature_cols


def benchmark_latency(explainers: Dict, X_test: np.ndarray, n_samples: int = 200) -> pd.DataFrame:
    """Benchmark explanation latency."""
    results = []
    
    for name, explainer in explainers.items():
        logger.info(f"Benchmarking {name}...")
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
        'pearson_per_instance': np.mean(per_instance),
        'top5_rank_overlap': np.mean(top5_corr),
        'mse': mse,
        'mae': mae
    }


def main():
    logger.info("=" * 80)
    logger.info("REAL-TIME XAI FOR FRAUD DETECTION - COMPLETE DEMO")
    logger.info("Using Realistic Synthetic ULB Data")
    logger.info("=" * 80)
    
    # 1. Generate Data
    logger.info("\n[1/6] Generating synthetic ULB data...")
    generator = ULBSyntheticGenerator(n_samples=20000, fraud_rate=0.00172)
    df = generator.generate()
    
    # 2. Preprocess
    logger.info("\n[2/6] Preprocessing...")
    df_processed, feature_cols = preprocess_data(df)
    
    # Temporal split
    df_sorted = df_processed.sort_values('TransactionDT')
    n = len(df_sorted)
    train_df = df_sorted.iloc[:int(0.7*n)]
    val_df = df_sorted.iloc[int(0.7*n):int(0.85*n)]
    test_df = df_sorted.iloc[int(0.85*n):]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['isFraud'].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['isFraud'].values
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Train: {len(X_train)}, Val: {len(val_df)}, Test: {len(X_test)}")
    
    # 3. Train Models
    logger.info("\n[3/6] Training models...")
    
    models = {}
    
    # Random Forest
    logger.info("  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    logger.info("  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # Evaluate
    logger.info("\n  Model Performance:")
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        logger.info(f"    {name}: AUC={auc:.4f}, F1={f1:.4f}")
    
    # Select best model
    primary_model = models['Gradient Boosting']
    
    # 4. Train Explainers
    logger.info("\n[4/6] Training explainers...")
    
    # TreeSHAP (exact-ish)
    tree_shap = TreeSHAPExplainer(primary_model, feature_cols)
    tree_shap.fit(X_train[:1000])
    
    # FastSHAP
    fast_shap = FastSHAPSurrogate(primary_model, feature_cols)
    fast_shap.fit(X_train[:2000])
    
    # 5. Benchmark Latency
    logger.info("\n[5/6] Benchmarking latency...")
    explainers = {
        'TreeSHAP (approx)': tree_shap,
        'FastSHAP': fast_shap
    }
    
    bench_df = benchmark_latency(explainers, X_test, n_samples=200)
    
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(bench_df.to_string(index=False))
    
    # 6. Evaluate Fidelity
    logger.info("\n[6/6] Evaluating fidelity...")
    
    n_eval = 500
    tree_result = tree_shap.explain(X_test[:n_eval])
    fast_result = fast_shap.explain(X_test[:n_eval])
    
    fidelity = compute_fidelity(fast_result['shap_values'], tree_result['shap_values'])
    
    print("\n" + "=" * 80)
    print("FIDELITY EVALUATION (FastSHAP vs TreeSHAP)")
    print("=" * 80)
    for metric, value in fidelity.items():
        print(f"  {metric}: {value:.4f}")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    # Model performance
    y_prob = primary_model.predict_proba(X_test)[:, 1]
    y_pred = primary_model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"  AUC-ROC: {auc:.4f} (target >0.98: {'PASS' if auc > 0.98 else 'FAIL'})")
    print(f"  F1 Score: {f1:.4f} (target >0.95: {'PASS' if f1 > 0.95 else 'FAIL'})")
    
    # FastSHAP performance
    fast_row = bench_df[bench_df['Method'] == 'FastSHAP'].iloc[0]
    print(f"\nFastSHAP Performance:")
    print(f"  P95 Latency: {fast_row['P95 (ms)']:.2f} ms (target <50ms: {'PASS' if fast_row['P95 (ms)'] < 50 else 'FAIL'})")
    print(f"  P99 Latency: {fast_row['P99 (ms)']:.2f} ms (target <100ms: {'PASS' if fast_row['P99 (ms)'] < 100 else 'FAIL'})")
    print(f"  Throughput: {fast_row['Throughput (TPS)']:.0f} TPS")
    print(f"  Fidelity (Pearson r): {fidelity['pearson_r']:.4f}")
    print(f"  Top-5 Rank Overlap: {fidelity['top5_rank_overlap']:.4f}")
    
    # Example explanation
    print("\n" + "=" * 80)
    print("EXAMPLE EXPLANATION")
    print("=" * 80)
    
    idx = 0
    x = X_test[idx]
    y_true = y_test[idx]
    y_prob = primary_model.predict_proba(x.reshape(1, -1))[0, 1]
    
    fast_exp = fast_shap.explain_single(x)
    shap_vals = fast_exp['shap_values'][0]
    
    print(f"Transaction ID: {idx}")
    print(f"True Label: {'FRAUD' if y_true else 'LEGITIMATE'}")
    print(f"Predicted: {'FRAUD' if y_prob > 0.5 else 'LEGITIMATE'} (prob={y_prob:.4f})")
    print(f"Explanation Latency: {fast_exp['latency_ms']:.2f} ms")
    print(f"\nTop 5 Risk Factors:")
    
    top5 = np.argsort(np.abs(shap_vals))[-5:][::-1]
    for rank, i in enumerate(top5, 1):
        direction = "increases" if shap_vals[i] > 0 else "decreases"
        print(f"  {rank}. {feature_cols[i]}: {shap_vals[i]:+.6f} ({direction} risk)")
    
    # Save results
    results = {
        'model_performance': {'auc': auc, 'f1': f1},
        'latency': fast_row.to_dict(),
        'fidelity': fidelity,
        'targets_met': {
            'p95_under_50ms': fast_row['P95 (ms)'] < 50,
            'p99_under_100ms': fast_row['P99 (ms)'] < 100,
            'auc_over_0.98': auc > 0.98,
            'f1_over_0.95': f1 > 0.95
        }
    }
    
    with open('demo_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("Results saved to demo_results.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
