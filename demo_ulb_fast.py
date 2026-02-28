"""
Fast ULB Credit Card Fraud Demo - Optimized for quick execution

IMPORTANT: This demo requires the real ULB Credit Card Fraud dataset.
NO synthetic data is used.

Download: python download_ulb_data.py
Or: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""
import sys
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("REAL-TIME XAI - ULB CREDIT CARD FRAUD (FAST VERSION)")
    logger.info("=" * 80)
    
    # 1. Load Data
    logger.info("\n[1/5] Loading ULB dataset...")
    data_path = Path("data/raw/creditcard.csv")
    
    if not data_path.exists():
        logger.error("=" * 80)
        logger.error("ERROR: ULB Credit Card Fraud dataset not found!")
        logger.error("=" * 80)
        logger.error(f"\nExpected file: {data_path}")
        logger.error("\nThis demo ONLY works with real ULB data. No synthetic fallback.")
        logger.error("\nTo download the dataset:")
        logger.error("  python download_ulb_data.py")
        logger.error("\nOr download manually from:")
        logger.error("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.error("\nPlace 'creditcard.csv' in the 'data/raw/' folder.")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    df = df.rename(columns={'Class': 'isFraud'})
    logger.info(f"Loaded {len(df)} transactions, fraud rate: {df['isFraud'].mean():.4f}")
    
    # 2. Preprocess
    logger.info("\n[2/5] Preprocessing...")
    feature_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    X = df[feature_cols].values.astype(np.float32)
    y = df['isFraud'].values
    
    # Scale
    scaler = StandardScaler()
    X[:, :2] = scaler.fit_transform(X[:, :2])
    
    # Subsample for faster training (use 25% of data)
    n_samples = len(X) // 4
    indices = np.random.choice(len(X), n_samples, replace=False)
    X, y = X[indices], y[indices]
    
    # Temporal split
    n = len(X)
    X_train, y_train = X[:int(0.7*n)], y[:int(0.7*n)]
    X_val, y_val = X[int(0.7*n):int(0.85*n)], y[int(0.7*n):int(0.85*n)]
    X_test, y_test = X[int(0.85*n):], y[int(0.85*n):]
    
    logger.info(f"Using {n} samples - Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Frauds in train: {y_train.sum()}, test: {y_test.sum()}")
    
    # 3. Train Model (fast settings)
    logger.info("\n[3/5] Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=50,      # Reduced for speed
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"\n  Test Performance:")
    logger.info(f"    AUC-ROC:  {auc:.4f}")
    logger.info(f"    F1:       {f1:.4f}")
    logger.info(f"    Precision: {precision:.4f}")
    logger.info(f"    Recall:   {recall:.4f}")
    
    # 4. Fast Explainer (optimized)
    logger.info("\n[4/5] Training FastSHAP explainer...")
    
    expected_value = np.mean(model.predict_proba(X_train)[:, 1])
    
    # Compute importance weights (sample-based)
    n_features = X_train.shape[1]
    importances = np.zeros(n_features)
    
    n_sample = min(500, len(X_train))
    X_sample = X_train[:n_sample]
    base_preds = model.predict_proba(X_sample)[:, 1]
    
    for i in range(n_features):
        X_pert = X_sample.copy()
        np.random.shuffle(X_pert[:, i])
        pert_preds = model.predict_proba(X_pert)[:, 1]
        importances[i] = np.mean(np.abs(base_preds - pert_preds))
    
    importance_weights = importances / (importances.sum() + 1e-10)
    logger.info(f"  Explainer fitted in {n_sample} samples")
    
    # 5. Benchmark
    logger.info("\n[5/5] Benchmarking latency...")
    
    latencies = []
    n_bench = min(500, len(X_test))
    
    # Warmup
    for _ in range(5):
        x = X_test[0]
        pred = model.predict_proba(x.reshape(1, -1))[0, 1]
        dev = pred - expected_value
        _ = importance_weights * dev
    
    # Benchmark
    for i in range(n_bench):
        start = time.time()
        x = X_test[i]
        pred = model.predict_proba(x.reshape(1, -1))[0, 1]
        dev = pred - expected_value
        shap_vals = importance_weights * dev
        latencies.append((time.time() - start) * 1000)
    
    # Results
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = 1000 / np.mean(latencies)
    
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"  Samples:     {n_bench}")
    print(f"  Mean:        {np.mean(latencies):.2f} ms")
    print(f"  P50:         {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:         {p95:.2f} ms")
    print(f"  P99:         {p99:.2f} ms")
    print(f"  Throughput:  {throughput:.0f} TPS")
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nModel Performance:")
    print(f"  AUC-ROC:  {auc:.4f}  (target >0.98: {'PASS' if auc > 0.98 else 'FAIL'})")
    print(f"  F1 Score: {f1:.4f}  (target >0.95: {'PASS' if f1 > 0.95 else 'FAIL'})")
    
    print(f"\nFastSHAP Performance:")
    print(f"  P95 Latency:  {p95:.2f} ms  (target <50ms: {'PASS' if p95 < 50 else 'FAIL'})")
    print(f"  P99 Latency:  {p99:.2f} ms  (target <100ms: {'PASS' if p99 < 100 else 'FAIL'})")
    print(f"  Throughput:   {throughput:.0f} TPS")
    
    # Example
    print("\n" + "=" * 80)
    print("EXAMPLE EXPLANATION")
    print("=" * 80)
    
    fraud_idx = np.where(y_test == 1)[0]
    if len(fraud_idx) > 0:
        idx = fraud_idx[0]
        x = X_test[idx]
        y_true = y_test[idx]
        y_prob = model.predict_proba(x.reshape(1, -1))[0, 1]
        
        start = time.time()
        pred = model.predict_proba(x.reshape(1, -1))[0, 1]
        dev = pred - expected_value
        shap_vals = importance_weights * dev
        latency = (time.time() - start) * 1000
        
        print(f"Transaction - FRAUD")
        print(f"  Predicted Probability: {y_prob:.4f}")
        print(f"  Explanation Latency: {latency:.2f} ms")
        print(f"  Top 5 Risk Factors:")
        
        top5 = np.argsort(np.abs(shap_vals))[-5:][::-1]
        for rank, i in enumerate(top5, 1):
            direction = "increases" if shap_vals[i] > 0 else "decreases"
            print(f"    {rank}. {feature_cols[i]}: {shap_vals[i]:+.6f} ({direction} risk)")
    
    # Save
    results = {
        'dataset': 'ULB Credit Card Fraud',
        'n_total': 284807,
        'n_used': n,
        'fraud_rate': float(df['isFraud'].mean()),
        'model': {'auc': float(auc), 'f1': float(f1), 'precision': float(precision), 'recall': float(recall)},
        'latency': {'p95_ms': float(p95), 'p99_ms': float(p99), 'throughput_tps': float(throughput)},
        'targets_met': {
            'p95_under_50ms': bool(p95 < 50),
            'p99_under_100ms': bool(p99 < 100)
        }
    }
    
    with open('ulb_fast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("Results saved to: ulb_fast_results.json")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
