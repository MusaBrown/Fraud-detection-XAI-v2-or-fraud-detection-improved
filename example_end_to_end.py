"""
End-to-end example demonstrating the complete Real-Time XAI pipeline.
"""
import logging
import numpy as np
import pandas as pd

# NOTE: This example was NOT used in the project. Only ULB data was used.
from src.data.load_datasets import ULBLoader, temporal_split
from src.data.preprocessing import FraudDataPreprocessor
from src.models.train_models import FraudModelTrainer
from src.explainers.baseline_shap import TreeSHAPExplainer
from src.explainers.fastshap_implementation import FastSHAPExplainer
from src.evaluation.latency_benchmark import LatencyBenchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("Real-Time XAI for Fraud Detection - End-to-End Example")
    logger.info("=" * 80)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    logger.info("\n[1/6] Loading synthetic data...")
    loader = ULBLoader()
    df = loader._create_synthetic_data(n_samples=10000, n_features=30)
    
    # =========================================================================
    # 2. Preprocess
    # =========================================================================
    logger.info("\n[2/6] Preprocessing...")
    train_df, val_df, test_df = temporal_split(df, test_size=0.2, val_size=0.1)
    
    preprocessor = FraudDataPreprocessor()
    train_processed = preprocessor.fit_transform(train_df, target_col='isFraud')
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)
    
    feature_cols = [c for c in train_processed.columns if c not in ['isFraud', 'TransactionID']]
    X_train = train_processed[feature_cols].values.astype(np.float32)
    y_train = train_processed['isFraud'].values
    X_test = test_processed[feature_cols].values.astype(np.float32)
    y_test = test_processed['isFraud'].values
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # =========================================================================
    # 3. Train Model
    # =========================================================================
    logger.info("\n[3/6] Training XGBoost model...")
    model_trainer = FraudModelTrainer(model_type='xgboost')
    model_trainer.build_model(n_estimators=100, max_depth=6)
    metrics = model_trainer.fit(X_train, y_train, X_val=None, y_val=None)
    
    test_metrics = model_trainer.evaluate(X_test, y_test)
    logger.info(f"Test AUC-ROC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    
    # =========================================================================
    # 4. Train FastSHAP
    # =========================================================================
    logger.info("\n[4/6] Training FastSHAP explainer...")
    
    # Create TreeSHAP teacher
    tree_explainer = TreeSHAPExplainer(model_trainer.model, feature_names=feature_cols)
    tree_explainer.fit()
    
    # Generate training data
    n_train = min(1000, len(X_train))
    train_result = tree_explainer.explain(X_train[:n_train])
    shap_train = train_result['shap_values']
    expected_value = train_result['expected_value']
    
    val_result = tree_explainer.explain(X_test[:200])
    shap_val = val_result['shap_values']
    
    # Train FastSHAP
    fastshap = FastSHAPExplainer(
        input_dim=X_train.shape[1],
        hidden_dims=[128, 64],
        epochs=20,
        patience=5
    )
    
    fastshap.fit(
        X_train=X_train[:n_train],
        shap_values_train=shap_train,
        X_val=X_test[:200],
        shap_values_val=shap_val,
        expected_value=expected_value
    )
    
    # =========================================================================
    # 5. Evaluate Fidelity
    # =========================================================================
    logger.info("\n[5/6] Evaluating FastSHAP fidelity...")
    n_eval = min(200, len(X_test))
    test_result = tree_explainer.explain(X_test[:n_eval])
    true_shap = test_result['shap_values']
    
    fidelity = fastshap.compute_fidelity(X_test[:n_eval], true_shap, top_k=10)
    logger.info(f"Pearson correlation: {fidelity['pearson']:.4f}")
    logger.info(f"Top-10 rank correlation: {fidelity['spearman_topk_mean']:.4f}")
    
    # =========================================================================
    # 6. Benchmark Latency
    # =========================================================================
    logger.info("\n[6/6] Benchmarking latency...")
    benchmark = LatencyBenchmark(warmup_runs=3)
    
    def tree_fn(X):
        return {'shap_values': tree_explainer.explain(X, check_additivity=False)['shap_values']}
    
    def fast_fn(X):
        return {'shap_values': fastshap.explain(X)['shap_values']}
    
    benchmark.benchmark_method("TreeSHAP", tree_fn, X_test[:50], n_samples=20)
    benchmark.benchmark_method("FastSHAP", fast_fn, X_test[:100], n_samples=100)
    
    print("\n" + benchmark.generate_report())
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model Performance: AUC={test_metrics['roc_auc']:.4f}, F1={test_metrics['f1']:.4f}")
    logger.info(f"FastSHAP Fidelity: {fidelity['pearson']:.4f} (target: >0.90)")
    
    for result in benchmark.results:
        logger.info(f"{result.method_name}: P95={result.p95_ms:.1f}ms (target: <50ms)")
    
    logger.info("\nExample completed successfully!")


if __name__ == '__main__':
    main()
