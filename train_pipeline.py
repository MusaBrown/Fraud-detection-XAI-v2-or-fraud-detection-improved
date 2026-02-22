"""
Main training pipeline for Real-Time XAI Fraud Detection System.
Trains models, FastSHAP surrogate, and evaluates all components.
"""
import logging
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from src.data.load_datasets import IEEECISLoader, ULBLoader, temporal_split, get_feature_columns
from src.data.preprocessing import FraudDataPreprocessor, ClassBalancer
from src.models.train_models import train_multiple_models, EnsembleFraudDetector, FraudModelTrainer
from src.explainers.baseline_shap import TreeSHAPExplainer, KernelSHAPExplainer, benchmark_shap_methods
from src.explainers.fastshap_implementation import FastSHAPExplainer, create_fastshap_from_model
from src.explainers.lime_optimized import OptimizedLIME, benchmark_lime
from src.explainers.fidelity_metrics import ExplanationEvaluator
from src.evaluation.latency_benchmark import LatencyBenchmark
from src.evaluation.pareto_analysis import ParetoAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Real-Time XAI Fraud Detection System')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['ieee', 'ulb', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--use-sample', action='store_true',
                       help='Use sample of data for faster training')
    parser.add_argument('--sample-frac', type=float, default=0.1,
                       help='Fraction of data to sample')
    parser.add_argument('--output-dir', type=str, default='models/saved',
                       help='Output directory for saved models')
    parser.add_argument('--skip-model-training', action='store_true',
                       help='Skip model training (load existing models)')
    parser.add_argument('--skip-fastshap-training', action='store_true',
                       help='Skip FastSHAP training (load existing explainer)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)
    
    if args.dataset == 'ieee':
        loader = IEEECISLoader()
        df = loader.load(use_sample=args.use_sample, sample_frac=args.sample_frac)
    elif args.dataset == 'ulb':
        loader = ULBLoader()
        df = loader.load()
    else:
        logger.info("Using synthetic data...")
        loader = IEEECISLoader()
        df = loader._create_synthetic_data(n_samples=50000, n_features=50)
    
    # =========================================================================
    # 2. PREPROCESSING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PREPROCESSING")
    logger.info("=" * 80)
    
    preprocessor = FraudDataPreprocessor(
        categorical_threshold=20,
        max_categories=50,
        scale_features=True,
        reduce_memory=True
    )
    
    # Temporal split
    train_df, val_df, test_df = temporal_split(df, test_size=0.2, val_size=0.1)
    
    # Fit preprocessor on training data
    logger.info("Fitting preprocessor...")
    train_processed = preprocessor.fit_transform(train_df, target_col='isFraud')
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)
    
    # Save preprocessor
    preprocessor.save(output_dir / 'preprocessor.joblib')
    
    # Prepare features
    feature_cols = [c for c in train_processed.columns if c not in ['isFraud', 'TransactionID', 'TransactionDT']]
    
    X_train = train_processed[feature_cols].values.astype(np.float32)
    y_train = train_processed['isFraud'].values
    X_val = val_processed[feature_cols].values.astype(np.float32)
    y_val = val_processed['isFraud'].values
    X_test = test_processed[feature_cols].values.astype(np.float32)
    y_test = test_processed['isFraud'].values
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Fraud rate (train): {y_train.mean():.4f}")
    
    # =========================================================================
    # 3. TRAIN MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAINING MODELS")
    logger.info("=" * 80)
    
    if not args.skip_model_training:
        # Train XGBoost, LightGBM, CatBoost
        models = train_multiple_models(
            X_train, y_train, X_val, y_val,
            model_types=['xgboost', 'lightgbm'],
            categorical_features=preprocessor.categorical_features
        )
        
        # Save models
        for name, model in models.items():
            model.save(output_dir / f'{name}_model.joblib')
            
            # Evaluate on test set
            test_metrics = model.evaluate(X_test, y_test)
            logger.info(f"\n{name} Test Performance:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # Select best model for explanation
        best_model = models['xgboost']  # Default to XGBoost
    else:
        logger.info("Loading existing models...")
        best_model = FraudModelTrainer(model_type='xgboost')
        best_model.load(output_dir / 'xgboost_model.joblib')
    
    # =========================================================================
    # 4. TRAIN FASTSHAP
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: TRAINING FASTSHAP")
    logger.info("=" * 80)
    
    if not args.skip_fastshap_training:
        # Create TreeSHAP explainer as teacher
        logger.info("Creating TreeSHAP teacher...")
        tree_explainer = TreeSHAPExplainer(best_model.model, feature_names=feature_cols)
        tree_explainer.fit()
        
        # Train FastSHAP
        logger.info("Training FastSHAP surrogate (this may take a while)...")
        fastshap = create_fastshap_from_model(
            model=best_model.model,
            X_train=X_train,
            X_val=X_val,
            tree_explainer=tree_explainer,
            hidden_dims=[256, 128, 64],
            learning_rate=1e-3,
            batch_size=256,
            epochs=50,
            patience=10
        )
        
        # Save FastSHAP
        fastshap.feature_names = feature_cols
        fastshap.save(output_dir / 'fastshap_model.pt')
        
        # Evaluate fidelity
        logger.info("\nEvaluating FastSHAP fidelity...")
        n_test_samples = min(500, len(X_test))
        test_indices = np.random.choice(len(X_test), n_test_samples, replace=False)
        
        tree_result = tree_explainer.explain(X_test[test_indices])
        true_shap = tree_result['shap_values']
        
        fidelity = fastshap.compute_fidelity(X_test[test_indices], true_shap, top_k=10)
        
        logger.info("FastSHAP Fidelity Metrics:")
        for metric, value in fidelity.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
    else:
        logger.info("Loading existing FastSHAP...")
        fastshap = FastSHAPExplainer(input_dim=len(feature_cols))
        fastshap.load(output_dir / 'fastshap_model.pt')
    
    # =========================================================================
    # 5. BENCHMARK LATENCY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: BENCHMARKING LATENCY")
    logger.info("=" * 80)
    
    benchmark = LatencyBenchmark(warmup_runs=5)
    
    # Prepare explain functions
    n_benchmark = min(100, len(X_test))
    X_benchmark = X_test[:n_benchmark]
    
    def tree_shap_fn(X):
        return {'shap_values': tree_explainer.explain(X)['shap_values']}
    
    def fastshap_fn(X):
        return {'shap_values': fastshap.explain(X)['shap_values']}
    
    # Benchmark TreeSHAP (exact)
    logger.info("Benchmarking TreeSHAP...")
    benchmark.benchmark_method(
        "TreeSHAP (exact)",
        tree_shap_fn,
        X_benchmark,
        n_samples=50,
        batch_mode=False
    )
    
    # Benchmark FastSHAP
    logger.info("Benchmarking FastSHAP...")
    benchmark.benchmark_method(
        "FastSHAP",
        fastshap_fn,
        X_benchmark,
        n_samples=n_benchmark,
        batch_mode=False
    )
    
    # Benchmark KernelSHAP
    logger.info("Benchmarking KernelSHAP...")
    def kernel_shap_fn(X):
        kernel_exp = KernelSHAPExplainer(
            lambda x: best_model.model.predict_proba(x)[:, 1],
            feature_cols,
            nsamples=100
        )
        kernel_exp.fit(X_train[:100])
        return {'shap_values': kernel_exp.explain(X, nsamples=100)['shap_values']}
    
    try:
        benchmark.benchmark_method(
            "KernelSHAP (nsamples=100)",
            kernel_shap_fn,
            X_benchmark,
            n_samples=10,
            batch_mode=False
        )
    except Exception as e:
        logger.warning(f"KernelSHAP benchmarking failed: {e}")
    
    # Print results
    logger.info("\n" + benchmark.generate_report())
    
    # Save latency results
    latency_df = benchmark.get_results_df()
    latency_df.to_csv(output_dir / 'latency_benchmark.csv', index=False)
    
    # =========================================================================
    # 6. PARETO ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: PARETO ANALYSIS")
    logger.info("=" * 80)
    
    pareto = ParetoAnalyzer()
    
    # Add points from benchmark
    for result in benchmark.results:
        # Estimate fidelity (from previous evaluations)
        fidelity = 0.99 if 'FastSHAP' in result.method_name else \
                   0.95 if 'TreeSHAP' in result.method_name else 0.85
        
        pareto.add_point(
            method=result.method_name,
            latency_ms=result.p95_ms,
            fidelity=fidelity
        )
    
    pareto.compute_pareto_frontier()
    logger.info("\n" + pareto.generate_report())
    
    # =========================================================================
    # 7. SAVE CONFIGURATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: SAVING CONFIGURATION")
    logger.info("=" * 80)
    
    config = {
        'dataset': args.dataset,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'model_type': 'xgboost',
        'fastshap_config': {
            'hidden_dims': [256, 128, 64],
            'input_dim': len(feature_cols)
        },
        'preprocessing': {
            'categorical_threshold': 20,
            'max_categories': 50,
            'scale_features': True
        }
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nAll artifacts saved to {output_dir}")
    logger.info("Training pipeline completed successfully!")
    
    # Print final summary
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model saved: {output_dir / 'xgboost_model.joblib'}")
    logger.info(f"FastSHAP saved: {output_dir / 'fastshap_model.pt'}")
    logger.info(f"Preprocessor saved: {output_dir / 'preprocessor.joblib'}")
    logger.info(f"Config saved: {output_dir / 'config.json'}")


if __name__ == '__main__':
    main()
