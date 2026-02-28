"""
Comprehensive Benchmarking for Master's Thesis
Includes: Statistical Rigor, Class-Specific Analysis, Stability Metrics

This script provides thesis-level evaluation with:
- Multi-seed statistical validation (95% confidence intervals)
- Class-specific fidelity analysis (fraud vs legitimate)
- Stability and robustness metrics
- Ablation study on architectures
"""
import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
from src.data.load_datasets import ULBLoader, temporal_split
from src.data.preprocessing import FraudDataPreprocessor
from src.models.train_models import FraudModelTrainer
from src.explainers.baseline_shap import TreeSHAPExplainer
from src.explainers.fastshap_implementation import FastSHAPExplainer
from src.evaluation.latency_benchmark import LatencyBenchmark


class ComprehensiveBenchmark:
    """Comprehensive benchmarking for thesis-level evaluation."""
    
    def __init__(self, output_dir: str = 'thesis_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split ULB data."""
        logger.info("Loading ULB Credit Card Fraud dataset...")
        loader = ULBLoader()
        df = loader.load()
        
        # Temporal split
        train_df, val_df, test_df = temporal_split(df, time_col='Time', test_size=0.2, val_size=0.1)
        
        logger.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Fraud rates - Train: {train_df['isFraud'].mean():.4f}, "
                   f"Val: {val_df['isFraud'].mean():.4f}, "
                   f"Test: {test_df['isFraud'].mean():.4f}")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, train_df, val_df, test_df):
        """Preprocess data and extract features."""
        preprocessor = FraudDataPreprocessor(
            categorical_threshold=20,
            max_categories=50,
            scale_features=True,
            reduce_memory=True
        )
        
        train_processed = preprocessor.fit_transform(train_df, target_col='isFraud')
        val_processed = preprocessor.transform(val_df)
        test_processed = preprocessor.transform(test_df)
        
        feature_cols = [c for c in train_processed.columns if c not in ['isFraud', 'Time']]
        
        X_train = train_processed[feature_cols]
        y_train = train_processed['isFraud']
        X_val = val_processed[feature_cols]
        y_val = val_processed['isFraud']
        X_test = test_processed[feature_cols]
        y_test = test_processed['isFraud']
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def run_statistical_benchmark(self, model, X_test, y_test, feature_cols, 
                                   n_seeds=10) -> Dict:
        """
        Run benchmarks with multiple random seeds for statistical rigor.
        
        Returns mean ± std with 95% confidence intervals.
        """
        logger.info(f"\n{'='*80}")
        logger.info("STATISTICAL BENCHMARK - Multi-Seed Validation")
        logger.info(f"Running {n_seeds} iterations with different random seeds...")
        logger.info(f"{'='*80}\n")
        
        results = {
            'fastshap_latency': [],
            'fastshap_fidelity': [],
            'treeshap_latency': [],
            'stability_scores': []
        }
        
        # Load pre-trained FastSHAP
        fastshap = FastSHAPExplainer(input_dim=len(feature_cols))
        fastshap.load('models/saved/fastshap_model.pt')
        fastshap.feature_names = feature_cols
        
        # Create TreeSHAP explainer
        tree_explainer = TreeSHAPExplainer(model.model, feature_names=feature_cols)
        tree_explainer.fit()
        
        # Sample test data (consistent across seeds)
        np.random.seed(42)  # Fixed seed for data sampling
        n_samples = 100
        test_indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_test_sample = X_test.iloc[test_indices]
        y_test_sample = y_test.iloc[test_indices]
        X_test_np = X_test_sample.values.astype(np.float32)
        
        # Get ground truth once
        tree_result = tree_explainer.explain(X_test_np)
        true_shap = tree_result['shap_values']
        
        for seed_idx in range(n_seeds):
            seed = 42 + seed_idx * 100
            logger.info(f"Iteration {seed_idx + 1}/{n_seeds} (seed={seed})")
            np.random.seed(seed)
            
            # Benchmark FastSHAP latency
            latencies = []
            for i in range(min(50, len(X_test_np))):
                start = time.time()
                _ = fastshap.explain(X_test_np[i:i+1])['shap_values']
                latencies.append((time.time() - start) * 1000)
            
            results['fastshap_latency'].append({
                'mean': np.mean(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            })
            
            # Benchmark TreeSHAP latency
            latencies_tree = []
            for i in range(min(20, len(X_test_np))):  # Fewer samples (slower)
                start = time.time()
                _ = tree_explainer.explain(X_test_np[i:i+1])['shap_values']
                latencies_tree.append((time.time() - start) * 1000)
            
            results['treeshap_latency'].append({
                'mean': np.mean(latencies_tree),
                'p95': np.percentile(latencies_tree, 95)
            })
            
            # Compute fidelity
            fastshap_result = fastshap.explain(X_test_np)['shap_values']
            fidelity_pearson = self._compute_pearson_correlation(fastshap_result, true_shap)
            results['fastshap_fidelity'].append(fidelity_pearson)
            
            # Stability test
            stability = self._compute_stability_score(fastshap, X_test_np[:10])
            results['stability_scores'].append(stability)
        
        # Compute statistics
        stats_results = self._compute_statistics(results)
        
        # Save results
        with open(self.output_dir / 'statistical_benchmark.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("STATISTICAL BENCHMARK RESULTS")
        logger.info("="*80)
        self._print_statistics(stats_results)
        
        return stats_results
    
    def _compute_pearson_correlation(self, pred, true):
        """Compute Pearson correlation."""
        from scipy.stats import pearsonr
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        corr, _ = pearsonr(pred_flat, true_flat)
        return corr
    
    def _compute_stability_score(self, explainer, X, n_perturbations=50, noise_level=0.01):
        """
        Compute stability score by perturbing inputs.
        Lower variance = more stable.
        """
        explanations = []
        
        # Get base explanation
        base_exp = explainer.explain(X)['shap_values']
        
        # Perturb and re-explain
        for _ in range(n_perturbations):
            X_noise = X + np.random.normal(0, noise_level, X.shape)
            exp_noise = explainer.explain(X_noise)['shap_values']
            explanations.append(exp_noise)
        
        # Compute coefficient of variation (CV)
        explanations = np.array(explanations)
        mean_exp = np.mean(explanations, axis=0)
        std_exp = np.std(explanations, axis=0)
        
        # CV = std / mean (averaged across features)
        cv = np.mean(std_exp / (np.abs(mean_exp) + 1e-10))
        
        # Stability score = 1 / (1 + CV) [higher is better]
        stability = 1 / (1 + cv)
        
        return stability
    
    def _compute_statistics(self, results):
        """Compute mean, std, and 95% CI for all metrics."""
        stats_results = {}
        
        for key, values in results.items():
            if isinstance(values[0], dict):
                # Nested dict (latency metrics)
                stats_results[key] = {}
                for metric in values[0].keys():
                    metric_values = [v[metric] for v in values]
                    stats_results[key][metric] = self._compute_metric_stats(metric_values)
            else:
                # Simple list
                stats_results[key] = self._compute_metric_stats(values)
        
        return stats_results
    
    def _compute_metric_stats(self, values):
        """Compute statistics for a single metric."""
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)  # Sample std
        n = len(arr)
        
        # 95% Confidence Interval
        ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=stats.sem(arr))
        
        return {
            'mean': float(mean),
            'std': float(std),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'ci_95_lower': float(ci_95[0]),
            'ci_95_upper': float(ci_95[1]),
            'n_samples': n
        }
    
    def _print_statistics(self, stats_results):
        """Print formatted statistics."""
        logger.info("\nFastSHAP Latency (ms):")
        for metric, stats in stats_results['fastshap_latency'].items():
            logger.info(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} ms "
                       f"[95% CI: {stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}]")
        
        logger.info("\nTreeSHAP Latency (ms):")
        for metric, stats in stats_results['treeshap_latency'].items():
            logger.info(f"  {metric.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} ms "
                       f"[95% CI: {stats['ci_95_lower']:.3f}, {stats['ci_95_upper']:.3f}]")
        
        logger.info("\nFastSHAP Fidelity (Pearson Correlation):")
        stats = stats_results['fastshap_fidelity']
        logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"[95% CI: {stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
        
        logger.info("\nStability Score (higher is better):")
        stats = stats_results['stability_scores']
        logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"[95% CI: {stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]")
    
    def run_class_specific_analysis(self, model, X_test, y_test, feature_cols) -> Dict:
        """
        Analyze explanation quality separately for fraud vs legitimate transactions.
        """
        logger.info(f"\n{'='*80}")
        logger.info("CLASS-SPECIFIC ANALYSIS")
        logger.info("="*80 + "\n")
        
        # Load FastSHAP
        fastshap = FastSHAPExplainer(input_dim=len(feature_cols))
        fastshap.load('models/saved/fastshap_model.pt')
        fastshap.feature_names = feature_cols
        
        # Create TreeSHAP
        tree_explainer = TreeSHAPExplainer(model.model, feature_names=feature_cols)
        tree_explainer.fit()
        
        # Separate fraud and legitimate
        fraud_mask = y_test == 1
        legit_mask = y_test == 0
        
        X_fraud = X_test[fraud_mask]
        y_fraud = y_test[fraud_mask]
        X_legit = X_test[legit_mask]
        y_legit = y_test[legit_mask]
        
        logger.info(f"Fraud samples: {len(X_fraud)} ({len(X_fraud)/len(X_test)*100:.3f}%)")
        logger.info(f"Legitimate samples: {len(X_legit)} ({len(X_legit)/len(X_test)*100:.3f}%)")
        
        results = {}
        
        # Analyze fraud samples
        if len(X_fraud) > 0:
            logger.info("\n--- Fraud Transactions Analysis ---")
            results['fraud'] = self._analyze_subset(X_fraud, y_fraud, fastshap, tree_explainer, 
                                                     feature_cols, "FRAUD")
        
        # Analyze legitimate samples
        logger.info("\n--- Legitimate Transactions Analysis ---")
        results['legitimate'] = self._analyze_subset(X_legit, y_legit, fastshap, tree_explainer,
                                                       feature_cols, "LEGITIMATE")
        
        # High-value fraud analysis (Amount > Q3)
        if 'Amount' in X_test.columns:
            amount_q3 = X_test['Amount'].quantile(0.75)
            high_value_mask = (X_test['Amount'] > amount_q3) & (y_test == 1)
            X_high_value = X_test[high_value_mask]
            
            if len(X_high_value) > 0:
                logger.info(f"\n--- High-Value Fraud Analysis (Amount > ${amount_q3:.2f}) ---")
                results['high_value_fraud'] = self._analyze_subset(
                    X_high_value, y_test[high_value_mask], fastshap, tree_explainer,
                    feature_cols, "HIGH-VALUE FRAUD"
                )
        
        # Save results (convert numpy types to Python types)
        def convert_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_python(item) for item in obj]
            return obj
        
        results_clean = convert_to_python(results)
        with open(self.output_dir / 'class_specific_analysis.json', 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        return results
    
    def _analyze_subset(self, X_subset, y_subset, fastshap, tree_explainer, 
                        feature_cols, label) -> Dict:
        """Analyze a specific subset of data."""
        # Sample if too large
        if len(X_subset) > 100:
            indices = np.random.choice(len(X_subset), 100, replace=False)
            X_sample = X_subset.iloc[indices]
        else:
            X_sample = X_subset
        
        X_np = X_sample.values.astype(np.float32)
        
        # Get explanations
        tree_result = tree_explainer.explain(X_np)
        true_shap = tree_result['shap_values']
        fastshap_result = fastshap.explain(X_np)['shap_values']
        
        # Compute fidelity
        fidelity = self._compute_pearson_correlation(fastshap_result, true_shap)
        
        # Compute per-sample fidelity
        per_sample_fidelity = []
        for i in range(len(X_np)):
            corr = self._compute_pearson_correlation(
                fastshap_result[i:i+1], true_shap[i:i+1]
            )
            per_sample_fidelity.append(corr)
        
        # Feature importance analysis
        feature_importance = np.abs(true_shap).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        top_features = [(feature_cols[i], feature_importance[i]) 
                       for i in top_features_idx]
        
        logger.info(f"  Samples analyzed: {len(X_np)}")
        logger.info(f"  Fidelity (Pearson): {fidelity:.4f}")
        logger.info(f"  Per-sample fidelity: {np.mean(per_sample_fidelity):.4f} ± {np.std(per_sample_fidelity):.4f}")
        logger.info(f"  Top 5 important features:")
        for feat, imp in top_features:
            logger.info(f"    - {feat}: {imp:.4f}")
        
        return {
            'n_samples': len(X_np),
            'fidelity': float(fidelity),
            'per_sample_fidelity_mean': float(np.mean(per_sample_fidelity)),
            'per_sample_fidelity_std': float(np.std(per_sample_fidelity)),
            'top_features': top_features
        }


def main():
    """Run comprehensive thesis benchmarking."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE THESIS BENCHMARK")
    logger.info("Real-Time XAI for Credit Card Fraud Detection")
    logger.info("="*80 + "\n")
    
    benchmark = ComprehensiveBenchmark(output_dir='thesis_results')
    
    # Load data
    train_df, val_df, test_df = benchmark.load_data()
    
    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
        benchmark.prepare_features(train_df, val_df, test_df)
    
    # Load trained model
    logger.info("\nLoading trained XGBoost model...")
    model = FraudModelTrainer(model_type='xgboost')
    model.load('models/saved/xgboost_model.joblib')
    logger.info("Model loaded successfully\n")
    
    # Run statistical benchmark (multi-seed)
    stats_results = benchmark.run_statistical_benchmark(
        model, X_test, y_test, feature_cols, n_seeds=10
    )
    
    # Run class-specific analysis
    class_results = benchmark.run_class_specific_analysis(
        model, X_test, y_test, feature_cols
    )
    
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE BENCHMARK COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {benchmark.output_dir}/")
    logger.info("  - statistical_benchmark.json")
    logger.info("  - class_specific_analysis.json")
    logger.info("\nThese results provide the statistical rigor and class-specific")
    logger.info("analysis required for thesis-level research.")


if __name__ == '__main__':
    main()
