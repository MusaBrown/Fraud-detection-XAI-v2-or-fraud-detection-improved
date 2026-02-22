"""
Latency benchmarking module for explanation methods.
Measures P50, P95, P99 response times and throughput.
"""
import logging
import time
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LatencyBenchmarkResult:
    """Result of latency benchmark."""
    method_name: str
    latencies_ms: List[float]
    n_samples: int
    
    @property
    def mean_ms(self) -> float:
        return np.mean(self.latencies_ms)
    
    @property
    def std_ms(self) -> float:
        return np.std(self.latencies_ms)
    
    @property
    def min_ms(self) -> float:
        return np.min(self.latencies_ms)
    
    @property
    def max_ms(self) -> float:
        return np.max(self.latencies_ms)
    
    @property
    def p50_ms(self) -> float:
        return np.percentile(self.latencies_ms, 50)
    
    @property
    def p95_ms(self) -> float:
        return np.percentile(self.latencies_ms, 95)
    
    @property
    def p99_ms(self) -> float:
        return np.percentile(self.latencies_ms, 99)
    
    @property
    def throughput_tps(self) -> float:
        """Calculate throughput in transactions per second."""
        total_time_sec = sum(self.latencies_ms) / 1000
        return self.n_samples / total_time_sec if total_time_sec > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method_name,
            'n_samples': self.n_samples,
            'mean_ms': self.mean_ms,
            'std_ms': self.std_ms,
            'min_ms': self.min_ms,
            'p50_ms': self.p50_ms,
            'p95_ms': self.p95_ms,
            'p99_ms': self.p99_ms,
            'max_ms': self.max_ms,
            'throughput_tps': self.throughput_tps
        }


class LatencyBenchmark:
    """
    Benchmark latency of different explanation methods.
    """
    
    def __init__(self, warmup_runs: int = 5):
        self.warmup_runs = warmup_runs
        self.results: List[LatencyBenchmarkResult] = []
        
    def benchmark_method(
        self,
        method_name: str,
        explain_fn: Callable[[np.ndarray], Dict],
        X_test: np.ndarray,
        n_samples: Optional[int] = None,
        batch_mode: bool = False
    ) -> LatencyBenchmarkResult:
        """
        Benchmark a single explanation method.
        
        Args:
            method_name: Name of the method
            explain_fn: Function that takes X and returns dict with 'shap_values'
            X_test: Test data
            n_samples: Number of samples to benchmark (None = all)
            batch_mode: If True, pass all samples at once; else one-by-one
            
        Returns:
            LatencyBenchmarkResult
        """
        logger.info(f"Benchmarking {method_name}...")
        
        if n_samples:
            indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
            X_benchmark = X_test[indices]
        else:
            X_benchmark = X_test
        
        # Warmup runs
        logger.info(f"  Running {self.warmup_runs} warmup iterations...")
        for i in range(self.warmup_runs):
            if batch_mode:
                explain_fn(X_benchmark[:1])
            else:
                explain_fn(X_benchmark[0])
        
        # Actual benchmark
        latencies = []
        
        if batch_mode:
            # Single batch call
            start = time.time()
            result = explain_fn(X_benchmark)
            elapsed = (time.time() - start) * 1000
            latencies = [elapsed / len(X_benchmark)] * len(X_benchmark)
        else:
            # Individual calls
            for i in range(len(X_benchmark)):
                start = time.time()
                explain_fn(X_benchmark[i:i+1])
                elapsed = (time.time() - start) * 1000
                latencies.append(elapsed)
        
        result = LatencyBenchmarkResult(
            method_name=method_name,
            latencies_ms=latencies,
            n_samples=len(X_benchmark)
        )
        
        self.results.append(result)
        
        logger.info(f"  Results: Mean={result.mean_ms:.2f}ms, P95={result.p95_ms:.2f}ms, P99={result.p99_ms:.2f}ms")
        
        return result
    
    def benchmark_all(
        self,
        methods: Dict[str, Callable],
        X_test: np.ndarray,
        n_samples: int = 100
    ) -> pd.DataFrame:
        """
        Benchmark multiple methods.
        
        Args:
            methods: Dict mapping method name to explain function
            X_test: Test data
            n_samples: Number of samples per method
            
        Returns:
            DataFrame with results
        """
        for name, fn in methods.items():
            self.benchmark_method(name, fn, X_test, n_samples)
        
        return self.get_results_df()
    
    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def plot_latency_distribution(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ):
        """Plot latency distribution for all methods."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Box plot
        data = [r.latencies_ms for r in self.results]
        labels = [r.method_name for r in self.results]
        
        axes[0].boxplot(data, labels=labels)
        axes[0].set_ylabel('Latency (ms)')
        axes[0].set_title('Latency Distribution by Method')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=100, color='r', linestyle='--', label='100ms threshold')
        axes[0].axhline(y=50, color='orange', linestyle='--', label='50ms target')
        axes[0].legend()
        
        # CDF
        for result in self.results:
            sorted_latencies = np.sort(result.latencies_ms)
            cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
            axes[1].plot(sorted_latencies, cdf * 100, label=result.method_name, linewidth=2)
        
        axes[1].set_xlabel('Latency (ms)')
        axes[1].set_ylabel('Percentile')
        axes[1].set_title('Latency CDF')
        axes[1].axvline(x=100, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(x=50, color='orange', linestyle='--', alpha=0.5)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved latency plot to {save_path}")
        
        return fig
    
    def plot_percentile_comparison(
        self,
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None
    ):
        """Plot percentile comparison bar chart."""
        df = self.get_results_df()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(df))
        width = 0.2
        
        ax.bar(x - 1.5*width, df['p50_ms'], width, label='P50', alpha=0.8)
        ax.bar(x - 0.5*width, df['p95_ms'], width, label='P95', alpha=0.8)
        ax.bar(x + 0.5*width, df['p99_ms'], width, label='P99', alpha=0.8)
        ax.bar(x + 1.5*width, df['mean_ms'], width, label='Mean', alpha=0.8)
        
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Percentile Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['method'], rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100ms threshold')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50ms target')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved percentile plot to {save_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """Generate text report."""
        lines = [
            "=" * 80,
            "LATENCY BENCHMARK REPORT",
            "=" * 80,
            ""
        ]
        
        for result in self.results:
            lines.extend([
                f"Method: {result.method_name}",
                "-" * 40,
                f"  Samples:        {result.n_samples}",
                f"  Mean:           {result.mean_ms:.2f} ms",
                f"  Std Dev:        {result.std_ms:.2f} ms",
                f"  Min:            {result.min_ms:.2f} ms",
                f"  P50 (Median):   {result.p50_ms:.2f} ms",
                f"  P95:            {result.p95_ms:.2f} ms",
                f"  P99:            {result.p99_ms:.2f} ms",
                f"  Max:            {result.max_ms:.2f} ms",
                f"  Throughput:     {result.throughput_tps:.1f} TPS",
                f"  Target P95<50ms: {'✓ PASS' if result.p95_ms < 50 else '✗ FAIL'}",
                f"  Target P99<100ms: {'✓ PASS' if result.p99_ms < 100 else '✗ FAIL'}",
                ""
            ])
        
        return "\n".join(lines)
    
    def check_constraints(self, p95_threshold: float = 50, p99_threshold: float = 100) -> Dict:
        """Check if methods meet latency constraints."""
        results = {}
        for r in self.results:
            results[r.method_name] = {
                'p95_pass': r.p95_ms < p95_threshold,
                'p99_pass': r.p99_ms < p99_threshold,
                'p95_ms': r.p95_ms,
                'p99_ms': r.p99_ms
            }
        return results
