"""
Streaming simulation engine for load testing.
Simulates production load: 1000+ TPS with sliding window evaluation.
"""
import logging
import time
import asyncio
import threading
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import pandas as pd

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransactionMetrics:
    """Metrics for a single transaction."""
    transaction_id: str
    timestamp: float
    latency_ms: float
    prediction: int
    fraud_probability: float
    from_cache: bool = False
    is_fallback: bool = False
    error: Optional[str] = None


@dataclass
class WindowMetrics:
    """Metrics for a time window."""
    window_start: float
    window_end: float
    transaction_count: int = 0
    error_count: int = 0
    latencies: List[float] = field(default_factory=list)
    cache_hits: int = 0
    fallback_count: int = 0
    fraud_count: int = 0
    
    @property
    def throughput_tps(self) -> float:
        duration = self.window_end - self.window_start
        return self.transaction_count / duration if duration > 0 else 0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.transaction_count, 1)
    
    @property
    def p50_latency(self) -> float:
        return np.percentile(self.latencies, 50) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        return np.percentile(self.latencies, 95) if self.latencies else 0
    
    @property
    def p99_latency(self) -> float:
        return np.percentile(self.latencies, 99) if self.latencies else 0
    
    def to_dict(self) -> Dict:
        return {
            'window_start': self.window_start,
            'window_end': self.window_end,
            'transaction_count': self.transaction_count,
            'throughput_tps': self.throughput_tps,
            'error_rate': self.error_rate,
            'p50_latency_ms': self.p50_latency,
            'p95_latency_ms': self.p95_latency,
            'p99_latency_ms': self.p99_latency,
            'cache_hit_rate': self.cache_hits / max(self.transaction_count, 1),
            'fallback_rate': self.fallback_count / max(self.transaction_count, 1),
            'fraud_rate': self.fraud_count / max(self.transaction_count, 1)
        }


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    
    @classmethod
    def collect(cls) -> 'SystemMetrics':
        """Collect current system metrics."""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            return cls(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=mem.percent,
                memory_used_mb=mem.used / (1024 * 1024)
            )
        return cls(timestamp=time.time(), cpu_percent=0, memory_percent=0, memory_used_mb=0)


class StreamingSimulator:
    """
    Simulates production streaming load for fraud detection.
    Supports configurable TPS, sliding windows, and system monitoring.
    """
    
    def __init__(
        self,
        process_fn: Callable[[Dict], Dict],
        target_tps: int = 1000,
        window_sizes: List[int] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize streaming simulator.
        
        Args:
            process_fn: Function to process a transaction (returns dict with latency, prediction, etc.)
            target_tps: Target transactions per second
            window_sizes: List of window sizes in seconds (default: [60, 300, 900])
            enable_monitoring: Whether to collect system metrics
        """
        self.process_fn = process_fn
        self.target_tps = target_tps
        self.window_sizes = window_sizes or [60, 300, 900]  # 1min, 5min, 15min
        self.enable_monitoring = enable_monitoring
        
        self._running = False
        self._transaction_count = 0
        self._error_count = 0
        self._start_time: Optional[float] = None
        
        # Sliding windows
        self._windows: Dict[int, deque] = {
            size: deque() for size in self.window_sizes
        }
        
        # Metrics storage
        self._all_transactions: List[TransactionMetrics] = []
        self._system_metrics: List[SystemMetrics] = []
        self._window_summaries: List[WindowMetrics] = []
        
        # Threading
        self._lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        
    def _generate_transaction(self) -> Dict:
        """Generate a synthetic transaction for testing."""
        # Generate features similar to IEEE-CIS
        tx = {
            'TransactionAmt': np.random.lognormal(4, 1.5),
            'card1': np.random.randint(1000, 20000),
            'card2': np.random.randint(100, 1000) if np.random.random() > 0.1 else None,
            'card3': np.random.randint(100, 200),
            'card5': np.random.randint(100, 250),
            'addr1': np.random.randint(100, 500),
            'addr2': np.random.randint(1, 100) if np.random.random() > 0.3 else None,
            'dist1': np.random.exponential(10) if np.random.random() > 0.2 else None,
            'dist2': np.random.exponential(10) if np.random.random() > 0.4 else None,
            'C1': np.random.poisson(5),
            'C2': np.random.poisson(3),
            'C3': np.random.poisson(2),
            'C4': np.random.poisson(3),
            'C5': np.random.poisson(2),
            'C6': np.random.poisson(2),
            'C7': np.random.poisson(2),
            'C8': np.random.poisson(2),
            'C9': np.random.poisson(2),
            'C10': np.random.poisson(2),
            'C11': np.random.poisson(2),
            'C12': np.random.poisson(2),
            'C13': np.random.poisson(2),
            'C14': np.random.poisson(2),
            'D1': np.random.exponential(50),
            'D2': np.random.exponential(100) if np.random.random() > 0.2 else None,
            'D3': np.random.exponential(50) if np.random.random() > 0.3 else None,
            'D4': np.random.exponential(30) if np.random.random() > 0.4 else None,
            'V1': np.random.normal(0, 1),
            'V2': np.random.normal(0, 1),
            'V3': np.random.normal(0, 1),
            'V4': np.random.normal(0, 1),
            'V5': np.random.normal(0, 1),
            'V6': np.random.normal(0, 1),
            'V7': np.random.normal(0, 1),
            'V8': np.random.normal(0, 1),
            'V9': np.random.normal(0, 1),
            'V10': np.random.normal(0, 1),
        }
        return tx
    
    def _process_single(self, tx_id: str) -> TransactionMetrics:
        """Process a single transaction."""
        tx_data = self._generate_transaction()
        timestamp = time.time()
        
        try:
            result = self.process_fn(tx_data)
            
            return TransactionMetrics(
                transaction_id=tx_id,
                timestamp=timestamp,
                latency_ms=result.get('latency_ms', 0),
                prediction=result.get('prediction', 0),
                fraud_probability=result.get('fraud_probability', 0),
                from_cache=result.get('from_cache', False),
                is_fallback=result.get('is_fallback', False)
            )
        except Exception as e:
            logger.error(f"Error processing transaction {tx_id}: {e}")
            return TransactionMetrics(
                transaction_id=tx_id,
                timestamp=timestamp,
                latency_ms=0,
                prediction=0,
                fraud_probability=0,
                error=str(e)
            )
    
    def _worker_thread(self, thread_id: int, tx_queue: queue.Queue):
        """Worker thread for processing transactions."""
        while self._running:
            try:
                tx_id = tx_queue.get(timeout=1)
                if tx_id is None:  # Poison pill
                    break
                
                metrics = self._process_single(tx_id)
                
                with self._lock:
                    self._all_transactions.append(metrics)
                    
                    if metrics.error:
                        self._error_count += 1
                    
                    # Add to sliding windows
                    for size in self.window_sizes:
                        self._windows[size].append(metrics)
                        # Remove old entries
                        cutoff = time.time() - size
                        while self._windows[size] and self._windows[size][0].timestamp < cutoff:
                            self._windows[size].popleft()
                
                tx_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _monitor_thread(self, interval: float = 1.0):
        """Thread for collecting system metrics."""
        while self._running:
            if self.enable_monitoring:
                metrics = SystemMetrics.collect()
                self._system_metrics.append(metrics)
            time.sleep(interval)
    
    def run(
        self,
        duration_seconds: float = 60.0,
        num_threads: int = 4,
        warmup_seconds: float = 5.0
    ) -> Dict:
        """
        Run the streaming simulation.
        
        Args:
            duration_seconds: Total simulation duration
            num_threads: Number of worker threads
            warmup_seconds: Warmup period before measuring
            
        Returns:
            Simulation results summary
        """
        logger.info(f"Starting streaming simulation: {self.target_tps} TPS, {duration_seconds}s duration")
        
        self._running = True
        self._start_time = time.time()
        self._transaction_count = 0
        self._error_count = 0
        
        # Create queue and worker threads
        tx_queue = queue.Queue(maxsize=self.target_tps * 2)
        
        for i in range(num_threads):
            t = threading.Thread(target=self._worker_thread, args=(i, tx_queue))
            t.daemon = True
            t.start()
            self._threads.append(t)
        
        # Start monitor thread
        monitor_t = threading.Thread(target=self._monitor_thread)
        monitor_t.daemon = True
        monitor_t.start()
        self._threads.append(monitor_t)
        
        # Main loop - generate transactions at target TPS
        interval = 1.0 / self.target_tps
        end_time = self._start_time + duration_seconds + warmup_seconds
        warmup_end = self._start_time + warmup_seconds
        tx_counter = 0
        
        try:
            while time.time() < end_time and self._running:
                tx_counter += 1
                tx_id = f"tx_{tx_counter}"
                
                try:
                    tx_queue.put(tx_id, timeout=0.1)
                    
                    # Only count after warmup
                    if time.time() >= warmup_end:
                        self._transaction_count += 1
                        
                except queue.Full:
                    logger.warning("Queue full, dropping transaction")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted")
        finally:
            self.stop()
        
        return self.get_results()
    
    def stop(self):
        """Stop the simulation."""
        self._running = False
        
        for t in self._threads:
            t.join(timeout=2.0)
        
        self._threads.clear()
        logger.info("Simulation stopped")
    
    def get_window_metrics(self, window_size: int) -> WindowMetrics:
        """Get metrics for a specific window size."""
        with self._lock:
            window_data = list(self._windows[window_size])
        
        if not window_data:
            return WindowMetrics(window_start=time.time(), window_end=time.time())
        
        metrics = WindowMetrics(
            window_start=window_data[0].timestamp,
            window_end=window_data[-1].timestamp,
            transaction_count=len(window_data),
            error_count=sum(1 for tx in window_data if tx.error),
            latencies=[tx.latency_ms for tx in window_data if not tx.error],
            cache_hits=sum(1 for tx in window_data if tx.from_cache),
            fallback_count=sum(1 for tx in window_data if tx.is_fallback),
            fraud_count=sum(1 for tx in window_data if tx.prediction == 1)
        )
        
        return metrics
    
    def get_results(self) -> Dict:
        """Get comprehensive simulation results."""
        # Filter out warmup period
        if self._start_time:
            valid_transactions = [
                tx for tx in self._all_transactions 
                if tx.timestamp >= self._start_time + 5  # Skip first 5s
            ]
        else:
            valid_transactions = self._all_transactions
        
        if not valid_transactions:
            return {"error": "No transactions processed"}
        
        latencies = [tx.latency_ms for tx in valid_transactions if not tx.error]
        
        results = {
            'total_transactions': len(valid_transactions),
            'errors': sum(1 for tx in valid_transactions if tx.error),
            'duration_seconds': valid_transactions[-1].timestamp - valid_transactions[0].timestamp if len(valid_transactions) > 1 else 0,
            'throughput_tps': len(valid_transactions) / (valid_transactions[-1].timestamp - valid_transactions[0].timestamp) if len(valid_transactions) > 1 else 0,
            'latency': {
                'mean_ms': np.mean(latencies) if latencies else 0,
                'std_ms': np.std(latencies) if latencies else 0,
                'p50_ms': np.percentile(latencies, 50) if latencies else 0,
                'p95_ms': np.percentile(latencies, 95) if latencies else 0,
                'p99_ms': np.percentile(latencies, 99) if latencies else 0,
                'max_ms': np.max(latencies) if latencies else 0
            },
            'window_metrics': {
                f'{size}s': self.get_window_metrics(size).to_dict()
                for size in self.window_sizes
            },
            'system_metrics': {
                'cpu_mean': np.mean([m.cpu_percent for m in self._system_metrics]),
                'memory_mean': np.mean([m.memory_percent for m in self._system_metrics]),
                'memory_max': np.max([m.memory_used_mb for m in self._system_metrics]) if self._system_metrics else 0
            },
            'cache_metrics': {
                'cache_hit_rate': sum(1 for tx in valid_transactions if tx.from_cache) / len(valid_transactions),
                'fallback_rate': sum(1 for tx in valid_transactions if tx.is_fallback) / len(valid_transactions)
            },
            'fraud_detection': {
                'fraud_rate': sum(1 for tx in valid_transactions if tx.prediction == 1) / len(valid_transactions),
                'avg_fraud_probability': np.mean([tx.fraud_probability for tx in valid_transactions])
            }
        }
        
        return results
    
    def get_latency_timeseries(self) -> pd.DataFrame:
        """Get latency as a time series."""
        data = []
        for tx in self._all_transactions:
            if not tx.error:
                data.append({
                    'timestamp': tx.timestamp,
                    'latency_ms': tx.latency_ms,
                    'from_cache': tx.from_cache
                })
        return pd.DataFrame(data)
    
    def get_sliding_window_report(self) -> pd.DataFrame:
        """Get report of all sliding window metrics."""
        rows = []
        for size in self.window_sizes:
            metrics = self.get_window_metrics(size)
            row = metrics.to_dict()
            row['window_size'] = f"{size}s"
            rows.append(row)
        return pd.DataFrame(rows)


class AsyncStreamingSimulator:
    """Async version of streaming simulator for async/await compatibility."""
    
    def __init__(
        self,
        process_fn: Callable[[Dict], Dict],
        target_tps: int = 1000
    ):
        self.process_fn = process_fn
        self.target_tps = target_tps
        self._running = False
        self._metrics: List[TransactionMetrics] = []
    
    async def run(self, duration_seconds: float = 60.0) -> Dict:
        """Run async simulation."""
        self._running = True
        interval = 1.0 / self.target_tps
        end_time = time.time() + duration_seconds
        counter = 0
        
        async def process_one(tx_id: str):
            tx = self._generate_transaction()
            start = time.time()
            try:
                result = self.process_fn(tx)
                latency = (time.time() - start) * 1000
                self._metrics.append(TransactionMetrics(
                    transaction_id=tx_id,
                    timestamp=start,
                    latency_ms=latency,
                    prediction=result.get('prediction', 0),
                    fraud_probability=result.get('fraud_probability', 0)
                ))
            except Exception as e:
                logger.error(f"Error: {e}")
        
        tasks = []
        while time.time() < end_time and self._running:
            counter += 1
            task = asyncio.create_task(process_one(f"tx_{counter}"))
            tasks.append(task)
            await asyncio.sleep(interval)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        return self._summarize()
    
    def _generate_transaction(self) -> Dict:
        """Generate synthetic transaction."""
        return {
            'TransactionAmt': np.random.lognormal(4, 1.5),
            'card1': np.random.randint(1000, 20000),
            'C1': np.random.poisson(5),
            'V1': np.random.normal(0, 1),
        }
    
    def _summarize(self) -> Dict:
        """Summarize results."""
        if not self._metrics:
            return {}
        latencies = [m.latency_ms for m in self._metrics]
        return {
            'total': len(self._metrics),
            'mean_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
    
    def stop(self):
        self._running = False
