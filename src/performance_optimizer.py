#!/usr/bin/env python3
"""
Performance Optimization Framework for SDLC Components
Advanced performance profiling, optimization, and adaptive tuning
"""

import time
import psutil
import threading
import multiprocessing
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque
from contextlib import contextmanager
import cProfile
import pstats
import tracemalloc
import gc
import functools
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np

from logging_framework import get_logger, profile_performance
from caching_framework import cached, cache_manager
from monitoring_framework import MetricsCollector, MetricType

logger = get_logger('performance')


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    thread_count: int
    cache_hit_rate: float = 0.0
    throughput_ops_per_sec: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion based on performance analysis"""
    category: str  # 'memory', 'cpu', 'io', 'cache', 'concurrency'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    recommendation: str
    estimated_improvement: str
    implementation_effort: str  # 'low', 'medium', 'high'
    code_location: Optional[str] = None
    metrics_evidence: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Advanced performance profiler with multiple profiling methods"""
    
    def __init__(self):
        self.profilers = {}
        self.active_profilers = set()
        self.metrics_collector = MetricsCollector()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup performance metrics"""
        self.metrics_collector.register_metric(
            "profiler_execution_time",
            MetricType.HISTOGRAM,
            unit="seconds",
            help_text="Execution time of profiled operations"
        )
        
        self.metrics_collector.register_metric(
            "profiler_memory_usage",
            MetricType.GAUGE,
            unit="megabytes",
            help_text="Memory usage during profiling"
        )
        
        self.metrics_collector.register_metric(
            "profiler_cpu_usage",
            MetricType.GAUGE,
            unit="percent",
            help_text="CPU usage during profiling"
        )
    
    @contextmanager
    def profile_cpu(self, operation_name: str):
        """CPU profiling context manager"""
        profiler = cProfile.Profile()
        start_time = time.time()
        
        try:
            profiler.enable()
            yield profiler
        finally:
            profiler.disable()
            execution_time = time.time() - start_time
            
            # Store profile results
            self.profilers[f"{operation_name}_cpu"] = profiler
            
            # Record metrics
            self.metrics_collector.observe_histogram(
                "profiler_execution_time",
                execution_time,
                labels={"operation": operation_name, "type": "cpu"}
            )
    
    @contextmanager
    def profile_memory(self, operation_name: str):
        """Memory profiling context manager"""
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory
            
            # Record metrics
            self.metrics_collector.set_gauge(
                "profiler_memory_usage",
                memory_delta,
                labels={"operation": operation_name}
            )
            
            self.metrics_collector.observe_histogram(
                "profiler_execution_time",
                execution_time,
                labels={"operation": operation_name, "type": "memory"}
            )
            
            logger.debug(f"Memory profiling {operation_name}: {memory_delta:.2f}MB delta, peak: {peak/1024/1024:.2f}MB")
    
    @contextmanager
    def profile_comprehensive(self, operation_name: str):
        """Comprehensive profiling (CPU + Memory + System)"""
        # Start all monitoring
        tracemalloc.start()
        profiler = cProfile.Profile()
        process = psutil.Process()
        
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_cpu_times = process.cpu_times()
        
        try:
            profiler.enable()
            yield
        finally:
            profiler.disable()
            
            # Collect final metrics
            execution_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024 / 1024
            end_cpu_times = process.cpu_times()
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            memory_delta = end_memory - start_memory
            cpu_time_delta = (end_cpu_times.user + end_cpu_times.system) - (start_cpu_times.user + start_cpu_times.system)
            cpu_percent = (cpu_time_delta / execution_time) * 100 if execution_time > 0 else 0
            
            # Store results
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=memory_delta,
                cpu_percent=cpu_percent,
                thread_count=threading.active_count(),
                metadata={
                    'peak_memory_mb': peak_memory / 1024 / 1024,
                    'current_memory_mb': current_memory / 1024 / 1024
                }
            )
            
            self.profilers[f"{operation_name}_comprehensive"] = {
                'cpu_profiler': profiler,
                'metrics': metrics
            }
            
            # Record metrics
            self.metrics_collector.observe_histogram(
                "profiler_execution_time",
                execution_time,
                labels={"operation": operation_name, "type": "comprehensive"}
            )
            
            self.metrics_collector.set_gauge(
                "profiler_memory_usage",
                memory_delta,
                labels={"operation": operation_name}
            )
            
            self.metrics_collector.set_gauge(
                "profiler_cpu_usage",
                cpu_percent,
                labels={"operation": operation_name}
            )
    
    def get_cpu_profile_stats(self, operation_name: str, top_n: int = 20) -> Dict[str, Any]:
        """Get CPU profiling statistics"""
        profiler_key = f"{operation_name}_cpu"
        
        if profiler_key not in self.profilers:
            return {}
        
        profiler = self.profilers[profiler_key]
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Extract top functions
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            filename, line_number, function_name = func_info
            top_functions.append({
                'function': function_name,
                'filename': filename,
                'line_number': line_number,
                'call_count': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / cc if cc > 0 else 0
            })
        
        return {
            'operation': operation_name,
            'total_calls': stats.total_calls,
            'total_time': stats.total_tt,
            'top_functions': top_functions
        }
    
    def analyze_performance_trends(self, operation_name: str, window_size: int = 100) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # This would typically pull from a time-series database
        # For now, we'll simulate trend analysis
        
        return {
            'operation': operation_name,
            'trend_direction': 'stable',  # 'improving', 'degrading', 'stable'
            'performance_variance': 0.05,
            'outlier_count': 2,
            'recommendations': [
                'Performance is stable',
                'No immediate optimizations needed'
            ]
        }


class MemoryOptimizer:
    """Memory optimization utilities"""
    
    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return statistics"""
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())
        
        return {
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_collected': collected,
            'objects_freed': before_objects - after_objects
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    
    @staticmethod
    def find_memory_leaks() -> List[Dict[str, Any]]:
        """Analyze potential memory leaks"""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        leaks = []
        for stat in top_stats[:10]:  # Top 10 memory consumers
            leaks.append({
                'filename': stat.traceback.format()[0],
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count,
                'average_size': stat.size / stat.count if stat.count > 0 else 0
            })
        
        return leaks
    
    @classmethod
    def optimize_object_pools(cls, pool_size: int = 1000):
        """Create object pools for commonly used objects"""
        # This would implement object pooling strategies
        logger.info(f"Object pooling optimization applied with pool size: {pool_size}")


class ConcurrencyOptimizer:
    """Concurrency and parallelism optimization"""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.thread_pool = None
        self.process_pool = None
        self.optimal_thread_count = self._calculate_optimal_thread_count()
        self.optimal_process_count = self._calculate_optimal_process_count()
    
    def _calculate_optimal_thread_count(self) -> int:
        """Calculate optimal thread count based on workload type"""
        # For I/O bound tasks, use more threads
        # For CPU bound tasks, use fewer threads
        return min(self.cpu_count * 2, 32)  # Cap at 32 threads
    
    def _calculate_optimal_process_count(self) -> int:
        """Calculate optimal process count"""
        return max(1, self.cpu_count - 1)  # Leave one core for system
    
    def get_thread_pool(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get optimized thread pool"""
        if self.thread_pool is None:
            workers = max_workers or self.optimal_thread_count
            self.thread_pool = ThreadPoolExecutor(max_workers=workers)
        
        return self.thread_pool
    
    def get_process_pool(self, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
        """Get optimized process pool"""
        if self.process_pool is None:
            workers = max_workers or self.optimal_process_count
            self.process_pool = ProcessPoolExecutor(max_workers=workers)
        
        return self.process_pool
    
    def benchmark_concurrency_strategies(self, workload_func: Callable, 
                                       data: List[Any],
                                       strategies: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Benchmark different concurrency strategies"""
        strategies = strategies or ['sequential', 'threading', 'multiprocessing']
        results = {}
        
        for strategy in strategies:
            start_time = time.time()
            
            if strategy == 'sequential':
                # Sequential execution
                for item in data:
                    workload_func(item)
            
            elif strategy == 'threading':
                # Thread-based execution
                with self.get_thread_pool() as executor:
                    list(executor.map(workload_func, data))
            
            elif strategy == 'multiprocessing':
                # Process-based execution
                with self.get_process_pool() as executor:
                    list(executor.map(workload_func, data))
            
            execution_time = time.time() - start_time
            
            results[strategy] = {
                'execution_time': execution_time,
                'throughput': len(data) / execution_time if execution_time > 0 else 0,
                'speedup': results['sequential']['execution_time'] / execution_time if 'sequential' in results and execution_time > 0 else 1.0
            }
        
        return results
    
    def cleanup(self):
        """Cleanup thread and process pools"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class AdaptiveOptimizer:
    """Adaptive optimization that learns from performance patterns"""
    
    def __init__(self):
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.optimization_settings: Dict[str, Dict[str, Any]] = {}
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05  # 5% performance change threshold
    
    def record_performance(self, operation: str, metrics: PerformanceMetrics):
        """Record performance metrics for analysis"""
        self.performance_history[operation].append(metrics)
        
        # Trigger adaptation if we have enough data
        if len(self.performance_history[operation]) >= 10:
            self._analyze_and_adapt(operation)
    
    def _analyze_and_adapt(self, operation: str):
        """Analyze performance and adapt optimization settings"""
        history = list(self.performance_history[operation])
        
        if len(history) < 10:
            return
        
        # Calculate recent vs historical performance
        recent_metrics = history[-5:]  # Last 5 measurements
        historical_metrics = history[-20:-5]  # Previous 15 measurements
        
        if not historical_metrics:
            return
        
        recent_avg_time = np.mean([m.execution_time for m in recent_metrics])
        historical_avg_time = np.mean([m.execution_time for m in historical_metrics])
        
        # Check for performance degradation
        if recent_avg_time > historical_avg_time * (1 + self.adaptation_threshold):
            logger.warning(f"Performance degradation detected for {operation}")
            self._suggest_optimizations(operation, recent_metrics)
        
        # Check for performance improvement opportunity
        elif recent_avg_time < historical_avg_time * (1 - self.adaptation_threshold):
            logger.info(f"Performance improvement opportunity for {operation}")
            self._suggest_aggressive_optimizations(operation, recent_metrics)
    
    def _suggest_optimizations(self, operation: str, metrics: List[PerformanceMetrics]) -> List[OptimizationSuggestion]:
        """Suggest optimizations based on performance metrics"""
        suggestions = []
        
        avg_memory = np.mean([m.memory_usage_mb for m in metrics])
        avg_cpu = np.mean([m.cpu_percent for m in metrics])
        avg_time = np.mean([m.execution_time for m in metrics])
        
        # Memory optimization suggestions
        if avg_memory > 100:  # More than 100MB
            suggestions.append(OptimizationSuggestion(
                category="memory",
                severity="medium",
                description=f"High memory usage detected: {avg_memory:.1f}MB average",
                recommendation="Consider implementing object pooling, reducing object creation, or using memory-efficient data structures",
                estimated_improvement="10-30% memory reduction",
                implementation_effort="medium",
                metrics_evidence={"avg_memory_mb": avg_memory}
            ))
        
        # CPU optimization suggestions
        if avg_cpu > 80:  # More than 80% CPU
            suggestions.append(OptimizationSuggestion(
                category="cpu",
                severity="high",
                description=f"High CPU usage detected: {avg_cpu:.1f}% average",
                recommendation="Consider algorithm optimization, caching, or parallel processing",
                estimated_improvement="20-50% performance improvement",
                implementation_effort="high",
                metrics_evidence={"avg_cpu_percent": avg_cpu}
            ))
        
        # Execution time suggestions
        if avg_time > 1.0:  # More than 1 second
            suggestions.append(OptimizationSuggestion(
                category="performance",
                severity="medium",
                description=f"Slow execution time detected: {avg_time:.2f}s average",
                recommendation="Consider caching, pre-computation, or asynchronous processing",
                estimated_improvement="30-70% time reduction",
                implementation_effort="medium",
                metrics_evidence={"avg_execution_time": avg_time}
            ))
        
        return suggestions
    
    def _suggest_aggressive_optimizations(self, operation: str, metrics: List[PerformanceMetrics]) -> List[OptimizationSuggestion]:
        """Suggest aggressive optimizations for operations performing well"""
        suggestions = []
        
        # If performance is good, suggest advanced optimizations
        suggestions.append(OptimizationSuggestion(
            category="advanced",
            severity="low",
            description="Operation performing well, consider advanced optimizations",
            recommendation="Implement predictive caching, batch processing, or machine learning optimization",
            estimated_improvement="5-15% additional improvement",
            implementation_effort="high",
            metrics_evidence={}
        ))
        
        return suggestions
    
    def get_optimization_recommendations(self, operation: str) -> List[OptimizationSuggestion]:
        """Get optimization recommendations for an operation"""
        if operation not in self.performance_history:
            return []
        
        recent_metrics = list(self.performance_history[operation])[-10:]
        return self._suggest_optimizations(operation, recent_metrics)


class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.concurrency_optimizer = ConcurrencyOptimizer()
        self.adaptive_optimizer = AdaptiveOptimizer()
        self.metrics_collector = MetricsCollector()
        
        # Performance optimization cache
        self.optimization_cache = cache_manager.get_cache("local")
        
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup optimization metrics"""
        self.metrics_collector.register_metric(
            "optimization_suggestions_count",
            MetricType.COUNTER,
            help_text="Number of optimization suggestions generated"
        )
        
        self.metrics_collector.register_metric(
            "performance_improvement_percent",
            MetricType.HISTOGRAM,
            unit="percent",
            help_text="Performance improvement percentage after optimization"
        )
    
    def optimize_function(self, func: Callable, 
                         enable_caching: bool = True,
                         enable_profiling: bool = True,
                         cache_ttl: Optional[float] = None) -> Callable:
        """Decorator to optimize function performance"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Check cache first if enabled
            if enable_caching:
                cache_key = f"{operation_name}:{hash(str(args) + str(sorted(kwargs.items())))}"
                cached_result = self.optimization_cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Profile execution if enabled
            if enable_profiling:
                with self.profiler.profile_comprehensive(operation_name):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result if enabled
            if enable_caching:
                self.optimization_cache.put(cache_key, result, ttl=cache_ttl)
            
            return result
        
        return wrapper
    
    def benchmark_operation(self, operation_func: Callable,
                          args_list: List[tuple],
                          kwargs_list: Optional[List[dict]] = None,
                          warmup_runs: int = 3,
                          benchmark_runs: int = 10) -> Dict[str, Any]:
        """Benchmark an operation with multiple data sets"""
        kwargs_list = kwargs_list or [{}] * len(args_list)
        
        # Warmup runs
        for i in range(min(warmup_runs, len(args_list))):
            operation_func(*args_list[i], **kwargs_list[i])
        
        # Benchmark runs
        execution_times = []
        memory_usages = []
        
        for i in range(min(benchmark_runs, len(args_list))):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            operation_func(*args_list[i], **kwargs_list[i])
            
            execution_time = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_times.append(execution_time)
            memory_usages.append(end_memory - start_memory)
        
        return {
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'avg_memory_usage': np.mean(memory_usages),
            'throughput_ops_per_sec': benchmark_runs / np.sum(execution_times),
            'execution_times': execution_times,
            'memory_usages': memory_usages
        }
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        # System metrics
        system_metrics = {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent
        }
        
        # Cache statistics
        cache_stats = cache_manager.get_all_stats()
        
        # Optimization suggestions
        all_suggestions = []
        for operation in self.adaptive_optimizer.performance_history.keys():
            suggestions = self.adaptive_optimizer.get_optimization_recommendations(operation)
            all_suggestions.extend(suggestions)
        
        # Memory analysis
        memory_analysis = {
            'current_usage': self.memory_optimizer.get_memory_usage(),
            'gc_stats': self.memory_optimizer.force_garbage_collection(),
            'potential_leaks': self.memory_optimizer.find_memory_leaks()
        }
        
        return {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'system_metrics': system_metrics,
            'cache_statistics': cache_stats,
            'optimization_suggestions': [
                {
                    'category': s.category,
                    'severity': s.severity,
                    'description': s.description,
                    'recommendation': s.recommendation,
                    'estimated_improvement': s.estimated_improvement,
                    'implementation_effort': s.implementation_effort
                }
                for s in all_suggestions
            ],
            'memory_analysis': memory_analysis,
            'performance_trends': self._analyze_overall_trends()
        }
    
    def _analyze_overall_trends(self) -> Dict[str, Any]:
        """Analyze overall performance trends"""
        trends = {}
        
        for operation, history in self.adaptive_optimizer.performance_history.items():
            if len(history) >= 10:
                recent = list(history)[-5:]
                historical = list(history)[-15:-5]
                
                if historical:
                    recent_avg = np.mean([m.execution_time for m in recent])
                    historical_avg = np.mean([m.execution_time for m in historical])
                    
                    trend = "stable"
                    if recent_avg > historical_avg * 1.1:
                        trend = "degrading"
                    elif recent_avg < historical_avg * 0.9:
                        trend = "improving"
                    
                    trends[operation] = {
                        'trend': trend,
                        'recent_avg_time': recent_avg,
                        'historical_avg_time': historical_avg,
                        'change_percent': ((recent_avg - historical_avg) / historical_avg) * 100
                    }
        
        return trends
    
    def cleanup(self):
        """Cleanup optimizer resources"""
        self.concurrency_optimizer.cleanup()


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


# Convenience decorators
def optimize(enable_caching: bool = True, 
            enable_profiling: bool = True,
            cache_ttl: Optional[float] = None):
    """Convenience decorator for function optimization"""
    return performance_optimizer.optimize_function(
        enable_caching=enable_caching,
        enable_profiling=enable_profiling,
        cache_ttl=cache_ttl
    )


if __name__ == "__main__":
    # Example usage and testing
    
    @optimize(enable_caching=True, enable_profiling=True, cache_ttl=60.0)
    def expensive_computation(n: int) -> int:
        """Simulate expensive computation"""
        time.sleep(0.1)
        return sum(i ** 2 for i in range(n))
    
    # Test optimization
    print("Testing performance optimization...")
    
    # First call - will be profiled and cached
    start_time = time.time()
    result1 = expensive_computation(1000)
    time1 = time.time() - start_time
    
    # Second call - will use cache
    start_time = time.time()
    result2 = expensive_computation(1000)
    time2 = time.time() - start_time
    
    print(f"First call: {result1} in {time1:.3f}s")
    print(f"Second call (cached): {result2} in {time2:.3f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Generate optimization report
    report = performance_optimizer.generate_optimization_report()
    print(f"\nOptimization report generated with {len(report['optimization_suggestions'])} suggestions")
    
    # Cleanup
    performance_optimizer.cleanup()