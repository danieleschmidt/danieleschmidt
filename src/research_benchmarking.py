#!/usr/bin/env python3
"""
Research Benchmarking Suite for Bioneuro-Olfactory Fusion
Comprehensive evaluation, comparison, and statistical analysis framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import yaml
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from abc import ABC, abstractmethod
import pickle
import warnings
from collections import defaultdict, deque
import time
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
import itertools

# Import our frameworks
from logging_framework import get_logger, EventType, profile_performance
from monitoring_framework import MetricsCollector, MetricType
from bioneuro_olfactory_fusion import (
    SpikingNeuralNetwork, OlfactoryReceptorField, MultiSensoryFusion,
    SensorModality, NeuralArchitecture, FusionStrategy,
    SensoryStimulus, ChemicalSignal
)

logger = get_logger('research_benchmarking')

class BenchmarkType(Enum):
    """Types of benchmark evaluations"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    SCALABILITY = "scalability"
    COMPARISON = "comparison"
    STATISTICAL = "statistical"

class ExperimentDesign(Enum):
    """Experimental design types"""
    SINGLE_FACTOR = "single_factor"
    FACTORIAL = "factorial"
    REPEATED_MEASURES = "repeated_measures"
    CROSS_VALIDATION = "cross_validation"
    BOOTSTRAP = "bootstrap"
    ABLATION_STUDY = "ablation"

@dataclass
class BenchmarkResult:
    """Result of a benchmark evaluation"""
    benchmark_id: str
    benchmark_type: BenchmarkType
    algorithm_name: str
    configuration: Dict[str, Any]
    metrics: Dict[str, Union[float, List[float]]]
    statistical_significance: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    processing_time: float = 0.0
    memory_usage: Optional[float] = None
    error_analysis: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ExperimentResults:
    """Results of a complete experimental study"""
    experiment_id: str
    experiment_design: ExperimentDesign
    hypothesis: str
    conditions: List[Dict[str, Any]]
    results: List[BenchmarkResult]
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)
    publication_ready: bool = False
    peer_review_notes: Optional[str] = None

@dataclass
class DatasetInfo:
    """Information about benchmark datasets"""
    dataset_id: str
    description: str
    n_samples: int
    n_features: int
    task_type: str  # classification, regression, clustering
    modalities: List[SensorModality]
    ground_truth_available: bool
    synthetic: bool = False
    source: Optional[str] = None

class SyntheticDataGenerator:
    """Generate synthetic datasets for benchmarking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random_state = config.get('random_state', 42)
        np.random.seed(self.random_state)
    
    def generate_olfactory_dataset(self, n_samples: int = 1000, n_chemicals: int = 50, 
                                  n_receptors: int = 100, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic olfactory response dataset"""
        
        # Generate chemical properties
        chemical_properties = np.random.random((n_chemicals, 5))  # MW, volatility, etc.
        
        # Generate receptor sensitivity profiles
        receptor_profiles = np.random.random((n_receptors, 5))
        
        # Generate realistic concentration patterns
        concentrations = np.random.lognormal(mean=-2, sigma=1, size=(n_samples, n_chemicals))
        concentrations = np.clip(concentrations, 0, 10)  # Realistic concentration range
        
        # Calculate receptor responses using realistic binding model
        responses = np.zeros((n_samples, n_receptors))
        
        for sample_idx in range(n_samples):
            for chem_idx in range(n_chemicals):
                concentration = concentrations[sample_idx, chem_idx]
                
                if concentration > 0.01:  # Threshold for detection
                    # Calculate binding affinity (dot product of properties)
                    affinities = np.dot(receptor_profiles, chemical_properties[chem_idx])
                    
                    # Hill equation for dose-response
                    hill_coeff = 1.0
                    ec50 = np.random.uniform(0.1, 2.0, n_receptors)
                    
                    chem_responses = (concentration ** hill_coeff) / ((ec50 ** hill_coeff) + (concentration ** hill_coeff))
                    chem_responses *= affinities
                    
                    responses[sample_idx] += chem_responses
        
        # Add realistic noise
        responses += np.random.normal(0, noise_level, responses.shape)
        responses = np.clip(responses, 0, None)  # Non-negative responses
        
        # Generate labels (chemical categories)
        chemical_categories = np.random.randint(0, 5, n_chemicals)  # 5 odor categories
        labels = np.zeros(n_samples)
        
        for sample_idx in range(n_samples):
            # Label based on dominant chemical category
            weighted_categories = concentrations[sample_idx] * (chemical_categories + 1)
            labels[sample_idx] = np.argmax(np.bincount(chemical_categories, weights=concentrations[sample_idx]))
        
        return responses, labels
    
    def generate_multisensory_dataset(self, n_samples: int = 500) -> Tuple[List[SensoryStimulus], np.ndarray]:
        """Generate synthetic multi-sensory dataset"""
        
        stimuli = []
        labels = np.random.randint(0, 3, n_samples)  # 3 stimulus categories
        
        for sample_idx in range(n_samples):
            label = labels[sample_idx]
            
            # Generate correlated multi-sensory data based on label
            base_signal = np.random.random(50) + label * 0.5
            
            # Add modality-specific variations
            olfactory_data = base_signal[:30] + np.random.normal(0, 0.2, 30)
            visual_data = base_signal[:25] + np.random.normal(0, 0.3, 25)
            auditory_data = base_signal[:20] + np.random.normal(0, 0.25, 20)
            
            # Add temporal correlation
            temporal_sync = np.array([0.0, 0.1 + label * 0.05, 0.2 + label * 0.03])
            
            stimulus = SensoryStimulus(
                stimulus_id=f"synthetic_{sample_idx}",
                modalities={
                    SensorModality.OLFACTORY: olfactory_data,
                    SensorModality.VISUAL: visual_data,
                    SensorModality.AUDITORY: auditory_data
                },
                temporal_sync=temporal_sync,
                onset_time=np.random.uniform(0, 0.5),
                duration=np.random.uniform(0.5, 2.0)
            )
            
            stimuli.append(stimulus)
        
        return stimuli, labels

class PerformanceBenchmark:
    """Performance benchmarking for neuromorphic algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup performance metrics"""
        
        metrics = [
            ("benchmark_execution_time", MetricType.HISTOGRAM, "seconds"),
            ("memory_peak_usage", MetricType.GAUGE, "MB"),
            ("throughput", MetricType.GAUGE, "samples/second"),
            ("accuracy", MetricType.HISTOGRAM, "percent"),
            ("cpu_utilization", MetricType.GAUGE, "percent")
        ]
        
        for name, metric_type, unit in metrics:
            self.metrics_collector.register_metric(name, metric_type, unit=unit)
    
    @profile_performance('research_benchmarking')
    def benchmark_snn_performance(self, snn_configs: List[Dict[str, Any]], 
                                 datasets: List[Tuple[np.ndarray, np.ndarray]]) -> List[BenchmarkResult]:
        """Benchmark SNN performance across configurations and datasets"""
        
        results = []
        
        for config_idx, config in enumerate(snn_configs):
            for dataset_idx, (X, y) in enumerate(datasets):
                
                logger.info(f"Benchmarking SNN config {config_idx + 1}/{len(snn_configs)} on dataset {dataset_idx + 1}/{len(datasets)}")
                
                # Initialize SNN
                snn = SpikingNeuralNetwork(f"benchmark_snn_{config_idx}", config)
                
                # Performance metrics
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                # Process samples
                outputs = []
                for sample in X:
                    output = snn.process(sample)
                    outputs.append(output)
                
                end_time = time.time()
                end_memory = self._get_memory_usage()
                
                processing_time = end_time - start_time
                memory_usage = end_memory - start_memory
                throughput = len(X) / processing_time if processing_time > 0 else 0
                
                # Calculate accuracy if labels available
                accuracy = None
                if y is not None and len(np.unique(y)) > 1:
                    # Convert outputs to predictions (simple classification)
                    predictions = np.array([np.argmax(output) for output in outputs])
                    accuracy = accuracy_score(y, predictions % len(np.unique(y)))
                
                # Create benchmark result
                result = BenchmarkResult(
                    benchmark_id=f"snn_perf_{config_idx}_{dataset_idx}",
                    benchmark_type=BenchmarkType.PERFORMANCE,
                    algorithm_name="SpikingNeuralNetwork",
                    configuration=config,
                    metrics={
                        'processing_time': processing_time,
                        'throughput': throughput,
                        'accuracy': accuracy if accuracy is not None else 0.0
                    },
                    processing_time=processing_time,
                    memory_usage=memory_usage,
                    metadata={
                        'n_samples': len(X),
                        'n_features': X[0].shape[0] if len(X) > 0 else 0,
                        'dataset_index': dataset_idx
                    }
                )
                
                results.append(result)
                
                # Record metrics
                self.metrics_collector.observe_histogram("benchmark_execution_time", processing_time)
                self.metrics_collector.set_gauge("memory_peak_usage", memory_usage)
                self.metrics_collector.set_gauge("throughput", throughput)
                if accuracy is not None:
                    self.metrics_collector.observe_histogram("accuracy", accuracy * 100)
        
        return results
    
    @profile_performance('research_benchmarking')
    def benchmark_fusion_strategies(self, fusion_system: MultiSensoryFusion,
                                   stimuli: List[SensoryStimulus], 
                                   strategies: List[FusionStrategy]) -> List[BenchmarkResult]:
        """Benchmark different fusion strategies"""
        
        results = []
        
        for strategy in strategies:
            logger.info(f"Benchmarking fusion strategy: {strategy.value}")
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            fusion_results = []
            processing_times = []
            confidences = []
            uncertainties = []
            
            for stimulus in stimuli:
                stimulus_start = time.time()
                fusion_result = fusion_system.fuse_sensory_inputs(stimulus, strategy)
                stimulus_time = time.time() - stimulus_start
                
                fusion_results.append(fusion_result)
                processing_times.append(stimulus_time)
                confidences.append(np.mean(list(fusion_result.confidence_scores.values())))
                uncertainties.append(fusion_result.uncertainty_estimate or 0.0)
            
            total_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_usage = end_memory - start_memory
            
            # Calculate aggregate metrics
            avg_processing_time = np.mean(processing_times)
            std_processing_time = np.std(processing_times)
            avg_confidence = np.mean(confidences)
            avg_uncertainty = np.mean(uncertainties)
            throughput = len(stimuli) / total_time if total_time > 0 else 0
            
            result = BenchmarkResult(
                benchmark_id=f"fusion_{strategy.value}",
                benchmark_type=BenchmarkType.PERFORMANCE,
                algorithm_name=f"MultiSensoryFusion_{strategy.value}",
                configuration={'fusion_strategy': strategy.value},
                metrics={
                    'avg_processing_time': avg_processing_time,
                    'std_processing_time': std_processing_time,
                    'total_processing_time': total_time,
                    'throughput': throughput,
                    'avg_confidence': avg_confidence,
                    'avg_uncertainty': avg_uncertainty,
                    'processing_times': processing_times
                },
                processing_time=total_time,
                memory_usage=memory_usage,
                metadata={
                    'n_stimuli': len(stimuli),
                    'strategy': strategy.value
                }
            )
            
            results.append(result)
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0

class AccuracyBenchmark:
    """Accuracy and quality benchmarking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
    
    def benchmark_classification_accuracy(self, algorithms: Dict[str, Any], 
                                        datasets: List[Tuple[np.ndarray, np.ndarray]],
                                        cv_folds: int = 5) -> List[BenchmarkResult]:
        """Benchmark classification accuracy using cross-validation"""
        
        results = []
        
        for dataset_idx, (X, y) in enumerate(datasets):
            logger.info(f"Benchmarking accuracy on dataset {dataset_idx + 1}/{len(datasets)}")
            
            for alg_name, algorithm in algorithms.items():
                logger.info(f"Testing algorithm: {alg_name}")
                
                # Cross-validation evaluation
                kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_scores = []
                cv_predictions = []
                cv_true_labels = []
                
                for train_idx, test_idx in kfold.split(X):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train/fit algorithm (simplified)
                    if hasattr(algorithm, 'fit'):
                        algorithm.fit(X_train, y_train)
                    
                    # Predict
                    predictions = self._predict_with_algorithm(algorithm, X_test)
                    
                    # Calculate metrics
                    fold_accuracy = accuracy_score(y_test, predictions)
                    cv_scores.append(fold_accuracy)
                    cv_predictions.extend(predictions)
                    cv_true_labels.extend(y_test)
                
                # Calculate comprehensive metrics
                overall_accuracy = accuracy_score(cv_true_labels, cv_predictions)
                precision = precision_score(cv_true_labels, cv_predictions, average='weighted', zero_division=0)
                recall = recall_score(cv_true_labels, cv_predictions, average='weighted', zero_division=0)
                f1 = f1_score(cv_true_labels, cv_predictions, average='weighted', zero_division=0)
                
                # Statistical analysis
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                confidence_interval = stats.t.interval(
                    0.95, len(cv_scores) - 1,
                    loc=mean_cv_score,
                    scale=stats.sem(cv_scores)
                )
                
                result = BenchmarkResult(
                    benchmark_id=f"accuracy_{alg_name}_{dataset_idx}",
                    benchmark_type=BenchmarkType.ACCURACY,
                    algorithm_name=alg_name,
                    configuration=getattr(algorithm, 'config', {}),
                    metrics={
                        'accuracy': overall_accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_mean': mean_cv_score,
                        'cv_std': std_cv_score,
                        'cv_scores': cv_scores
                    },
                    confidence_intervals={
                        'accuracy': confidence_interval
                    },
                    metadata={
                        'dataset_index': dataset_idx,
                        'cv_folds': cv_folds,
                        'n_samples': len(X),
                        'n_classes': len(np.unique(y))
                    }
                )
                
                results.append(result)
        
        return results
    
    def _predict_with_algorithm(self, algorithm: Any, X: np.ndarray) -> np.ndarray:
        """Make predictions with different algorithm types"""
        
        if hasattr(algorithm, 'predict'):
            return algorithm.predict(X)
        elif hasattr(algorithm, 'process'):
            # For neuromorphic algorithms
            predictions = []
            for sample in X:
                output = algorithm.process(sample)
                pred = np.argmax(output) if hasattr(output, 'shape') and output.shape else 0
                predictions.append(pred)
            return np.array(predictions)
        else:
            # Fallback: random predictions
            logger.warning(f"Algorithm {type(algorithm)} has no predict method, using random predictions")
            return np.random.randint(0, 2, len(X))

class RobustnessBenchmark:
    """Robustness and reliability benchmarking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def benchmark_noise_robustness(self, algorithm: Any, clean_data: np.ndarray,
                                  noise_levels: List[float] = [0.0, 0.1, 0.3, 0.5, 1.0]) -> BenchmarkResult:
        """Benchmark algorithm robustness to noise"""
        
        logger.info("Benchmarking noise robustness")
        
        performance_by_noise = {}
        
        for noise_level in noise_levels:
            logger.info(f"Testing noise level: {noise_level}")
            
            # Add Gaussian noise
            noisy_data = clean_data + np.random.normal(0, noise_level, clean_data.shape)
            
            # Test algorithm performance
            start_time = time.time()
            
            if hasattr(algorithm, 'process'):
                outputs = []
                for sample in noisy_data:
                    output = algorithm.process(sample)
                    outputs.append(np.mean(output) if hasattr(output, 'shape') else output)
                
                # Calculate performance degradation
                clean_outputs = []
                for sample in clean_data:
                    output = algorithm.process(sample)
                    clean_outputs.append(np.mean(output) if hasattr(output, 'shape') else output)
                
                performance_degradation = np.mean(np.abs(np.array(outputs) - np.array(clean_outputs)))
                
            else:
                performance_degradation = noise_level  # Placeholder
            
            processing_time = time.time() - start_time
            
            performance_by_noise[noise_level] = {
                'degradation': performance_degradation,
                'processing_time': processing_time
            }
        
        # Calculate robustness metrics
        degradation_values = [perf['degradation'] for perf in performance_by_noise.values()]
        robustness_score = 1.0 - np.mean(degradation_values)  # Higher is better
        
        result = BenchmarkResult(
            benchmark_id=f"robustness_{type(algorithm).__name__}",
            benchmark_type=BenchmarkType.ROBUSTNESS,
            algorithm_name=type(algorithm).__name__,
            configuration=getattr(algorithm, 'config', {}),
            metrics={
                'robustness_score': robustness_score,
                'performance_by_noise': performance_by_noise,
                'max_degradation': max(degradation_values),
                'avg_degradation': np.mean(degradation_values)
            },
            metadata={
                'noise_levels_tested': noise_levels,
                'n_samples': len(clean_data)
            }
        )
        
        return result

class StatisticalAnalysis:
    """Statistical analysis and hypothesis testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def compare_algorithms(self, results_dict: Dict[str, List[BenchmarkResult]],
                          metric: str = 'accuracy', alpha: float = 0.05) -> Dict[str, Any]:
        """Statistical comparison of multiple algorithms"""
        
        logger.info(f"Performing statistical comparison on metric: {metric}")
        
        # Extract metric values for each algorithm
        algorithm_metrics = {}
        for alg_name, results in results_dict.items():
            metrics_values = []
            for result in results:
                if metric in result.metrics:
                    value = result.metrics[metric]
                    if isinstance(value, list):
                        metrics_values.extend(value)
                    else:
                        metrics_values.append(value)
            algorithm_metrics[alg_name] = metrics_values
        
        # Remove algorithms with insufficient data
        algorithm_metrics = {name: values for name, values in algorithm_metrics.items() 
                           if len(values) >= 3}
        
        if len(algorithm_metrics) < 2:
            logger.warning("Insufficient data for statistical comparison")
            return {'error': 'Insufficient data for comparison'}
        
        # Descriptive statistics
        descriptive_stats = {}
        for alg_name, values in algorithm_metrics.items():
            descriptive_stats[alg_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
        
        # Normality tests
        normality_tests = {}
        for alg_name, values in algorithm_metrics.items():
            if len(values) >= 8:  # Minimum for Shapiro-Wilk
                statistic, p_value = stats.shapiro(values)
                normality_tests[alg_name] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': statistic,
                    'p_value': p_value,
                    'normal': p_value > alpha
                }
        
        # Statistical tests
        statistical_tests = {}
        
        # ANOVA or Kruskal-Wallis test
        all_values = list(algorithm_metrics.values())
        all_normal = all(test.get('normal', True) for test in normality_tests.values())
        
        if len(algorithm_metrics) > 2:
            if all_normal and all(len(values) >= 5 for values in all_values):
                # One-way ANOVA
                f_stat, p_value = stats.f_oneway(*all_values)
                statistical_tests['omnibus'] = {
                    'test': 'One-way ANOVA',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha
                }
            else:
                # Kruskal-Wallis test (non-parametric)
                h_stat, p_value = stats.kruskal(*all_values)
                statistical_tests['omnibus'] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < alpha
                }
        
        # Pairwise comparisons
        pairwise_tests = {}
        algorithm_names = list(algorithm_metrics.keys())
        
        for i in range(len(algorithm_names)):
            for j in range(i + 1, len(algorithm_names)):
                alg1, alg2 = algorithm_names[i], algorithm_names[j]
                values1, values2 = algorithm_metrics[alg1], algorithm_metrics[alg2]
                
                # Check if both distributions are normal
                normal1 = normality_tests.get(alg1, {}).get('normal', True)
                normal2 = normality_tests.get(alg2, {}).get('normal', True)
                
                if normal1 and normal2:
                    # T-test
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    test_name = 'Independent t-test'
                    statistic = t_stat
                else:
                    # Mann-Whitney U test
                    u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                    statistic = u_stat
                
                pairwise_tests[f"{alg1}_vs_{alg2}"] = {
                    'test': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'effect_size': self._calculate_effect_size(values1, values2)
                }
        
        # Effect sizes and practical significance
        best_algorithm = max(algorithm_metrics.keys(), 
                           key=lambda x: descriptive_stats[x]['mean'])
        
        return {
            'metric': metric,
            'alpha': alpha,
            'n_algorithms': len(algorithm_metrics),
            'descriptive_statistics': descriptive_stats,
            'normality_tests': normality_tests,
            'statistical_tests': statistical_tests,
            'pairwise_comparisons': pairwise_tests,
            'best_algorithm': best_algorithm,
            'recommendations': self._generate_recommendations(descriptive_stats, statistical_tests, pairwise_tests)
        }
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        if pooled_std > 0:
            d = (np.mean(group1) - np.mean(group2)) / pooled_std
        else:
            d = 0.0
        
        return abs(d)
    
    def _generate_recommendations(self, descriptive_stats: Dict, statistical_tests: Dict, 
                                pairwise_tests: Dict) -> List[str]:
        """Generate recommendations based on statistical analysis"""
        
        recommendations = []
        
        # Check for significant omnibus test
        omnibus = statistical_tests.get('omnibus', {})
        if omnibus.get('significant', False):
            recommendations.append(
                f"Significant differences found between algorithms ({omnibus['test']}: p={omnibus['p_value']:.4f})"
            )
        else:
            recommendations.append("No significant differences found between algorithms overall")
        
        # Identify best performer
        best_alg = max(descriptive_stats.keys(), key=lambda x: descriptive_stats[x]['mean'])
        recommendations.append(f"Best performing algorithm: {best_alg} (mean={descriptive_stats[best_alg]['mean']:.4f})")
        
        # Check for large effect sizes
        large_effects = []
        for comparison, results in pairwise_tests.items():
            if results['effect_size'] > 0.8:  # Large effect size threshold
                large_effects.append((comparison, results['effect_size']))
        
        if large_effects:
            recommendations.append("Large effect sizes found in comparisons:")
            for comp, effect in large_effects:
                recommendations.append(f"  - {comp}: Cohen's d = {effect:.3f}")
        
        return recommendations

class ExperimentRunner:
    """Main experiment orchestration and management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_generator = SyntheticDataGenerator(config.get('data_generation', {}))
        self.performance_benchmark = PerformanceBenchmark(config.get('performance', {}))
        self.accuracy_benchmark = AccuracyBenchmark(config.get('accuracy', {}))
        self.robustness_benchmark = RobustnessBenchmark(config.get('robustness', {}))
        self.statistical_analysis = StatisticalAnalysis(config.get('statistics', {}))
        
        self.results_storage = config.get('results_dir', 'benchmark_results')
        Path(self.results_storage).mkdir(parents=True, exist_ok=True)
    
    @profile_performance('research_benchmarking')
    def run_comprehensive_benchmark(self) -> ExperimentResults:
        """Run comprehensive benchmarking experiment"""
        
        experiment_id = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting comprehensive benchmark experiment: {experiment_id}")
        
        # Generate datasets
        logger.info("Generating synthetic datasets...")
        olfactory_data, olfactory_labels = self.data_generator.generate_olfactory_dataset(n_samples=500)
        multisensory_stimuli, multisensory_labels = self.data_generator.generate_multisensory_dataset(n_samples=200)
        
        datasets = [(olfactory_data, olfactory_labels)]
        
        # Initialize algorithms
        logger.info("Initializing algorithms...")
        snn_configs = [
            {
                'n_input': olfactory_data.shape[1],
                'n_hidden': [64, 32],
                'n_output': len(np.unique(olfactory_labels)),
                'learning_rate': 0.001
            },
            {
                'n_input': olfactory_data.shape[1],
                'n_hidden': [128, 64, 32],
                'n_output': len(np.unique(olfactory_labels)),
                'learning_rate': 0.0005
            }
        ]
        
        fusion_system = MultiSensoryFusion(self.config.get('fusion', {}))
        
        # Run benchmarks
        all_results = []
        
        # 1. Performance benchmarking
        logger.info("Running performance benchmarks...")
        perf_results = self.performance_benchmark.benchmark_snn_performance(snn_configs, datasets)
        all_results.extend(perf_results)
        
        # 2. Fusion strategy benchmarking
        logger.info("Running fusion strategy benchmarks...")
        fusion_strategies = [FusionStrategy.EARLY_FUSION, FusionStrategy.LATE_FUSION, 
                           FusionStrategy.HYBRID_FUSION, FusionStrategy.ATTENTION_FUSION]
        fusion_results = self.performance_benchmark.benchmark_fusion_strategies(
            fusion_system, multisensory_stimuli[:50], fusion_strategies
        )
        all_results.extend(fusion_results)
        
        # 3. Accuracy benchmarking
        logger.info("Running accuracy benchmarks...")
        algorithms = {
            f"SNN_{i}": SpikingNeuralNetwork(f"snn_{i}", config) 
            for i, config in enumerate(snn_configs)
        }
        accuracy_results = self.accuracy_benchmark.benchmark_classification_accuracy(algorithms, datasets)
        all_results.extend(accuracy_results)
        
        # 4. Robustness benchmarking
        logger.info("Running robustness benchmarks...")
        for alg_name, algorithm in algorithms.items():
            robustness_result = self.robustness_benchmark.benchmark_noise_robustness(
                algorithm, olfactory_data[:100]
            )
            all_results.append(robustness_result)
        
        # 5. Statistical analysis
        logger.info("Performing statistical analysis...")
        results_by_algorithm = defaultdict(list)
        for result in all_results:
            results_by_algorithm[result.algorithm_name].append(result)
        
        statistical_analysis = self.statistical_analysis.compare_algorithms(
            dict(results_by_algorithm), metric='accuracy'
        )
        
        # Create experiment results
        experiment_results = ExperimentResults(
            experiment_id=experiment_id,
            experiment_design=ExperimentDesign.FACTORIAL,
            hypothesis="Neuromorphic algorithms show superior performance in bioneuro-olfactory tasks",
            conditions=[
                {'algorithm_type': 'SNN', 'dataset_type': 'olfactory'},
                {'algorithm_type': 'Fusion', 'dataset_type': 'multisensory'}
            ],
            results=all_results,
            statistical_analysis=statistical_analysis,
            conclusions=[
                "Comprehensive benchmarking completed successfully",
                f"Tested {len(snn_configs)} SNN configurations",
                f"Evaluated {len(fusion_strategies)} fusion strategies",
                f"Statistical analysis completed with {len(results_by_algorithm)} algorithms"
            ]
        )
        
        # Save results
        self.save_experiment_results(experiment_results)
        
        logger.info(f"Comprehensive benchmark experiment completed: {experiment_id}")
        return experiment_results
    
    def save_experiment_results(self, experiment: ExperimentResults) -> None:
        """Save experiment results to files"""
        
        results_dir = Path(self.results_storage) / experiment.experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        with open(results_dir / "experiment_results.json", 'w') as f:
            json.dump(self._serialize_experiment(experiment), f, indent=2, default=str)
        
        # Save as YAML for readability
        with open(results_dir / "experiment_results.yaml", 'w') as f:
            yaml.dump(self._serialize_experiment(experiment), f, default_flow_style=False)
        
        # Generate summary report
        self.generate_summary_report(experiment, results_dir / "summary_report.md")
        
        # Generate plots if matplotlib available
        try:
            self.generate_plots(experiment, results_dir)
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot generation")
        
        logger.info(f"Experiment results saved to: {results_dir}")
    
    def _serialize_experiment(self, experiment: ExperimentResults) -> Dict:
        """Convert experiment results to serializable format"""
        
        return {
            'experiment_id': experiment.experiment_id,
            'experiment_design': experiment.experiment_design.value,
            'hypothesis': experiment.hypothesis,
            'conditions': experiment.conditions,
            'results': [self._serialize_result(result) for result in experiment.results],
            'statistical_analysis': experiment.statistical_analysis,
            'conclusions': experiment.conclusions,
            'publication_ready': experiment.publication_ready,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _serialize_result(self, result: BenchmarkResult) -> Dict:
        """Convert benchmark result to serializable format"""
        
        return {
            'benchmark_id': result.benchmark_id,
            'benchmark_type': result.benchmark_type.value,
            'algorithm_name': result.algorithm_name,
            'configuration': result.configuration,
            'metrics': result.metrics,
            'statistical_significance': result.statistical_significance,
            'confidence_intervals': result.confidence_intervals,
            'processing_time': result.processing_time,
            'memory_usage': result.memory_usage,
            'error_analysis': result.error_analysis,
            'metadata': result.metadata,
            'timestamp': result.timestamp
        }
    
    def generate_summary_report(self, experiment: ExperimentResults, output_path: Path) -> None:
        """Generate markdown summary report"""
        
        with open(output_path, 'w') as f:
            f.write(f"# Experiment Report: {experiment.experiment_id}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Hypothesis\n{experiment.hypothesis}\n\n")
            
            f.write(f"## Experiment Design\n")
            f.write(f"**Design Type:** {experiment.experiment_design.value}\n\n")
            
            f.write(f"**Conditions:**\n")
            for i, condition in enumerate(experiment.conditions, 1):
                f.write(f"{i}. {condition}\n")
            f.write("\n")
            
            f.write(f"## Results Summary\n")
            f.write(f"**Total Benchmarks:** {len(experiment.results)}\n")
            
            # Group results by type
            results_by_type = defaultdict(list)
            for result in experiment.results:
                results_by_type[result.benchmark_type].append(result)
            
            for bench_type, results in results_by_type.items():
                f.write(f"\n### {bench_type.value.title()} Benchmarks\n")
                f.write(f"**Count:** {len(results)}\n\n")
                
                # Summary statistics
                if results:
                    algorithms = set(r.algorithm_name for r in results)
                    f.write(f"**Algorithms Tested:** {', '.join(algorithms)}\n\n")
            
            # Statistical Analysis
            if experiment.statistical_analysis:
                f.write(f"## Statistical Analysis\n")
                
                stats_analysis = experiment.statistical_analysis
                if 'best_algorithm' in stats_analysis:
                    f.write(f"**Best Algorithm:** {stats_analysis['best_algorithm']}\n\n")
                
                if 'recommendations' in stats_analysis:
                    f.write(f"**Recommendations:**\n")
                    for rec in stats_analysis['recommendations']:
                        f.write(f"- {rec}\n")
                    f.write("\n")
            
            # Conclusions
            f.write(f"## Conclusions\n")
            for conclusion in experiment.conclusions:
                f.write(f"- {conclusion}\n")
            
            # Detailed Results
            f.write(f"\n## Detailed Results\n")
            for result in experiment.results:
                f.write(f"\n### {result.benchmark_id}\n")
                f.write(f"**Algorithm:** {result.algorithm_name}\n")
                f.write(f"**Type:** {result.benchmark_type.value}\n")
                f.write(f"**Processing Time:** {result.processing_time:.4f}s\n")
                
                if result.metrics:
                    f.write(f"**Metrics:**\n")
                    for metric, value in result.metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  - {metric}: {value:.4f}\n")
                        elif isinstance(value, list) and len(value) <= 5:
                            f.write(f"  - {metric}: {value}\n")
                        else:
                            f.write(f"  - {metric}: [complex data]\n")
                f.write("\n")
        
        logger.info(f"Summary report generated: {output_path}")
    
    def generate_plots(self, experiment: ExperimentResults, output_dir: Path) -> None:
        """Generate visualization plots"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Performance comparison plot
            perf_results = [r for r in experiment.results if r.benchmark_type == BenchmarkType.PERFORMANCE]
            if perf_results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('Performance Benchmark Results', fontsize=16)
                
                # Processing time comparison
                algorithms = [r.algorithm_name for r in perf_results]
                times = [r.processing_time for r in perf_results]
                
                axes[0, 0].bar(range(len(algorithms)), times)
                axes[0, 0].set_title('Processing Time by Algorithm')
                axes[0, 0].set_xlabel('Algorithm')
                axes[0, 0].set_ylabel('Time (seconds)')
                axes[0, 0].set_xticks(range(len(algorithms)))
                axes[0, 0].set_xticklabels(algorithms, rotation=45, ha='right')
                
                # Throughput comparison
                throughputs = [r.metrics.get('throughput', 0) for r in perf_results]
                axes[0, 1].bar(range(len(algorithms)), throughputs)
                axes[0, 1].set_title('Throughput by Algorithm')
                axes[0, 1].set_xlabel('Algorithm')
                axes[0, 1].set_ylabel('Samples/second')
                axes[0, 1].set_xticks(range(len(algorithms)))
                axes[0, 1].set_xticklabels(algorithms, rotation=45, ha='right')
                
                # Accuracy comparison
                accuracies = [r.metrics.get('accuracy', 0) for r in perf_results]
                axes[1, 0].bar(range(len(algorithms)), accuracies)
                axes[1, 0].set_title('Accuracy by Algorithm')
                axes[1, 0].set_xlabel('Algorithm')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].set_xticks(range(len(algorithms)))
                axes[1, 0].set_xticklabels(algorithms, rotation=45, ha='right')
                
                # Memory usage comparison
                memory_usage = [r.memory_usage or 0 for r in perf_results]
                axes[1, 1].bar(range(len(algorithms)), memory_usage)
                axes[1, 1].set_title('Memory Usage by Algorithm')
                axes[1, 1].set_xlabel('Algorithm')
                axes[1, 1].set_ylabel('Memory (MB)')
                axes[1, 1].set_xticks(range(len(algorithms)))
                axes[1, 1].set_xticklabels(algorithms, rotation=45, ha='right')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Accuracy results plot
            acc_results = [r for r in experiment.results if r.benchmark_type == BenchmarkType.ACCURACY]
            if acc_results:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                algorithms = [r.algorithm_name for r in acc_results]
                accuracies = [r.metrics.get('accuracy', 0) for r in acc_results]
                cv_stds = [r.metrics.get('cv_std', 0) for r in acc_results]
                
                bars = ax.bar(range(len(algorithms)), accuracies, yerr=cv_stds, capsize=5)
                ax.set_title('Classification Accuracy with Error Bars')
                ax.set_xlabel('Algorithm')
                ax.set_ylabel('Accuracy')
                ax.set_xticks(range(len(algorithms)))
                ax.set_xticklabels(algorithms, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Plots generated in: {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

def main():
    """CLI interface for research benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Benchmarking Suite")
    parser.add_argument("--run-benchmark", choices=['comprehensive', 'performance', 'accuracy', 'robustness'], 
                       help="Type of benchmark to run")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of samples for synthetic data")
    parser.add_argument("--generate-data", action="store_true", help="Generate and save synthetic datasets")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'results_dir': args.output_dir,
            'data_generation': {
                'random_state': 42
            },
            'performance': {},
            'accuracy': {},
            'robustness': {},
            'statistics': {},
            'fusion': {}
        }
    
    # Update config with CLI arguments
    config['results_dir'] = args.output_dir
    config['data_generation']['n_samples'] = args.n_samples
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    logger.info("Research Benchmarking Suite initialized")
    
    if args.generate_data:
        print("Generating synthetic datasets...")
        
        # Generate and save datasets
        olfactory_data, olfactory_labels = runner.data_generator.generate_olfactory_dataset(n_samples=args.n_samples)
        multisensory_stimuli, multisensory_labels = runner.data_generator.generate_multisensory_dataset(n_samples=args.n_samples//2)
        
        data_dir = Path(args.output_dir) / "datasets"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(data_dir / "olfactory_data.npy", olfactory_data)
        np.save(data_dir / "olfactory_labels.npy", olfactory_labels)
        
        with open(data_dir / "multisensory_stimuli.pkl", 'wb') as f:
            pickle.dump((multisensory_stimuli, multisensory_labels), f)
        
        print(f"Datasets saved to: {data_dir}")
        print(f"Olfactory dataset: {olfactory_data.shape} samples")
        print(f"Multisensory dataset: {len(multisensory_stimuli)} stimuli")
    
    if args.run_benchmark:
        print(f"Running {args.run_benchmark} benchmark...")
        
        if args.run_benchmark == 'comprehensive':
            results = runner.run_comprehensive_benchmark()
            print(f"Comprehensive benchmark completed: {results.experiment_id}")
            print(f"Total results: {len(results.results)}")
            print(f"Results saved to: {Path(args.output_dir) / results.experiment_id}")
        else:
            print(f"Specific benchmark '{args.run_benchmark}' not yet implemented")
    
    print("Research benchmarking completed.")

if __name__ == "__main__":
    main()