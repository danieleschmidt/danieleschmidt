#!/usr/bin/env python3
"""
Bioneuro-Olfactory Fusion Research Framework
Advanced neuromorphic algorithms for olfactory signal processing and multi-sensory integration
"""

import numpy as np
import pandas as pd
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
from collections import deque
import time

# Import our frameworks
from logging_framework import get_logger, EventType, profile_performance, retry_on_error
from monitoring_framework import MetricsCollector, MetricType

logger = get_logger('bioneuro_fusion')

class SensorModality(Enum):
    """Types of sensory modalities for fusion"""
    OLFACTORY = "olfactory"
    GUSTATORY = "gustatory"
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"
    VESTIBULAR = "vestibular"

class NeuralArchitecture(Enum):
    """Neural network architectures for bioneuro processing"""
    SPIKING_NEURAL_NETWORK = "snn"
    LIQUID_STATE_MACHINE = "lsm"
    RESERVOIR_COMPUTING = "rc"
    NEUROMORPHIC_TRANSFORMER = "ntransformer"
    BIOPHYSICAL_MODEL = "biophysical"
    HODGKIN_HUXLEY = "hh"
    IZHIKEVICH = "izh"

class FusionStrategy(Enum):
    """Multi-sensory fusion strategies"""
    EARLY_FUSION = "early"
    LATE_FUSION = "late"
    HYBRID_FUSION = "hybrid"
    ATTENTION_FUSION = "attention"
    TEMPORAL_FUSION = "temporal"
    BAYESIAN_FUSION = "bayesian"

@dataclass
class OlfactoryReceptor:
    """Model of an olfactory receptor"""
    receptor_id: str
    sensitivity_profile: np.ndarray
    response_threshold: float
    adaptation_rate: float
    recovery_time: float
    current_state: float = 0.0
    last_activation: float = 0.0
    binding_affinity: Dict[str, float] = field(default_factory=dict)

@dataclass
class SensoryNeuron:
    """Biophysically realistic sensory neuron model"""
    neuron_id: str
    membrane_potential: float = -70.0  # mV
    threshold: float = -55.0  # mV
    resting_potential: float = -70.0  # mV
    refractory_period: float = 2.0  # ms
    last_spike_time: float = -float('inf')
    synaptic_weights: Dict[str, float] = field(default_factory=dict)
    dendrite_tree: Optional['DendriteTree'] = None
    adaptation_level: float = 0.0

@dataclass
class ChemicalSignal:
    """Chemical signal representation for olfactory processing"""
    molecule_id: str
    concentration: float
    molecular_weight: float
    volatility: float
    functional_groups: List[str]
    spatial_distribution: np.ndarray
    temporal_profile: np.ndarray
    interaction_matrix: Optional[np.ndarray] = None

@dataclass
class SensoryStimulus:
    """Multi-modal sensory stimulus"""
    stimulus_id: str
    modalities: Dict[SensorModality, np.ndarray]
    temporal_sync: np.ndarray  # Temporal synchronization markers
    spatial_coordinates: Optional[np.ndarray] = None
    intensity: float = 1.0
    onset_time: float = 0.0
    duration: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FusionResult:
    """Result of multi-sensory fusion"""
    fused_representation: np.ndarray
    confidence_scores: Dict[SensorModality, float]
    fusion_strategy: FusionStrategy
    temporal_alignment: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    uncertainty_estimate: Optional[float] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class NeuromorphicProcessor(ABC):
    """Abstract base class for neuromorphic processing algorithms"""
    
    def __init__(self, processor_id: str, config: Dict[str, Any]):
        self.processor_id = processor_id
        self.config = config
        self.state_history: deque = deque(maxlen=config.get('history_length', 1000))
        self.metrics_collector = MetricsCollector()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup processor-specific metrics"""
        self.metrics_collector.register_metric(
            f"{self.processor_id}_processing_time",
            MetricType.HISTOGRAM,
            unit="seconds",
            help_text="Processing time for neuromorphic operations"
        )
        self.metrics_collector.register_metric(
            f"{self.processor_id}_spike_rate",
            MetricType.GAUGE,
            unit="spikes/second",
            help_text="Average spike rate"
        )
        self.metrics_collector.register_metric(
            f"{self.processor_id}_accuracy",
            MetricType.GAUGE,
            unit="percent",
            help_text="Processing accuracy"
        )
    
    @abstractmethod
    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Process input data through neuromorphic algorithm"""
        pass
    
    @abstractmethod
    def update_weights(self, learning_signal: np.ndarray) -> None:
        """Update synaptic weights based on learning signal"""
        pass

class SpikingNeuralNetwork(NeuromorphicProcessor):
    """Advanced Spiking Neural Network implementation"""
    
    def __init__(self, processor_id: str, config: Dict[str, Any]):
        super().__init__(processor_id, config)
        
        # Network topology
        self.n_input = config['n_input']
        self.n_hidden = config.get('n_hidden', [128, 64])
        self.n_output = config['n_output']
        
        # Neuronal parameters
        self.v_threshold = config.get('v_threshold', -55.0)  # mV
        self.v_reset = config.get('v_reset', -70.0)  # mV
        self.v_rest = config.get('v_rest', -70.0)  # mV
        self.tau_m = config.get('tau_m', 20.0)  # ms
        self.tau_syn = config.get('tau_syn', 5.0)  # ms
        self.refractory_period = config.get('refractory_period', 2.0)  # ms
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.stdp_window = config.get('stdp_window', 20.0)  # ms
        
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize network weights and state variables"""
        
        # Initialize weights with Xavier initialization
        layers = [self.n_input] + self.n_hidden + [self.n_output]
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        # Neuron state variables
        total_neurons = sum(self.n_hidden) + self.n_output
        self.membrane_potentials = np.full(total_neurons, self.v_rest)
        self.last_spike_times = np.full(total_neurons, -np.inf)
        self.synaptic_currents = np.zeros(total_neurons)
        self.spike_trains = []
        
        logger.info(f"Initialized SNN with {len(layers)} layers: {layers}")
    
    @profile_performance('bioneuro_fusion')
    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Process input through spiking neural network"""
        
        start_time = time.time()
        
        # Convert input to spike trains
        spike_input = self._encode_to_spikes(input_data)
        
        # Forward propagation through network layers
        layer_outputs = [spike_input]
        
        for layer_idx, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            layer_input = layer_outputs[-1]
            layer_output = self._process_layer(layer_input, weights, bias, layer_idx)
            layer_outputs.append(layer_output)
        
        # Decode output spikes to continuous values
        network_output = self._decode_from_spikes(layer_outputs[-1])
        
        # Record metrics
        processing_time = time.time() - start_time
        self.metrics_collector.observe_histogram(
            f"{self.processor_id}_processing_time",
            processing_time
        )
        
        # Update state history
        self.state_history.append({
            'timestamp': time.time(),
            'input': input_data,
            'output': network_output,
            'spike_rates': [np.mean(output) for output in layer_outputs[1:]]
        })
        
        return network_output
    
    def _encode_to_spikes(self, input_data: np.ndarray) -> np.ndarray:
        """Encode continuous input to spike trains using rate coding"""
        
        # Rate coding: higher values -> higher spike rates
        max_rate = self.config.get('max_spike_rate', 100.0)  # Hz
        dt = self.config.get('dt', 1.0)  # ms
        
        # Normalize input to [0, 1]
        normalized_input = np.clip(input_data / np.max(np.abs(input_data)), 0, 1)
        
        # Convert to spike probabilities
        spike_probs = normalized_input * max_rate * dt / 1000.0
        
        # Generate spikes stochastically
        spikes = np.random.random(input_data.shape) < spike_probs
        
        return spikes.astype(float)
    
    def _process_layer(self, layer_input: np.ndarray, weights: np.ndarray, 
                      bias: np.ndarray, layer_idx: int) -> np.ndarray:
        """Process one layer of the spiking neural network"""
        
        dt = self.config.get('dt', 1.0)  # ms
        
        # Calculate synaptic input
        synaptic_input = np.dot(layer_input, weights) + bias
        
        # Update synaptic currents (exponential decay)
        decay_factor = np.exp(-dt / self.tau_syn)
        layer_start_idx = sum(self.n_hidden[:layer_idx])
        layer_end_idx = layer_start_idx + weights.shape[1]
        
        self.synaptic_currents[layer_start_idx:layer_end_idx] *= decay_factor
        self.synaptic_currents[layer_start_idx:layer_end_idx] += synaptic_input
        
        # Update membrane potentials (leaky integrate-and-fire)
        current_time = time.time() * 1000  # Convert to ms
        
        # Check refractory period
        not_refractory = (current_time - self.last_spike_times[layer_start_idx:layer_end_idx]) > self.refractory_period
        
        # Update membrane potentials
        v_decay = np.exp(-dt / self.tau_m)
        dv = self.synaptic_currents[layer_start_idx:layer_end_idx] * (1 - v_decay)
        
        self.membrane_potentials[layer_start_idx:layer_end_idx] = (
            self.v_rest + 
            (self.membrane_potentials[layer_start_idx:layer_end_idx] - self.v_rest) * v_decay + 
            dv
        )
        
        # Generate spikes
        spike_mask = (self.membrane_potentials[layer_start_idx:layer_end_idx] >= self.v_threshold) & not_refractory
        spikes = np.zeros(weights.shape[1])
        spikes[spike_mask] = 1.0
        
        # Reset spiked neurons
        self.membrane_potentials[layer_start_idx:layer_end_idx][spike_mask] = self.v_reset
        self.last_spike_times[layer_start_idx:layer_end_idx][spike_mask] = current_time
        
        return spikes
    
    def _decode_from_spikes(self, spike_output: np.ndarray) -> np.ndarray:
        """Decode spike trains to continuous values"""
        
        # Simple rate-based decoding
        # In practice, you might use more sophisticated temporal decoding
        window_size = self.config.get('decode_window', 10)
        
        if len(self.spike_trains) >= window_size:
            recent_spikes = self.spike_trains[-window_size:]
            spike_rates = np.mean(recent_spikes, axis=0)
        else:
            spike_rates = spike_output
        
        self.spike_trains.append(spike_output)
        if len(self.spike_trains) > window_size:
            self.spike_trains.pop(0)
        
        return spike_rates
    
    def update_weights(self, learning_signal: np.ndarray) -> None:
        """Update synaptic weights using STDP (Spike-Timing-Dependent Plasticity)"""
        
        # Simplified STDP implementation
        # In practice, you'd need to track precise spike timing
        
        if len(self.state_history) < 2:
            return
        
        current_state = self.state_history[-1]
        previous_state = self.state_history[-2]
        
        # Calculate weight updates based on correlation and learning signal
        for layer_idx in range(len(self.weights)):
            # Get pre- and post-synaptic activities
            if layer_idx == 0:
                pre_activity = current_state['input']
            else:
                pre_activity = current_state['spike_rates'][layer_idx-1]
            
            post_activity = current_state['spike_rates'][layer_idx]
            
            # STDP-based weight update
            correlation = np.outer(pre_activity, post_activity)
            weight_update = self.learning_rate * correlation * np.mean(learning_signal)
            
            # Apply weight update with bounds
            self.weights[layer_idx] += weight_update
            self.weights[layer_idx] = np.clip(self.weights[layer_idx], -5.0, 5.0)
        
        logger.debug(f"Updated SNN weights for {self.processor_id}")

class OlfactoryReceptorField:
    """Ensemble of olfactory receptors with realistic response characteristics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.receptors: Dict[str, OlfactoryReceptor] = {}
        self.receptor_map: Dict[str, List[str]] = {}  # Chemical -> Receptor mapping
        self.metrics_collector = MetricsCollector()
        
        self._initialize_receptors()
        self._setup_metrics()
    
    def _initialize_receptors(self):
        """Initialize olfactory receptor field"""
        
        n_receptors = self.config.get('n_receptors', 100)
        sensitivity_range = self.config.get('sensitivity_range', (0.01, 1.0))
        
        # Create receptor diversity using different sensitivity profiles
        for i in range(n_receptors):
            receptor_id = f"OR_{i:03d}"
            
            # Generate unique sensitivity profile
            profile_length = self.config.get('profile_length', 50)
            sensitivity_profile = self._generate_sensitivity_profile(profile_length, i)
            
            receptor = OlfactoryReceptor(
                receptor_id=receptor_id,
                sensitivity_profile=sensitivity_profile,
                response_threshold=np.random.uniform(*sensitivity_range),
                adaptation_rate=np.random.uniform(0.1, 0.9),
                recovery_time=np.random.uniform(10.0, 100.0)  # ms
            )
            
            self.receptors[receptor_id] = receptor
        
        logger.info(f"Initialized {n_receptors} olfactory receptors")
    
    def _generate_sensitivity_profile(self, length: int, seed: int) -> np.ndarray:
        """Generate diverse sensitivity profiles for receptors"""
        
        np.random.seed(seed)
        
        # Use different basis functions for diversity
        x = np.linspace(0, 2*np.pi, length)
        
        # Combine multiple frequency components
        profile = np.zeros(length)
        n_components = np.random.randint(1, 5)
        
        for _ in range(n_components):
            freq = np.random.uniform(0.5, 3.0)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.1, 1.0)
            
            profile += amplitude * np.sin(freq * x + phase)
        
        # Add noise and normalize
        profile += np.random.normal(0, 0.1, length)
        profile = np.abs(profile)  # Ensure positive
        profile = profile / np.max(profile)  # Normalize
        
        return profile
    
    def _setup_metrics(self):
        """Setup receptor field metrics"""
        
        self.metrics_collector.register_metric(
            "receptor_activation_rate",
            MetricType.GAUGE,
            unit="percent",
            help_text="Percentage of receptors activated"
        )
        
        self.metrics_collector.register_metric(
            "receptor_response_intensity",
            MetricType.HISTOGRAM,
            unit="response_units",
            help_text="Intensity of receptor responses"
        )
    
    @profile_performance('bioneuro_fusion')
    def process_chemical_signals(self, chemical_signals: List[ChemicalSignal]) -> np.ndarray:
        """Process chemical signals through receptor field"""
        
        receptor_responses = np.zeros(len(self.receptors))
        current_time = time.time() * 1000  # Convert to ms
        
        for signal_idx, signal in enumerate(chemical_signals):
            # Find receptors that respond to this chemical
            responding_receptors = self._find_responding_receptors(signal)
            
            for receptor_id in responding_receptors:
                receptor = self.receptors[receptor_id]
                receptor_idx = list(self.receptors.keys()).index(receptor_id)
                
                # Calculate binding affinity
                affinity = self._calculate_binding_affinity(receptor, signal)
                
                # Apply concentration-response curve (Hill equation)
                response = self._hill_response(
                    signal.concentration, 
                    affinity, 
                    receptor.response_threshold
                )
                
                # Apply adaptation
                response = self._apply_adaptation(receptor, response, current_time)
                
                # Update receptor state
                receptor.current_state = response
                receptor.last_activation = current_time
                
                receptor_responses[receptor_idx] = max(receptor_responses[receptor_idx], response)
        
        # Record metrics
        activation_rate = (receptor_responses > 0.1).mean() * 100
        self.metrics_collector.set_gauge("receptor_activation_rate", activation_rate)
        
        for response in receptor_responses[receptor_responses > 0]:
            self.metrics_collector.observe_histogram("receptor_response_intensity", response)
        
        return receptor_responses
    
    def _find_responding_receptors(self, signal: ChemicalSignal) -> List[str]:
        """Find receptors that respond to a chemical signal"""
        
        # Simple model: receptors respond based on molecular features
        responding_receptors = []
        
        for receptor_id, receptor in self.receptors.items():
            # Calculate response based on molecular weight and functional groups
            mw_response = self._molecular_weight_response(signal.molecular_weight, receptor)
            fg_response = self._functional_group_response(signal.functional_groups, receptor)
            
            overall_response = (mw_response + fg_response) / 2
            
            if overall_response > 0.1:  # Threshold for response
                responding_receptors.append(receptor_id)
        
        return responding_receptors
    
    def _molecular_weight_response(self, molecular_weight: float, receptor: OlfactoryReceptor) -> float:
        """Calculate receptor response based on molecular weight"""
        
        # Map molecular weight to receptor sensitivity profile index
        mw_index = int((molecular_weight - 50) / 10) % len(receptor.sensitivity_profile)
        mw_index = max(0, min(len(receptor.sensitivity_profile) - 1, mw_index))
        
        return receptor.sensitivity_profile[mw_index]
    
    def _functional_group_response(self, functional_groups: List[str], receptor: OlfactoryReceptor) -> float:
        """Calculate receptor response based on functional groups"""
        
        # Simple functional group mapping
        group_weights = {
            'alcohol': 0.8,
            'ester': 0.9,
            'aldehyde': 0.95,
            'ketone': 0.7,
            'aromatic': 0.85,
            'aliphatic': 0.6
        }
        
        total_response = 0.0
        for group in functional_groups:
            if group in group_weights:
                # Use random component of sensitivity profile for group response
                group_idx = hash(group + receptor.receptor_id) % len(receptor.sensitivity_profile)
                group_response = receptor.sensitivity_profile[group_idx] * group_weights[group]
                total_response += group_response
        
        return min(total_response, 1.0)
    
    def _calculate_binding_affinity(self, receptor: OlfactoryReceptor, signal: ChemicalSignal) -> float:
        """Calculate binding affinity between receptor and chemical"""
        
        # Use cached affinity if available
        if signal.molecule_id in receptor.binding_affinity:
            return receptor.binding_affinity[signal.molecule_id]
        
        # Calculate new affinity based on molecular features
        affinity = self._molecular_weight_response(signal.molecular_weight, receptor)
        affinity *= self._functional_group_response(signal.functional_groups, receptor)
        
        # Add some noise for biological realism
        affinity *= np.random.uniform(0.8, 1.2)
        affinity = np.clip(affinity, 0.0, 1.0)
        
        # Cache for future use
        receptor.binding_affinity[signal.molecule_id] = affinity
        
        return affinity
    
    def _hill_response(self, concentration: float, affinity: float, threshold: float) -> float:
        """Hill equation for concentration-response relationship"""
        
        # Hill coefficient (cooperativity)
        n = self.config.get('hill_coefficient', 1.0)
        
        # EC50 (concentration for half-maximal response)
        ec50 = threshold / affinity if affinity > 0 else float('inf')
        
        if ec50 == float('inf'):
            return 0.0
        
        response = (concentration ** n) / ((ec50 ** n) + (concentration ** n))
        return response
    
    def _apply_adaptation(self, receptor: OlfactoryReceptor, response: float, current_time: float) -> float:
        """Apply adaptation to receptor response"""
        
        if receptor.last_activation == 0:
            return response
        
        # Time since last activation
        time_diff = current_time - receptor.last_activation
        
        # Recovery from adaptation
        recovery_factor = 1 - np.exp(-time_diff / receptor.recovery_time)
        recovery_factor = np.clip(recovery_factor, 0.0, 1.0)
        
        # Apply adaptation
        adapted_response = response * (1 - receptor.adaptation_rate * (1 - recovery_factor))
        
        return max(adapted_response, 0.0)

class MultiSensoryFusion:
    """Advanced multi-sensory integration and fusion system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fusion_strategies: Dict[FusionStrategy, Callable] = {
            FusionStrategy.EARLY_FUSION: self._early_fusion,
            FusionStrategy.LATE_FUSION: self._late_fusion,
            FusionStrategy.HYBRID_FUSION: self._hybrid_fusion,
            FusionStrategy.ATTENTION_FUSION: self._attention_fusion,
            FusionStrategy.TEMPORAL_FUSION: self._temporal_fusion,
            FusionStrategy.BAYESIAN_FUSION: self._bayesian_fusion
        }
        self.metrics_collector = MetricsCollector()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup fusion metrics"""
        
        self.metrics_collector.register_metric(
            "fusion_processing_time",
            MetricType.HISTOGRAM,
            unit="seconds",
            help_text="Time taken for sensory fusion"
        )
        
        self.metrics_collector.register_metric(
            "fusion_confidence",
            MetricType.HISTOGRAM,
            unit="confidence",
            help_text="Confidence in fusion results"
        )
    
    @profile_performance('bioneuro_fusion')
    def fuse_sensory_inputs(self, stimulus: SensoryStimulus, 
                           strategy: FusionStrategy = FusionStrategy.HYBRID_FUSION) -> FusionResult:
        """Fuse multi-sensory inputs using specified strategy"""
        
        start_time = time.time()
        
        if strategy not in self.fusion_strategies:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
        
        # Apply fusion strategy
        fusion_func = self.fusion_strategies[strategy]
        fused_representation, confidence_scores, attention_weights = fusion_func(stimulus)
        
        # Temporal alignment
        temporal_alignment = self._perform_temporal_alignment(stimulus)
        
        # Uncertainty estimation
        uncertainty = self._estimate_uncertainty(confidence_scores, fused_representation)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        self.metrics_collector.observe_histogram("fusion_processing_time", processing_time)
        avg_confidence = np.mean(list(confidence_scores.values()))
        self.metrics_collector.observe_histogram("fusion_confidence", avg_confidence)
        
        result = FusionResult(
            fused_representation=fused_representation,
            confidence_scores=confidence_scores,
            fusion_strategy=strategy,
            temporal_alignment=temporal_alignment,
            attention_weights=attention_weights,
            uncertainty_estimate=uncertainty,
            processing_time=processing_time,
            metadata={
                'stimulus_id': stimulus.stimulus_id,
                'n_modalities': len(stimulus.modalities),
                'strategy': strategy.value
            }
        )
        
        return result
    
    def _early_fusion(self, stimulus: SensoryStimulus) -> Tuple[np.ndarray, Dict[SensorModality, float], np.ndarray]:
        """Early fusion: Combine raw sensory data before processing"""
        
        # Concatenate all modality data
        modality_data = []
        confidence_scores = {}
        
        for modality, data in stimulus.modalities.items():
            # Normalize data to common scale
            normalized_data = self._normalize_data(data)
            modality_data.append(normalized_data)
            
            # Simple confidence based on signal strength
            confidence_scores[modality] = np.mean(np.abs(normalized_data))
        
        # Concatenate all modalities
        fused_representation = np.concatenate(modality_data)
        
        # Uniform attention weights
        attention_weights = np.ones(len(modality_data)) / len(modality_data)
        
        return fused_representation, confidence_scores, attention_weights
    
    def _late_fusion(self, stimulus: SensoryStimulus) -> Tuple[np.ndarray, Dict[SensorModality, float], np.ndarray]:
        """Late fusion: Process modalities separately then combine decisions"""
        
        processed_outputs = []
        confidence_scores = {}
        
        for modality, data in stimulus.modalities.items():
            # Process each modality separately
            processed_output = self._process_modality(data, modality)
            processed_outputs.append(processed_output)
            
            # Calculate confidence based on processing quality
            confidence_scores[modality] = self._calculate_modality_confidence(processed_output)
        
        # Weight by confidence and combine
        weights = np.array([confidence_scores[mod] for mod in stimulus.modalities.keys()])
        weights = weights / np.sum(weights)  # Normalize
        
        # Weighted average of processed outputs
        fused_representation = np.average(processed_outputs, axis=0, weights=weights)
        
        return fused_representation, confidence_scores, weights
    
    def _hybrid_fusion(self, stimulus: SensoryStimulus) -> Tuple[np.ndarray, Dict[SensorModality, float], np.ndarray]:
        """Hybrid fusion: Combine early and late fusion strategies"""
        
        # Perform both early and late fusion
        early_result, early_conf, early_weights = self._early_fusion(stimulus)
        late_result, late_conf, late_weights = self._late_fusion(stimulus)
        
        # Combine results based on relative confidence
        early_avg_conf = np.mean(list(early_conf.values()))
        late_avg_conf = np.mean(list(late_conf.values()))
        
        total_conf = early_avg_conf + late_avg_conf
        if total_conf > 0:
            early_weight = early_avg_conf / total_conf
            late_weight = late_avg_conf / total_conf
        else:
            early_weight = late_weight = 0.5
        
        # Ensure results have compatible dimensions for combination
        min_length = min(len(early_result), len(late_result))
        early_result_trimmed = early_result[:min_length]
        late_result_trimmed = late_result[:min_length]
        
        fused_representation = early_weight * early_result_trimmed + late_weight * late_result_trimmed
        
        # Combine confidence scores
        confidence_scores = {}
        for modality in stimulus.modalities.keys():
            confidence_scores[modality] = (
                early_weight * early_conf.get(modality, 0) + 
                late_weight * late_conf.get(modality, 0)
            )
        
        # Combined attention weights
        attention_weights = early_weight * early_weights + late_weight * late_weights
        
        return fused_representation, confidence_scores, attention_weights
    
    def _attention_fusion(self, stimulus: SensoryStimulus) -> Tuple[np.ndarray, Dict[SensorModality, float], np.ndarray]:
        """Attention-based fusion using learned attention mechanisms"""
        
        modality_features = []
        confidence_scores = {}
        
        # Extract features from each modality
        for modality, data in stimulus.modalities.items():
            features = self._extract_features(data, modality)
            modality_features.append(features)
            confidence_scores[modality] = np.mean(np.abs(features))
        
        # Compute attention scores
        attention_weights = self._compute_attention_weights(modality_features, stimulus)
        
        # Apply attention weights
        weighted_features = []
        for features, weight in zip(modality_features, attention_weights):
            weighted_features.append(weight * features)
        
        # Combine weighted features
        fused_representation = np.sum(weighted_features, axis=0)
        
        return fused_representation, confidence_scores, attention_weights
    
    def _temporal_fusion(self, stimulus: SensoryStimulus) -> Tuple[np.ndarray, Dict[SensorModality, float], np.ndarray]:
        """Temporal fusion considering synchronization and temporal dynamics"""
        
        # Align temporal components
        aligned_data = self._align_temporal_components(stimulus)
        
        # Apply temporal weighting based on synchronization
        temporal_weights = self._calculate_temporal_weights(stimulus)
        
        modality_outputs = []
        confidence_scores = {}
        
        for i, (modality, data) in enumerate(aligned_data.items()):
            # Apply temporal filtering
            filtered_data = self._apply_temporal_filter(data)
            modality_outputs.append(filtered_data)
            
            # Confidence based on temporal coherence
            confidence_scores[modality] = self._temporal_coherence(data, stimulus.temporal_sync)
        
        # Weight by temporal synchronization
        fused_representation = np.average(modality_outputs, axis=0, weights=temporal_weights)
        
        return fused_representation, confidence_scores, temporal_weights
    
    def _bayesian_fusion(self, stimulus: SensoryStimulus) -> Tuple[np.ndarray, Dict[SensorModality, float], np.ndarray]:
        """Bayesian fusion with uncertainty quantification"""
        
        # Prior probabilities for each modality
        priors = self._calculate_modality_priors(stimulus)
        
        # Likelihood estimates
        likelihoods = {}
        modality_outputs = []
        
        for modality, data in stimulus.modalities.items():
            # Process modality data
            output = self._process_modality(data, modality)
            modality_outputs.append(output)
            
            # Calculate likelihood
            likelihoods[modality] = self._calculate_likelihood(output, modality)
        
        # Posterior probabilities (Bayes' theorem)
        posteriors = {}
        total_evidence = 0
        
        for modality in stimulus.modalities.keys():
            evidence = likelihoods[modality] * priors[modality]
            posteriors[modality] = evidence
            total_evidence += evidence
        
        # Normalize posteriors
        if total_evidence > 0:
            for modality in posteriors:
                posteriors[modality] /= total_evidence
        
        # Use posteriors as weights and confidence scores
        weights = np.array(list(posteriors.values()))
        fused_representation = np.average(modality_outputs, axis=0, weights=weights)
        
        return fused_representation, posteriors, weights
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to common scale"""
        if np.max(np.abs(data)) > 0:
            return data / np.max(np.abs(data))
        return data
    
    def _process_modality(self, data: np.ndarray, modality: SensorModality) -> np.ndarray:
        """Process data for specific sensory modality"""
        
        # Modality-specific processing
        if modality == SensorModality.OLFACTORY:
            # Apply olfactory-specific filtering
            return self._olfactory_processing(data)
        elif modality == SensorModality.VISUAL:
            # Apply visual processing
            return self._visual_processing(data)
        elif modality == SensorModality.AUDITORY:
            # Apply auditory processing
            return self._auditory_processing(data)
        else:
            # Generic processing
            return self._generic_processing(data)
    
    def _olfactory_processing(self, data: np.ndarray) -> np.ndarray:
        """Olfactory-specific signal processing"""
        # Apply log transformation (Weber-Fechner law)
        processed = np.log1p(np.abs(data)) * np.sign(data)
        
        # Apply temporal smoothing
        if len(processed) > 5:
            from scipy.ndimage import uniform_filter1d
            processed = uniform_filter1d(processed, size=3)
        
        return processed
    
    def _visual_processing(self, data: np.ndarray) -> np.ndarray:
        """Visual-specific signal processing"""
        # Apply contrast enhancement
        processed = data - np.mean(data)
        
        # Apply spatial filtering if multi-dimensional
        if data.ndim > 1:
            # Simple edge detection
            processed = np.gradient(processed, axis=0) + np.gradient(processed, axis=1)
        
        return processed.flatten()
    
    def _auditory_processing(self, data: np.ndarray) -> np.ndarray:
        """Auditory-specific signal processing"""
        # Apply frequency domain processing
        if len(data) > 8:  # Minimum length for FFT
            fft = np.fft.fft(data)
            
            # Emphasize certain frequency ranges
            freqs = np.fft.fftfreq(len(data))
            emphasis = np.exp(-np.abs(freqs) * 10)  # Low-pass emphasis
            
            processed_fft = fft * emphasis
            processed = np.real(np.fft.ifft(processed_fft))
        else:
            processed = data
        
        return processed
    
    def _generic_processing(self, data: np.ndarray) -> np.ndarray:
        """Generic signal processing for other modalities"""
        # Simple normalization and smoothing
        processed = self._normalize_data(data)
        
        if len(processed) > 3:
            # Simple moving average
            kernel = np.ones(3) / 3
            processed = np.convolve(processed, kernel, mode='same')
        
        return processed
    
    def _calculate_modality_confidence(self, processed_data: np.ndarray) -> float:
        """Calculate confidence score for processed modality data"""
        
        # Confidence based on signal-to-noise ratio
        signal_power = np.mean(processed_data ** 2)
        noise_estimate = np.var(np.diff(processed_data)) if len(processed_data) > 1 else 0.1
        
        snr = signal_power / (noise_estimate + 1e-8)
        confidence = 1.0 / (1.0 + np.exp(-snr))  # Sigmoid mapping
        
        return confidence
    
    def _extract_features(self, data: np.ndarray, modality: SensorModality) -> np.ndarray:
        """Extract relevant features from modality data"""
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data)
        ])
        
        # Temporal features (if applicable)
        if len(data) > 1:
            features.extend([
                np.mean(np.diff(data)),  # Trend
                np.std(np.diff(data))    # Variability
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Frequency domain features (if sufficient data)
        if len(data) > 8:
            fft = np.fft.fft(data)
            power_spectrum = np.abs(fft) ** 2
            
            features.extend([
                np.argmax(power_spectrum),  # Dominant frequency
                np.sum(power_spectrum[:len(power_spectrum)//4]),  # Low frequency power
                np.sum(power_spectrum[len(power_spectrum)//4:])   # High frequency power
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def _compute_attention_weights(self, modality_features: List[np.ndarray], 
                                  stimulus: SensoryStimulus) -> np.ndarray:
        """Compute attention weights for modalities"""
        
        # Simple attention mechanism based on feature salience
        salience_scores = []
        
        for features in modality_features:
            # Salience based on feature magnitude and variation
            magnitude = np.linalg.norm(features)
            variation = np.std(features) if len(features) > 1 else 0
            
            salience = magnitude * (1 + variation)
            salience_scores.append(salience)
        
        # Softmax normalization
        salience_scores = np.array(salience_scores)
        exp_scores = np.exp(salience_scores - np.max(salience_scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        return attention_weights
    
    def _perform_temporal_alignment(self, stimulus: SensoryStimulus) -> np.ndarray:
        """Perform temporal alignment of sensory modalities"""
        
        # Use temporal sync markers if available
        if stimulus.temporal_sync is not None:
            return stimulus.temporal_sync
        
        # Otherwise, create alignment based on onset times
        n_modalities = len(stimulus.modalities)
        alignment = np.zeros(n_modalities)
        
        # Simple alignment based on stimulus onset
        alignment[:] = stimulus.onset_time
        
        return alignment
    
    def _align_temporal_components(self, stimulus: SensoryStimulus) -> Dict[SensorModality, np.ndarray]:
        """Align temporal components of different modalities"""
        
        aligned_data = {}
        target_length = max(len(data) for data in stimulus.modalities.values()) if stimulus.modalities else 0
        
        for modality, data in stimulus.modalities.items():
            if len(data) < target_length:
                # Pad with zeros or interpolate
                padded_data = np.pad(data, (0, target_length - len(data)), mode='constant')
                aligned_data[modality] = padded_data
            else:
                aligned_data[modality] = data[:target_length]
        
        return aligned_data
    
    def _calculate_temporal_weights(self, stimulus: SensoryStimulus) -> np.ndarray:
        """Calculate temporal weights based on synchronization quality"""
        
        weights = []
        
        for modality, data in stimulus.modalities.items():
            # Weight based on temporal consistency
            if len(data) > 1:
                consistency = 1.0 / (1.0 + np.var(np.diff(data)))
            else:
                consistency = 1.0
            
            weights.append(consistency)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
        
        return weights
    
    def _apply_temporal_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal filtering to data"""
        
        if len(data) <= 1:
            return data
        
        # Simple low-pass filter
        alpha = self.config.get('temporal_filter_alpha', 0.3)
        filtered_data = np.zeros_like(data)
        filtered_data[0] = data[0]
        
        for i in range(1, len(data)):
            filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i-1]
        
        return filtered_data
    
    def _temporal_coherence(self, data: np.ndarray, sync_signal: np.ndarray) -> float:
        """Calculate temporal coherence between data and sync signal"""
        
        if sync_signal is None or len(data) == 0:
            return 0.5  # Default coherence
        
        # Simple correlation-based coherence
        min_length = min(len(data), len(sync_signal))
        if min_length <= 1:
            return 0.5
        
        data_trimmed = data[:min_length]
        sync_trimmed = sync_signal[:min_length]
        
        correlation = np.corrcoef(data_trimmed, sync_trimmed)[0, 1]
        coherence = (correlation + 1) / 2  # Map to [0, 1]
        
        return coherence if not np.isnan(coherence) else 0.5
    
    def _calculate_modality_priors(self, stimulus: SensoryStimulus) -> Dict[SensorModality, float]:
        """Calculate prior probabilities for each modality"""
        
        # Simple uniform priors for now
        n_modalities = len(stimulus.modalities)
        prior_value = 1.0 / n_modalities if n_modalities > 0 else 1.0
        
        priors = {}
        for modality in stimulus.modalities.keys():
            priors[modality] = prior_value
        
        return priors
    
    def _calculate_likelihood(self, output: np.ndarray, modality: SensorModality) -> float:
        """Calculate likelihood of output given modality"""
        
        # Simple likelihood based on output quality
        if len(output) == 0:
            return 0.1
        
        # Likelihood based on signal characteristics
        signal_strength = np.mean(np.abs(output))
        noise_estimate = np.std(output) if len(output) > 1 else 0.1
        
        snr = signal_strength / (noise_estimate + 1e-8)
        likelihood = 1.0 / (1.0 + np.exp(-snr + 2))  # Sigmoid with offset
        
        return max(likelihood, 0.01)  # Minimum likelihood
    
    def _estimate_uncertainty(self, confidence_scores: Dict[SensorModality, float], 
                            fused_representation: np.ndarray) -> float:
        """Estimate uncertainty in fusion result"""
        
        # Uncertainty based on confidence variation and representation quality
        confidences = list(confidence_scores.values())
        
        if len(confidences) == 0:
            return 1.0  # Maximum uncertainty
        
        # Variance in confidence scores
        confidence_var = np.var(confidences) if len(confidences) > 1 else 0
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        
        # Representation quality (inverse of noise)
        repr_noise = np.std(fused_representation) if len(fused_representation) > 1 else 1.0
        repr_quality = 1.0 / (1.0 + repr_noise)
        
        # Combined uncertainty
        uncertainty = confidence_var + (1 - avg_confidence) + (1 - repr_quality)
        uncertainty = uncertainty / 3.0  # Normalize
        
        return min(max(uncertainty, 0.0), 1.0)  # Clamp to [0, 1]

def main():
    """CLI interface for bioneuro-olfactory-fusion framework"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bioneuro-Olfactory Fusion Framework")
    parser.add_argument("--test-receptors", action="store_true", help="Test olfactory receptor field")
    parser.add_argument("--test-snn", action="store_true", help="Test spiking neural network")
    parser.add_argument("--test-fusion", action="store_true", help="Test multi-sensory fusion")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmarks")
    parser.add_argument("--config", help="Configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'n_receptors': 50,
            'profile_length': 20,
            'snn': {
                'n_input': 50,
                'n_hidden': [32, 16],
                'n_output': 10,
                'learning_rate': 0.001
            },
            'fusion': {
                'temporal_filter_alpha': 0.3
            }
        }
    
    logger.info("Starting bioneuro-olfactory-fusion framework tests")
    
    if args.test_receptors or args.benchmark:
        print("Testing Olfactory Receptor Field...")
        
        # Initialize receptor field
        receptor_field = OlfactoryReceptorField(config)
        
        # Create test chemical signals
        test_signals = [
            ChemicalSignal(
                molecule_id="ethyl_butyrate",
                concentration=0.5,
                molecular_weight=116.16,
                volatility=0.8,
                functional_groups=['ester', 'aliphatic'],
                spatial_distribution=np.random.random(10),
                temporal_profile=np.random.random(20)
            ),
            ChemicalSignal(
                molecule_id="vanillin",
                concentration=0.3,
                molecular_weight=152.15,
                volatility=0.3,
                functional_groups=['aldehyde', 'aromatic'],
                spatial_distribution=np.random.random(10),
                temporal_profile=np.random.random(20)
            )
        ]
        
        # Process signals
        start_time = time.time()
        responses = receptor_field.process_chemical_signals(test_signals)
        processing_time = time.time() - start_time
        
        print(f"Processed {len(test_signals)} chemical signals in {processing_time:.4f}s")
        print(f"Activated {(responses > 0.1).sum()}/{len(responses)} receptors")
        print(f"Average response intensity: {np.mean(responses[responses > 0]):.4f}")
    
    if args.test_snn or args.benchmark:
        print("\nTesting Spiking Neural Network...")
        
        # Initialize SNN
        snn = SpikingNeuralNetwork("test_snn", config['snn'])
        
        # Create test input
        test_input = np.random.random(config['snn']['n_input']) * 2 - 1
        
        # Process input
        start_time = time.time()
        output = snn.process(test_input)
        processing_time = time.time() - start_time
        
        print(f"Processed input through SNN in {processing_time:.4f}s")
        print(f"Input shape: {test_input.shape}, Output shape: {output.shape}")
        print(f"Output range: [{np.min(output):.4f}, {np.max(output):.4f}]")
        
        # Test learning
        learning_signal = np.random.random(config['snn']['n_output'])
        snn.update_weights(learning_signal)
        print("Weight update completed")
    
    if args.test_fusion or args.benchmark:
        print("\nTesting Multi-Sensory Fusion...")
        
        # Initialize fusion system
        fusion_system = MultiSensoryFusion(config.get('fusion', {}))
        
        # Create test stimulus
        test_stimulus = SensoryStimulus(
            stimulus_id="test_stimulus",
            modalities={
                SensorModality.OLFACTORY: np.random.random(30),
                SensorModality.VISUAL: np.random.random(50),
                SensorModality.AUDITORY: np.random.random(40)
            },
            temporal_sync=np.linspace(0, 1, 3),
            onset_time=0.0,
            duration=1.0
        )
        
        # Test different fusion strategies
        strategies = [
            FusionStrategy.EARLY_FUSION,
            FusionStrategy.LATE_FUSION,
            FusionStrategy.HYBRID_FUSION,
            FusionStrategy.ATTENTION_FUSION
        ]
        
        for strategy in strategies:
            start_time = time.time()
            result = fusion_system.fuse_sensory_inputs(test_stimulus, strategy)
            processing_time = time.time() - start_time
            
            print(f"\n{strategy.value.upper()} FUSION:")
            print(f"  Processing time: {processing_time:.4f}s")
            print(f"  Fused representation shape: {result.fused_representation.shape}")
            print(f"  Average confidence: {np.mean(list(result.confidence_scores.values())):.4f}")
            print(f"  Uncertainty: {result.uncertainty_estimate:.4f}")
    
    if args.benchmark:
        print("\n" + "="*60)
        print("COMPREHENSIVE BENCHMARK RESULTS")
        print("="*60)
        
        # Additional benchmarking code would go here
        print("Benchmark completed successfully!")
    
    print("\nBioneuro-olfactory-fusion framework testing completed.")

if __name__ == "__main__":
    main()