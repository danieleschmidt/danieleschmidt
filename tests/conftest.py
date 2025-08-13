#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Terragon SDLC Framework
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import yaml
import json
import numpy as np

@pytest.fixture
def temp_project_dir() -> Generator[Path, None, None]:
    """Create a temporary project directory for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create basic project structure
        (temp_dir / "src").mkdir()
        (temp_dir / "tests").mkdir()
        (temp_dir / "docs").mkdir()
        
        # Create some test files
        (temp_dir / "src" / "__init__.py").touch()
        (temp_dir / "tests" / "__init__.py").touch()
        (temp_dir / "README.md").write_text("# Test Project")
        
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

@pytest.fixture
def sample_backlog_config() -> Dict[str, Any]:
    """Sample backlog configuration for testing"""
    return {
        'items': [
            {
                'id': 'TEST-001',
                'title': 'Test Feature Implementation',
                'type': 'feature',
                'description': 'Implement test feature for validation',
                'acceptance_criteria': [
                    'Feature works correctly',
                    'Tests pass',
                    'Documentation updated'
                ],
                'effort': 5,
                'value': 8,
                'time_criticality': 3,
                'risk_reduction': 2,
                'wsjf_score': 2.6,
                'status': 'READY',
                'risk_tier': 'medium',
                'created_at': '2025-08-13T10:00:00Z',
                'links': []
            },
            {
                'id': 'TEST-002',
                'title': 'Security Enhancement',
                'type': 'security',
                'description': 'Enhance security measures',
                'acceptance_criteria': [
                    'Security scan passes',
                    'Vulnerabilities fixed'
                ],
                'effort': 3,
                'value': 13,
                'time_criticality': 8,
                'risk_reduction': 5,
                'wsjf_score': 8.67,
                'status': 'NEW',
                'risk_tier': 'high',
                'created_at': '2025-08-13T11:00:00Z',
                'links': []
            }
        ],
        'metrics': {
            'last_updated': '2025-08-13T12:00:00Z',
            'total_items': 2,
            'by_status': {
                'NEW': 1,
                'REFINED': 0,
                'READY': 1,
                'DOING': 0,
                'PR': 0,
                'DONE': 0,
                'BLOCKED': 0
            },
            'avg_wsjf_score': 5.64,
            'high_risk_items': 1
        }
    }

@pytest.fixture
def sample_backlog_file(temp_project_dir: Path, sample_backlog_config: Dict[str, Any]) -> Path:
    """Create a sample backlog file for testing"""
    backlog_file = temp_project_dir / "test_backlog.yml"
    with open(backlog_file, 'w') as f:
        yaml.dump(sample_backlog_config, f, default_flow_style=False)
    return backlog_file

@pytest.fixture
def sample_wsjf_weights():
    """Sample WSJF weights for testing"""
    from src.wsjf_engine import WSJFWeights
    return WSJFWeights(
        value_weight=1.5,
        time_criticality_weight=1.2,
        risk_reduction_weight=1.0,
        effort_penalty=1.0,
        confidence_factor=0.9
    )

@pytest.fixture
def sample_chemical_signals():
    """Sample chemical signals for bioneuro testing"""
    from src.bioneuro_olfactory_fusion import ChemicalSignal
    return [
        ChemicalSignal(
            molecule_id="test_molecule_1",
            concentration=0.5,
            molecular_weight=100.0,
            volatility=0.8,
            functional_groups=['alcohol', 'aromatic'],
            spatial_distribution=np.random.random(10),
            temporal_profile=np.random.random(20)
        ),
        ChemicalSignal(
            molecule_id="test_molecule_2", 
            concentration=0.3,
            molecular_weight=150.0,
            volatility=0.6,
            functional_groups=['ester', 'aliphatic'],
            spatial_distribution=np.random.random(10),
            temporal_profile=np.random.random(20)
        )
    ]

@pytest.fixture
def sample_sensory_stimulus():
    """Sample multi-sensory stimulus for fusion testing"""
    from src.bioneuro_olfactory_fusion import SensoryStimulus, SensorModality
    return SensoryStimulus(
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

@pytest.fixture
def mock_security_config():
    """Mock security scanner configuration"""
    return {
        'static_analysis': {
            'enabled': True,
            'tools': ['bandit', 'semgrep'],
            'severity_threshold': 'medium'
        },
        'dependency_scan': {
            'enabled': True,
            'tools': ['safety', 'pip-audit'],
            'severity_threshold': 'high'
        },
        'secret_scan': {
            'enabled': True,
            'patterns': [
                r'api[_-]?key[_-]?=',
                r'secret[_-]?key[_-]?=',
                r'password[_-]?='
            ]
        }
    }

@pytest.fixture
def mock_quality_gate_config():
    """Mock quality gate configuration"""
    return {
        'code_quality': {
            'enabled': True,
            'thresholds': {
                'complexity': 10,
                'duplication': 5,
                'maintainability': 'A'
            }
        },
        'testing': {
            'enabled': True,
            'thresholds': {
                'coverage': 85,
                'pass_rate': 100
            }
        },
        'security': {
            'enabled': True,
            'thresholds': {
                'vulnerabilities': 0,
                'security_score': 70
            }
        }
    }

@pytest.fixture
def mock_monitoring_config():
    """Mock monitoring configuration"""
    return {
        'metrics': {
            'collection_interval': 30,
            'retention_days': 7,
            'export_format': 'prometheus'
        },
        'alerts': {
            'enabled': True,
            'channels': ['console', 'file']
        },
        'health_checks': {
            'interval': 60,
            'timeout': 30
        }
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment with necessary environment variables"""
    monkeypatch.setenv("TERRAGON_ENVIRONMENT", "test")
    monkeypatch.setenv("TERRAGON_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TERRAGON_DISABLE_EXTERNAL_CALLS", "true")

@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run for testing external commands"""
    import subprocess
    
    def mock_run(*args, **kwargs):
        # Return success by default
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=0,
            stdout="Mock output",
            stderr=""
        )
    
    monkeypatch.setattr(subprocess, "run", mock_run)

@pytest.fixture
def sample_test_results():
    """Sample test results for quality gate testing"""
    return {
        'summary': {
            'total': 50,
            'passed': 47,
            'failed': 2,
            'skipped': 1,
            'errors': 0
        },
        'coverage': {
            'line_coverage': 87.5,
            'branch_coverage': 83.2,
            'function_coverage': 92.1
        },
        'duration': 15.7,
        'timestamp': '2025-08-13T12:00:00Z'
    }

# Performance testing utilities
@pytest.fixture
def performance_benchmark():
    """Fixture for performance benchmarking"""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.measurements = []
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            if self.start_time is not None:
                duration = time.perf_counter() - self.start_time
                self.measurements.append(duration)
                return duration
            return 0
        
        def average_time(self):
            return sum(self.measurements) / len(self.measurements) if self.measurements else 0
        
        def max_time(self):
            return max(self.measurements) if self.measurements else 0
    
    return PerformanceBenchmark()

# Async testing utilities
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Security testing utilities
@pytest.fixture
def security_test_files(temp_project_dir: Path):
    """Create test files with various security patterns"""
    
    # File with potential security issue
    vulnerable_file = temp_project_dir / "src" / "vulnerable.py"
    vulnerable_file.write_text('''
import os
import subprocess

# Potential security issues for testing
api_key = "sk-1234567890abcdef"  # Hardcoded secret
password = "admin123"  # Hardcoded password

def unsafe_command(user_input):
    # Command injection vulnerability
    os.system(f"echo {user_input}")

def sql_query(user_id):
    # SQL injection vulnerability  
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return query
''')
    
    # Safe file
    safe_file = temp_project_dir / "src" / "safe.py"
    safe_file.write_text('''
import os
import logging

logger = logging.getLogger(__name__)

def safe_function(data):
    """This is a safe function with no security issues"""
    processed_data = data.strip().lower()
    logger.info("Processing data safely")
    return processed_data

class SecureClass:
    def __init__(self):
        self.config = os.environ.get("CONFIG", "default")
    
    def process(self, input_data):
        return f"Processed: {input_data}"
''')
    
    return {
        'vulnerable': vulnerable_file,
        'safe': safe_file
    }

# Data validation utilities
@pytest.fixture
def data_validator():
    """Utility for validating test data structures"""
    
    class DataValidator:
        @staticmethod
        def validate_backlog_item(item):
            required_fields = ['id', 'title', 'type', 'description', 'effort', 'value']
            return all(field in item for field in required_fields)
        
        @staticmethod
        def validate_wsjf_score(score):
            return isinstance(score, (int, float)) and score >= 0
        
        @staticmethod
        def validate_security_issue(issue):
            required_fields = ['id', 'title', 'severity', 'scan_type']
            return all(field in issue.__dict__ for field in required_fields)
        
        @staticmethod
        def validate_quality_result(result):
            required_fields = ['gate_name', 'result', 'timestamp']
            return all(field in result.__dict__ for field in required_fields)
    
    return DataValidator()