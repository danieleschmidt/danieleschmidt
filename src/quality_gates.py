#!/usr/bin/env python3
"""
Quality Gate Automation Framework
Automated quality checks and gates for SDLC processes
"""

import json
import subprocess
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Import our logging framework
import sys
sys.path.insert(0, str(Path(__file__).parent))
from logging_framework import get_logger, EventType

logger = get_logger('quality_gates')

class QualityGateResult(Enum):
    """Quality gate results"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

class GateType(Enum):
    """Types of quality gates"""
    CODE_QUALITY = "code_quality"
    SECURITY = "security" 
    TESTING = "testing"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    COMPLIANCE = "compliance"
    DEPENDENCIES = "dependencies"

@dataclass
class QualityMetric:
    """Individual quality metric measurement"""
    name: str
    value: Union[float, int, str, bool]
    threshold: Optional[Union[float, int]] = None
    comparison_operator: str = ">="  # >=, >, <=, <, ==, !=
    unit: Optional[str] = None
    description: Optional[str] = None

@dataclass
class GateResult:
    """Result of a quality gate check"""
    gate_name: str
    gate_type: GateType
    result: QualityGateResult
    score: Optional[float] = None
    message: Optional[str] = None
    metrics: List[QualityMetric] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    artifacts: List[str] = field(default_factory=list)

@dataclass
class QualityGateConfig:
    """Configuration for a quality gate"""
    name: str
    gate_type: GateType
    enabled: bool = True
    required: bool = True
    timeout: int = 300  # seconds
    retry_count: int = 0
    retry_delay: int = 5  # seconds
    thresholds: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

class QualityGate(ABC):
    """Abstract base class for quality gates"""
    
    def __init__(self, config: QualityGateConfig, project_root: str):
        self.config = config
        self.project_root = Path(project_root)
        self.logger = get_logger(f'quality_gates.{config.name}')
    
    @abstractmethod
    def execute(self) -> GateResult:
        """Execute the quality gate check"""
        pass
    
    def _evaluate_metric(self, metric: QualityMetric) -> bool:
        """Evaluate if a metric passes its threshold"""
        if metric.threshold is None:
            return True
        
        try:
            value = float(metric.value)
            threshold = float(metric.threshold)
            
            if metric.comparison_operator == ">=":
                return value >= threshold
            elif metric.comparison_operator == ">":
                return value > threshold
            elif metric.comparison_operator == "<=":
                return value <= threshold
            elif metric.comparison_operator == "<":
                return value < threshold
            elif metric.comparison_operator == "==":
                return value == threshold
            elif metric.comparison_operator == "!=":
                return value != threshold
            else:
                self.logger.warning(f"Unknown comparison operator: {metric.comparison_operator}")
                return True
                
        except (ValueError, TypeError):
            # For non-numeric values, only support == and !=
            if metric.comparison_operator == "==":
                return metric.value == metric.threshold
            elif metric.comparison_operator == "!=":
                return metric.value != metric.threshold
            else:
                self.logger.warning(f"Cannot compare non-numeric values with operator: {metric.comparison_operator}")
                return True
    
    def _run_command(self, command: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
        """Run shell command and return results"""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            return -1, "", "Command timed out"
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            return -1, "", str(e)

class CodeQualityGate(QualityGate):
    """Code quality gate using linting and static analysis"""
    
    def execute(self) -> GateResult:
        start_time = datetime.now()
        metrics = []
        details = {}
        artifacts = []
        
        try:
            # Python code quality checks
            python_files = list(self.project_root.rglob("*.py"))
            if python_files:
                metrics.extend(self._check_python_quality())
            
            # JavaScript/TypeScript code quality checks  
            js_files = list(self.project_root.rglob("*.js")) + list(self.project_root.rglob("*.ts"))
            if js_files:
                metrics.extend(self._check_javascript_quality())
            
            # Calculate overall score
            passing_metrics = sum(1 for m in metrics if self._evaluate_metric(m))
            total_metrics = len(metrics)
            score = (passing_metrics / total_metrics * 100) if total_metrics > 0 else 100
            
            # Determine result
            min_score = self.config.thresholds.get('min_score', 80)
            result = QualityGateResult.PASS if score >= min_score else QualityGateResult.FAIL
            
            message = f"Code quality score: {score:.1f}% ({passing_metrics}/{total_metrics} checks passed)"
            
        except Exception as e:
            self.logger.error(f"Code quality gate failed: {e}")
            result = QualityGateResult.FAIL
            score = 0.0
            message = f"Code quality check failed: {e}"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return GateResult(
            gate_name=self.config.name,
            gate_type=self.config.gate_type,
            result=result,
            score=score,
            message=message,
            metrics=metrics,
            details=details,
            execution_time=execution_time,
            artifacts=artifacts
        )
    
    def _check_python_quality(self) -> List[QualityMetric]:
        """Check Python code quality"""
        metrics = []
        
        # Flake8 linting
        returncode, stdout, stderr = self._run_command(['flake8', 'src/', 'tests/', '--count', '--statistics'])
        if returncode == 0:
            # Extract metrics from flake8 output
            lines = stdout.strip().split('\n')
            error_count = 0
            for line in lines:
                if line.strip() and line[0].isdigit():
                    error_count += int(line.split()[0])
            
            metrics.append(QualityMetric(
                name="flake8_errors",
                value=error_count,
                threshold=self.config.thresholds.get('max_flake8_errors', 0),
                comparison_operator="<=",
                description="Number of flake8 linting errors"
            ))
        
        # Black formatting check
        returncode, stdout, stderr = self._run_command(['black', '--check', 'src/', 'tests/'])
        metrics.append(QualityMetric(
            name="black_formatting",
            value=returncode == 0,
            threshold=True,
            comparison_operator="==",
            description="Code properly formatted with Black"
        ))
        
        # isort import sorting check
        returncode, stdout, stderr = self._run_command(['isort', '--check-only', 'src/', 'tests/'])
        metrics.append(QualityMetric(
            name="isort_imports",
            value=returncode == 0,
            threshold=True,
            comparison_operator="==", 
            description="Imports properly sorted with isort"
        ))
        
        # MyPy type checking
        returncode, stdout, stderr = self._run_command(['mypy', 'src/', '--ignore-missing-imports'])
        mypy_errors = len([line for line in stderr.split('\n') if 'error:' in line])
        metrics.append(QualityMetric(
            name="mypy_errors",
            value=mypy_errors,
            threshold=self.config.thresholds.get('max_mypy_errors', 0),
            comparison_operator="<=",
            description="Number of MyPy type errors"
        ))
        
        return metrics
    
    def _check_javascript_quality(self) -> List[QualityMetric]:
        """Check JavaScript/TypeScript code quality"""
        metrics = []
        
        # ESLint checking
        returncode, stdout, stderr = self._run_command(['npx', 'eslint', 'src/', '--format', 'json'])
        if returncode in [0, 1]:  # ESLint returns 1 when issues found but command succeeds
            try:
                eslint_results = json.loads(stdout) if stdout else []
                error_count = sum(len(file_result.get('messages', [])) for file_result in eslint_results)
                
                metrics.append(QualityMetric(
                    name="eslint_errors",
                    value=error_count,
                    threshold=self.config.thresholds.get('max_eslint_errors', 0),
                    comparison_operator="<=",
                    description="Number of ESLint errors"
                ))
            except json.JSONDecodeError:
                pass
        
        # Prettier formatting check
        returncode, stdout, stderr = self._run_command(['npx', 'prettier', '--check', 'src/'])
        metrics.append(QualityMetric(
            name="prettier_formatting", 
            value=returncode == 0,
            threshold=True,
            comparison_operator="==",
            description="Code properly formatted with Prettier"
        ))
        
        # TypeScript compilation check
        if (self.project_root / 'tsconfig.json').exists():
            returncode, stdout, stderr = self._run_command(['npx', 'tsc', '--noEmit'])
            metrics.append(QualityMetric(
                name="typescript_compilation",
                value=returncode == 0,
                threshold=True,
                comparison_operator="==",
                description="TypeScript code compiles without errors"
            ))
        
        return metrics

class SecurityGate(QualityGate):
    """Security quality gate"""
    
    def execute(self) -> GateResult:
        start_time = datetime.now()
        metrics = []
        details = {}
        
        try:
            # Import our security scanner
            from security_scanner import SecurityScanner, ScanType
            
            scanner = SecurityScanner(str(self.project_root))
            
            # Run security scans
            scan_types = [ScanType.STATIC_ANALYSIS, ScanType.SECRET_SCAN, ScanType.DEPENDENCY_SCAN]
            results = scanner.run_comprehensive_scan(scan_types)
            
            # Extract metrics
            total_issues = 0
            critical_issues = 0
            high_issues = 0
            
            for scan_result in results.values():
                total_issues += len(scan_result.issues)
                for issue in scan_result.issues:
                    if issue.severity.value == 'critical':
                        critical_issues += 1
                    elif issue.severity.value == 'high':
                        high_issues += 1
            
            metrics.extend([
                QualityMetric(
                    name="total_security_issues",
                    value=total_issues,
                    threshold=self.config.thresholds.get('max_total_issues', 10),
                    comparison_operator="<=",
                    description="Total number of security issues"
                ),
                QualityMetric(
                    name="critical_security_issues",
                    value=critical_issues,
                    threshold=self.config.thresholds.get('max_critical_issues', 0),
                    comparison_operator="<=",
                    description="Number of critical security issues"
                ),
                QualityMetric(
                    name="high_security_issues", 
                    value=high_issues,
                    threshold=self.config.thresholds.get('max_high_issues', 2),
                    comparison_operator="<=",
                    description="Number of high severity security issues"
                )
            ])
            
            # Get security score
            security_score = scanner.get_security_score()
            score = security_score['overall_score']
            
            min_score = self.config.thresholds.get('min_security_score', 70)
            result = QualityGateResult.PASS if score >= min_score else QualityGateResult.FAIL
            
            message = f"Security score: {score:.1f}/100, Issues: {total_issues} ({critical_issues} critical, {high_issues} high)"
            details = security_score['details']
            
        except Exception as e:
            self.logger.error(f"Security gate failed: {e}")
            result = QualityGateResult.FAIL
            score = 0.0
            message = f"Security check failed: {e}"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return GateResult(
            gate_name=self.config.name,
            gate_type=self.config.gate_type,
            result=result,
            score=score,
            message=message,
            metrics=metrics,
            details=details,
            execution_time=execution_time
        )

class TestingGate(QualityGate):
    """Testing quality gate"""
    
    def execute(self) -> GateResult:
        start_time = datetime.now()
        metrics = []
        details = {}
        artifacts = []
        
        try:
            # Run Python tests with coverage
            if (self.project_root / 'tests').exists():
                test_results = self._run_python_tests()
                metrics.extend(test_results['metrics'])
                details.update(test_results['details'])
                artifacts.extend(test_results['artifacts'])
            
            # Run JavaScript/Node.js tests
            if (self.project_root / 'package.json').exists():
                js_test_results = self._run_javascript_tests()
                metrics.extend(js_test_results['metrics'])
                details.update(js_test_results['details'])
            
            # Calculate overall score
            passing_metrics = sum(1 for m in metrics if self._evaluate_metric(m))
            total_metrics = len(metrics)
            score = (passing_metrics / total_metrics * 100) if total_metrics > 0 else 100
            
            min_score = self.config.thresholds.get('min_score', 80)
            result = QualityGateResult.PASS if score >= min_score else QualityGateResult.FAIL
            
            message = f"Testing score: {score:.1f}% ({passing_metrics}/{total_metrics} checks passed)"
            
        except Exception as e:
            self.logger.error(f"Testing gate failed: {e}")
            result = QualityGateResult.FAIL
            score = 0.0
            message = f"Testing check failed: {e}"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return GateResult(
            gate_name=self.config.name,
            gate_type=self.config.gate_type,
            result=result,
            score=score,
            message=message,
            metrics=metrics,
            details=details,
            execution_time=execution_time,
            artifacts=artifacts
        )
    
    def _run_python_tests(self) -> Dict[str, Any]:
        """Run Python tests with pytest and coverage"""
        metrics = []
        details = {}
        artifacts = []
        
        # Run pytest with coverage
        returncode, stdout, stderr = self._run_command([
            'pytest', 'tests/', '-v', '--cov=src', '--cov-report=json', '--cov-report=html',
            '--junit-xml=test-results.xml'
        ])
        
        # Parse test results
        if returncode in [0, 1]:  # 0 = success, 1 = tests failed but pytest ran
            # Count passed/failed tests from stdout
            lines = stdout.split('\n')
            passed_tests = len([line for line in lines if '::' in line and 'PASSED' in line])
            failed_tests = len([line for line in lines if '::' in line and 'FAILED' in line])
            total_tests = passed_tests + failed_tests
            
            metrics.extend([
                QualityMetric(
                    name="test_pass_rate",
                    value=passed_tests / total_tests * 100 if total_tests > 0 else 100,
                    threshold=self.config.thresholds.get('min_pass_rate', 95),
                    comparison_operator=">=",
                    unit="%",
                    description="Percentage of tests that passed"
                ),
                QualityMetric(
                    name="failed_tests",
                    value=failed_tests,
                    threshold=self.config.thresholds.get('max_failed_tests', 0),
                    comparison_operator="<=",
                    description="Number of failed tests"
                )
            ])
            
            details.update({
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests
            })
        
        # Parse coverage results
        coverage_json = self.project_root / 'coverage.json'
        if coverage_json.exists():
            try:
                with open(coverage_json, 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data['totals']['percent_covered']
                
                metrics.append(QualityMetric(
                    name="code_coverage",
                    value=total_coverage,
                    threshold=self.config.thresholds.get('min_coverage', 80),
                    comparison_operator=">=",
                    unit="%",
                    description="Code coverage percentage"
                ))
                
                details['coverage'] = coverage_data['totals']
                artifacts.append('htmlcov/index.html')  # Coverage HTML report
                
            except Exception as e:
                self.logger.warning(f"Failed to parse coverage results: {e}")
        
        if (self.project_root / 'test-results.xml').exists():
            artifacts.append('test-results.xml')
        
        return {
            'metrics': metrics,
            'details': details,
            'artifacts': artifacts
        }
    
    def _run_javascript_tests(self) -> Dict[str, Any]:
        """Run JavaScript/Node.js tests"""
        metrics = []
        details = {}
        
        # Run npm test
        returncode, stdout, stderr = self._run_command(['npm', 'test'])
        
        metrics.append(QualityMetric(
            name="javascript_tests_pass",
            value=returncode == 0,
            threshold=True,
            comparison_operator="==",
            description="JavaScript tests pass"
        ))
        
        return {
            'metrics': metrics,
            'details': {'npm_test_exit_code': returncode}
        }

class PerformanceGate(QualityGate):
    """Performance quality gate"""
    
    def execute(self) -> GateResult:
        start_time = datetime.now()
        metrics = []
        details = {}
        
        try:
            # Run performance tests if available
            if (self.project_root / 'tests' / 'performance').exists():
                perf_results = self._run_performance_tests()
                metrics.extend(perf_results['metrics'])
                details.update(perf_results['details'])
            
            # Check bundle sizes for web projects
            if (self.project_root / 'package.json').exists():
                bundle_metrics = self._check_bundle_sizes()
                metrics.extend(bundle_metrics)
            
            # Calculate score
            passing_metrics = sum(1 for m in metrics if self._evaluate_metric(m))
            total_metrics = len(metrics)
            score = (passing_metrics / total_metrics * 100) if total_metrics > 0 else 100
            
            min_score = self.config.thresholds.get('min_score', 80)
            result = QualityGateResult.PASS if score >= min_score else QualityGateResult.FAIL
            
            message = f"Performance score: {score:.1f}% ({passing_metrics}/{total_metrics} checks passed)"
            
        except Exception as e:
            self.logger.error(f"Performance gate failed: {e}")
            result = QualityGateResult.FAIL
            score = 0.0
            message = f"Performance check failed: {e}"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return GateResult(
            gate_name=self.config.name,
            gate_type=self.config.gate_type,
            result=result,
            score=score,
            message=message,
            metrics=metrics,
            details=details,
            execution_time=execution_time
        )
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        metrics = []
        details = {}
        
        # Run pytest benchmarks if available
        returncode, stdout, stderr = self._run_command([
            'pytest', 'tests/performance/', '--benchmark-json=benchmark-results.json'
        ])
        
        # Parse benchmark results
        benchmark_file = self.project_root / 'benchmark-results.json'
        if benchmark_file.exists():
            try:
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                for benchmark in benchmark_data.get('benchmarks', []):
                    name = benchmark['name']
                    mean_time = benchmark['stats']['mean']
                    
                    metrics.append(QualityMetric(
                        name=f"benchmark_{name}_mean_time",
                        value=mean_time,
                        threshold=self.config.thresholds.get(f'max_{name}_time', 1.0),
                        comparison_operator="<=",
                        unit="seconds",
                        description=f"Mean execution time for {name}"
                    ))
                
                details['benchmarks'] = benchmark_data
                
            except Exception as e:
                self.logger.warning(f"Failed to parse benchmark results: {e}")
        
        return {
            'metrics': metrics,
            'details': details
        }
    
    def _check_bundle_sizes(self) -> List[QualityMetric]:
        """Check JavaScript bundle sizes"""
        metrics = []
        
        # Check if build directory exists
        build_dirs = ['dist', 'build', 'public']
        for build_dir in build_dirs:
            build_path = self.project_root / build_dir
            if build_path.exists():
                # Find JavaScript files
                js_files = list(build_path.rglob('*.js'))
                for js_file in js_files:
                    size_bytes = js_file.stat().st_size
                    size_kb = size_bytes / 1024
                    
                    metrics.append(QualityMetric(
                        name=f"bundle_size_{js_file.name}",
                        value=size_kb,
                        threshold=self.config.thresholds.get('max_bundle_size_kb', 500),
                        comparison_operator="<=",
                        unit="KB",
                        description=f"Size of {js_file.name} bundle"
                    ))
                break
        
        return metrics

class DocumentationGate(QualityGate):
    """Documentation quality gate"""
    
    def execute(self) -> GateResult:
        start_time = datetime.now()
        metrics = []
        details = {}
        
        try:
            # Check for required documentation files
            required_files = ['README.md', 'CONTRIBUTING.md', 'LICENSE']
            missing_files = []
            
            for file_name in required_files:
                if not (self.project_root / file_name).exists():
                    missing_files.append(file_name)
            
            metrics.append(QualityMetric(
                name="required_docs_present",
                value=len(missing_files) == 0,
                threshold=True,
                comparison_operator="==",
                description="All required documentation files present"
            ))
            
            # Check README.md quality
            readme_score = self._check_readme_quality()
            metrics.append(QualityMetric(
                name="readme_quality_score",
                value=readme_score,
                threshold=self.config.thresholds.get('min_readme_score', 60),
                comparison_operator=">=",
                unit="%",
                description="README.md quality score"
            ))
            
            # Check API documentation coverage
            api_doc_score = self._check_api_documentation()
            if api_doc_score is not None:
                metrics.append(QualityMetric(
                    name="api_documentation_coverage",
                    value=api_doc_score,
                    threshold=self.config.thresholds.get('min_api_doc_coverage', 70),
                    comparison_operator=">=",
                    unit="%",
                    description="API documentation coverage"
                ))
            
            details.update({
                'missing_files': missing_files,
                'required_files': required_files
            })
            
            # Calculate score
            passing_metrics = sum(1 for m in metrics if self._evaluate_metric(m))
            total_metrics = len(metrics)
            score = (passing_metrics / total_metrics * 100) if total_metrics > 0 else 100
            
            min_score = self.config.thresholds.get('min_score', 80)
            result = QualityGateResult.PASS if score >= min_score else QualityGateResult.FAIL
            
            message = f"Documentation score: {score:.1f}% ({passing_metrics}/{total_metrics} checks passed)"
            
        except Exception as e:
            self.logger.error(f"Documentation gate failed: {e}")
            result = QualityGateResult.FAIL
            score = 0.0
            message = f"Documentation check failed: {e}"
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return GateResult(
            gate_name=self.config.name,
            gate_type=self.config.gate_type,
            result=result,
            score=score,
            message=message,
            metrics=metrics,
            details=details,
            execution_time=execution_time
        )
    
    def _check_readme_quality(self) -> float:
        """Check README.md quality"""
        readme_file = self.project_root / 'README.md'
        if not readme_file.exists():
            return 0.0
        
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            score = 0
            max_score = 100
            
            # Check for common sections
            sections = {
                'title': 20,
                'description': 15,
                'installation': 15,
                'usage': 15,
                'contributing': 10,
                'license': 10,
                'examples': 15
            }
            
            for section, points in sections.items():
                if section.lower() in content.lower():
                    score += points
            
            # Check length (should be substantial but not too long)
            length = len(content)
            if 500 <= length <= 10000:
                score += 0  # Already counted in sections
            elif length < 500:
                score *= 0.8  # Penalize short READMEs
            
            return min(score, max_score)
            
        except Exception as e:
            self.logger.warning(f"Failed to check README quality: {e}")
            return 0.0
    
    def _check_api_documentation(self) -> Optional[float]:
        """Check API documentation coverage"""
        # This is a simplified check - in practice, you'd use tools like pydoc-markdown,
        # jsdoc, or similar to analyze actual documentation coverage
        
        # Check for common API doc files/patterns
        api_doc_patterns = [
            'docs/api/**/*.md',
            'docs/**/*.md',
            '**/*.rst',
            'openapi.yaml',
            'swagger.yaml'
        ]
        
        doc_files_found = 0
        for pattern in api_doc_patterns:
            files = list(self.project_root.glob(pattern))
            doc_files_found += len(files)
        
        if doc_files_found == 0:
            return None
        
        # Simple heuristic: assume some level of documentation exists
        # In practice, this would analyze actual docstring coverage, etc.
        return min(doc_files_found * 25, 100)

class QualityGateRunner:
    """Orchestrates execution of quality gates"""
    
    def __init__(self, project_root: str, config_file: Optional[str] = None):
        self.project_root = Path(project_root)
        self.logger = get_logger('quality_gates.runner')
        self.config = self._load_config(config_file)
        self.gates = self._initialize_gates()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load quality gates configuration"""
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        return json.load(f)
                    else:
                        return yaml.safe_load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_file}: {e}")
        
        # Default configuration
        return {
            'gates': [
                {
                    'name': 'code_quality',
                    'gate_type': 'code_quality',
                    'enabled': True,
                    'required': True,
                    'thresholds': {
                        'min_score': 80,
                        'max_flake8_errors': 0,
                        'max_eslint_errors': 0
                    }
                },
                {
                    'name': 'security',
                    'gate_type': 'security',
                    'enabled': True,
                    'required': True,
                    'thresholds': {
                        'min_security_score': 70,
                        'max_critical_issues': 0,
                        'max_high_issues': 2
                    }
                },
                {
                    'name': 'testing',
                    'gate_type': 'testing',
                    'enabled': True,
                    'required': True,
                    'thresholds': {
                        'min_pass_rate': 95,
                        'min_coverage': 80,
                        'max_failed_tests': 0
                    }
                },
                {
                    'name': 'documentation',
                    'gate_type': 'documentation',
                    'enabled': True,
                    'required': False,
                    'thresholds': {
                        'min_score': 60
                    }
                }
            ]
        }
    
    def _initialize_gates(self) -> List[QualityGate]:
        """Initialize quality gate instances"""
        gates = []
        
        gate_classes = {
            'code_quality': CodeQualityGate,
            'security': SecurityGate,
            'testing': TestingGate,
            'performance': PerformanceGate,
            'documentation': DocumentationGate
        }
        
        for gate_config in self.config.get('gates', []):
            if not gate_config.get('enabled', True):
                continue
            
            gate_type = gate_config['gate_type']
            if gate_type not in gate_classes:
                self.logger.warning(f"Unknown gate type: {gate_type}")
                continue
            
            config = QualityGateConfig(
                name=gate_config['name'],
                gate_type=GateType(gate_type),
                enabled=gate_config.get('enabled', True),
                required=gate_config.get('required', True),
                timeout=gate_config.get('timeout', 300),
                retry_count=gate_config.get('retry_count', 0),
                retry_delay=gate_config.get('retry_delay', 5),
                thresholds=gate_config.get('thresholds', {}),
                parameters=gate_config.get('parameters', {}),
                dependencies=gate_config.get('dependencies', [])
            )
            
            gate_class = gate_classes[gate_type]
            gate = gate_class(config, str(self.project_root))
            gates.append(gate)
        
        return gates
    
    def run_all_gates(self, fail_fast: bool = False) -> List[GateResult]:
        """Run all enabled quality gates"""
        
        self.logger.info(f"Running {len(self.gates)} quality gates")
        results = []
        
        for gate in self.gates:
            try:
                self.logger.info(f"Running gate: {gate.config.name}")
                
                # Execute with retries if configured
                result = self._execute_with_retry(gate)
                results.append(result)
                
                self.logger.info(f"Gate {gate.config.name}: {result.result.value} (score: {result.score})")
                
                # Fail fast if required gate fails
                if (fail_fast and gate.config.required and 
                    result.result == QualityGateResult.FAIL):
                    self.logger.error(f"Required gate {gate.config.name} failed, stopping execution")
                    break
                    
            except Exception as e:
                self.logger.error(f"Gate {gate.config.name} execution failed: {e}")
                
                error_result = GateResult(
                    gate_name=gate.config.name,
                    gate_type=gate.config.gate_type,
                    result=QualityGateResult.FAIL,
                    message=f"Execution error: {e}"
                )
                results.append(error_result)
        
        self._log_summary(results)
        return results
    
    def _execute_with_retry(self, gate: QualityGate) -> GateResult:
        """Execute gate with retry logic"""
        
        last_result = None
        
        for attempt in range(gate.config.retry_count + 1):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying gate {gate.config.name} (attempt {attempt + 1})")
                    import time
                    time.sleep(gate.config.retry_delay)
                
                result = gate.execute()
                
                # If gate passes or this is the last attempt, return result
                if result.result == QualityGateResult.PASS or attempt == gate.config.retry_count:
                    return result
                
                last_result = result
                
            except Exception as e:
                if attempt == gate.config.retry_count:
                    raise
                self.logger.warning(f"Gate {gate.config.name} attempt {attempt + 1} failed: {e}")
        
        return last_result or GateResult(
            gate_name=gate.config.name,
            gate_type=gate.config.gate_type,
            result=QualityGateResult.FAIL,
            message="All retry attempts failed"
        )
    
    def _log_summary(self, results: List[GateResult]) -> None:
        """Log execution summary"""
        
        total_gates = len(results)
        passed_gates = sum(1 for r in results if r.result == QualityGateResult.PASS)
        failed_gates = sum(1 for r in results if r.result == QualityGateResult.FAIL)
        warning_gates = sum(1 for r in results if r.result == QualityGateResult.WARNING)
        
        self.logger.info(f"Quality Gates Summary:")
        self.logger.info(f"  Total: {total_gates}")
        self.logger.info(f"  Passed: {passed_gates}")
        self.logger.info(f"  Failed: {failed_gates}")
        self.logger.info(f"  Warnings: {warning_gates}")
        
        overall_success = all(
            r.result in [QualityGateResult.PASS, QualityGateResult.WARNING] 
            for r in results
        )
        
        if overall_success:
            self.logger.info("✅ All quality gates passed!")
        else:
            self.logger.error("❌ Some quality gates failed!")
    
    def generate_report(self, results: List[GateResult], output_file: Optional[str] = None) -> str:
        """Generate quality gates report"""
        
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'project_path': str(self.project_root),
            'summary': {
                'total_gates': len(results),
                'passed': sum(1 for r in results if r.result == QualityGateResult.PASS),
                'failed': sum(1 for r in results if r.result == QualityGateResult.FAIL),
                'warnings': sum(1 for r in results if r.result == QualityGateResult.WARNING),
                'overall_success': all(r.result != QualityGateResult.FAIL for r in results)
            },
            'gates': [
                {
                    'name': result.gate_name,
                    'type': result.gate_type.value,
                    'result': result.result.value,
                    'score': result.score,
                    'message': result.message,
                    'execution_time': result.execution_time,
                    'metrics': [
                        {
                            'name': m.name,
                            'value': m.value,
                            'threshold': m.threshold,
                            'comparison_operator': m.comparison_operator,
                            'unit': m.unit,
                            'description': m.description,
                            'passed': gate._evaluate_metric(m) if hasattr(gate, '_evaluate_metric') else None
                        }
                        for m in result.metrics
                    ] if hasattr(result, 'metrics') else [],
                    'details': result.details,
                    'artifacts': result.artifacts
                }
                for result in results
                for gate in self.gates if gate.config.name == result.gate_name
            ]
        }
        
        report_json = json.dumps(report, indent=2, default=str)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_json)
            self.logger.info(f"Quality gates report written to {output_file}")
        
        return report_json

def main():
    """CLI interface for quality gates"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Gates Runner")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-report", help="Output report file path")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first required gate failure")
    parser.add_argument("--gates", nargs="*", help="Specific gates to run")
    
    args = parser.parse_args()
    
    runner = QualityGateRunner(args.project_root, args.config)
    
    # Filter gates if specified
    if args.gates:
        runner.gates = [gate for gate in runner.gates if gate.config.name in args.gates]
    
    # Run gates
    results = runner.run_all_gates(fail_fast=args.fail_fast)
    
    # Generate report
    report = runner.generate_report(results, args.output_report)
    
    if not args.output_report:
        print(report)
    
    # Exit with appropriate code
    failed_required_gates = [
        r for r in results 
        if r.result == QualityGateResult.FAIL and
        any(gate.config.required for gate in runner.gates if gate.config.name == r.gate_name)
    ]
    
    if failed_required_gates:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()