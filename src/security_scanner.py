#!/usr/bin/env python3
"""
Security Scanner and Validation Framework
Comprehensive security analysis for SDLC processes and code
"""

import os
import subprocess
import json
import yaml
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    """Security issue severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ScanType(Enum):
    """Types of security scans"""
    STATIC_ANALYSIS = "static_analysis"
    DEPENDENCY_SCAN = "dependency_scan"
    SECRET_SCAN = "secret_scan"
    CONTAINER_SCAN = "container_scan"
    COMPLIANCE_SCAN = "compliance_scan"
    LICENSE_SCAN = "license_scan"

@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning"""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    scan_type: ScanType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    cwe_id: Optional[str] = None
    cve_id: Optional[str] = None
    remediation: Optional[str] = None
    confidence: float = 1.0
    evidence: Dict = field(default_factory=dict)
    first_seen: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ScanResult:
    """Results of a security scan"""
    scan_type: ScanType
    tool_name: str
    scan_time: str
    issues: List[SecurityIssue]
    summary: Dict[str, int]
    metadata: Dict = field(default_factory=dict)

class SecurityScanner:
    """Comprehensive security scanner for SDLC processes"""
    
    def __init__(self, project_root: str = "."):
        """Initialize security scanner"""
        self.project_root = Path(project_root).resolve()
        self.scan_results: List[ScanResult] = []
        self.patterns = self._load_security_patterns()
        
    def run_comprehensive_scan(self, scan_types: Optional[List[ScanType]] = None) -> Dict[str, ScanResult]:
        """Run comprehensive security scan"""
        
        if scan_types is None:
            scan_types = list(ScanType)
        
        results = {}
        
        logger.info(f"Starting comprehensive security scan on {self.project_root}")
        
        for scan_type in scan_types:
            try:
                logger.info(f"Running {scan_type.value} scan")
                
                if scan_type == ScanType.STATIC_ANALYSIS:
                    result = self._run_static_analysis_scan()
                elif scan_type == ScanType.DEPENDENCY_SCAN:
                    result = self._run_dependency_scan()
                elif scan_type == ScanType.SECRET_SCAN:
                    result = self._run_secret_scan()
                elif scan_type == ScanType.CONTAINER_SCAN:
                    result = self._run_container_scan()
                elif scan_type == ScanType.COMPLIANCE_SCAN:
                    result = self._run_compliance_scan()
                elif scan_type == ScanType.LICENSE_SCAN:
                    result = self._run_license_scan()
                else:
                    continue
                
                results[scan_type.value] = result
                self.scan_results.append(result)
                
            except Exception as e:
                logger.error(f"Error running {scan_type.value} scan: {e}")
                continue
        
        logger.info(f"Completed security scan. Found {self._count_total_issues(results)} total issues")
        return results
    
    def _run_static_analysis_scan(self) -> ScanResult:
        """Run static code analysis for security issues"""
        
        issues = []
        
        # Scan Python files for common security issues
        python_files = list(self.project_root.rglob("*.py"))
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = self._scan_python_file(file_path, content)
                issues.extend(file_issues)
                
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        # Scan JavaScript/TypeScript files
        js_files = list(self.project_root.rglob("*.js")) + list(self.project_root.rglob("*.ts"))
        for file_path in js_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = self._scan_javascript_file(file_path, content)
                issues.extend(file_issues)
                
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        # Try to run Bandit if available
        bandit_issues = self._run_bandit_scan()
        issues.extend(bandit_issues)
        
        summary = self._create_summary(issues)
        
        return ScanResult(
            scan_type=ScanType.STATIC_ANALYSIS,
            tool_name="custom_static_analyzer",
            scan_time=datetime.now(timezone.utc).isoformat(),
            issues=issues,
            summary=summary
        )
    
    def _run_dependency_scan(self) -> ScanResult:
        """Scan for vulnerable dependencies"""
        
        issues = []
        
        # Scan Python dependencies
        requirements_files = list(self.project_root.rglob("requirements*.txt"))
        for req_file in requirements_files:
            try:
                dep_issues = self._scan_python_dependencies(req_file)
                issues.extend(dep_issues)
            except Exception as e:
                logger.warning(f"Could not scan {req_file}: {e}")
        
        # Scan package.json for Node.js dependencies
        package_json_files = list(self.project_root.rglob("package.json"))
        for package_file in package_json_files:
            try:
                dep_issues = self._scan_nodejs_dependencies(package_file)
                issues.extend(dep_issues)
            except Exception as e:
                logger.warning(f"Could not scan {package_file}: {e}")
        
        # Try to run Safety if available for Python
        safety_issues = self._run_safety_scan()
        issues.extend(safety_issues)
        
        summary = self._create_summary(issues)
        
        return ScanResult(
            scan_type=ScanType.DEPENDENCY_SCAN,
            tool_name="dependency_scanner",
            scan_time=datetime.now(timezone.utc).isoformat(),
            issues=issues,
            summary=summary
        )
    
    def _run_secret_scan(self) -> ScanResult:
        """Scan for hardcoded secrets and sensitive data"""
        
        issues = []
        
        # Common patterns for secrets
        secret_patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', 'hardcoded_password'),
            (r'api_key\s*=\s*["\']([^"\']+)["\']', 'api_key'),
            (r'secret_key\s*=\s*["\']([^"\']+)["\']', 'secret_key'),
            (r'access_token\s*=\s*["\']([^"\']+)["\']', 'access_token'),
            (r'private_key\s*=\s*["\']([^"\']+)["\']', 'private_key'),
            (r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----', 'rsa_private_key'),
            (r'sk_live_[0-9a-zA-Z]{24}', 'stripe_secret_key'),
            (r'rk_live_[0-9a-zA-Z]{24}', 'stripe_restricted_key'),
            (r'AKIA[0-9A-Z]{16}', 'aws_access_key'),
            (r'ya29\.[0-9A-Za-z\-_]+', 'google_oauth_token'),
        ]
        
        # Scan all text files
        text_files = []
        for pattern in ['*.py', '*.js', '*.ts', '*.json', '*.yml', '*.yaml', '*.env*', '*.conf', '*.config']:
            text_files.extend(self.project_root.rglob(pattern))
        
        for file_path in text_files:
            if self._should_skip_file(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, secret_type in secret_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            # Skip if it looks like a template or example
                            if any(keyword in line.lower() for keyword in ['example', 'template', 'placeholder', 'todo', 'fixme']):
                                continue
                            
                            issue = SecurityIssue(
                                id=f"secret_{hashlib.md5(f'{file_path}:{line_num}:{secret_type}'.encode()).hexdigest()[:8]}",
                                title=f"Potential hardcoded {secret_type.replace('_', ' ')}",
                                description=f"Found potential {secret_type} in source code",
                                severity=SeverityLevel.HIGH,
                                scan_type=ScanType.SECRET_SCAN,
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                remediation="Move sensitive data to environment variables or secure configuration",
                                evidence={
                                    'line_content': line.strip()[:100],  # Truncate for security
                                    'pattern_matched': secret_type
                                }
                            )
                            issues.append(issue)
                
            except Exception as e:
                logger.warning(f"Could not scan {file_path} for secrets: {e}")
        
        summary = self._create_summary(issues)
        
        return ScanResult(
            scan_type=ScanType.SECRET_SCAN,
            tool_name="secret_scanner",
            scan_time=datetime.now(timezone.utc).isoformat(),
            issues=issues,
            summary=summary
        )
    
    def _run_container_scan(self) -> ScanResult:
        """Scan Docker containers and images for security issues"""
        
        issues = []
        
        # Find Dockerfile
        dockerfiles = list(self.project_root.rglob("Dockerfile*"))
        
        for dockerfile in dockerfiles:
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Check for security issues
                    docker_issues = self._analyze_dockerfile_line(dockerfile, line_num, line)
                    issues.extend(docker_issues)
                    
            except Exception as e:
                logger.warning(f"Could not scan {dockerfile}: {e}")
        
        # Try to run container scanning tools if available
        trivy_issues = self._run_trivy_scan()
        issues.extend(trivy_issues)
        
        summary = self._create_summary(issues)
        
        return ScanResult(
            scan_type=ScanType.CONTAINER_SCAN,
            tool_name="container_scanner",
            scan_time=datetime.now(timezone.utc).isoformat(),
            issues=issues,
            summary=summary
        )
    
    def _run_compliance_scan(self) -> ScanResult:
        """Scan for compliance with security standards"""
        
        issues = []
        
        # Check for required security files
        required_files = {
            'SECURITY.md': 'Security policy documentation',
            'LICENSE': 'License file',
            '.gitignore': 'Git ignore file'
        }
        
        for filename, description in required_files.items():
            if not (self.project_root / filename).exists():
                issue = SecurityIssue(
                    id=f"compliance_missing_{filename.lower().replace('.', '_')}",
                    title=f"Missing {filename}",
                    description=f"Required file {filename} is missing ({description})",
                    severity=SeverityLevel.MEDIUM,
                    scan_type=ScanType.COMPLIANCE_SCAN,
                    remediation=f"Create {filename} file with appropriate content"
                )
                issues.append(issue)
        
        # Check for secure defaults in configuration
        config_files = ['config.yml', 'config.yaml', 'settings.py', 'config.py', '.env.example']
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                config_issues = self._scan_configuration_file(config_path)
                issues.extend(config_issues)
        
        # Check CI/CD security
        github_workflows = list((self.project_root / ".github" / "workflows").glob("*.yml")) if (self.project_root / ".github" / "workflows").exists() else []
        for workflow_file in github_workflows:
            workflow_issues = self._scan_github_workflow(workflow_file)
            issues.extend(workflow_issues)
        
        summary = self._create_summary(issues)
        
        return ScanResult(
            scan_type=ScanType.COMPLIANCE_SCAN,
            tool_name="compliance_scanner",
            scan_time=datetime.now(timezone.utc).isoformat(),
            issues=issues,
            summary=summary
        )
    
    def _run_license_scan(self) -> ScanResult:
        """Scan for license compliance issues"""
        
        issues = []
        
        # Check Python dependencies for license issues
        requirements_files = list(self.project_root.rglob("requirements*.txt"))
        for req_file in requirements_files:
            try:
                license_issues = self._scan_python_licenses(req_file)
                issues.extend(license_issues)
            except Exception as e:
                logger.warning(f"Could not scan licenses in {req_file}: {e}")
        
        # Check Node.js dependencies
        package_json_files = list(self.project_root.rglob("package.json"))
        for package_file in package_json_files:
            try:
                license_issues = self._scan_nodejs_licenses(package_file)
                issues.extend(license_issues)
            except Exception as e:
                logger.warning(f"Could not scan licenses in {package_file}: {e}")
        
        summary = self._create_summary(issues)
        
        return ScanResult(
            scan_type=ScanType.LICENSE_SCAN,
            tool_name="license_scanner",
            scan_time=datetime.now(timezone.utc).isoformat(),
            issues=issues,
            summary=summary
        )
    
    def _scan_python_file(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan Python file for security issues"""
        
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check for dangerous function usage
            dangerous_patterns = [
                (r'eval\s*\(', 'eval_usage', 'Use of eval() function'),
                (r'exec\s*\(', 'exec_usage', 'Use of exec() function'),
                (r'__import__\s*\(', 'import_usage', 'Use of __import__() function'),
                (r'pickle\.loads?\s*\(', 'pickle_usage', 'Unsafe pickle usage'),
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', 'shell_injection', 'Potential shell injection'),
                (r'os\.system\s*\(', 'os_system', 'Use of os.system()'),
                (r'input\s*\([^)]*\)', 'input_usage', 'Use of input() function'),
            ]
            
            for pattern, issue_type, description in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = SecurityIssue(
                        id=f"static_{hashlib.md5(f'{file_path}:{line_num}:{issue_type}'.encode()).hexdigest()[:8]}",
                        title=f"Potential security issue: {description}",
                        description=f"Found {description} which may pose security risks",
                        severity=SeverityLevel.HIGH if issue_type in ['eval_usage', 'exec_usage', 'shell_injection'] else SeverityLevel.MEDIUM,
                        scan_type=ScanType.STATIC_ANALYSIS,
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        cwe_id="CWE-94" if issue_type in ['eval_usage', 'exec_usage'] else "CWE-78" if issue_type == 'shell_injection' else None,
                        remediation=self._get_remediation_advice(issue_type),
                        evidence={'line_content': line.strip()}
                    )
                    issues.append(issue)
        
        return issues
    
    def _scan_javascript_file(self, file_path: Path, content: str) -> List[SecurityIssue]:
        """Scan JavaScript/TypeScript file for security issues"""
        
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check for dangerous patterns
            dangerous_patterns = [
                (r'eval\s*\(', 'eval_usage', 'Use of eval() function'),
                (r'innerHTML\s*=', 'innerHTML_usage', 'Use of innerHTML (XSS risk)'),
                (r'document\.write\s*\(', 'document_write', 'Use of document.write()'),
                (r'location\.href\s*=.*(?:window\.location|document\.URL)', 'open_redirect', 'Potential open redirect'),
                (r'setTimeout\s*\(\s*["\'][^"\']*["\']', 'settimeout_string', 'setTimeout with string argument'),
            ]
            
            for pattern, issue_type, description in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = SecurityIssue(
                        id=f"static_{hashlib.md5(f'{file_path}:{line_num}:{issue_type}'.encode()).hexdigest()[:8]}",
                        title=f"Potential security issue: {description}",
                        description=f"Found {description} which may pose security risks",
                        severity=SeverityLevel.HIGH if issue_type in ['eval_usage', 'open_redirect'] else SeverityLevel.MEDIUM,
                        scan_type=ScanType.STATIC_ANALYSIS,
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        cwe_id="CWE-79" if 'XSS' in description else "CWE-94" if issue_type == 'eval_usage' else None,
                        remediation=self._get_remediation_advice(issue_type),
                        evidence={'line_content': line.strip()}
                    )
                    issues.append(issue)
        
        return issues
    
    def _run_bandit_scan(self) -> List[SecurityIssue]:
        """Run Bandit static analysis tool if available"""
        
        issues = []
        
        try:
            result = subprocess.run(
                ['bandit', '-r', str(self.project_root), '-f', 'json'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0 or result.stdout:
                bandit_data = json.loads(result.stdout)
                
                for bandit_issue in bandit_data.get('results', []):
                    severity = SeverityLevel.LOW
                    if bandit_issue.get('issue_confidence') == 'HIGH' and bandit_issue.get('issue_severity') == 'HIGH':
                        severity = SeverityLevel.HIGH
                    elif bandit_issue.get('issue_severity') == 'MEDIUM':
                        severity = SeverityLevel.MEDIUM
                    
                    issue = SecurityIssue(
                        id=f"bandit_{bandit_issue.get('test_id', 'unknown')}_{hashlib.md5(str(bandit_issue).encode()).hexdigest()[:8]}",
                        title=f"Bandit: {bandit_issue.get('test_name', 'Security Issue')}",
                        description=bandit_issue.get('issue_text', 'Security issue detected by Bandit'),
                        severity=severity,
                        scan_type=ScanType.STATIC_ANALYSIS,
                        file_path=bandit_issue.get('filename', '').replace(str(self.project_root) + '/', ''),
                        line_number=bandit_issue.get('line_number'),
                        cwe_id=bandit_issue.get('test_id'),
                        confidence=0.8 if bandit_issue.get('issue_confidence') == 'HIGH' else 0.5,
                        evidence={
                            'code': bandit_issue.get('code', ''),
                            'severity': bandit_issue.get('issue_severity'),
                            'confidence': bandit_issue.get('issue_confidence')
                        }
                    )
                    issues.append(issue)
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            logger.debug("Bandit not available or failed to run")
        
        return issues
    
    def _run_safety_scan(self) -> List[SecurityIssue]:
        """Run Safety dependency scanner if available"""
        
        issues = []
        
        try:
            result = subprocess.run(
                ['safety', 'check', '--json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0 and result.stdout:
                safety_data = json.loads(result.stdout)
                
                for vuln in safety_data:
                    issue = SecurityIssue(
                        id=f"safety_{vuln.get('id', 'unknown')}",
                        title=f"Vulnerable dependency: {vuln.get('package_name')}",
                        description=vuln.get('advisory', 'Known security vulnerability'),
                        severity=SeverityLevel.HIGH,
                        scan_type=ScanType.DEPENDENCY_SCAN,
                        cve_id=vuln.get('cve', ''),
                        remediation=f"Update {vuln.get('package_name')} to version {vuln.get('safe_version', 'latest')}",
                        evidence={
                            'package': vuln.get('package_name'),
                            'installed_version': vuln.get('installed_version'),
                            'safe_version': vuln.get('safe_version')
                        }
                    )
                    issues.append(issue)
        
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            logger.debug("Safety not available or failed to run")
        
        return issues
    
    def _run_trivy_scan(self) -> List[SecurityIssue]:
        """Run Trivy container scanner if available"""
        
        issues = []
        
        # Look for Dockerfile
        dockerfiles = list(self.project_root.rglob("Dockerfile*"))
        
        for dockerfile in dockerfiles:
            try:
                result = subprocess.run(
                    ['trivy', 'config', '--format', 'json', str(dockerfile)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0 and result.stdout:
                    trivy_data = json.loads(result.stdout)
                    
                    for result_item in trivy_data.get('Results', []):
                        for vuln in result_item.get('Misconfigurations', []):
                            severity = SeverityLevel.MEDIUM
                            if vuln.get('Severity') == 'HIGH':
                                severity = SeverityLevel.HIGH
                            elif vuln.get('Severity') == 'CRITICAL':
                                severity = SeverityLevel.CRITICAL
                            
                            issue = SecurityIssue(
                                id=f"trivy_{vuln.get('ID', 'unknown')}",
                                title=f"Container misconfiguration: {vuln.get('Title')}",
                                description=vuln.get('Description', ''),
                                severity=severity,
                                scan_type=ScanType.CONTAINER_SCAN,
                                file_path=str(dockerfile.relative_to(self.project_root)),
                                remediation=vuln.get('Resolution', ''),
                                evidence={
                                    'rule_id': vuln.get('ID'),
                                    'severity': vuln.get('Severity')
                                }
                            )
                            issues.append(issue)
            
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
                logger.debug("Trivy not available or failed to run")
                break
        
        return issues
    
    def generate_security_report(self, output_format: str = "json", output_file: Optional[str] = None) -> str:
        """Generate comprehensive security report"""
        
        if not self.scan_results:
            logger.warning("No scan results available. Run scan first.")
            return ""
        
        # Aggregate results
        all_issues = []
        scan_summary = {}
        
        for result in self.scan_results:
            all_issues.extend(result.issues)
            scan_summary[result.scan_type.value] = result.summary
        
        # Create overall summary
        overall_summary = {
            'total_issues': len(all_issues),
            'by_severity': {severity.value: 0 for severity in SeverityLevel},
            'by_scan_type': {scan_type.value: 0 for scan_type in ScanType},
            'high_priority_issues': 0
        }
        
        for issue in all_issues:
            overall_summary['by_severity'][issue.severity.value] += 1
            overall_summary['by_scan_type'][issue.scan_type.value] += 1
            
            if issue.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                overall_summary['high_priority_issues'] += 1
        
        # Create report
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'project_path': str(self.project_root),
            'overall_summary': overall_summary,
            'scan_summary': scan_summary,
            'issues': [
                {
                    'id': issue.id,
                    'title': issue.title,
                    'description': issue.description,
                    'severity': issue.severity.value,
                    'scan_type': issue.scan_type.value,
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'cwe_id': issue.cwe_id,
                    'cve_id': issue.cve_id,
                    'remediation': issue.remediation,
                    'confidence': issue.confidence,
                    'evidence': issue.evidence,
                    'first_seen': issue.first_seen
                }
                for issue in sorted(all_issues, key=lambda x: (x.severity.value, x.scan_type.value))
            ]
        }
        
        if output_format.lower() == "json":
            report_content = json.dumps(report, indent=2)
        elif output_format.lower() == "yaml":
            report_content = yaml.dump(report, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Security report written to {output_file}")
        
        return report_content
    
    def get_security_score(self) -> Dict[str, float]:
        """Calculate security score based on findings"""
        
        if not self.scan_results:
            return {'overall_score': 0.0, 'details': {}}
        
        # Collect all issues
        all_issues = []
        for result in self.scan_results:
            all_issues.extend(result.issues)
        
        if not all_issues:
            return {'overall_score': 100.0, 'details': {'no_issues_found': True}}
        
        # Calculate penalty points based on severity
        penalty_points = 0
        severity_weights = {
            SeverityLevel.INFO: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 5,
            SeverityLevel.HIGH: 15,
            SeverityLevel.CRITICAL: 30
        }
        
        severity_counts = {severity: 0 for severity in SeverityLevel}
        
        for issue in all_issues:
            penalty_points += severity_weights[issue.severity]
            severity_counts[issue.severity] += 1
        
        # Calculate score (100 - penalty points, minimum 0)
        overall_score = max(0, 100 - penalty_points)
        
        # Create detailed breakdown
        details = {
            'penalty_points': penalty_points,
            'severity_breakdown': {severity.value: count for severity, count in severity_counts.items()},
            'total_issues': len(all_issues),
            'scan_coverage': len(self.scan_results)
        }
        
        return {
            'overall_score': round(overall_score, 1),
            'details': details
        }
    
    def _load_security_patterns(self) -> Dict:
        """Load security patterns and rules"""
        
        # Basic security patterns - in a real implementation, these would be loaded from external files
        return {
            'dangerous_functions': {
                'python': ['eval', 'exec', '__import__', 'compile'],
                'javascript': ['eval', 'setTimeout', 'setInterval']
            },
            'secret_patterns': {
                'api_key': r'api[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9\-_]{20,})',
                'password': r'password\s*[:=]\s*["\']?([^\s"\']+)',
                'token': r'token\s*[:=]\s*["\']?([a-zA-Z0-9\-_]{20,})'
            }
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during scanning"""
        
        skip_patterns = [
            '*/node_modules/*',
            '*/venv/*',
            '*/.git/*',
            '*/build/*',
            '*/dist/*',
            '*/__pycache__/*',
            '*.pyc',
            '*.min.js',
            '*.min.css'
        ]
        
        file_str = str(file_path)
        
        for pattern in skip_patterns:
            if Path(file_str).match(pattern) or pattern.replace('*/', '') in file_str:
                return True
        
        return False
    
    def _create_summary(self, issues: List[SecurityIssue]) -> Dict[str, int]:
        """Create summary of issues by severity"""
        
        summary = {severity.value: 0 for severity in SeverityLevel}
        
        for issue in issues:
            summary[issue.severity.value] += 1
        
        summary['total'] = len(issues)
        return summary
    
    def _get_remediation_advice(self, issue_type: str) -> str:
        """Get remediation advice for specific issue types"""
        
        advice = {
            'eval_usage': "Avoid using eval(). Use safer alternatives like ast.literal_eval() for Python or JSON.parse() for JavaScript.",
            'exec_usage': "Avoid using exec(). Refactor code to use safer alternatives.",
            'shell_injection': "Use parameterized commands or input validation to prevent shell injection.",
            'innerHTML_usage': "Use textContent or innerText instead of innerHTML to prevent XSS attacks.",
            'input_usage': "Validate and sanitize all user inputs.",
            'os_system': "Use subprocess module with proper argument handling instead of os.system()."
        }
        
        return advice.get(issue_type, "Review code and follow security best practices.")
    
    def _scan_python_dependencies(self, requirements_file: Path) -> List[SecurityIssue]:
        """Scan Python dependencies for known vulnerabilities"""
        # Implementation would check against vulnerability databases
        return []
    
    def _scan_nodejs_dependencies(self, package_file: Path) -> List[SecurityIssue]:
        """Scan Node.js dependencies for known vulnerabilities"""
        # Implementation would check against vulnerability databases
        return []
    
    def _scan_python_licenses(self, requirements_file: Path) -> List[SecurityIssue]:
        """Scan Python dependencies for license compliance"""
        # Implementation would check license compatibility
        return []
    
    def _scan_nodejs_licenses(self, package_file: Path) -> List[SecurityIssue]:
        """Scan Node.js dependencies for license compliance"""
        # Implementation would check license compatibility
        return []
    
    def _analyze_dockerfile_line(self, dockerfile: Path, line_num: int, line: str) -> List[SecurityIssue]:
        """Analyze Dockerfile line for security issues"""
        
        issues = []
        
        # Check for common Docker security issues
        if line.upper().startswith('USER') and 'root' in line.lower():
            issue = SecurityIssue(
                id=f"docker_root_user_{hashlib.md5(f'{dockerfile}:{line_num}'.encode()).hexdigest()[:8]}",
                title="Running as root user",
                description="Container runs as root user, which is a security risk",
                severity=SeverityLevel.HIGH,
                scan_type=ScanType.CONTAINER_SCAN,
                file_path=str(dockerfile.relative_to(self.project_root)),
                line_number=line_num,
                remediation="Create and use a non-root user for the container",
                evidence={'line_content': line.strip()}
            )
            issues.append(issue)
        
        if line.upper().startswith('ADD') and 'http' in line.lower():
            issue = SecurityIssue(
                id=f"docker_add_http_{hashlib.md5(f'{dockerfile}:{line_num}'.encode()).hexdigest()[:8]}",
                title="Using ADD with HTTP URL",
                description="ADD instruction with HTTP URL can be a security risk",
                severity=SeverityLevel.MEDIUM,
                scan_type=ScanType.CONTAINER_SCAN,
                file_path=str(dockerfile.relative_to(self.project_root)),
                line_number=line_num,
                remediation="Use COPY instead of ADD, or download files separately with proper verification",
                evidence={'line_content': line.strip()}
            )
            issues.append(issue)
        
        return issues
    
    def _scan_configuration_file(self, config_path: Path) -> List[SecurityIssue]:
        """Scan configuration file for security issues"""
        
        issues = []
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for insecure configurations
            insecure_patterns = [
                (r'debug\s*[:=]\s*true', 'debug_enabled', 'Debug mode enabled in configuration'),
                (r'ssl\s*[:=]\s*false', 'ssl_disabled', 'SSL/TLS disabled in configuration'),
                (r'verify_ssl\s*[:=]\s*false', 'ssl_verification_disabled', 'SSL verification disabled')
            ]
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                for pattern, issue_type, description in insecure_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issue = SecurityIssue(
                            id=f"config_{issue_type}_{hashlib.md5(f'{config_path}:{line_num}'.encode()).hexdigest()[:8]}",
                            title=f"Insecure configuration: {description}",
                            description=description,
                            severity=SeverityLevel.MEDIUM,
                            scan_type=ScanType.COMPLIANCE_SCAN,
                            file_path=str(config_path.relative_to(self.project_root)),
                            line_number=line_num,
                            remediation=f"Review and secure configuration setting",
                            evidence={'line_content': line.strip()}
                        )
                        issues.append(issue)
        
        except Exception as e:
            logger.warning(f"Could not scan configuration file {config_path}: {e}")
        
        return issues
    
    def _scan_github_workflow(self, workflow_file: Path) -> List[SecurityIssue]:
        """Scan GitHub Actions workflow for security issues"""
        
        issues = []
        
        try:
            with open(workflow_file, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            # Check for security issues in workflows
            if isinstance(workflow_data, dict):
                jobs = workflow_data.get('jobs', {})
                
                for job_name, job_config in jobs.items():
                    if isinstance(job_config, dict):
                        steps = job_config.get('steps', [])
                        
                        for step_idx, step in enumerate(steps):
                            if isinstance(step, dict):
                                # Check for dangerous patterns
                                run_command = step.get('run', '')
                                if 'curl' in run_command and '|' in run_command and 'sh' in run_command:
                                    issue = SecurityIssue(
                                        id=f"workflow_curl_sh_{hashlib.md5(f'{workflow_file}:{job_name}:{step_idx}'.encode()).hexdigest()[:8]}",
                                        title="Dangerous curl | sh pattern in workflow",
                                        description="Workflow downloads and executes code directly from the internet",
                                        severity=SeverityLevel.HIGH,
                                        scan_type=ScanType.COMPLIANCE_SCAN,
                                        file_path=str(workflow_file.relative_to(self.project_root)),
                                        remediation="Download files separately and verify their integrity before execution",
                                        evidence={
                                            'job': job_name,
                                            'step': step_idx,
                                            'command': run_command[:100]
                                        }
                                    )
                                    issues.append(issue)
        
        except Exception as e:
            logger.warning(f"Could not scan GitHub workflow {workflow_file}: {e}")
        
        return issues
    
    def _count_total_issues(self, results: Dict[str, ScanResult]) -> int:
        """Count total issues across all scan results"""
        
        total = 0
        for result in results.values():
            total += len(result.issues)
        return total

def main():
    """CLI interface for security scanner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SDLC Security Scanner")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--scan-types", nargs="*", 
                       choices=[st.value for st in ScanType],
                       help="Types of scans to run")
    parser.add_argument("--output-format", choices=["json", "yaml"], default="json",
                       help="Output format")
    parser.add_argument("--output-file", help="Output file path")
    parser.add_argument("--score", action="store_true", help="Calculate security score")
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.project_root)
    
    # Convert scan type strings to enums
    scan_types = None
    if args.scan_types:
        scan_types = [ScanType(st) for st in args.scan_types]
    
    # Run scans
    results = scanner.run_comprehensive_scan(scan_types)
    
    if args.score:
        score = scanner.get_security_score()
        print(f"Security Score: {score['overall_score']}/100")
        print(json.dumps(score, indent=2))
    
    if args.output_file or not args.score:
        report = scanner.generate_security_report(args.output_format, args.output_file)
        if not args.output_file:
            print(report)

if __name__ == "__main__":
    main()