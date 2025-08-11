#!/usr/bin/env python3
"""
Unit tests for security_scanner module
Comprehensive test coverage for security scanning functionality
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from security_scanner import (
    SecurityScanner, SecurityIssue, ScanResult, SeverityLevel, ScanType
)

class TestSecurityIssue:
    """Test SecurityIssue dataclass"""
    
    def test_security_issue_creation(self):
        """Test creating a security issue"""
        issue = SecurityIssue(
            id="TEST-001",
            title="Test Security Issue",
            description="A test security issue",
            severity=SeverityLevel.HIGH,
            scan_type=ScanType.STATIC_ANALYSIS,
            file_path="test.py",
            line_number=42,
            cwe_id="CWE-94",
            remediation="Fix the issue"
        )
        
        assert issue.id == "TEST-001"
        assert issue.severity == SeverityLevel.HIGH
        assert issue.scan_type == ScanType.STATIC_ANALYSIS
        assert issue.line_number == 42
        assert issue.confidence == 1.0  # Default value

class TestScanResult:
    """Test ScanResult dataclass"""
    
    def test_scan_result_creation(self):
        """Test creating a scan result"""
        issues = [
            SecurityIssue("ID1", "Issue 1", "Description", SeverityLevel.HIGH, ScanType.STATIC_ANALYSIS),
            SecurityIssue("ID2", "Issue 2", "Description", SeverityLevel.MEDIUM, ScanType.STATIC_ANALYSIS)
        ]
        
        summary = {"high": 1, "medium": 1, "total": 2}
        
        result = ScanResult(
            scan_type=ScanType.STATIC_ANALYSIS,
            tool_name="test_scanner",
            scan_time="2025-01-01T00:00:00Z",
            issues=issues,
            summary=summary
        )
        
        assert result.scan_type == ScanType.STATIC_ANALYSIS
        assert result.tool_name == "test_scanner"
        assert len(result.issues) == 2
        assert result.summary["total"] == 2

class TestSecurityScanner:
    """Test SecurityScanner functionality"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create sample Python file with security issues
            python_file = project_path / "test.py"
            python_file.write_text('''
def dangerous_function():
    # This is a security issue
    eval("print('hello')")
    exec("x = 1")
    os.system("ls")
    
def safe_function():
    print("This is safe")
''')
            
            # Create JavaScript file with issues
            js_file = project_path / "test.js"
            js_file.write_text('''
function dangerous() {
    eval("alert('xss')");
    document.write("<script>alert('xss')</script>");
}
''')
            
            # Create Dockerfile with issues
            dockerfile = project_path / "Dockerfile"
            dockerfile.write_text('''
FROM ubuntu:20.04
USER root
ADD http://example.com/file.tar.gz /tmp/
RUN apt-get update
''')
            
            # Create config file
            config_file = project_path / "config.yml"
            config_file.write_text('''
debug: true
ssl: false
api_key: "hardcoded-api-key-12345"
''')
            
            # Create requirements.txt
            requirements = project_path / "requirements.txt"
            requirements.write_text('''
flask==1.0.0
requests==2.25.0
''')
            
            yield project_path
    
    @pytest.fixture
    def scanner(self, temp_project):
        """Create scanner instance for testing"""
        return SecurityScanner(str(temp_project))
    
    def test_scanner_initialization(self, temp_project):
        """Test scanner initialization"""
        scanner = SecurityScanner(str(temp_project))
        
        assert scanner.project_root == temp_project
        assert isinstance(scanner.scan_results, list)
        assert len(scanner.scan_results) == 0
        assert isinstance(scanner.patterns, dict)
    
    def test_should_skip_file(self, scanner):
        """Test file skip logic"""
        
        # Should skip
        assert scanner._should_skip_file(Path("/project/node_modules/package/file.js"))
        assert scanner._should_skip_file(Path("/project/venv/lib/python.py"))
        assert scanner._should_skip_file(Path("/project/.git/config"))
        assert scanner._should_skip_file(Path("/project/build/output.js"))
        assert scanner._should_skip_file(Path("/project/file.pyc"))
        assert scanner._should_skip_file(Path("/project/script.min.js"))
        
        # Should not skip
        assert not scanner._should_skip_file(Path("/project/src/main.py"))
        assert not scanner._should_skip_file(Path("/project/test.js"))
        assert not scanner._should_skip_file(Path("/project/Dockerfile"))
    
    def test_scan_python_file_dangerous_patterns(self, scanner):
        """Test scanning Python file for dangerous patterns"""
        
        file_path = scanner.project_root / "test.py"
        content = '''
def test():
    eval("dangerous")
    exec("more dangerous")
    subprocess.call("ls", shell=True)
    os.system("rm file")
    pickle.loads(data)
'''
        
        issues = scanner._scan_python_file(file_path, content)
        
        assert len(issues) >= 4  # Should find eval, exec, shell=True, os.system
        
        # Check that issues have required fields
        for issue in issues:
            assert issue.id is not None
            assert issue.title is not None
            assert issue.severity in [SeverityLevel.HIGH, SeverityLevel.MEDIUM]
            assert issue.scan_type == ScanType.STATIC_ANALYSIS
            assert issue.file_path is not None
            assert issue.line_number is not None
    
    def test_scan_javascript_file_dangerous_patterns(self, scanner):
        """Test scanning JavaScript file for dangerous patterns"""
        
        file_path = scanner.project_root / "test.js"
        content = '''
function test() {
    eval("dangerous");
    element.innerHTML = userInput;
    document.write("<script></script>");
    setTimeout("alert('xss')", 1000);
}
'''
        
        issues = scanner._scan_javascript_file(file_path, content)
        
        assert len(issues) >= 3  # Should find eval, innerHTML, document.write, setTimeout
        
        for issue in issues:
            assert issue.severity in [SeverityLevel.HIGH, SeverityLevel.MEDIUM]
            assert issue.scan_type == ScanType.STATIC_ANALYSIS
    
    @patch('subprocess.run')
    def test_run_bandit_scan_success(self, mock_run, scanner):
        """Test successful Bandit scan"""
        
        # Mock Bandit output
        bandit_output = {
            "results": [
                {
                    "filename": str(scanner.project_root / "test.py"),
                    "test_id": "B102",
                    "test_name": "exec_used",
                    "issue_text": "Use of exec detected.",
                    "issue_severity": "MEDIUM",
                    "issue_confidence": "HIGH",
                    "line_number": 5,
                    "code": "exec('malicious code')"
                }
            ]
        }
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps(bandit_output)
        )
        
        issues = scanner._run_bandit_scan()
        
        assert len(issues) == 1
        assert issues[0].title.startswith("Bandit:")
        assert issues[0].severity == SeverityLevel.MEDIUM
        assert issues[0].cwe_id == "B102"
    
    @patch('subprocess.run')
    def test_run_bandit_scan_not_available(self, mock_run, scanner):
        """Test Bandit scan when tool not available"""
        
        mock_run.side_effect = FileNotFoundError()
        
        issues = scanner._run_bandit_scan()
        
        assert issues == []
    
    @patch('subprocess.run')
    def test_run_safety_scan_success(self, mock_run, scanner):
        """Test successful Safety scan"""
        
        # Mock Safety output (Safety returns non-zero when vulnerabilities found)
        safety_output = [
            {
                "id": "12345",
                "package_name": "flask",
                "installed_version": "1.0.0",
                "safe_version": "1.1.0",
                "advisory": "Flask before 1.1.0 has XSS vulnerability",
                "cve": "CVE-2019-1010083"
            }
        ]
        
        mock_run.return_value = MagicMock(
            returncode=1,  # Safety returns 1 when vulnerabilities found
            stdout=json.dumps(safety_output)
        )
        
        issues = scanner._run_safety_scan()
        
        assert len(issues) == 1
        assert issues[0].title.startswith("Vulnerable dependency:")
        assert issues[0].severity == SeverityLevel.HIGH
        assert issues[0].cve_id == "CVE-2019-1010083"
    
    def test_run_static_analysis_scan(self, scanner):
        """Test running complete static analysis scan"""
        
        result = scanner._run_static_analysis_scan()
        
        assert result.scan_type == ScanType.STATIC_ANALYSIS
        assert result.tool_name == "custom_static_analyzer"
        assert isinstance(result.issues, list)
        assert isinstance(result.summary, dict)
        assert "total" in result.summary
        
        # Should find issues in the test files
        assert len(result.issues) > 0
    
    def test_run_secret_scan(self, scanner):
        """Test secret scanning"""
        
        result = scanner._run_secret_scan()
        
        assert result.scan_type == ScanType.SECRET_SCAN
        assert isinstance(result.issues, list)
        
        # Should find the hardcoded API key in config.yml
        api_key_issues = [issue for issue in result.issues if "api_key" in issue.title.lower()]
        assert len(api_key_issues) > 0
        
        for issue in api_key_issues:
            assert issue.severity == SeverityLevel.HIGH
            assert issue.remediation is not None
    
    def test_run_container_scan(self, scanner):
        """Test container scanning"""
        
        result = scanner._run_container_scan()
        
        assert result.scan_type == ScanType.CONTAINER_SCAN
        assert isinstance(result.issues, list)
        
        # Should find issues in Dockerfile
        root_user_issues = [issue for issue in result.issues if "root" in issue.title.lower()]
        assert len(root_user_issues) > 0
        
        add_http_issues = [issue for issue in result.issues if "ADD" in issue.title and "HTTP" in issue.title]
        assert len(add_http_issues) > 0
    
    def test_run_compliance_scan(self, scanner):
        """Test compliance scanning"""
        
        result = scanner._run_compliance_scan()
        
        assert result.scan_type == ScanType.COMPLIANCE_SCAN
        assert isinstance(result.issues, list)
        
        # Should find missing required files
        missing_file_issues = [issue for issue in result.issues if "Missing" in issue.title]
        assert len(missing_file_issues) > 0
        
        # Should find insecure configuration
        config_issues = [issue for issue in result.issues if "debug" in issue.description.lower()]
        assert len(config_issues) > 0
    
    def test_run_comprehensive_scan(self, scanner):
        """Test running comprehensive scan"""
        
        scan_types = [ScanType.STATIC_ANALYSIS, ScanType.SECRET_SCAN, ScanType.CONTAINER_SCAN]
        results = scanner.run_comprehensive_scan(scan_types)
        
        assert len(results) == 3
        assert ScanType.STATIC_ANALYSIS.value in results
        assert ScanType.SECRET_SCAN.value in results
        assert ScanType.CONTAINER_SCAN.value in results
        
        # Check that scan results were stored
        assert len(scanner.scan_results) == 3
    
    def test_run_comprehensive_scan_all_types(self, scanner):
        """Test running comprehensive scan with all types"""
        
        results = scanner.run_comprehensive_scan()  # Should run all types
        
        assert len(results) >= 5  # Should have most scan types
        
        for scan_type in results:
            assert scan_type in [st.value for st in ScanType]
    
    def test_generate_security_report_json(self, scanner):
        """Test generating security report in JSON format"""
        
        # Run some scans first
        scanner.run_comprehensive_scan([ScanType.STATIC_ANALYSIS, ScanType.SECRET_SCAN])
        
        report_content = scanner.generate_security_report("json")
        
        # Should be valid JSON
        report_data = json.loads(report_content)
        
        assert "generated_at" in report_data
        assert "project_path" in report_data
        assert "overall_summary" in report_data
        assert "scan_summary" in report_data
        assert "issues" in report_data
        
        # Check overall summary structure
        overall_summary = report_data["overall_summary"]
        assert "total_issues" in overall_summary
        assert "by_severity" in overall_summary
        assert "by_scan_type" in overall_summary
        assert "high_priority_issues" in overall_summary
    
    def test_generate_security_report_to_file(self, scanner, tmp_path):
        """Test generating security report to file"""
        
        scanner.run_comprehensive_scan([ScanType.STATIC_ANALYSIS])
        
        output_file = tmp_path / "security_report.json"
        report_content = scanner.generate_security_report("json", str(output_file))
        
        assert output_file.exists()
        
        # File content should match returned content
        with open(output_file, 'r') as f:
            file_content = f.read()
        
        assert file_content == report_content
    
    def test_get_security_score_no_results(self, scanner):
        """Test getting security score with no scan results"""
        
        score = scanner.get_security_score()
        
        assert score["overall_score"] == 0.0
        assert "details" in score
    
    def test_get_security_score_with_issues(self, scanner):
        """Test getting security score with issues"""
        
        # Run scans to generate issues
        scanner.run_comprehensive_scan([ScanType.STATIC_ANALYSIS, ScanType.SECRET_SCAN])
        
        score = scanner.get_security_score()
        
        assert "overall_score" in score
        assert 0 <= score["overall_score"] <= 100
        assert "details" in score
        
        details = score["details"]
        assert "penalty_points" in details
        assert "severity_breakdown" in details
        assert "total_issues" in details
        assert "scan_coverage" in details
        
        # Should have some issues
        assert details["total_issues"] > 0
    
    def test_get_security_score_penalty_calculation(self, scanner):
        """Test security score penalty calculation"""
        
        # Create mock issues with known severities
        mock_issues = [
            SecurityIssue("1", "Critical", "desc", SeverityLevel.CRITICAL, ScanType.STATIC_ANALYSIS),
            SecurityIssue("2", "High", "desc", SeverityLevel.HIGH, ScanType.STATIC_ANALYSIS),
            SecurityIssue("3", "Medium", "desc", SeverityLevel.MEDIUM, ScanType.STATIC_ANALYSIS),
            SecurityIssue("4", "Low", "desc", SeverityLevel.LOW, ScanType.STATIC_ANALYSIS),
        ]
        
        # Mock scan results
        mock_result = ScanResult(
            scan_type=ScanType.STATIC_ANALYSIS,
            tool_name="test",
            scan_time="2025-01-01T00:00:00Z",
            issues=mock_issues,
            summary={}
        )
        scanner.scan_results = [mock_result]
        
        score = scanner.get_security_score()
        
        # Expected penalty: CRITICAL(30) + HIGH(15) + MEDIUM(5) + LOW(1) = 51
        expected_score = max(0, 100 - 51)  # 49
        
        assert score["overall_score"] == expected_score
        assert score["details"]["penalty_points"] == 51
    
    def test_create_summary(self, scanner):
        """Test creating issue summary"""
        
        issues = [
            SecurityIssue("1", "Issue 1", "desc", SeverityLevel.HIGH, ScanType.STATIC_ANALYSIS),
            SecurityIssue("2", "Issue 2", "desc", SeverityLevel.HIGH, ScanType.STATIC_ANALYSIS),
            SecurityIssue("3", "Issue 3", "desc", SeverityLevel.MEDIUM, ScanType.STATIC_ANALYSIS),
            SecurityIssue("4", "Issue 4", "desc", SeverityLevel.LOW, ScanType.STATIC_ANALYSIS),
        ]
        
        summary = scanner._create_summary(issues)
        
        assert summary["high"] == 2
        assert summary["medium"] == 1
        assert summary["low"] == 1
        assert summary["critical"] == 0
        assert summary["info"] == 0
        assert summary["total"] == 4
    
    def test_get_remediation_advice(self, scanner):
        """Test getting remediation advice"""
        
        advice = scanner._get_remediation_advice("eval_usage")
        assert "eval()" in advice
        assert "safer" in advice.lower()
        
        advice = scanner._get_remediation_advice("shell_injection")
        assert "injection" in advice.lower()
        
        advice = scanner._get_remediation_advice("unknown_issue")
        assert "security best practices" in advice.lower()
    
    def test_analyze_dockerfile_line_root_user(self, scanner):
        """Test analyzing Dockerfile line for root user"""
        
        dockerfile = scanner.project_root / "Dockerfile"
        line = "USER root"
        
        issues = scanner._analyze_dockerfile_line(dockerfile, 1, line)
        
        assert len(issues) == 1
        assert "root user" in issues[0].title.lower()
        assert issues[0].severity == SeverityLevel.HIGH
    
    def test_analyze_dockerfile_line_add_http(self, scanner):
        """Test analyzing Dockerfile line for ADD with HTTP"""
        
        dockerfile = scanner.project_root / "Dockerfile"
        line = "ADD http://example.com/file.tar.gz /tmp/"
        
        issues = scanner._analyze_dockerfile_line(dockerfile, 2, line)
        
        assert len(issues) == 1
        assert "ADD" in issues[0].title
        assert "HTTP" in issues[0].title
        assert issues[0].severity == SeverityLevel.MEDIUM
    
    def test_scan_configuration_file(self, scanner):
        """Test scanning configuration file"""
        
        config_path = scanner.project_root / "config.yml"
        
        issues = scanner._scan_configuration_file(config_path)
        
        assert len(issues) >= 2  # debug: true and ssl: false
        
        debug_issues = [issue for issue in issues if "debug" in issue.title.lower()]
        assert len(debug_issues) == 1
        
        ssl_issues = [issue for issue in issues if "ssl" in issue.title.lower()]
        assert len(ssl_issues) == 1
    
    def test_count_total_issues(self, scanner):
        """Test counting total issues across results"""
        
        results = {
            "static": ScanResult(
                ScanType.STATIC_ANALYSIS, "test", "2025-01-01T00:00:00Z",
                [SecurityIssue("1", "Issue", "desc", SeverityLevel.HIGH, ScanType.STATIC_ANALYSIS),
                 SecurityIssue("2", "Issue", "desc", SeverityLevel.MEDIUM, ScanType.STATIC_ANALYSIS)],
                {}
            ),
            "secrets": ScanResult(
                ScanType.SECRET_SCAN, "test", "2025-01-01T00:00:00Z",
                [SecurityIssue("3", "Issue", "desc", SeverityLevel.HIGH, ScanType.SECRET_SCAN)],
                {}
            )
        }
        
        total = scanner._count_total_issues(results)
        
        assert total == 3
    
    def test_load_security_patterns(self, scanner):
        """Test loading security patterns"""
        
        patterns = scanner._load_security_patterns()
        
        assert isinstance(patterns, dict)
        assert "dangerous_functions" in patterns
        assert "secret_patterns" in patterns
        
        assert "python" in patterns["dangerous_functions"]
        assert "javascript" in patterns["dangerous_functions"]
        
        assert isinstance(patterns["dangerous_functions"]["python"], list)
        assert "eval" in patterns["dangerous_functions"]["python"]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=security_scanner", "--cov-report=html"])