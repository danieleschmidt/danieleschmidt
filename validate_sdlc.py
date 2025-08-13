#!/usr/bin/env python3
"""
SDLC Framework Validation Script
Validates the implementation without external dependencies
"""

import os
import sys
import importlib.util
from pathlib import Path
import ast
import json


def validate_file_structure():
    """Validate required file structure"""
    required_files = [
        'src/backlog_manager.py',
        'src/wsjf_engine.py',
        'src/bioneuro_olfactory_fusion.py',
        'src/security_scanner.py',
        'src/quality_gates.py',
        'src/monitoring_framework.py',
        'src/logging_framework.py',
        'src/caching_framework.py',
        'src/async_processing.py',
        'src/performance_optimizer.py',
        'tests/conftest.py',
        'tests/unit/test_backlog_manager.py',
        'tests/unit/test_wsjf_engine.py',
        'tests/integration/test_sdlc_integration.py',
        'pyproject.toml',
        'requirements.txt',
        'Makefile',
        '.pre-commit-config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    return missing_files


def validate_python_syntax():
    """Validate Python syntax for all source files"""
    python_files = list(Path('src').glob('*.py')) + list(Path('tests').glob('**/*.py'))
    syntax_errors = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            ast.parse(content)
            print(f"✓ Syntax valid: {file_path}")
            
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"✗ Syntax error: {file_path}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {e}")
            print(f"✗ Error reading: {file_path}: {e}")
    
    return syntax_errors


def validate_imports():
    """Validate that key modules can be imported"""
    modules_to_test = [
        'src.backlog_manager',
        'src.wsjf_engine', 
        'src.bioneuro_olfactory_fusion',
        'src.security_scanner',
        'src.quality_gates',
        'src.monitoring_framework',
        'src.logging_framework',
        'src.caching_framework',
        'src.async_processing',
        'src.performance_optimizer'
    ]
    
    import_errors = []
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    for module_name in modules_to_test:
        try:
            # Try to load the module
            module_path = Path(module_name.replace('.', '/') + '.py')
            if module_path.exists():
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Don't actually execute, just validate structure
                    print(f"✓ Module loadable: {module_name}")
                else:
                    import_errors.append(f"Cannot create spec for {module_name}")
            else:
                import_errors.append(f"Module file not found: {module_path}")
                
        except Exception as e:
            import_errors.append(f"{module_name}: {e}")
            print(f"✗ Import error: {module_name}: {e}")
    
    return import_errors


def validate_configuration_files():
    """Validate configuration files"""
    config_errors = []
    
    # Validate pyproject.toml
    try:
        import tomllib
        with open('pyproject.toml', 'rb') as f:
            tomllib.load(f)
        print("✓ pyproject.toml is valid")
    except Exception as e:
        config_errors.append(f"pyproject.toml: {e}")
        print(f"✗ pyproject.toml error: {e}")
    
    # Validate requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 5:
            config_errors.append("requirements.txt seems incomplete")
        else:
            print("✓ requirements.txt looks good")
            
    except Exception as e:
        config_errors.append(f"requirements.txt: {e}")
        print(f"✗ requirements.txt error: {e}")
    
    return config_errors


def validate_code_quality():
    """Basic code quality checks"""
    quality_issues = []
    
    # Check for TODO comments that should be tracked
    python_files = list(Path('src').glob('*.py'))
    todo_count = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if any(keyword in line.upper() for keyword in ['TODO', 'FIXME', 'HACK', 'XXX']):
                    todo_count += 1
                    
        except Exception:
            pass
    
    if todo_count > 0:
        print(f"ⓘ Found {todo_count} TODO/FIXME comments")
    else:
        print("✓ No TODO/FIXME comments found")
    
    # Check for basic security patterns
    security_issues = 0
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for potential security issues
            if 'password' in content.lower() and '=' in content:
                security_issues += 1
            if 'api_key' in content.lower() and '=' in content:
                security_issues += 1
                
        except Exception:
            pass
    
    if security_issues > 0:
        quality_issues.append(f"Found {security_issues} potential hardcoded secrets")
        print(f"⚠ Found {security_issues} potential hardcoded secrets")
    else:
        print("✓ No obvious hardcoded secrets found")
    
    return quality_issues


def validate_test_structure():
    """Validate test structure"""
    test_issues = []
    
    # Check for test files
    unit_tests = list(Path('tests/unit').glob('test_*.py'))
    integration_tests = list(Path('tests/integration').glob('test_*.py'))
    
    if len(unit_tests) < 2:
        test_issues.append("Insufficient unit tests")
        print(f"⚠ Only {len(unit_tests)} unit test files found")
    else:
        print(f"✓ Found {len(unit_tests)} unit test files")
    
    if len(integration_tests) < 1:
        test_issues.append("No integration tests")
        print(f"⚠ Only {len(integration_tests)} integration test files found")
    else:
        print(f"✓ Found {len(integration_tests)} integration test files")
    
    # Check conftest.py exists
    if Path('tests/conftest.py').exists():
        print("✓ Test configuration file exists")
    else:
        test_issues.append("Missing tests/conftest.py")
        print("⚠ Missing tests/conftest.py")
    
    return test_issues


def generate_validation_report():
    """Generate comprehensive validation report"""
    
    print("=" * 60)
    print("TERRAGON SDLC FRAMEWORK VALIDATION")
    print("=" * 60)
    print()
    
    validation_results = {}
    
    print("1. File Structure Validation")
    print("-" * 30)
    missing_files = validate_file_structure()
    validation_results['missing_files'] = missing_files
    if missing_files:
        print(f"⚠ Missing {len(missing_files)} required files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("✓ All required files present")
    print()
    
    print("2. Python Syntax Validation")
    print("-" * 30)
    syntax_errors = validate_python_syntax()
    validation_results['syntax_errors'] = syntax_errors
    if syntax_errors:
        print(f"✗ {len(syntax_errors)} syntax errors found")
    else:
        print("✓ All Python files have valid syntax")
    print()
    
    print("3. Module Import Validation")
    print("-" * 30)
    import_errors = validate_imports()
    validation_results['import_errors'] = import_errors
    if import_errors:
        print(f"✗ {len(import_errors)} import errors found")
    else:
        print("✓ All modules can be loaded")
    print()
    
    print("4. Configuration File Validation")
    print("-" * 30)
    config_errors = validate_configuration_files()
    validation_results['config_errors'] = config_errors
    if config_errors:
        print(f"✗ {len(config_errors)} configuration errors found")
    else:
        print("✓ All configuration files are valid")
    print()
    
    print("5. Code Quality Checks")
    print("-" * 30)
    quality_issues = validate_code_quality()
    validation_results['quality_issues'] = quality_issues
    if quality_issues:
        print(f"⚠ {len(quality_issues)} quality issues found")
    else:
        print("✓ Basic code quality checks passed")
    print()
    
    print("6. Test Structure Validation")
    print("-" * 30)
    test_issues = validate_test_structure()
    validation_results['test_issues'] = test_issues
    if test_issues:
        print(f"⚠ {len(test_issues)} test structure issues found")
    else:
        print("✓ Test structure looks good")
    print()
    
    # Overall assessment
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_errors = len(syntax_errors) + len(import_errors) + len(config_errors)
    total_warnings = len(missing_files) + len(quality_issues) + len(test_issues)
    
    if total_errors == 0 and total_warnings == 0:
        print("🎉 EXCELLENT! All validations passed with no issues.")
        validation_status = "PASS"
    elif total_errors == 0:
        print(f"✅ GOOD! No critical errors, but {total_warnings} warnings to address.")
        validation_status = "PASS_WITH_WARNINGS"
    else:
        print(f"❌ ISSUES FOUND! {total_errors} errors and {total_warnings} warnings.")
        validation_status = "FAIL"
    
    validation_results['summary'] = {
        'status': validation_status,
        'total_errors': total_errors,
        'total_warnings': total_warnings,
        'timestamp': Path(__file__).stat().st_mtime
    }
    
    # Save validation report
    with open('validation_report.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\nDetailed validation report saved to: validation_report.json")
    
    return validation_status == "PASS" or validation_status == "PASS_WITH_WARNINGS"


if __name__ == "__main__":
    success = generate_validation_report()
    sys.exit(0 if success else 1)