#!/usr/bin/env python3
"""
Integration tests for SDLC system components
Tests the interaction between backlog manager, WSJF engine, and project generator
"""

import pytest
import yaml
import json
import tempfile
import shutil
import os
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from backlog_manager import BacklogManager, BacklogItem
from wsjf_engine import WSJFEngine, WSJFWeights
from project_generator import ProjectGenerator

class TestSDLCIntegration:
    """Test integration between SDLC components"""
    
    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for integration tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_backlog_data(self):
        """Create sample backlog data for testing"""
        return {
            'items': [
                {
                    'id': 'FEAT-001',
                    'title': 'User Authentication System',
                    'type': 'feature',
                    'description': 'Implement JWT-based authentication',
                    'acceptance_criteria': [
                        'Users can register with email/password',
                        'Users can login and receive JWT token',
                        'Protected endpoints validate tokens'
                    ],
                    'effort': 8,
                    'value': 13,
                    'time_criticality': 8,
                    'risk_reduction': 5,
                    'wsjf_score': 3.25,
                    'status': 'READY',
                    'risk_tier': 'medium',
                    'created_at': '2025-01-01T00:00:00Z',
                    'links': []
                },
                {
                    'id': 'INFRA-001',
                    'title': 'CI/CD Pipeline Setup',
                    'type': 'infrastructure',
                    'description': 'Set up automated testing and deployment',
                    'acceptance_criteria': [
                        'GitHub Actions workflow created',
                        'Automated testing on PRs',
                        'Deployment to staging environment'
                    ],
                    'effort': 5,
                    'value': 8,
                    'time_criticality': 13,
                    'risk_reduction': 8,
                    'wsjf_score': 5.8,
                    'status': 'DOING',
                    'risk_tier': 'high',
                    'created_at': '2025-01-01T00:00:00Z',
                    'links': []
                },
                {
                    'id': 'BUG-001',
                    'title': 'Fix Memory Leak in API',
                    'type': 'bug',
                    'description': 'Memory usage grows over time in production',
                    'acceptance_criteria': [
                        'Memory leak identified',
                        'Fix implemented and tested',
                        'Memory usage stable in production'
                    ],
                    'effort': 3,
                    'value': 5,
                    'time_criticality': 13,
                    'risk_reduction': 13,
                    'wsjf_score': 10.33,
                    'status': 'READY',
                    'risk_tier': 'critical',
                    'created_at': '2025-01-01T00:00:00Z',
                    'links': []
                }
            ],
            'metrics': {
                'last_updated': '2025-01-01T00:00:00Z',
                'total_items': 3,
                'by_status': {
                    'NEW': 0, 'REFINED': 0, 'READY': 2, 'DOING': 1,
                    'PR': 0, 'DONE': 0, 'BLOCKED': 0
                },
                'avg_wsjf_score': 6.46,
                'high_risk_items': 1
            }
        }
    
    def test_backlog_manager_wsjf_engine_integration(self, temp_directory, sample_backlog_data):
        """Test integration between backlog manager and WSJF engine"""
        
        # Create backlog file
        backlog_file = Path(temp_directory) / "backlog.yml"
        with open(backlog_file, 'w') as f:
            yaml.dump(sample_backlog_data, f)
        
        # Initialize components
        backlog_manager = BacklogManager(str(backlog_file))
        wsjf_engine = WSJFEngine()
        
        # Verify backlog loaded correctly
        assert len(backlog_manager.items) == 3
        
        # Convert backlog items to format expected by WSJF engine
        items_for_analysis = []
        for item in backlog_manager.items:
            items_for_analysis.append({
                'id': item.id,
                'value': item.value,
                'time_criticality': item.time_criticality,
                'risk_reduction': item.risk_reduction,
                'effort': item.effort,
                'type': item.type,
                'context': {},
                'features': {}
            })
        
        # Run comprehensive analytics
        analytics = wsjf_engine.calculate_comprehensive_analytics(items_for_analysis)
        
        # Verify analytics results
        assert len(analytics) == 3
        
        # Should be sorted by WSJF score (BUG-001 should be first due to highest WSJF)
        assert analytics[0].item_id == 'BUG-001'
        assert analytics[0].wsjf_scores['classic'] == 10.33
        
        # Verify all analytics have required fields
        for item_analytics in analytics:
            assert item_analytics.percentile_ranking >= 0
            assert item_analytics.risk_assessment in ['low', 'medium', 'high', 'critical']
            assert item_analytics.recommended_sprint > 0
    
    def test_backlog_manager_project_generator_integration(self, temp_directory):
        """Test integration between backlog manager and project generator"""
        
        # Initialize components
        backlog_manager = BacklogManager()
        project_generator = ProjectGenerator(temp_directory)
        
        # Add items to backlog that could inform project generation
        backlog_manager.add_item(
            title="API Development",
            description="Build REST API with authentication",
            item_type="feature",
            acceptance_criteria=["API endpoints created", "Authentication implemented"],
            effort=8,
            value=13,
            time_criticality=8,
            risk_reduction=5
        )
        
        backlog_manager.add_item(
            title="Frontend Dashboard",
            description="Create React dashboard",
            item_type="feature",
            acceptance_criteria=["Dashboard UI created", "API integration"],
            effort=5,
            value=8,
            time_criticality=5,
            risk_reduction=3
        )
        
        # Generate project based on backlog insights
        # In a real scenario, this would analyze backlog to determine best template
        project_path = project_generator.generate_project(
            template_name="python-api",
            project_name="test_project"
        )
        
        # Verify project structure was created
        assert project_path.exists()
        assert (project_path / "src").exists()
        assert (project_path / "tests").exists()
        assert (project_path / "README.md").exists()
        assert (project_path / "requirements.txt").exists()
        
        # Verify backlog items are preserved
        assert len(backlog_manager.items) == 2
        
        # Could save backlog to new project
        project_backlog_file = project_path / "backlog.yml"
        backlog_manager.backlog_file = project_backlog_file
        backlog_manager.save_backlog()
        
        assert project_backlog_file.exists()
    
    def test_full_sdlc_workflow(self, temp_directory, sample_backlog_data):
        """Test complete SDLC workflow integration"""
        
        # Step 1: Set up backlog
        backlog_file = Path(temp_directory) / "backlog.yml"
        with open(backlog_file, 'w') as f:
            yaml.dump(sample_backlog_data, f)
        
        backlog_manager = BacklogManager(str(backlog_file))
        
        # Step 2: Analyze backlog with WSJF engine
        wsjf_engine = WSJFEngine()
        
        items_for_analysis = []
        for item in backlog_manager.items:
            items_for_analysis.append({
                'id': item.id,
                'value': item.value,
                'time_criticality': item.time_criticality,
                'risk_reduction': item.risk_reduction,
                'effort': item.effort,
                'type': item.type,
                'context': {},
                'features': {}
            })
        
        analytics = wsjf_engine.calculate_comprehensive_analytics(items_for_analysis)
        insights = wsjf_engine.generate_insights(analytics)
        
        # Step 3: Optimize portfolio selection
        constraints = {
            'max_effort': 10,  # Sprint capacity
            'max_items': 2,    # Sprint limit
            'min_value': 5     # Minimum value threshold
        }
        
        portfolio = wsjf_engine.optimize_portfolio(items_for_analysis, constraints)
        
        # Verify portfolio optimization
        assert len(portfolio['selected_items']) <= 2
        assert portfolio['portfolio_metrics']['total_effort'] <= 10
        
        # Step 4: Update backlog based on optimization results
        selected_item_ids = [item['item']['id'] for item in portfolio['selected_items']]
        
        for item in backlog_manager.items:
            if item.id in selected_item_ids and item.status == 'READY':
                backlog_manager.update_item_status(item.id, 'DOING')
        
        # Step 5: Generate project if needed (simulate project creation decision)
        project_types_in_selection = set([item['item']['type'] for item in portfolio['selected_items']])
        
        if 'infrastructure' in project_types_in_selection:
            project_generator = ProjectGenerator(temp_directory)
            project_path = project_generator.generate_project(
                template_name="python-api",
                project_name="infrastructure_project"
            )
            
            assert project_path.exists()
        
        # Step 6: Export comprehensive report
        report_file = Path(temp_directory) / "sdlc_report.json"
        report_content = wsjf_engine.export_scoring_report(
            analytics, insights, str(report_file)
        )
        
        assert report_file.exists()
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        assert 'generated_at' in report_data
        assert len(report_data['items']) == 3
        assert 'insights' in report_data
        
        # Step 7: Save updated backlog
        backlog_manager.save_backlog()
        
        # Verify workflow completed successfully
        final_metrics = backlog_manager.export_metrics()
        assert final_metrics['summary']['total_items'] == 3
    
    def test_todo_discovery_integration(self, temp_directory):
        """Test TODO discovery integration across components"""
        
        # Step 1: Create project with TODO items
        project_generator = ProjectGenerator(temp_directory)
        project_path = project_generator.generate_project(
            template_name="python-api",
            project_name="todo_test_project"
        )
        
        # Step 2: Add files with TODO comments
        src_file = project_path / "src" / "sample.py"
        src_file.write_text('''
def process_data():
    # TODO: Add input validation
    # FIXME: Handle edge cases
    # BUG: Memory leak in this function
    pass

def calculate_metrics():
    # TODO: Optimize performance
    return {}
''')
        
        test_file = project_path / "tests" / "test_sample.py"
        test_file.write_text('''
def test_process_data():
    # TODO: Add more test cases
    assert True
''')
        
        # Step 3: Initialize backlog manager and discover TODOs
        backlog_manager = BacklogManager()
        discovered_todos = backlog_manager.discover_todo_items(str(project_path / "src"))
        
        # Should find TODO/FIXME/BUG items
        assert len(discovered_todos) >= 4
        
        # Step 4: Convert discovered TODOs to backlog items
        for todo in discovered_todos[:2]:  # Add first 2 as examples
            backlog_manager.add_item(
                title=todo['suggested_title'],
                description=f"Address {todo['pattern']} comment: {todo['content']}",
                item_type=todo['suggested_type'],
                acceptance_criteria=[f"Review and address comment in {todo['file']}:{todo['line']}"],
                effort=2,  # Default effort for TODO items
                value=3,   # Default value
                time_criticality=2,
                risk_reduction=1
            )
        
        # Step 5: Analyze TODO-based backlog with WSJF
        wsjf_engine = WSJFEngine()
        
        items_for_analysis = []
        for item in backlog_manager.items:
            items_for_analysis.append({
                'id': item.id,
                'value': item.value,
                'time_criticality': item.time_criticality,
                'risk_reduction': item.risk_reduction,
                'effort': item.effort,
                'type': item.type,
                'context': {},
                'features': {}
            })
        
        analytics = wsjf_engine.calculate_comprehensive_analytics(items_for_analysis)
        
        # Verify TODO-based analytics
        assert len(analytics) >= 2
        for item_analytics in analytics:
            assert item_analytics.wsjf_scores['classic'] > 0
        
        # Step 6: Save backlog to project
        project_backlog = project_path / "discovered_backlog.yml"
        backlog_manager.backlog_file = project_backlog
        backlog_manager.save_backlog()
        
        assert project_backlog.exists()
    
    def test_portfolio_optimization_with_constraints(self, temp_directory, sample_backlog_data):
        """Test portfolio optimization with various constraint scenarios"""
        
        # Initialize components
        backlog_file = Path(temp_directory) / "backlog.yml"
        with open(backlog_file, 'w') as f:
            yaml.dump(sample_backlog_data, f)
        
        backlog_manager = BacklogManager(str(backlog_file))
        wsjf_engine = WSJFEngine()
        
        items_for_analysis = []
        for item in backlog_manager.items:
            items_for_analysis.append({
                'id': item.id,
                'value': item.value,
                'time_criticality': item.time_criticality,
                'risk_reduction': item.risk_reduction,
                'effort': item.effort,
                'type': item.type
            })
        
        # Test scenario 1: Tight effort constraint
        tight_constraints = {'max_effort': 5}
        tight_portfolio = wsjf_engine.optimize_portfolio(items_for_analysis, tight_constraints)
        
        assert tight_portfolio['portfolio_metrics']['total_effort'] <= 5
        assert len(tight_portfolio['selected_items']) <= 2  # Can't fit all items
        
        # Test scenario 2: Type-specific requirements
        bug_only_constraints = {'required_types': ['bug']}
        bug_portfolio = wsjf_engine.optimize_portfolio(items_for_analysis, bug_only_constraints)
        
        assert len(bug_portfolio['selected_items']) == 1
        assert bug_portfolio['selected_items'][0]['item']['type'] == 'bug'
        
        # Test scenario 3: Minimum value threshold
        high_value_constraints = {'min_value': 8}
        high_value_portfolio = wsjf_engine.optimize_portfolio(items_for_analysis, high_value_constraints)
        
        for selected_item in high_value_portfolio['selected_items']:
            assert selected_item['item']['value'] >= 8
        
        # Test scenario 4: Combined constraints
        combined_constraints = {
            'max_effort': 12,
            'max_items': 2,
            'min_value': 5
        }
        combined_portfolio = wsjf_engine.optimize_portfolio(items_for_analysis, combined_constraints)
        
        assert len(combined_portfolio['selected_items']) <= 2
        assert combined_portfolio['portfolio_metrics']['total_effort'] <= 12
        for selected_item in combined_portfolio['selected_items']:
            assert selected_item['item']['value'] >= 5
    
    def test_scoring_calibration_integration(self, temp_directory):
        """Test scoring calibration with historical data"""
        
        # Create backlog with completed items
        backlog_manager = BacklogManager()
        wsjf_engine = WSJFEngine()
        
        # Add items that represent "completed" work
        completed_items = [
            {
                'effort': 5, 'actual_effort': 6,
                'value': 8, 'actual_value': 7,
                'completion_time': 7
            },
            {
                'effort': 3, 'actual_effort': 4,
                'value': 13, 'actual_value': 12,
                'completion_time': 5
            },
            {
                'effort': 8, 'actual_effort': 10,
                'value': 5, 'actual_value': 6,
                'completion_time': 12
            }
        ]
        
        # Calibrate scoring based on completed items
        calibration_results = wsjf_engine.calibrate_scoring(completed_items)
        
        assert 'accuracy_metrics' in calibration_results
        assert 'effort_mae' in calibration_results['accuracy_metrics']
        assert 'value_mae' in calibration_results['accuracy_metrics']
        
        # If errors are high, should have recommendations
        if calibration_results['accuracy_metrics']['effort_mae'] > 0.3:
            assert len(calibration_results['recommendations']) > 0
        
        # Use calibration insights to adjust future scoring
        # This would typically involve updating WSJFWeights based on calibration
        if calibration_results['accuracy_metrics']['effort_mae'] > 0.2:
            # Increase effort penalty to account for underestimation
            adjusted_weights = WSJFWeights(effort_penalty=1.2)
            calibrated_engine = WSJFEngine(adjusted_weights)
            
            # Test that adjusted engine produces different scores
            original_score = wsjf_engine.calculate_weighted_wsjf(8, 5, 2, 3)
            adjusted_score = calibrated_engine.calculate_weighted_wsjf(8, 5, 2, 3)
            
            assert original_score != adjusted_score
    
    def test_error_handling_integration(self, temp_directory):
        """Test error handling across integrated components"""
        
        # Test handling of corrupted backlog file
        corrupted_backlog = Path(temp_directory) / "corrupted.yml"
        corrupted_backlog.write_text("invalid: yaml: content: [broken")
        
        # Should handle corrupted file gracefully
        backlog_manager = BacklogManager(str(corrupted_backlog))
        assert len(backlog_manager.items) == 0  # Should start empty
        
        # Test handling of invalid WSJF inputs
        wsjf_engine = WSJFEngine()
        
        # Should handle zero effort
        score = wsjf_engine.calculate_classic_wsjf(8, 5, 2, 0)
        assert score > 0  # Should use minimum effort
        
        # Test handling of invalid project template
        project_generator = ProjectGenerator(temp_directory)
        
        with pytest.raises(ValueError, match="Unknown template"):
            project_generator.generate_project("nonexistent-template", "test_project")
        
        # Test handling of existing project directory
        existing_project = Path(temp_directory) / "existing_project"
        existing_project.mkdir()
        
        with pytest.raises(FileExistsError, match="Project directory already exists"):
            project_generator.generate_project("python-api", "existing_project")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html"])