#!/usr/bin/env python3
"""
Unit tests for backlog_manager module
Comprehensive test coverage for all backlog management functionality
"""

import pytest
import yaml
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, mock_open, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from backlog_manager import (
    BacklogItem, BacklogMetrics, BacklogManager,
    main
)

class TestBacklogItem:
    """Test BacklogItem dataclass functionality"""
    
    def test_backlog_item_creation(self):
        """Test creating a backlog item with all required fields"""
        item = BacklogItem(
            id="TEST-001",
            title="Test Item",
            type="feature",
            description="A test backlog item",
            acceptance_criteria=["AC1", "AC2"],
            effort=3,
            value=8,
            time_criticality=5,
            risk_reduction=2,
            wsjf_score=5.0,
            status="NEW",
            risk_tier="medium",
            created_at="2025-01-01T00:00:00Z",
            links=[]
        )
        
        assert item.id == "TEST-001"
        assert item.title == "Test Item"
        assert item.type == "feature"
        assert len(item.acceptance_criteria) == 2
        assert item.wsjf_score == 5.0
    
    def test_cost_of_delay_calculation(self):
        """Test cost of delay calculation"""
        item = BacklogItem(
            id="TEST-001", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=3, value=8, time_criticality=5,
            risk_reduction=2, wsjf_score=0, status="NEW", risk_tier="medium",
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        assert item.cost_of_delay == 15  # 8 + 5 + 2
    
    def test_wsjf_calculation(self):
        """Test WSJF score calculation"""
        item = BacklogItem(
            id="TEST-001", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=3, value=8, time_criticality=5,
            risk_reduction=2, wsjf_score=0, status="NEW", risk_tier="medium",
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        calculated_wsjf = item.calculate_wsjf()
        assert calculated_wsjf == 5.0  # (8 + 5 + 2) / 3
    
    def test_wsjf_calculation_zero_effort(self):
        """Test WSJF calculation with zero effort"""
        item = BacklogItem(
            id="TEST-001", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=0, value=8, time_criticality=5,
            risk_reduction=2, wsjf_score=0, status="NEW", risk_tier="medium",
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        calculated_wsjf = item.calculate_wsjf()
        assert calculated_wsjf == 0.0
    
    def test_update_wsjf(self):
        """Test updating WSJF score"""
        item = BacklogItem(
            id="TEST-001", title="Test", type="feature", description="Test",
            acceptance_criteria=[], effort=3, value=8, time_criticality=5,
            risk_reduction=2, wsjf_score=0, status="NEW", risk_tier="medium",
            created_at="2025-01-01T00:00:00Z", links=[]
        )
        
        item.update_wsjf()
        assert item.wsjf_score == 5.0

class TestBacklogManager:
    """Test BacklogManager functionality"""
    
    @pytest.fixture
    def temp_backlog_file(self):
        """Create a temporary backlog file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            backlog_data = {
                'items': [
                    {
                        'id': 'TEST-001',
                        'title': 'Test Item 1',
                        'type': 'feature',
                        'description': 'Test description',
                        'acceptance_criteria': ['AC1', 'AC2'],
                        'effort': 3,
                        'value': 8,
                        'time_criticality': 5,
                        'risk_reduction': 2,
                        'wsjf_score': 5.0,
                        'status': 'NEW',
                        'risk_tier': 'medium',
                        'created_at': '2025-01-01T00:00:00Z',
                        'links': []
                    }
                ],
                'metrics': {
                    'last_updated': '2025-01-01T00:00:00Z',
                    'total_items': 1,
                    'by_status': {'NEW': 1, 'REFINED': 0, 'READY': 0, 'DOING': 0, 'PR': 0, 'DONE': 0, 'BLOCKED': 0},
                    'avg_wsjf_score': 5.0,
                    'high_risk_items': 0
                }
            }
            yaml.dump(backlog_data, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    def empty_manager(self):
        """Create an empty backlog manager for testing"""
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as f:
            manager = BacklogManager(f.name)
            yield manager
        os.unlink(f.name)
    
    def test_manager_initialization(self, temp_backlog_file):
        """Test backlog manager initialization with existing file"""
        manager = BacklogManager(temp_backlog_file)
        
        assert len(manager.items) == 1
        assert manager.items[0].id == 'TEST-001'
        assert manager.metrics is not None
        assert manager.metrics.total_items == 1
    
    def test_manager_initialization_empty(self, empty_manager):
        """Test backlog manager initialization with empty file"""
        assert len(empty_manager.items) == 0
        assert empty_manager.metrics is None
    
    def test_load_backlog(self, temp_backlog_file):
        """Test loading backlog from file"""
        manager = BacklogManager()
        manager.backlog_file = Path(temp_backlog_file)
        manager.load_backlog()
        
        assert len(manager.items) == 1
        assert manager.items[0].id == 'TEST-001'
    
    def test_load_backlog_invalid_file(self):
        """Test loading backlog from invalid file"""
        manager = BacklogManager("nonexistent.yml")
        # Should not raise exception, just log warning
        assert len(manager.items) == 0
    
    def test_save_backlog(self, empty_manager):
        """Test saving backlog to file"""
        empty_manager.add_item(
            title="Test Item",
            description="Test description",
            item_type="feature",
            acceptance_criteria=["AC1"],
            effort=3,
            value=8,
            time_criticality=5,
            risk_reduction=2
        )
        
        empty_manager.save_backlog()
        
        # Reload and verify
        empty_manager.load_backlog()
        assert len(empty_manager.items) == 1
        assert empty_manager.items[0].title == "Test Item"
    
    def test_add_item_valid(self, empty_manager):
        """Test adding a valid item to backlog"""
        item = empty_manager.add_item(
            title="New Feature",
            description="A new feature description",
            item_type="feature",
            acceptance_criteria=["User can do X", "System validates Y"],
            effort=5,
            value=13,
            time_criticality=8,
            risk_reduction=3
        )
        
        assert len(empty_manager.items) == 1
        assert item.title == "New Feature"
        assert item.id.startswith("FEAT-")
        assert item.wsjf_score == 4.8  # (13 + 8 + 3) / 5
        assert item.status == "NEW"
    
    def test_add_item_invalid_type(self, empty_manager):
        """Test adding item with invalid type"""
        with pytest.raises(ValueError, match="Invalid type"):
            empty_manager.add_item(
                title="Test",
                description="Test",
                item_type="invalid_type",
                acceptance_criteria=["AC1"],
                effort=3,
                value=8,
                time_criticality=5,
                risk_reduction=2
            )
    
    def test_add_item_invalid_scores(self, empty_manager):
        """Test adding item with invalid scores"""
        with pytest.raises(ValueError, match="All scoring values must be between 1-13"):
            empty_manager.add_item(
                title="Test",
                description="Test",
                item_type="feature",
                acceptance_criteria=["AC1"],
                effort=15,  # Invalid: > 13
                value=8,
                time_criticality=5,
                risk_reduction=2
            )
    
    def test_update_item_status_valid(self, empty_manager):
        """Test updating item status with valid status"""
        item = empty_manager.add_item(
            title="Test Item",
            description="Test",
            item_type="feature",
            acceptance_criteria=["AC1"],
            effort=3,
            value=8,
            time_criticality=5,
            risk_reduction=2
        )
        
        result = empty_manager.update_item_status(item.id, "READY")
        assert result is True
        assert item.status == "READY"
    
    def test_update_item_status_invalid(self, empty_manager):
        """Test updating item status with invalid status"""
        item = empty_manager.add_item(
            title="Test Item",
            description="Test",
            item_type="feature",
            acceptance_criteria=["AC1"],
            effort=3,
            value=8,
            time_criticality=5,
            risk_reduction=2
        )
        
        with pytest.raises(ValueError, match="Invalid status"):
            empty_manager.update_item_status(item.id, "INVALID_STATUS")
    
    def test_update_item_status_nonexistent(self, empty_manager):
        """Test updating status for nonexistent item"""
        result = empty_manager.update_item_status("NONEXISTENT-001", "READY")
        assert result is False
    
    def test_get_prioritized_items(self, empty_manager):
        """Test getting prioritized items"""
        # Add items with different WSJF scores
        empty_manager.add_item("High Priority", "Test", "feature", ["AC1"], 2, 13, 8, 5)  # WSJF = 13
        empty_manager.add_item("Medium Priority", "Test", "feature", ["AC1"], 5, 8, 5, 2)  # WSJF = 3
        empty_manager.add_item("Low Priority", "Test", "feature", ["AC1"], 8, 3, 2, 1)    # WSJF = 0.75
        
        prioritized = empty_manager.get_prioritized_items()
        
        assert len(prioritized) == 3
        assert prioritized[0].title == "High Priority"
        assert prioritized[1].title == "Medium Priority"
        assert prioritized[2].title == "Low Priority"
    
    def test_get_prioritized_items_with_filter(self, empty_manager):
        """Test getting prioritized items with status filter"""
        item1 = empty_manager.add_item("Item 1", "Test", "feature", ["AC1"], 2, 13, 8, 5)
        item2 = empty_manager.add_item("Item 2", "Test", "feature", ["AC1"], 5, 8, 5, 2)
        
        empty_manager.update_item_status(item1.id, "READY")
        # item2 stays in NEW status
        
        ready_items = empty_manager.get_prioritized_items("READY")
        
        assert len(ready_items) == 1
        assert ready_items[0].title == "Item 1"
    
    def test_get_next_work_items(self, empty_manager):
        """Test getting next work items"""
        # Add items and set some to READY status
        item1 = empty_manager.add_item("High Priority", "Test", "feature", ["AC1"], 2, 13, 8, 5)
        item2 = empty_manager.add_item("Medium Priority", "Test", "feature", ["AC1"], 5, 8, 5, 2)
        item3 = empty_manager.add_item("Low Priority", "Test", "feature", ["AC1"], 8, 3, 2, 1)
        
        empty_manager.update_item_status(item1.id, "READY")
        empty_manager.update_item_status(item2.id, "READY")
        empty_manager.update_item_status(item3.id, "READY")
        
        next_items = empty_manager.get_next_work_items(2)
        
        assert len(next_items) == 2
        assert next_items[0].title == "High Priority"
        assert next_items[1].title == "Medium Priority"
    
    def test_discover_todo_items(self, empty_manager):
        """Test discovering TODO items from source code"""
        # Create temporary source directory with files containing TODOs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Python file with TODOs
            py_file = Path(temp_dir) / "test.py"
            py_file.write_text('''
def example_function():
    # TODO: Implement error handling
    # FIXME: This is broken
    # BUG: Memory leak here
    pass
''')
            
            # Create JS file with TODOs
            js_file = Path(temp_dir) / "test.js"
            js_file.write_text('''
function example() {
    // TODO: Add validation
    // HACK: Temporary solution
}
''')
            
            todos = empty_manager.discover_todo_items(temp_dir)
            
            assert len(todos) >= 5  # Should find all TODO/FIXME/BUG/HACK comments
            
            # Check that all required fields are present
            for todo in todos:
                assert 'file' in todo
                assert 'line' in todo
                assert 'pattern' in todo
                assert 'content' in todo
                assert 'suggested_title' in todo
                assert 'suggested_type' in todo
    
    def test_discover_todo_items_nonexistent_dir(self, empty_manager):
        """Test discovering TODO items from nonexistent directory"""
        todos = empty_manager.discover_todo_items("nonexistent_directory")
        assert todos == []
    
    def test_export_metrics(self, empty_manager):
        """Test exporting backlog metrics"""
        # Add some items
        empty_manager.add_item("High Priority", "Test", "feature", ["AC1"], 2, 13, 8, 5)
        empty_manager.add_item("Medium Priority", "Test", "bug", ["AC1"], 5, 8, 5, 2)
        empty_manager.add_item("Low Priority", "Test", "documentation", ["AC1"], 8, 3, 2, 1, "high")
        
        metrics = empty_manager.export_metrics()
        
        # Check structure
        assert 'summary' in metrics
        assert 'distributions' in metrics
        assert 'top_priority_items' in metrics
        
        # Check summary
        assert metrics['summary']['total_items'] == 3
        assert metrics['summary']['high_risk_items'] == 1
        
        # Check distributions
        assert 'status' in metrics['distributions']
        assert 'type' in metrics['distributions']
        assert 'risk' in metrics['distributions']
        assert 'wsjf' in metrics['distributions']
        
        # Check top priority items
        assert len(metrics['top_priority_items']) == 3
        assert metrics['top_priority_items'][0]['wsjf_score'] == 13.0  # Highest WSJF first
    
    def test_generate_id_different_types(self, empty_manager):
        """Test ID generation for different item types"""
        # Test all supported types
        types_and_prefixes = [
            ("infrastructure", "INFRA"),
            ("feature", "FEAT"),
            ("documentation", "DOC"),
            ("bug", "BUG"),
            ("security", "SEC"),
            ("performance", "PERF")
        ]
        
        for item_type, expected_prefix in types_and_prefixes:
            item = empty_manager.add_item(
                title=f"Test {item_type}",
                description="Test",
                item_type=item_type,
                acceptance_criteria=["AC1"],
                effort=3,
                value=8,
                time_criticality=5,
                risk_reduction=2
            )
            
            assert item.id.startswith(expected_prefix + "-")
            assert item.id.endswith("001")  # First item of this type
    
    def test_generate_id_sequential(self, empty_manager):
        """Test that IDs are generated sequentially"""
        item1 = empty_manager.add_item("Feature 1", "Test", "feature", ["AC1"], 3, 8, 5, 2)
        item2 = empty_manager.add_item("Feature 2", "Test", "feature", ["AC1"], 3, 8, 5, 2)
        item3 = empty_manager.add_item("Feature 3", "Test", "feature", ["AC1"], 3, 8, 5, 2)
        
        assert item1.id == "FEAT-001"
        assert item2.id == "FEAT-002"
        assert item3.id == "FEAT-003"
    
    def test_update_metrics(self, empty_manager):
        """Test metrics update functionality"""
        # Add items with different statuses and risk levels
        item1 = empty_manager.add_item("Item 1", "Test", "feature", ["AC1"], 3, 8, 5, 2, "high")
        item2 = empty_manager.add_item("Item 2", "Test", "bug", ["AC1"], 3, 8, 5, 2, "low")
        
        empty_manager.update_item_status(item1.id, "DOING")
        
        # Force metrics update
        empty_manager._update_metrics()
        
        assert empty_manager.metrics.total_items == 2
        assert empty_manager.metrics.by_status["NEW"] == 1
        assert empty_manager.metrics.by_status["DOING"] == 1
        assert empty_manager.metrics.high_risk_items == 1
        assert empty_manager.metrics.avg_wsjf_score > 0

class TestMainFunction:
    """Test the main CLI function"""
    
    @patch('sys.argv', ['backlog_manager.py', '--help'])
    @patch('argparse.ArgumentParser.print_help')
    def test_main_help(self, mock_help):
        """Test main function with help argument"""
        with pytest.raises(SystemExit):
            main()
    
    @patch('sys.argv', ['backlog_manager.py', '--discover'])
    @patch.object(BacklogManager, 'discover_todo_items')
    def test_main_discover(self, mock_discover):
        """Test main function with discover argument"""
        mock_discover.return_value = [
            {'file': 'test.py', 'line': 1, 'content': '# TODO: test', 'pattern': 'TODO'}
        ]
        
        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()
    
    @patch('sys.argv', ['backlog_manager.py', '--metrics'])
    @patch.object(BacklogManager, 'export_metrics')
    def test_main_metrics(self, mock_metrics):
        """Test main function with metrics argument"""
        mock_metrics.return_value = {'summary': {'total_items': 5}}
        
        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()
    
    @patch('sys.argv', ['backlog_manager.py', '--next', '3'])
    @patch.object(BacklogManager, 'get_next_work_items')
    def test_main_next_items(self, mock_next_items):
        """Test main function with next items argument"""
        mock_next_items.return_value = [
            MagicMock(id="TEST-001", title="Test Item", wsjf_score=5.0)
        ]
        
        with patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=backlog_manager", "--cov-report=html"])