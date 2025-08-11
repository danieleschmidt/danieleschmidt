#!/usr/bin/env python3
"""
Unit tests for WSJF Engine
Comprehensive test coverage for all WSJF scoring strategies and analytics
"""

import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from wsjf_engine import (
    WSJFEngine, WSJFWeights, ScoreAdjustment, HistoricalData,
    BacklogItemAnalytics, ScoreComponent, PrioritizationStrategy
)

class TestWSJFWeights:
    """Test WSJFWeights dataclass"""
    
    def test_default_weights(self):
        """Test default weight values"""
        weights = WSJFWeights()
        
        assert weights.value_weight == 1.0
        assert weights.time_criticality_weight == 1.0
        assert weights.risk_reduction_weight == 1.0
        assert weights.effort_penalty == 1.0
        assert weights.confidence_factor == 1.0
    
    def test_custom_weights(self):
        """Test custom weight values"""
        weights = WSJFWeights(
            value_weight=2.0,
            time_criticality_weight=1.5,
            risk_reduction_weight=0.8,
            effort_penalty=1.2,
            confidence_factor=0.9
        )
        
        assert weights.value_weight == 2.0
        assert weights.time_criticality_weight == 1.5
        assert weights.risk_reduction_weight == 0.8
        assert weights.effort_penalty == 1.2
        assert weights.confidence_factor == 0.9

class TestScoreAdjustment:
    """Test ScoreAdjustment dataclass"""
    
    def test_default_adjustments(self):
        """Test default adjustment values"""
        adjustment = ScoreAdjustment()
        
        assert adjustment.stakeholder_priority == 0.0
        assert adjustment.technical_debt_factor == 0.0
        assert adjustment.dependency_complexity == 0.0
        assert adjustment.team_capacity == 0.0
        assert adjustment.learning_curve == 0.0
        assert adjustment.market_timing == 0.0
    
    def test_custom_adjustments(self):
        """Test custom adjustment values"""
        adjustment = ScoreAdjustment(
            stakeholder_priority=0.2,
            technical_debt_factor=-0.1,
            dependency_complexity=0.15,
            team_capacity=0.3,
            learning_curve=-0.2,
            market_timing=0.1
        )
        
        assert adjustment.stakeholder_priority == 0.2
        assert adjustment.technical_debt_factor == -0.1
        assert adjustment.dependency_complexity == 0.15

class TestWSJFEngine:
    """Test WSJFEngine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create a default WSJF engine for testing"""
        return WSJFEngine()
    
    @pytest.fixture
    def weighted_engine(self):
        """Create a WSJF engine with custom weights"""
        weights = WSJFWeights(
            value_weight=2.0,
            time_criticality_weight=1.5,
            risk_reduction_weight=0.8,
            effort_penalty=1.2,
            confidence_factor=0.9
        )
        return WSJFEngine(weights)
    
    def test_engine_initialization(self):
        """Test WSJF engine initialization"""
        engine = WSJFEngine()
        
        assert engine.weights is not None
        assert isinstance(engine.historical_data, dict)
        assert isinstance(engine.scoring_history, list)
        assert isinstance(engine.calibration_data, dict)
    
    def test_engine_initialization_with_weights(self, weighted_engine):
        """Test WSJF engine initialization with custom weights"""
        assert weighted_engine.weights.value_weight == 2.0
        assert weighted_engine.weights.time_criticality_weight == 1.5
    
    def test_calculate_classic_wsjf_normal(self, engine):
        """Test classic WSJF calculation with normal values"""
        score = engine.calculate_classic_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3
        )
        
        expected = (8 + 5 + 2) / 3  # 15 / 3 = 5.0
        assert score == 5.0
    
    def test_calculate_classic_wsjf_zero_effort(self, engine):
        """Test classic WSJF calculation with zero effort"""
        score = engine.calculate_classic_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=0
        )
        
        # Should use minimum effort of 0.1
        expected = (8 + 5 + 2) / 0.1  # 15 / 0.1 = 150.0
        assert score == 150.0
    
    def test_calculate_classic_wsjf_high_values(self, engine):
        """Test classic WSJF calculation with high values"""
        score = engine.calculate_classic_wsjf(
            value=13, time_criticality=13, risk_reduction=13, effort=1
        )
        
        expected = (13 + 13 + 13) / 1  # 39 / 1 = 39.0
        assert score == 39.0
    
    def test_calculate_weighted_wsjf_default_weights(self, engine):
        """Test weighted WSJF calculation with default weights"""
        score = engine.calculate_weighted_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3
        )
        
        # With default weights (all 1.0), should be same as classic
        assert score == 5.0
    
    def test_calculate_weighted_wsjf_custom_weights(self, weighted_engine):
        """Test weighted WSJF calculation with custom weights"""
        score = weighted_engine.calculate_weighted_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3
        )
        
        # With custom weights: (8*2.0 + 5*1.5 + 2*0.8) / (3*1.2) * 0.9
        weighted_cod = 16 + 7.5 + 1.6  # 25.1
        effort_penalty = 3 * 1.2  # 3.6
        base_score = 25.1 / 3.6  # ~6.972
        final_score = base_score * 0.9  # ~6.275
        
        assert abs(score - 6.275) < 0.01
    
    def test_calculate_weighted_wsjf_with_adjustments(self, engine):
        """Test weighted WSJF calculation with score adjustments"""
        adjustment = ScoreAdjustment(
            stakeholder_priority=0.2,
            technical_debt_factor=-0.1,
            dependency_complexity=0.0,
            team_capacity=0.1,
            learning_curve=0.0,
            market_timing=0.0
        )
        
        score = engine.calculate_weighted_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3,
            adjustment=adjustment
        )
        
        # Base score: 5.0
        # Adjustment factor: (0.2 - 0.1 + 0 + 0.1 + 0 + 0) / 6 = 0.2/6 = 0.0333...
        # Final score: 5.0 * (1 + 0.0333) = ~5.167
        assert abs(score - 5.167) < 0.01
    
    def test_calculate_dynamic_wsjf_no_context(self, engine):
        """Test dynamic WSJF calculation without context"""
        score = engine.calculate_dynamic_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3
        )
        
        # Without context, should be same as classic WSJF
        assert score == 5.0
    
    def test_calculate_dynamic_wsjf_with_context(self, engine):
        """Test dynamic WSJF calculation with context"""
        context = {
            'team_velocity': 20,
            'sprint_capacity': 25,
            'deadline_days': 15,
            'resource_availability': 0.8,
            'market_conditions': 1.2
        }
        
        score = engine.calculate_dynamic_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3,
            context=context
        )
        
        # Base WSJF: 5.0
        # Velocity multiplier: 20/25 = 0.8
        # Deadline pressure: max(0.5, 2 - 15/30) = max(0.5, 1.5) = 1.5
        # Resource multiplier: 0.8
        # Market multiplier: 1.2
        # Average multiplier: (0.8 + 1.5 + 0.8 + 1.2) / 4 = 1.075
        # Final score: 5.0 * 1.075 = 5.375
        
        assert abs(score - 5.375) < 0.01
    
    def test_calculate_ml_enhanced_wsjf_no_history(self, engine):
        """Test ML-enhanced WSJF calculation without historical data"""
        score, confidence = engine.calculate_ml_enhanced_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3
        )
        
        # Without historical data, should be same as classic WSJF
        assert score == 5.0
        assert confidence == 0.5  # Default confidence
    
    def test_calculate_ml_enhanced_wsjf_with_features(self, engine):
        """Test ML-enhanced WSJF calculation with item features"""
        features = {
            'complexity_score': 7,
            'team_experience': 8
        }
        
        score, confidence = engine.calculate_ml_enhanced_wsjf(
            value=8, time_criticality=5, risk_reduction=2, effort=3,
            item_features=features
        )
        
        # Base WSJF: 5.0
        # Complexity penalty: 1 - (7-5)*0.05 = 1 - 0.1 = 0.9
        # Experience boost: 1 + (8-5)*0.03 = 1 + 0.09 = 1.09
        # Enhancement factor: 1.0 * 0.9 * 1.09 = 0.981
        # Final score: 5.0 * 0.981 = 4.905
        
        assert abs(score - 4.905) < 0.01
        assert confidence == 0.5
    
    def test_calculate_comprehensive_analytics(self, engine):
        """Test comprehensive analytics calculation"""
        items = [
            {
                'id': 'TEST-001',
                'value': 8, 'time_criticality': 5, 'risk_reduction': 2, 'effort': 3,
                'type': 'feature', 'context': {}, 'features': {}
            },
            {
                'id': 'TEST-002', 
                'value': 13, 'time_criticality': 8, 'risk_reduction': 5, 'effort': 5,
                'type': 'bug', 'context': {}, 'features': {}
            },
            {
                'id': 'TEST-003',
                'value': 3, 'time_criticality': 2, 'risk_reduction': 1, 'effort': 8,
                'type': 'documentation', 'context': {}, 'features': {}
            }
        ]
        
        analytics = engine.calculate_comprehensive_analytics(items)
        
        assert len(analytics) == 3
        
        # Should be sorted by weighted WSJF score (highest first)
        assert analytics[0].item_id == 'TEST-002'  # Highest WSJF
        assert analytics[1].item_id == 'TEST-001'  # Medium WSJF
        assert analytics[2].item_id == 'TEST-003'  # Lowest WSJF
        
        # Check that all required fields are populated
        for item_analytics in analytics:
            assert item_analytics.item_id is not None
            assert 'classic' in item_analytics.wsjf_scores
            assert 'weighted' in item_analytics.wsjf_scores
            assert 'dynamic' in item_analytics.wsjf_scores
            assert 'ml_enhanced' in item_analytics.wsjf_scores
            assert 'confidence' in item_analytics.wsjf_scores
            assert item_analytics.percentile_ranking >= 0
            assert item_analytics.risk_assessment in ['low', 'medium', 'high', 'critical']
            assert item_analytics.recommended_sprint > 0
    
    def test_optimize_portfolio_no_constraints(self, engine):
        """Test portfolio optimization without constraints"""
        items = [
            {'id': 'TEST-001', 'value': 8, 'time_criticality': 5, 'risk_reduction': 2, 'effort': 3, 'type': 'feature'},
            {'id': 'TEST-002', 'value': 13, 'time_criticality': 8, 'risk_reduction': 5, 'effort': 5, 'type': 'bug'},
            {'id': 'TEST-003', 'value': 3, 'time_criticality': 2, 'risk_reduction': 1, 'effort': 8, 'type': 'documentation'}
        ]
        
        constraints = {}
        result = engine.optimize_portfolio(items, constraints)
        
        assert 'selected_items' in result
        assert 'portfolio_metrics' in result
        assert 'rejected_items' in result
        
        # Without constraints, all items should be selected
        assert len(result['selected_items']) == 3
        assert result['portfolio_metrics']['total_items'] == 3
        assert result['portfolio_metrics']['total_effort'] == 16  # 3 + 5 + 8
        assert result['portfolio_metrics']['total_value'] == 24   # 8 + 13 + 3
    
    def test_optimize_portfolio_with_effort_constraint(self, engine):
        """Test portfolio optimization with effort constraint"""
        items = [
            {'id': 'TEST-001', 'value': 8, 'time_criticality': 5, 'risk_reduction': 2, 'effort': 3, 'type': 'feature'},
            {'id': 'TEST-002', 'value': 13, 'time_criticality': 8, 'risk_reduction': 5, 'effort': 5, 'type': 'bug'},
            {'id': 'TEST-003', 'value': 3, 'time_criticality': 2, 'risk_reduction': 1, 'effort': 8, 'type': 'documentation'}
        ]
        
        constraints = {'max_effort': 10}  # Should exclude TEST-003
        result = engine.optimize_portfolio(items, constraints)
        
        assert len(result['selected_items']) == 2
        assert result['portfolio_metrics']['total_effort'] == 8  # 3 + 5
        assert result['portfolio_metrics']['effort_utilization'] == 80.0  # 8/10 * 100
    
    def test_optimize_portfolio_with_type_requirement(self, engine):
        """Test portfolio optimization with type requirements"""
        items = [
            {'id': 'TEST-001', 'value': 8, 'time_criticality': 5, 'risk_reduction': 2, 'effort': 3, 'type': 'feature'},
            {'id': 'TEST-002', 'value': 13, 'time_criticality': 8, 'risk_reduction': 5, 'effort': 5, 'type': 'bug'},
            {'id': 'TEST-003', 'value': 3, 'time_criticality': 2, 'risk_reduction': 1, 'effort': 8, 'type': 'documentation'}
        ]
        
        constraints = {'required_types': ['bug']}  # Only bug items
        result = engine.optimize_portfolio(items, constraints)
        
        assert len(result['selected_items']) == 1
        assert result['selected_items'][0]['item']['id'] == 'TEST-002'
        assert result['selected_items'][0]['item']['type'] == 'bug'
    
    def test_generate_insights_basic(self, engine):
        """Test insights generation with basic analytics"""
        analytics = [
            BacklogItemAnalytics(
                item_id='TEST-001',
                wsjf_scores={'weighted': 5.2},
                estimated_roi=2.67,
                risk_assessment='medium',
                recommended_sprint=1
            ),
            BacklogItemAnalytics(
                item_id='TEST-002',
                wsjf_scores={'weighted': 8.1},
                estimated_roi=3.25,
                risk_assessment='high',
                recommended_sprint=1
            ),
            BacklogItemAnalytics(
                item_id='TEST-003',
                wsjf_scores={'weighted': 0.75},
                estimated_roi=0.38,
                risk_assessment='low',
                recommended_sprint=2
            )
        ]
        
        insights = engine.generate_insights(analytics)
        
        assert 'summary' in insights
        assert 'recommendations' in insights
        assert 'risk_analysis' in insights
        assert 'capacity_planning' in insights
        
        # Check summary
        assert insights['summary']['total_items'] == 3
        assert insights['summary']['high_priority_items'] == 2  # Items with WSJF >= 5
        assert abs(insights['summary']['average_wsjf'] - 4.68) < 0.01  # (5.2+8.1+0.75)/3
        
        # Check risk analysis
        assert insights['risk_analysis']['high_risk_items'] == 1
        assert 'medium' in insights['risk_analysis']['distribution']
        assert 'high' in insights['risk_analysis']['distribution']
        assert 'low' in insights['risk_analysis']['distribution']
    
    def test_calibrate_scoring_empty_data(self, engine):
        """Test scoring calibration with empty completed items"""
        result = engine.calibrate_scoring([])
        
        assert 'accuracy_metrics' in result
        assert 'bias_analysis' in result
        assert 'recommendations' in result
        
        # Should be empty with no data
        assert result['accuracy_metrics'] == {}
        assert result['recommendations'] == []
    
    def test_calibrate_scoring_with_data(self, engine):
        """Test scoring calibration with completed items"""
        completed_items = [
            {
                'effort': 3, 'actual_effort': 4,
                'value': 8, 'actual_value': 7
            },
            {
                'effort': 5, 'actual_effort': 6,
                'value': 13, 'actual_value': 15
            }
        ]
        
        result = engine.calibrate_scoring(completed_items)
        
        assert 'accuracy_metrics' in result
        assert 'effort_mae' in result['accuracy_metrics']
        assert 'value_mae' in result['accuracy_metrics']
        
        # Calculate expected errors
        effort_error1 = abs(3 - 4) / 4  # 0.25
        effort_error2 = abs(5 - 6) / 6  # 0.167
        expected_effort_mae = (effort_error1 + effort_error2) / 2  # ~0.208
        
        assert abs(result['accuracy_metrics']['effort_mae'] - expected_effort_mae) < 0.01
    
    def test_export_scoring_report(self, engine):
        """Test exporting scoring report"""
        analytics = [
            BacklogItemAnalytics(
                item_id='TEST-001',
                wsjf_scores={'weighted': 5.2},
                percentile_ranking=66.7,
                risk_assessment='medium',
                estimated_roi=2.67,
                recommended_sprint=1,
                confidence_interval=(4.8, 5.6)
            )
        ]
        
        insights = {
            'summary': {
                'total_items': 1,
                'high_priority_items': 1,
                'average_wsjf': 5.2,
                'median_wsjf': 5.2
            },
            'recommendations': [],
            'risk_analysis': {},
            'capacity_planning': {}
        }
        
        report_json = engine.export_scoring_report(analytics, insights)
        report_data = json.loads(report_json)
        
        assert 'generated_at' in report_data
        assert 'summary' in report_data
        assert 'items' in report_data
        assert 'insights' in report_data
        
        assert len(report_data['items']) == 1
        assert report_data['items'][0]['id'] == 'TEST-001'
        assert report_data['items'][0]['wsjf_scores']['weighted'] == 5.2
    
    def test_export_scoring_report_to_file(self, engine):
        """Test exporting scoring report to file"""
        analytics = [
            BacklogItemAnalytics(
                item_id='TEST-001',
                wsjf_scores={'weighted': 5.2},
                percentile_ranking=66.7,
                risk_assessment='medium',
                estimated_roi=2.67,
                recommended_sprint=1,
                confidence_interval=(4.8, 5.6)
            )
        ]
        
        insights = {'summary': {}, 'recommendations': [], 'risk_analysis': {}, 'capacity_planning': {}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name
        
        try:
            engine.export_scoring_report(analytics, insights, filename)
            
            # Verify file was created and contains valid JSON
            with open(filename, 'r') as f:
                report_data = json.load(f)
            
            assert 'generated_at' in report_data
            assert len(report_data['items']) == 1
        
        finally:
            os.unlink(filename)
    
    def test_find_similar_items_empty_history(self, engine):
        """Test finding similar items with empty historical data"""
        similar = engine._find_similar_items(8, 5, 2, 3)
        assert similar == []
    
    def test_calculate_roi(self, engine):
        """Test ROI calculation"""
        item = {'value': 10, 'effort': 5}
        roi = engine._calculate_roi(item)
        assert roi == 2.0  # 10 / 5
        
        # Test zero effort
        item_zero_effort = {'value': 10, 'effort': 0}
        roi_zero = engine._calculate_roi(item_zero_effort)
        assert roi_zero == 0.0
    
    def test_assess_risk(self, engine):
        """Test risk assessment"""
        # Low risk item
        low_risk_item = {'effort': 3, 'complexity': 3, 'team_experience': 8}
        assert engine._assess_risk(low_risk_item) == 'low'
        
        # High risk item
        high_risk_item = {'effort': 8, 'complexity': 8, 'team_experience': 2}
        assert engine._assess_risk(high_risk_item) == 'critical'
        
        # Medium risk item
        medium_risk_item = {'effort': 5, 'complexity': 5}  # No team_experience defaults to 5
        assert engine._assess_risk(medium_risk_item) == 'medium'
    
    def test_calculate_percentile(self, engine):
        """Test percentile calculation"""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test exact match
        percentile = engine._calculate_percentile(3.0, scores)
        assert percentile == 40.0  # 3.0 is at index 2, so (2/5)*100 = 40%
        
        # Test highest score
        percentile_max = engine._calculate_percentile(5.0, scores)
        assert percentile_max == 80.0  # 5.0 is at index 4, so (4/5)*100 = 80%
        
        # Test with empty scores
        percentile_empty = engine._calculate_percentile(3.0, [])
        assert percentile_empty == 50.0  # Default percentile
    
    def test_calculate_confidence_interval(self, engine):
        """Test confidence interval calculation"""
        score = 5.0
        confidence = 0.8
        
        lower, upper = engine._calculate_confidence_interval(score, confidence)
        
        # Margin = 5.0 * (1 - 0.8) * 0.5 = 5.0 * 0.2 * 0.5 = 0.5
        # Lower = max(0, 5.0 - 0.5) = 4.5
        # Upper = 5.0 + 0.5 = 5.5
        assert lower == 4.5
        assert upper == 5.5
    
    def test_recommend_sprint(self, engine):
        """Test sprint recommendation"""
        # Create mock analytics sorted by WSJF
        analytics = [
            BacklogItemAnalytics(item_id='HIGH', wsjf_scores={'weighted': 10.0}),
            BacklogItemAnalytics(item_id='MEDIUM', wsjf_scores={'weighted': 5.0}),
            BacklogItemAnalytics(item_id='LOW', wsjf_scores={'weighted': 1.0})
        ]
        
        # Test recommendations (assuming 5 items per sprint)
        sprint_high = engine._recommend_sprint(analytics[0], analytics)
        sprint_medium = engine._recommend_sprint(analytics[1], analytics)
        sprint_low = engine._recommend_sprint(analytics[2], analytics)
        
        assert sprint_high == 1  # Position 0: (0 // 5) + 1 = 1
        assert sprint_medium == 1  # Position 1: (1 // 5) + 1 = 1  
        assert sprint_low == 1    # Position 2: (2 // 5) + 1 = 1
    
    def test_risk_to_numeric(self, engine):
        """Test risk to numeric conversion"""
        assert engine._risk_to_numeric('low') == 1.0
        assert engine._risk_to_numeric('medium') == 2.0
        assert engine._risk_to_numeric('high') == 3.0
        assert engine._risk_to_numeric('critical') == 4.0
        assert engine._risk_to_numeric('unknown') == 2.0  # Default

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=wsjf_engine", "--cov-report=html"])