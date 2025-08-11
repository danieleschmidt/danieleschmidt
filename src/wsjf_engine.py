#!/usr/bin/env python3
"""
Weighted Shortest Job First (WSJF) Scoring Engine
Implements advanced prioritization algorithms for autonomous backlog management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import math
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoreComponent(Enum):
    """WSJF scoring components"""
    VALUE = "value"
    TIME_CRITICALITY = "time_criticality"
    RISK_REDUCTION = "risk_reduction"
    EFFORT = "effort"

class PrioritizationStrategy(Enum):
    """Different prioritization strategies"""
    CLASSIC_WSJF = "classic_wsjf"
    WEIGHTED_WSJF = "weighted_wsjf"
    ADJUSTED_WSJF = "adjusted_wsjf"
    DYNAMIC_WSJF = "dynamic_wsjf"
    ML_ENHANCED_WSJF = "ml_enhanced_wsjf"

@dataclass
class WSJFWeights:
    """Configurable weights for WSJF components"""
    value_weight: float = 1.0
    time_criticality_weight: float = 1.0
    risk_reduction_weight: float = 1.0
    effort_penalty: float = 1.0
    confidence_factor: float = 1.0

@dataclass
class ScoreAdjustment:
    """Score adjustments based on various factors"""
    stakeholder_priority: float = 0.0
    technical_debt_factor: float = 0.0
    dependency_complexity: float = 0.0
    team_capacity: float = 0.0
    learning_curve: float = 0.0
    market_timing: float = 0.0

@dataclass
class HistoricalData:
    """Historical data for ML-enhanced scoring"""
    actual_effort: List[float] = field(default_factory=list)
    actual_value: List[float] = field(default_factory=list)
    completion_time: List[float] = field(default_factory=list)
    success_rate: List[float] = field(default_factory=list)

@dataclass
class BacklogItemAnalytics:
    """Comprehensive analytics for a backlog item"""
    item_id: str
    wsjf_scores: Dict[str, float] = field(default_factory=dict)
    percentile_ranking: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    risk_assessment: str = "medium"
    recommended_sprint: int = 0
    estimated_roi: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    opportunity_cost: float = 0.0

class WSJFEngine:
    """Advanced WSJF scoring engine with multiple strategies and ML enhancement"""
    
    def __init__(self, weights: Optional[WSJFWeights] = None):
        """Initialize WSJF engine with configurable weights"""
        self.weights = weights or WSJFWeights()
        self.historical_data: Dict[str, HistoricalData] = {}
        self.scoring_history: List[Dict] = []
        self.calibration_data: Dict = {}
        
    def calculate_classic_wsjf(self, value: float, time_criticality: float, 
                              risk_reduction: float, effort: float) -> float:
        """Calculate classic WSJF score: (Value + Time Criticality + Risk Reduction) / Effort"""
        if effort == 0:
            logger.warning("Effort cannot be zero, using minimum value of 0.1")
            effort = 0.1
        
        cost_of_delay = value + time_criticality + risk_reduction
        return round(cost_of_delay / effort, 3)
    
    def calculate_weighted_wsjf(self, value: float, time_criticality: float,
                               risk_reduction: float, effort: float,
                               adjustment: Optional[ScoreAdjustment] = None) -> float:
        """Calculate weighted WSJF with configurable component weights"""
        
        adjustment = adjustment or ScoreAdjustment()
        
        # Apply weights to cost of delay components
        weighted_value = value * self.weights.value_weight
        weighted_criticality = time_criticality * self.weights.time_criticality_weight
        weighted_risk = risk_reduction * self.weights.risk_reduction_weight
        
        # Calculate weighted cost of delay
        weighted_cod = weighted_value + weighted_criticality + weighted_risk
        
        # Apply effort penalty (higher penalty = lower priority for high effort items)
        effort_penalty = effort * self.weights.effort_penalty
        
        # Apply score adjustments
        adjustment_factor = (
            adjustment.stakeholder_priority +
            adjustment.technical_debt_factor +
            adjustment.dependency_complexity +
            adjustment.team_capacity +
            adjustment.learning_curve +
            adjustment.market_timing
        ) / 6  # Average of adjustments
        
        # Calculate final weighted score
        base_score = weighted_cod / max(effort_penalty, 0.1)
        adjusted_score = base_score * (1 + adjustment_factor)
        
        return round(adjusted_score * self.weights.confidence_factor, 3)
    
    def calculate_dynamic_wsjf(self, value: float, time_criticality: float,
                              risk_reduction: float, effort: float,
                              context: Dict = None) -> float:
        """Calculate dynamic WSJF that adapts based on context"""
        
        context = context or {}
        
        # Base WSJF calculation
        base_wsjf = self.calculate_classic_wsjf(value, time_criticality, risk_reduction, effort)
        
        # Dynamic adjustments based on context
        multipliers = []
        
        # Sprint velocity adjustment
        if 'team_velocity' in context and 'sprint_capacity' in context:
            velocity_ratio = context['team_velocity'] / context['sprint_capacity']
            velocity_multiplier = min(1.5, max(0.5, velocity_ratio))
            multipliers.append(velocity_multiplier)
        
        # Deadline pressure adjustment
        if 'deadline_days' in context:
            deadline_pressure = max(0.5, 2 - (context['deadline_days'] / 30))
            multipliers.append(deadline_pressure)
        
        # Resource availability adjustment
        if 'resource_availability' in context:
            resource_multiplier = context['resource_availability']
            multipliers.append(resource_multiplier)
        
        # Seasonal/market factors
        if 'market_conditions' in context:
            market_multiplier = context['market_conditions']
            multipliers.append(market_multiplier)
        
        # Apply all multipliers
        final_multiplier = np.mean(multipliers) if multipliers else 1.0
        dynamic_score = base_wsjf * final_multiplier
        
        return round(dynamic_score, 3)
    
    def calculate_ml_enhanced_wsjf(self, value: float, time_criticality: float,
                                  risk_reduction: float, effort: float,
                                  item_features: Dict = None) -> Tuple[float, float]:
        """Calculate ML-enhanced WSJF with confidence intervals"""
        
        item_features = item_features or {}
        
        # Base WSJF calculation
        base_wsjf = self.calculate_classic_wsjf(value, time_criticality, risk_reduction, effort)
        
        # Simple ML enhancement using historical patterns
        enhancement_factor = 1.0
        confidence = 0.5
        
        if self.historical_data:
            # Calculate enhancement based on historical success patterns
            similar_items = self._find_similar_items(value, time_criticality, risk_reduction, effort)
            
            if similar_items:
                historical_performance = np.mean([item['actual_value'] / item['estimated_value'] 
                                                for item in similar_items if item['estimated_value'] > 0])
                enhancement_factor = max(0.5, min(2.0, historical_performance))
                confidence = min(0.95, 0.5 + len(similar_items) * 0.1)
        
        # Feature-based adjustments
        if 'complexity_score' in item_features:
            complexity_penalty = 1 - (item_features['complexity_score'] - 5) * 0.05
            enhancement_factor *= complexity_penalty
        
        if 'team_experience' in item_features:
            experience_boost = 1 + (item_features['team_experience'] - 5) * 0.03
            enhancement_factor *= experience_boost
        
        enhanced_score = base_wsjf * enhancement_factor
        
        return round(enhanced_score, 3), round(confidence, 3)
    
    def calculate_comprehensive_analytics(self, items: List[Dict]) -> List[BacklogItemAnalytics]:
        """Generate comprehensive analytics for all backlog items"""
        
        analytics = []
        wsjf_scores = []
        
        # Calculate all WSJF variants for each item
        for item in items:
            item_analytics = BacklogItemAnalytics(item_id=item['id'])
            
            # Calculate different WSJF strategies
            classic_wsjf = self.calculate_classic_wsjf(
                item['value'], item['time_criticality'],
                item['risk_reduction'], item['effort']
            )
            
            weighted_wsjf = self.calculate_weighted_wsjf(
                item['value'], item['time_criticality'],
                item['risk_reduction'], item['effort']
            )
            
            dynamic_wsjf = self.calculate_dynamic_wsjf(
                item['value'], item['time_criticality'],
                item['risk_reduction'], item['effort'],
                item.get('context', {})
            )
            
            ml_enhanced_wsjf, confidence = self.calculate_ml_enhanced_wsjf(
                item['value'], item['time_criticality'],
                item['risk_reduction'], item['effort'],
                item.get('features', {})
            )
            
            # Store all scores
            item_analytics.wsjf_scores = {
                'classic': classic_wsjf,
                'weighted': weighted_wsjf,
                'dynamic': dynamic_wsjf,
                'ml_enhanced': ml_enhanced_wsjf,
                'confidence': confidence
            }
            
            wsjf_scores.append(weighted_wsjf)  # Use weighted as primary
            
            # Calculate additional metrics
            item_analytics.estimated_roi = self._calculate_roi(item)
            item_analytics.opportunity_cost = self._calculate_opportunity_cost(item, items)
            item_analytics.risk_assessment = self._assess_risk(item)
            
            analytics.append(item_analytics)
        
        # Calculate percentile rankings
        for i, item_analytics in enumerate(analytics):
            item_analytics.percentile_ranking = self._calculate_percentile(wsjf_scores[i], wsjf_scores)
            item_analytics.confidence_interval = self._calculate_confidence_interval(
                wsjf_scores[i], item_analytics.wsjf_scores['confidence']
            )
            item_analytics.recommended_sprint = self._recommend_sprint(item_analytics, analytics)
        
        return sorted(analytics, key=lambda x: x.wsjf_scores['weighted'], reverse=True)
    
    def optimize_portfolio(self, items: List[Dict], constraints: Dict) -> Dict:
        """Optimize portfolio selection based on constraints"""
        
        analytics = self.calculate_comprehensive_analytics(items)
        
        # Extract constraints
        max_effort = constraints.get('max_effort', float('inf'))
        max_items = constraints.get('max_items', len(items))
        min_value = constraints.get('min_value', 0)
        required_types = constraints.get('required_types', [])
        
        # Portfolio optimization using greedy approach with WSJF scores
        selected_items = []
        total_effort = 0
        total_value = 0
        type_counts = {}
        
        # Sort by weighted WSJF score
        sorted_analytics = sorted(analytics, key=lambda x: x.wsjf_scores['weighted'], reverse=True)
        
        for item_analytics in sorted_analytics:
            item = next(item for item in items if item['id'] == item_analytics.item_id)
            
            # Check constraints
            if len(selected_items) >= max_items:
                break
            if total_effort + item['effort'] > max_effort:
                continue
            if item['value'] < min_value:
                continue
            
            # Check type requirements
            item_type = item.get('type', 'unknown')
            if required_types and item_type not in required_types:
                continue
            
            # Add to portfolio
            selected_items.append({
                'item': item,
                'analytics': item_analytics
            })
            
            total_effort += item['effort']
            total_value += item['value']
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        # Calculate portfolio metrics
        portfolio_wsjf = total_value / max(total_effort, 1)
        portfolio_risk = np.mean([
            self._risk_to_numeric(selection['analytics'].risk_assessment) 
            for selection in selected_items
        ])
        
        return {
            'selected_items': selected_items,
            'portfolio_metrics': {
                'total_items': len(selected_items),
                'total_effort': total_effort,
                'total_value': total_value,
                'portfolio_wsjf': round(portfolio_wsjf, 3),
                'average_risk': round(portfolio_risk, 3),
                'type_distribution': type_counts,
                'effort_utilization': round(total_effort / max_effort * 100, 1) if max_effort != float('inf') else 0
            },
            'rejected_items': [
                analytics for analytics in sorted_analytics 
                if analytics.item_id not in [s['item']['id'] for s in selected_items]
            ]
        }
    
    def generate_insights(self, analytics: List[BacklogItemAnalytics]) -> Dict:
        """Generate strategic insights from backlog analytics"""
        
        insights = {
            'summary': {
                'total_items': len(analytics),
                'high_priority_items': len([a for a in analytics if a.wsjf_scores['weighted'] >= 5]),
                'average_wsjf': round(np.mean([a.wsjf_scores['weighted'] for a in analytics]), 2),
                'median_wsjf': round(np.median([a.wsjf_scores['weighted'] for a in analytics]), 2)
            },
            'recommendations': [],
            'risk_analysis': {},
            'capacity_planning': {},
            'trends': {}
        }
        
        # Generate recommendations
        high_value_low_effort = [a for a in analytics 
                               if a.estimated_roi > np.percentile([x.estimated_roi for x in analytics], 75)]
        if high_value_low_effort:
            insights['recommendations'].append({
                'type': 'quick_wins',
                'message': f'Identify {len(high_value_low_effort)} high-ROI items for immediate attention',
                'items': [a.item_id for a in high_value_low_effort[:3]]
            })
        
        # Risk analysis
        risk_distribution = {}
        for analytics_item in analytics:
            risk = analytics_item.risk_assessment
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        insights['risk_analysis'] = {
            'distribution': risk_distribution,
            'high_risk_items': len([a for a in analytics if a.risk_assessment in ['high', 'critical']]),
            'risk_mitigation_needed': len([a for a in analytics if a.risk_assessment == 'critical']) > 0
        }
        
        # Capacity planning insights
        effort_by_sprint = {}
        for analytics_item in analytics:
            sprint = analytics_item.recommended_sprint
            effort_by_sprint[sprint] = effort_by_sprint.get(sprint, 0) + 1
        
        insights['capacity_planning'] = {
            'sprint_distribution': effort_by_sprint,
            'bottleneck_sprints': [sprint for sprint, count in effort_by_sprint.items() if count > 5],
            'load_balancing_needed': max(effort_by_sprint.values()) > 3 * min(effort_by_sprint.values()) if effort_by_sprint else False
        }
        
        return insights
    
    def calibrate_scoring(self, completed_items: List[Dict]) -> Dict:
        """Calibrate scoring model based on completed items"""
        
        calibration_results = {
            'accuracy_metrics': {},
            'bias_analysis': {},
            'recommendations': []
        }
        
        if not completed_items:
            return calibration_results
        
        # Analyze prediction accuracy
        effort_errors = []
        value_errors = []
        
        for item in completed_items:
            if 'actual_effort' in item and 'actual_value' in item:
                effort_error = abs(item['effort'] - item['actual_effort']) / item['actual_effort']
                value_error = abs(item['value'] - item['actual_value']) / item['actual_value']
                
                effort_errors.append(effort_error)
                value_errors.append(value_error)
        
        if effort_errors:
            calibration_results['accuracy_metrics'] = {
                'effort_mae': round(np.mean(effort_errors), 3),
                'effort_median_error': round(np.median(effort_errors), 3),
                'value_mae': round(np.mean(value_errors), 3),
                'value_median_error': round(np.median(value_errors), 3)
            }
            
            # Generate calibration recommendations
            if np.mean(effort_errors) > 0.3:
                calibration_results['recommendations'].append(
                    "Consider breaking down large items - effort estimation accuracy is low"
                )
            
            if np.mean(value_errors) > 0.4:
                calibration_results['recommendations'].append(
                    "Improve value estimation process - significant value prediction errors detected"
                )
        
        return calibration_results
    
    def _find_similar_items(self, value: float, time_criticality: float, 
                          risk_reduction: float, effort: float) -> List[Dict]:
        """Find similar items in historical data"""
        similar_items = []
        
        for item_id, historical in self.historical_data.items():
            if not historical.actual_effort:
                continue
                
            # Simple similarity based on Euclidean distance
            distance = math.sqrt(
                (value - historical.actual_value[0]) ** 2 +
                (time_criticality - 5) ** 2 +  # Assume 5 as baseline
                (risk_reduction - 5) ** 2 +
                (effort - historical.actual_effort[0]) ** 2
            )
            
            if distance < 3:  # Similarity threshold
                similar_items.append({
                    'item_id': item_id,
                    'estimated_value': value,
                    'actual_value': historical.actual_value[0],
                    'estimated_effort': effort,
                    'actual_effort': historical.actual_effort[0]
                })
        
        return similar_items
    
    def _calculate_roi(self, item: Dict) -> float:
        """Calculate estimated ROI for an item"""
        if item['effort'] == 0:
            return 0.0
        return round(item['value'] / item['effort'], 2)
    
    def _calculate_opportunity_cost(self, item: Dict, all_items: List[Dict]) -> float:
        """Calculate opportunity cost of not doing this item"""
        # Simple heuristic: average value of items with similar time criticality
        similar_criticality_items = [
            other for other in all_items 
            if abs(other['time_criticality'] - item['time_criticality']) <= 1
            and other['id'] != item['id']
        ]
        
        if not similar_criticality_items:
            return 0.0
        
        avg_value = np.mean([other['value'] for other in similar_criticality_items])
        return round(max(0, avg_value - item['value']), 2)
    
    def _assess_risk(self, item: Dict) -> str:
        """Assess overall risk level of an item"""
        effort_risk = "high" if item['effort'] > 8 else "medium" if item['effort'] > 5 else "low"
        complexity_risk = item.get('complexity', 5)
        
        risk_score = item['effort'] + complexity_risk - item.get('team_experience', 5)
        
        if risk_score > 10:
            return "critical"
        elif risk_score > 7:
            return "high"
        elif risk_score > 4:
            return "medium"
        else:
            return "low"
    
    def _calculate_percentile(self, score: float, all_scores: List[float]) -> float:
        """Calculate percentile ranking of a score"""
        if not all_scores:
            return 50.0
        
        sorted_scores = sorted(all_scores)
        position = sorted_scores.index(score) if score in sorted_scores else 0
        percentile = (position / len(sorted_scores)) * 100
        return round(percentile, 1)
    
    def _calculate_confidence_interval(self, score: float, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for a score"""
        margin = score * (1 - confidence) * 0.5
        return (round(max(0, score - margin), 2), round(score + margin, 2))
    
    def _recommend_sprint(self, item_analytics: BacklogItemAnalytics, 
                         all_analytics: List[BacklogItemAnalytics]) -> int:
        """Recommend which sprint an item should be scheduled for"""
        
        # Simple heuristic based on WSJF ranking
        sorted_by_wsjf = sorted(all_analytics, key=lambda x: x.wsjf_scores['weighted'], reverse=True)
        position = sorted_by_wsjf.index(item_analytics)
        
        # Assume 5 items per sprint
        return (position // 5) + 1
    
    def _risk_to_numeric(self, risk: str) -> float:
        """Convert risk assessment to numeric value"""
        risk_values = {
            "low": 1.0,
            "medium": 2.0,
            "high": 3.0,
            "critical": 4.0
        }
        return risk_values.get(risk, 2.0)
    
    def export_scoring_report(self, analytics: List[BacklogItemAnalytics], 
                             insights: Dict, filename: str = None) -> str:
        """Export comprehensive scoring report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': insights['summary'],
            'items': [
                {
                    'id': item.item_id,
                    'wsjf_scores': item.wsjf_scores,
                    'percentile_ranking': item.percentile_ranking,
                    'risk_assessment': item.risk_assessment,
                    'estimated_roi': item.estimated_roi,
                    'recommended_sprint': item.recommended_sprint,
                    'confidence_interval': item.confidence_interval
                }
                for item in analytics
            ],
            'insights': insights
        }
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report exported to {filename}")
        
        return json.dumps(report, indent=2)

def main():
    """CLI interface for WSJF engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WSJF Scoring Engine")
    parser.add_argument("--calculate", action="store_true", help="Calculate WSJF scores")
    parser.add_argument("--strategy", choices=['classic', 'weighted', 'dynamic', 'ml'], 
                       default='weighted', help="Scoring strategy")
    parser.add_argument("--value", type=float, help="Value score")
    parser.add_argument("--criticality", type=float, help="Time criticality score")
    parser.add_argument("--risk", type=float, help="Risk reduction score")
    parser.add_argument("--effort", type=float, help="Effort score")
    parser.add_argument("--analyze-backlog", help="Analyze backlog from file")
    parser.add_argument("--export-report", help="Export analysis report to file")
    
    args = parser.parse_args()
    
    engine = WSJFEngine()
    
    if args.calculate and all([args.value, args.criticality, args.risk, args.effort]):
        if args.strategy == 'classic':
            score = engine.calculate_classic_wsjf(args.value, args.criticality, args.risk, args.effort)
        elif args.strategy == 'weighted':
            score = engine.calculate_weighted_wsjf(args.value, args.criticality, args.risk, args.effort)
        elif args.strategy == 'dynamic':
            score = engine.calculate_dynamic_wsjf(args.value, args.criticality, args.risk, args.effort)
        elif args.strategy == 'ml':
            score, confidence = engine.calculate_ml_enhanced_wsjf(args.value, args.criticality, args.risk, args.effort)
            print(f"WSJF Score: {score} (Confidence: {confidence})")
            return
        
        print(f"WSJF Score ({args.strategy}): {score}")
    
    elif args.analyze_backlog:
        try:
            import yaml
            with open(args.analyze_backlog, 'r') as f:
                data = yaml.safe_load(f)
            
            items = data.get('items', [])
            analytics = engine.calculate_comprehensive_analytics(items)
            insights = engine.generate_insights(analytics)
            
            print(f"Analyzed {len(analytics)} items")
            print(f"Average WSJF: {insights['summary']['average_wsjf']}")
            print(f"High priority items: {insights['summary']['high_priority_items']}")
            
            if args.export_report:
                engine.export_scoring_report(analytics, insights, args.export_report)
        
        except Exception as e:
            print(f"Error analyzing backlog: {e}")
    
    else:
        print("Use --help for available commands")

if __name__ == "__main__":
    main()