#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF scoring, status tracking, and automated prioritization
"""

import yaml
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacklogItem:
    """Represents a single backlog item with WSJF scoring"""
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: int
    value: int
    time_criticality: int
    risk_reduction: int
    wsjf_score: float
    status: str
    risk_tier: str
    created_at: str
    links: List[str]
    
    @property
    def cost_of_delay(self) -> int:
        """Calculate cost of delay (value + time_criticality + risk_reduction)"""
        return self.value + self.time_criticality + self.risk_reduction
    
    def calculate_wsjf(self) -> float:
        """Calculate WSJF score: Cost of Delay / Effort"""
        if self.effort == 0:
            return 0.0
        return round(self.cost_of_delay / self.effort, 2)
    
    def update_wsjf(self) -> None:
        """Update WSJF score based on current values"""
        self.wsjf_score = self.calculate_wsjf()

@dataclass
class BacklogMetrics:
    """Metrics for the entire backlog"""
    last_updated: str
    total_items: int
    by_status: Dict[str, int]
    avg_wsjf_score: float
    high_risk_items: int

class BacklogManager:
    """Manages autonomous backlog operations with WSJF prioritization"""
    
    VALID_STATUSES = ["NEW", "REFINED", "READY", "DOING", "PR", "DONE", "BLOCKED"]
    VALID_TYPES = ["infrastructure", "feature", "documentation", "bug", "security", "performance"]
    VALID_RISK_TIERS = ["low", "medium", "high", "critical"]
    
    def __init__(self, backlog_file: str = "backlog.yml"):
        """Initialize backlog manager with file path"""
        self.backlog_file = Path(backlog_file)
        self.items: List[BacklogItem] = []
        self.metrics: Optional[BacklogMetrics] = None
        
        if self.backlog_file.exists():
            self.load_backlog()
    
    def load_backlog(self) -> None:
        """Load backlog from YAML file"""
        try:
            with open(self.backlog_file, 'r') as file:
                data = yaml.safe_load(file)
                
            if not data or 'items' not in data:
                logger.warning("No items found in backlog file")
                return
                
            # Load items
            self.items = []
            for item_data in data['items']:
                item = BacklogItem(**item_data)
                self.items.append(item)
            
            # Load metrics if present
            if 'metrics' in data:
                self.metrics = BacklogMetrics(**data['metrics'])
                
            logger.info(f"Loaded {len(self.items)} backlog items")
            
        except Exception as e:
            logger.error(f"Error loading backlog: {e}")
            raise
    
    def save_backlog(self) -> None:
        """Save backlog to YAML file"""
        try:
            # Update metrics before saving
            self._update_metrics()
            
            data = {
                'items': [asdict(item) for item in self.items],
                'metrics': asdict(self.metrics)
            }
            
            with open(self.backlog_file, 'w') as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Saved {len(self.items)} items to {self.backlog_file}")
            
        except Exception as e:
            logger.error(f"Error saving backlog: {e}")
            raise
    
    def add_item(self, 
                 title: str, 
                 description: str, 
                 item_type: str,
                 acceptance_criteria: List[str],
                 effort: int, 
                 value: int, 
                 time_criticality: int,
                 risk_reduction: int,
                 risk_tier: str = "medium") -> BacklogItem:
        """Add new item to backlog with automatic WSJF calculation"""
        
        # Validation
        if item_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid type: {item_type}")
        if risk_tier not in self.VALID_RISK_TIERS:
            raise ValueError(f"Invalid risk tier: {risk_tier}")
        if not (1 <= effort <= 13 and 1 <= value <= 13 and 
                1 <= time_criticality <= 13 and 1 <= risk_reduction <= 13):
            raise ValueError("All scoring values must be between 1-13 (Fibonacci scale)")
        
        # Generate ID
        item_id = self._generate_id(item_type)
        
        # Create item
        item = BacklogItem(
            id=item_id,
            title=title,
            type=item_type,
            description=description,
            acceptance_criteria=acceptance_criteria,
            effort=effort,
            value=value,
            time_criticality=time_criticality,
            risk_reduction=risk_reduction,
            wsjf_score=0.0,
            status="NEW",
            risk_tier=risk_tier,
            created_at=datetime.now(timezone.utc).isoformat(),
            links=[]
        )
        
        item.update_wsjf()
        self.items.append(item)
        
        logger.info(f"Added item {item_id}: {title} (WSJF: {item.wsjf_score})")
        return item
    
    def update_item_status(self, item_id: str, new_status: str) -> bool:
        """Update item status"""
        if new_status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {new_status}")
        
        for item in self.items:
            if item.id == item_id:
                old_status = item.status
                item.status = new_status
                logger.info(f"Updated {item_id} status: {old_status} → {new_status}")
                return True
        
        logger.warning(f"Item not found: {item_id}")
        return False
    
    def get_prioritized_items(self, status_filter: Optional[str] = None) -> List[BacklogItem]:
        """Get items sorted by WSJF score (highest first)"""
        items = self.items
        
        if status_filter:
            items = [item for item in items if item.status == status_filter]
        
        return sorted(items, key=lambda x: x.wsjf_score, reverse=True)
    
    def get_next_work_items(self, limit: int = 5) -> List[BacklogItem]:
        """Get next items ready for work (READY status, highest WSJF)"""
        ready_items = self.get_prioritized_items("READY")
        return ready_items[:limit]
    
    def discover_todo_items(self, source_dir: str = "src") -> List[Dict[str, str]]:
        """Discover TODO/FIXME comments in codebase"""
        todo_patterns = ["TODO", "FIXME", "BUG", "HACK", "XXX"]
        discovered_items = []
        
        source_path = Path(source_dir)
        if not source_path.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return []
        
        for file_path in source_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in ['.py', '.js', '.ts', '.java', '.go']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line_content = line.strip()
                        for pattern in todo_patterns:
                            if pattern in line_content.upper():
                                discovered_items.append({
                                    'file': str(file_path),
                                    'line': line_num,
                                    'pattern': pattern,
                                    'content': line_content,
                                    'suggested_title': f"Address {pattern.lower()} in {file_path.name}",
                                    'suggested_type': 'bug' if pattern in ['BUG', 'FIXME'] else 'feature'
                                })
                
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        logger.info(f"Discovered {len(discovered_items)} TODO items")
        return discovered_items
    
    def export_metrics(self) -> Dict:
        """Export comprehensive backlog metrics"""
        self._update_metrics()
        
        status_distribution = {}
        type_distribution = {}
        risk_distribution = {}
        wsjf_distribution = {"high": 0, "medium": 0, "low": 0}
        
        for item in self.items:
            # Status distribution
            status_distribution[item.status] = status_distribution.get(item.status, 0) + 1
            
            # Type distribution
            type_distribution[item.type] = type_distribution.get(item.type, 0) + 1
            
            # Risk distribution
            risk_distribution[item.risk_tier] = risk_distribution.get(item.risk_tier, 0) + 1
            
            # WSJF distribution
            if item.wsjf_score >= 5:
                wsjf_distribution["high"] += 1
            elif item.wsjf_score >= 2:
                wsjf_distribution["medium"] += 1
            else:
                wsjf_distribution["low"] += 1
        
        return {
            'summary': asdict(self.metrics),
            'distributions': {
                'status': status_distribution,
                'type': type_distribution,
                'risk': risk_distribution,
                'wsjf': wsjf_distribution
            },
            'top_priority_items': [
                {'id': item.id, 'title': item.title, 'wsjf_score': item.wsjf_score}
                for item in self.get_prioritized_items()[:5]
            ]
        }
    
    def _generate_id(self, item_type: str) -> str:
        """Generate unique ID for new item"""
        type_prefixes = {
            "infrastructure": "INFRA",
            "feature": "FEAT",
            "documentation": "DOC",
            "bug": "BUG",
            "security": "SEC",
            "performance": "PERF"
        }
        
        prefix = type_prefixes.get(item_type, "ITEM")
        
        # Find highest existing number for this prefix
        max_num = 0
        for item in self.items:
            if item.id.startswith(prefix):
                try:
                    num = int(item.id.split('-')[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    continue
        
        return f"{prefix}-{max_num + 1:03d}"
    
    def _update_metrics(self) -> None:
        """Update backlog metrics"""
        status_counts = {status: 0 for status in self.VALID_STATUSES}
        high_risk_count = 0
        total_wsjf = 0
        
        for item in self.items:
            status_counts[item.status] += 1
            if item.risk_tier in ["high", "critical"]:
                high_risk_count += 1
            total_wsjf += item.wsjf_score
        
        avg_wsjf = round(total_wsjf / len(self.items), 2) if self.items else 0.0
        
        self.metrics = BacklogMetrics(
            last_updated=datetime.now(timezone.utc).isoformat(),
            total_items=len(self.items),
            by_status=status_counts,
            avg_wsjf_score=avg_wsjf,
            high_risk_items=high_risk_count
        )

def main():
    """CLI interface for backlog management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Backlog Manager")
    parser.add_argument("--file", default="backlog.yml", help="Backlog file path")
    parser.add_argument("--discover", action="store_true", help="Discover TODO items")
    parser.add_argument("--metrics", action="store_true", help="Show backlog metrics")
    parser.add_argument("--next", type=int, metavar="N", help="Show next N work items")
    
    args = parser.parse_args()
    
    manager = BacklogManager(args.file)
    
    if args.discover:
        todos = manager.discover_todo_items()
        print(f"Found {len(todos)} TODO items:")
        for todo in todos[:10]:  # Show first 10
            print(f"  {todo['file']}:{todo['line']} - {todo['content']}")
    
    elif args.metrics:
        metrics = manager.export_metrics()
        print(json.dumps(metrics, indent=2))
    
    elif args.next:
        items = manager.get_next_work_items(args.next)
        print(f"Next {len(items)} work items:")
        for item in items:
            print(f"  [{item.id}] {item.title} (WSJF: {item.wsjf_score})")
    
    else:
        print("Use --help for available commands")

if __name__ == "__main__":
    main()