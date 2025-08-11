#!/usr/bin/env python3
"""
Comprehensive Monitoring and Observability Framework
Real-time monitoring, metrics collection, and alerting for SDLC processes
"""

import time
import threading
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import our logging framework
from logging_framework import get_logger, EventType

logger = get_logger('monitoring')

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Individual metric measurement"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: Optional[str] = None
    help_text: Optional[str] = None

@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    condition: str
    severity: AlertSeverity
    message: str
    enabled: bool = True
    cooldown_minutes: int = 15
    labels: Dict[str, str] = field(default_factory=dict)
    last_triggered: Optional[float] = None
    trigger_count: int = 0

@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    timeout_seconds: int = 30
    interval_seconds: int = 60
    enabled: bool = True
    failure_threshold: int = 3
    current_failures: int = 0
    last_check: Optional[float] = None
    last_success: Optional[float] = None

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_definitions: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        self.collection_enabled = True
        
    def register_metric(self, name: str, metric_type: MetricType, 
                       labels: Optional[Dict[str, str]] = None,
                       unit: Optional[str] = None, help_text: Optional[str] = None) -> None:
        """Register a new metric"""
        
        with self._lock:
            self.metric_definitions[name] = Metric(
                name=name,
                value=0,
                metric_type=metric_type,
                labels=labels or {},
                unit=unit,
                help_text=help_text
            )
    
    def increment_counter(self, name: str, value: Union[int, float] = 1,
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        self._record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: Union[int, float],
                  labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        self._record_metric(name, value, MetricType.GAUGE, labels)
    
    def observe_histogram(self, name: str, value: Union[int, float],
                         labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation"""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    def observe_summary(self, name: str, value: Union[int, float],
                       labels: Optional[Dict[str, str]] = None) -> None:
        """Record a summary observation"""
        self._record_metric(name, value, MetricType.SUMMARY, labels)
    
    def _record_metric(self, name: str, value: Union[int, float], 
                      metric_type: MetricType, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric measurement"""
        
        if not self.collection_enabled:
            return
        
        with self._lock:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                labels=labels or {},
                timestamp=time.time()
            )
            
            metric_key = self._get_metric_key(name, labels)
            self.metrics[metric_key].append(metric)
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with labels"""
        if not labels:
            return name
        
        sorted_labels = sorted(labels.items())
        label_string = ",".join(f"{k}={v}" for k, v in sorted_labels)
        return f"{name}[{label_string}]"
    
    def get_current_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Union[int, float]]:
        """Get current value of a metric"""
        
        with self._lock:
            metric_key = self._get_metric_key(name, labels)
            if metric_key in self.metrics and self.metrics[metric_key]:
                return self.metrics[metric_key][-1].value
        return None
    
    def get_metric_history(self, name: str, labels: Optional[Dict[str, str]] = None,
                          duration_minutes: int = 60) -> List[Metric]:
        """Get metric history for specified duration"""
        
        cutoff_time = time.time() - (duration_minutes * 60)
        metric_key = self._get_metric_key(name, labels)
        
        with self._lock:
            if metric_key not in self.metrics:
                return []
            
            return [m for m in self.metrics[metric_key] if m.timestamp > cutoff_time]
    
    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all current metrics"""
        
        with self._lock:
            return {key: list(deque_val) for key, deque_val in self.metrics.items()}
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        
        lines = []
        
        with self._lock:
            for metric_key, metric_history in self.metrics.items():
                if not metric_history:
                    continue
                
                latest_metric = metric_history[-1]
                
                # Help text
                if latest_metric.help_text:
                    lines.append(f"# HELP {latest_metric.name} {latest_metric.help_text}")
                
                # Type
                type_map = {
                    MetricType.COUNTER: "counter",
                    MetricType.GAUGE: "gauge",
                    MetricType.HISTOGRAM: "histogram", 
                    MetricType.SUMMARY: "summary"
                }
                lines.append(f"# TYPE {latest_metric.name} {type_map[latest_metric.metric_type]}")
                
                # Value with labels
                if latest_metric.labels:
                    label_string = ",".join(f'{k}="{v}"' for k, v in latest_metric.labels.items())
                    lines.append(f"{latest_metric.name}{{{label_string}}} {latest_metric.value}")
                else:
                    lines.append(f"{latest_metric.name} {latest_metric.value}")
        
        return "\n".join(lines)

class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_enabled = True
        self._monitor_thread = None
        self._setup_system_metrics()
    
    def _setup_system_metrics(self) -> None:
        """Register system metrics"""
        
        system_metrics = [
            ("system_cpu_usage_percent", MetricType.GAUGE, None, "%", "CPU usage percentage"),
            ("system_memory_usage_bytes", MetricType.GAUGE, None, "bytes", "Memory usage in bytes"),
            ("system_memory_usage_percent", MetricType.GAUGE, None, "%", "Memory usage percentage"),
            ("system_disk_usage_bytes", MetricType.GAUGE, None, "bytes", "Disk usage in bytes"),
            ("system_disk_usage_percent", MetricType.GAUGE, None, "%", "Disk usage percentage"),
            ("system_network_bytes_sent", MetricType.COUNTER, None, "bytes", "Network bytes sent"),
            ("system_network_bytes_recv", MetricType.COUNTER, None, "bytes", "Network bytes received"),
            ("system_load_average_1m", MetricType.GAUGE, None, "load", "1-minute load average"),
            ("system_uptime_seconds", MetricType.GAUGE, None, "seconds", "System uptime"),
        ]
        
        for name, metric_type, labels, unit, help_text in system_metrics:
            self.metrics_collector.register_metric(name, metric_type, labels, unit, help_text)
    
    def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start system monitoring in background thread"""
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self.monitoring_enabled = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Started system monitoring (interval: {interval_seconds}s)")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        
        self.monitoring_enabled = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Stopped system monitoring")
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop"""
        
        while self.monitoring_enabled:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_usage_bytes", memory.used)
            self.metrics_collector.set_gauge("system_memory_usage_percent", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_usage_bytes", disk.used)
            self.metrics_collector.set_gauge("system_disk_usage_percent", (disk.used / disk.total) * 100)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics_collector.set_gauge("system_network_bytes_sent", net_io.bytes_sent)
            self.metrics_collector.set_gauge("system_network_bytes_recv", net_io.bytes_recv)
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0]
                self.metrics_collector.set_gauge("system_load_average_1m", load_avg)
            except AttributeError:
                pass  # Windows doesn't have load average
            
            # Uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            self.metrics_collector.set_gauge("system_uptime_seconds", uptime)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

class ApplicationMonitor:
    """Monitors application-specific metrics and events"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.start_time = time.time()
        self._setup_application_metrics()
    
    def _setup_application_metrics(self) -> None:
        """Register application metrics"""
        
        app_metrics = [
            ("app_requests_total", MetricType.COUNTER, None, "requests", "Total HTTP requests"),
            ("app_request_duration_seconds", MetricType.HISTOGRAM, None, "seconds", "HTTP request duration"),
            ("app_errors_total", MetricType.COUNTER, None, "errors", "Total application errors"),
            ("app_active_connections", MetricType.GAUGE, None, "connections", "Active connections"),
            ("app_database_connections", MetricType.GAUGE, None, "connections", "Database connections"),
            ("app_cache_hits", MetricType.COUNTER, None, "hits", "Cache hits"),
            ("app_cache_misses", MetricType.COUNTER, None, "misses", "Cache misses"),
            ("app_queue_size", MetricType.GAUGE, None, "items", "Queue size"),
            ("app_background_tasks", MetricType.GAUGE, None, "tasks", "Background tasks"),
            ("app_uptime_seconds", MetricType.GAUGE, None, "seconds", "Application uptime"),
        ]
        
        for name, metric_type, labels, unit, help_text in app_metrics:
            self.metrics_collector.register_metric(name, metric_type, labels, unit, help_text)
    
    def record_request(self, method: str, path: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics"""
        
        labels = {
            "method": method,
            "path": path,
            "status": str(status_code)
        }
        
        self.metrics_collector.increment_counter("app_requests_total", labels=labels)
        self.metrics_collector.observe_histogram("app_request_duration_seconds", duration, labels=labels)
        
        if status_code >= 400:
            self.metrics_collector.increment_counter("app_errors_total", labels={"status": str(status_code)})
    
    def record_database_operation(self, operation: str, duration: float, success: bool) -> None:
        """Record database operation metrics"""
        
        labels = {
            "operation": operation,
            "status": "success" if success else "error"
        }
        
        self.metrics_collector.observe_histogram("db_operation_duration_seconds", duration, labels=labels)
        self.metrics_collector.increment_counter("db_operations_total", labels=labels)
    
    def record_cache_operation(self, hit: bool) -> None:
        """Record cache operation"""
        
        if hit:
            self.metrics_collector.increment_counter("app_cache_hits")
        else:
            self.metrics_collector.increment_counter("app_cache_misses")
    
    def set_active_connections(self, count: int) -> None:
        """Set active connection count"""
        self.metrics_collector.set_gauge("app_active_connections", count)
    
    def set_queue_size(self, size: int, queue_name: str = "default") -> None:
        """Set queue size"""
        self.metrics_collector.set_gauge("app_queue_size", size, labels={"queue": queue_name})
    
    def update_uptime(self) -> None:
        """Update application uptime"""
        uptime = time.time() - self.start_time
        self.metrics_collector.set_gauge("app_uptime_seconds", uptime)

class AlertManager:
    """Manages alerts based on metric conditions"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert, float], None]] = []
        self.evaluation_enabled = True
        self._evaluation_thread = None
    
    def register_alert(self, alert: Alert) -> None:
        """Register a new alert"""
        self.alerts[alert.name] = alert
        logger.info(f"Registered alert: {alert.name}")
    
    def add_alert_handler(self, handler: Callable[[Alert, float], None]) -> None:
        """Add alert handler callback"""
        self.alert_handlers.append(handler)
    
    def start_evaluation(self, interval_seconds: int = 60) -> None:
        """Start alert evaluation in background thread"""
        
        if self._evaluation_thread and self._evaluation_thread.is_alive():
            return
        
        self.evaluation_enabled = True
        self._evaluation_thread = threading.Thread(
            target=self._evaluation_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._evaluation_thread.start()
        logger.info(f"Started alert evaluation (interval: {interval_seconds}s)")
    
    def stop_evaluation(self) -> None:
        """Stop alert evaluation"""
        self.evaluation_enabled = False
        if self._evaluation_thread:
            self._evaluation_thread.join(timeout=5)
        logger.info("Stopped alert evaluation")
    
    def _evaluation_loop(self, interval_seconds: int) -> None:
        """Main alert evaluation loop"""
        
        while self.evaluation_enabled:
            try:
                self._evaluate_alerts()
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error evaluating alerts: {e}")
                time.sleep(interval_seconds)
    
    def _evaluate_alerts(self) -> None:
        """Evaluate all alerts"""
        
        current_time = time.time()
        
        for alert_name, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            try:
                # Check cooldown
                if (alert.last_triggered and 
                    current_time - alert.last_triggered < alert.cooldown_minutes * 60):
                    continue
                
                # Evaluate condition
                if self._evaluate_condition(alert.condition):
                    self._trigger_alert(alert, current_time)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert {alert_name}: {e}")
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate alert condition"""
        
        # Simple condition evaluation - in practice, you'd want a more robust parser
        # Example conditions:
        # - "system_cpu_usage_percent > 80"
        # - "app_errors_total rate(5m) > 10"
        # - "system_memory_usage_percent > 90"
        
        try:
            # Parse simple comparison conditions
            for op in ['>=', '<=', '>', '<', '==', '!=']:
                if op in condition:
                    parts = condition.split(op)
                    if len(parts) == 2:
                        metric_name = parts[0].strip()
                        threshold = float(parts[1].strip())
                        
                        current_value = self.metrics_collector.get_current_value(metric_name)
                        if current_value is None:
                            return False
                        
                        if op == '>':
                            return current_value > threshold
                        elif op == '>=':
                            return current_value >= threshold
                        elif op == '<':
                            return current_value < threshold
                        elif op == '<=':
                            return current_value <= threshold
                        elif op == '==':
                            return current_value == threshold
                        elif op == '!=':
                            return current_value != threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _trigger_alert(self, alert: Alert, current_time: float) -> None:
        """Trigger an alert"""
        
        alert.last_triggered = current_time
        alert.trigger_count += 1
        
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.message}")
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert, current_time)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

class HealthCheckManager:
    """Manages application health checks"""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.overall_health = True
        self.checking_enabled = True
        self._check_thread = None
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def start_health_checks(self) -> None:
        """Start health check monitoring"""
        
        if self._check_thread and self._check_thread.is_alive():
            return
        
        self.checking_enabled = True
        self._check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._check_thread.start()
        logger.info("Started health check monitoring")
    
    def stop_health_checks(self) -> None:
        """Stop health check monitoring"""
        self.checking_enabled = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Stopped health check monitoring")
    
    def _health_check_loop(self) -> None:
        """Main health check loop"""
        
        while self.checking_enabled:
            try:
                self._run_health_checks()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(10)
    
    def _run_health_checks(self) -> None:
        """Run all enabled health checks"""
        
        current_time = time.time()
        overall_healthy = True
        
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            # Check if it's time to run this health check
            if (health_check.last_check and 
                current_time - health_check.last_check < health_check.interval_seconds):
                continue
            
            try:
                # Run health check with timeout
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(health_check.check_function)
                
                try:
                    is_healthy = future.result(timeout=health_check.timeout_seconds)
                    health_check.last_check = current_time
                    
                    if is_healthy:
                        health_check.current_failures = 0
                        health_check.last_success = current_time
                    else:
                        health_check.current_failures += 1
                        logger.warning(f"Health check failed: {name} ({health_check.current_failures}/{health_check.failure_threshold})")
                    
                except Exception as e:
                    health_check.current_failures += 1
                    logger.error(f"Health check error: {name} - {e}")
                
                finally:
                    executor.shutdown(wait=False)
                
                # Check if health check has failed too many times
                if health_check.current_failures >= health_check.failure_threshold:
                    overall_healthy = False
                    logger.error(f"Health check {name} has failed {health_check.current_failures} times")
                    
            except Exception as e:
                logger.error(f"Error running health check {name}: {e}")
                overall_healthy = False
        
        self.overall_health = overall_healthy
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        
        current_time = time.time()
        
        checks = {}
        for name, health_check in self.health_checks.items():
            checks[name] = {
                'enabled': health_check.enabled,
                'healthy': health_check.current_failures < health_check.failure_threshold,
                'current_failures': health_check.current_failures,
                'failure_threshold': health_check.failure_threshold,
                'last_check': health_check.last_check,
                'last_success': health_check.last_success,
                'time_since_last_check': current_time - health_check.last_check if health_check.last_check else None,
                'time_since_last_success': current_time - health_check.last_success if health_check.last_success else None
            }
        
        return {
            'overall_healthy': self.overall_health,
            'timestamp': current_time,
            'checks': checks
        }

class MonitoringFramework:
    """Main monitoring framework that coordinates all monitoring components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.app_monitor = ApplicationMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_check_manager = HealthCheckManager()
        
        self._setup_default_alerts()
        self._setup_default_health_checks()
        self._setup_alert_handlers()
    
    def _setup_default_alerts(self) -> None:
        """Setup default system alerts"""
        
        default_alerts = [
            Alert(
                name="high_cpu_usage",
                condition="system_cpu_usage_percent > 80",
                severity=AlertSeverity.WARNING,
                message="CPU usage is above 80%",
                cooldown_minutes=10
            ),
            Alert(
                name="high_memory_usage", 
                condition="system_memory_usage_percent > 85",
                severity=AlertSeverity.WARNING,
                message="Memory usage is above 85%",
                cooldown_minutes=15
            ),
            Alert(
                name="critical_memory_usage",
                condition="system_memory_usage_percent > 95",
                severity=AlertSeverity.CRITICAL,
                message="Memory usage is critically high (>95%)",
                cooldown_minutes=5
            ),
            Alert(
                name="high_disk_usage",
                condition="system_disk_usage_percent > 80",
                severity=AlertSeverity.WARNING,
                message="Disk usage is above 80%",
                cooldown_minutes=30
            ),
            Alert(
                name="high_error_rate",
                condition="app_errors_total > 10",
                severity=AlertSeverity.ERROR,
                message="Application error rate is high",
                cooldown_minutes=5
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.register_alert(alert)
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks"""
        
        def check_disk_space() -> bool:
            """Check if disk space is available"""
            disk_usage = psutil.disk_usage('/')
            return (disk_usage.used / disk_usage.total) < 0.95
        
        def check_memory() -> bool:
            """Check if memory usage is reasonable"""
            memory = psutil.virtual_memory()
            return memory.percent < 95
        
        def check_process_running() -> bool:
            """Check if current process is responsive"""
            return True  # If we can execute this, process is running
        
        health_checks = [
            HealthCheck(
                name="disk_space",
                check_function=check_disk_space,
                interval_seconds=300,  # Check every 5 minutes
                failure_threshold=2
            ),
            HealthCheck(
                name="memory_usage", 
                check_function=check_memory,
                interval_seconds=60,   # Check every minute
                failure_threshold=3
            ),
            HealthCheck(
                name="process_health",
                check_function=check_process_running,
                interval_seconds=30,   # Check every 30 seconds
                failure_threshold=1
            )
        ]
        
        for health_check in health_checks:
            self.health_check_manager.register_health_check(health_check)
    
    def _setup_alert_handlers(self) -> None:
        """Setup alert notification handlers"""
        
        def log_alert_handler(alert: Alert, trigger_time: float) -> None:
            """Log alert to application logger"""
            logger.warning(
                f"Alert triggered: {alert.name}",
                event_type=EventType.SYSTEM_EVENT,
                alert_name=alert.name,
                alert_severity=alert.severity.value,
                alert_message=alert.message,
                trigger_count=alert.trigger_count
            )
        
        # Could add more handlers: email, Slack, PagerDuty, etc.
        self.alert_manager.add_alert_handler(log_alert_handler)
    
    def start_monitoring(self) -> None:
        """Start all monitoring components"""
        
        logger.info("Starting monitoring framework")
        
        # Start system monitoring
        system_interval = self.config.get('system_monitor_interval', 30)
        self.system_monitor.start_monitoring(system_interval)
        
        # Start alert evaluation
        alert_interval = self.config.get('alert_evaluation_interval', 60)
        self.alert_manager.start_evaluation(alert_interval)
        
        # Start health checks
        self.health_check_manager.start_health_checks()
        
        logger.info("Monitoring framework started successfully")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components"""
        
        logger.info("Stopping monitoring framework")
        
        self.system_monitor.stop_monitoring()
        self.alert_manager.stop_evaluation()
        self.health_check_manager.stop_health_checks()
        
        logger.info("Monitoring framework stopped")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        
        return {
            'timestamp': time.time(),
            'system': {
                'cpu_percent': self.metrics_collector.get_current_value('system_cpu_usage_percent'),
                'memory_percent': self.metrics_collector.get_current_value('system_memory_usage_percent'),
                'disk_percent': self.metrics_collector.get_current_value('system_disk_usage_percent'),
                'uptime_seconds': self.metrics_collector.get_current_value('system_uptime_seconds')
            },
            'application': {
                'uptime_seconds': self.metrics_collector.get_current_value('app_uptime_seconds'),
                'total_requests': self.metrics_collector.get_current_value('app_requests_total'),
                'total_errors': self.metrics_collector.get_current_value('app_errors_total'),
                'active_connections': self.metrics_collector.get_current_value('app_active_connections')
            },
            'health': self.health_check_manager.get_health_status()
        }
    
    def export_metrics(self, format: str = 'prometheus') -> str:
        """Export metrics in specified format"""
        
        if format.lower() == 'prometheus':
            return self.metrics_collector.export_prometheus_format()
        elif format.lower() == 'json':
            return json.dumps(self.metrics_collector.get_all_metrics(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Context managers and decorators for easy monitoring
class MonitoredOperation:
    """Context manager for monitoring operations"""
    
    def __init__(self, name: str, monitoring_framework: MonitoringFramework,
                 labels: Optional[Dict[str, str]] = None):
        self.name = name
        self.monitoring = monitoring_framework
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.monitoring.metrics_collector.increment_counter(
            f"{self.name}_started", labels=self.labels
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        success_labels = {**self.labels, 'status': 'success' if exc_type is None else 'error'}
        
        self.monitoring.metrics_collector.increment_counter(
            f"{self.name}_completed", labels=success_labels
        )
        self.monitoring.metrics_collector.observe_histogram(
            f"{self.name}_duration_seconds", duration, labels=success_labels
        )

def monitored_function(operation_name: str, monitoring_framework: Optional[MonitoringFramework] = None):
    """Decorator for automatic function monitoring"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get monitoring framework from global state if not provided
            monitoring = monitoring_framework or getattr(wrapper, '_monitoring', None)
            
            if monitoring:
                with MonitoredOperation(operation_name, monitoring) as monitor:
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        # Allow setting monitoring framework later
        wrapper._monitoring = monitoring_framework
        return wrapper
    
    return decorator

def main():
    """CLI interface for monitoring framework"""
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description="Monitoring Framework")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--export-format", choices=['prometheus', 'json'], default='prometheus',
                       help="Export format")
    parser.add_argument("--export-file", help="Export metrics to file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize monitoring
    monitoring = MonitoringFramework(config)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down monitoring...")
        monitoring.stop_monitoring()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start monitoring
    monitoring.start_monitoring()
    
    try:
        print(f"Monitoring started. Running for {args.duration} seconds...")
        print("Press Ctrl+C to stop early")
        
        # Run for specified duration
        time.sleep(args.duration)
        
        # Export metrics
        exported_metrics = monitoring.export_metrics(args.export_format)
        
        if args.export_file:
            with open(args.export_file, 'w') as f:
                f.write(exported_metrics)
            print(f"Metrics exported to {args.export_file}")
        else:
            print("\n" + "="*50)
            print("EXPORTED METRICS")
            print("="*50)
            print(exported_metrics)
        
        # Show summary
        print("\n" + "="*50)
        print("MONITORING SUMMARY")
        print("="*50)
        summary = monitoring.get_metrics_summary()
        print(json.dumps(summary, indent=2, default=str))
    
    finally:
        monitoring.stop_monitoring()

if __name__ == "__main__":
    main()