#!/usr/bin/env python3
"""
Comprehensive Logging and Error Handling Framework
Structured logging, error tracking, and monitoring for SDLC processes
"""

import logging
import logging.handlers
import json
import traceback
import sys
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import functools
import inspect
from contextlib import contextmanager

class LogLevel(Enum):
    """Enhanced log levels with custom levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60

class EventType(Enum):
    """Types of events for structured logging"""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_EVENT = "performance_event"
    ERROR_EVENT = "error_event"
    AUDIT_EVENT = "audit_event"
    BUSINESS_EVENT = "business_event"

@dataclass
class LogContext:
    """Context information for structured logging"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorInfo:
    """Structured error information"""
    error_id: str
    error_type: str
    error_message: str
    error_code: Optional[str] = None
    severity: LogLevel = LogLevel.ERROR
    component: Optional[str] = None
    function: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def __init__(self, include_trace_info: bool = True):
        super().__init__()
        self.include_trace_info = include_trace_info
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Basic log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
        }
        
        # Add trace information if enabled
        if self.include_trace_info:
            log_data.update({
                'file': record.pathname,
                'line': record.lineno,
                'process': record.process,
                'thread': record.thread,
                'thread_name': record.threadName
            })
        
        # Add structured context if available
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'trace_id'):
            log_data['trace_id'] = record.trace_id
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data, default=str, separators=(',', ':'))

class SDLCLogger:
    """Enhanced logger for SDLC processes with structured logging"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize SDLC logger with configuration"""
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self.context = LogContext()
        self.error_handlers: List[Callable[[ErrorInfo], None]] = []
        self._setup_logger()
        
        # Add custom log level
        logging.addLevelName(LogLevel.TRACE.value, 'TRACE')
        logging.addLevelName(LogLevel.SECURITY.value, 'SECURITY')
    
    def _setup_logger(self) -> None:
        """Setup logger with handlers and formatters"""
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = self.config.get('level', 'INFO')
        self.logger.setLevel(getattr(LogLevel, log_level.upper()).value)
        
        # Console handler
        if self.config.get('console_enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            
            if self.config.get('json_format', False):
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_format = self.config.get(
                    'console_format',
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(logging.Formatter(console_format))
            
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('file_enabled', False):
            log_dir = Path(self.config.get('log_dir', 'logs'))
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"{self.name}.log"
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_bytes', 10 * 1024 * 1024),  # 10MB
                backupCount=self.config.get('backup_count', 5)
            )
            
            file_handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(file_handler)
        
        # Syslog handler
        if self.config.get('syslog_enabled', False):
            try:
                syslog_handler = logging.handlers.SysLogHandler(
                    address=self.config.get('syslog_address', '/dev/log')
                )
                syslog_handler.setFormatter(StructuredFormatter(include_trace_info=False))
                self.logger.addHandler(syslog_handler)
            except Exception as e:
                self.logger.warning(f"Failed to setup syslog handler: {e}")
    
    def set_context(self, **kwargs) -> None:
        """Set logging context"""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
            else:
                self.context.metadata[key] = value
    
    def clear_context(self) -> None:
        """Clear logging context"""
        self.context = LogContext()
    
    @contextmanager
    def context_scope(self, **kwargs):
        """Context manager for temporary logging context"""
        original_context = self.context
        try:
            # Create new context with updates
            new_context = LogContext(
                user_id=kwargs.get('user_id', original_context.user_id),
                session_id=kwargs.get('session_id', original_context.session_id),
                request_id=kwargs.get('request_id', original_context.request_id),
                trace_id=kwargs.get('trace_id', original_context.trace_id),
                component=kwargs.get('component', original_context.component),
                operation=kwargs.get('operation', original_context.operation),
                metadata={**original_context.metadata, **kwargs.get('metadata', {})}
            )
            self.context = new_context
            yield self
        finally:
            self.context = original_context
    
    def _log_with_context(self, level: int, msg: str, event_type: Optional[EventType] = None,
                         exc_info: Optional[tuple] = None, **kwargs) -> None:
        """Log message with context information"""
        
        extra = {
            'context': self.context.metadata,
            'user_id': self.context.user_id,
            'session_id': self.context.session_id,
            'request_id': self.context.request_id,
            'trace_id': self.context.trace_id,
        }
        
        if event_type:
            extra['event_type'] = event_type.value
        
        # Add any additional context
        extra.update(kwargs)
        
        self.logger.log(level, msg, exc_info=exc_info, extra=extra)
    
    def trace(self, msg: str, **kwargs) -> None:
        """Log trace message"""
        self._log_with_context(LogLevel.TRACE.value, msg, **kwargs)
    
    def debug(self, msg: str, **kwargs) -> None:
        """Log debug message"""
        self._log_with_context(LogLevel.DEBUG.value, msg, **kwargs)
    
    def info(self, msg: str, event_type: Optional[EventType] = None, **kwargs) -> None:
        """Log info message"""
        self._log_with_context(LogLevel.INFO.value, msg, event_type, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Log warning message"""
        self._log_with_context(LogLevel.WARNING.value, msg, **kwargs)
    
    def error(self, msg: str, exc_info: Optional[tuple] = None, **kwargs) -> None:
        """Log error message"""
        self._log_with_context(LogLevel.ERROR.value, msg, EventType.ERROR_EVENT,
                              exc_info=exc_info, **kwargs)
    
    def critical(self, msg: str, exc_info: Optional[tuple] = None, **kwargs) -> None:
        """Log critical message"""
        self._log_with_context(LogLevel.CRITICAL.value, msg, EventType.ERROR_EVENT,
                              exc_info=exc_info, **kwargs)
    
    def security(self, msg: str, **kwargs) -> None:
        """Log security event"""
        self._log_with_context(LogLevel.SECURITY.value, msg, EventType.SECURITY_EVENT, **kwargs)
    
    def audit(self, action: str, resource: str, result: str, **kwargs) -> None:
        """Log audit event"""
        audit_msg = f"Action: {action}, Resource: {resource}, Result: {result}"
        self._log_with_context(LogLevel.INFO.value, audit_msg, EventType.AUDIT_EVENT, **kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log performance event"""
        perf_msg = f"Operation: {operation}, Duration: {duration:.3f}s"
        extra_context = {'operation': operation, 'duration': duration, **kwargs}
        self._log_with_context(LogLevel.INFO.value, perf_msg, EventType.PERFORMANCE_EVENT,
                              **extra_context)
    
    def user_action(self, user_id: str, action: str, **kwargs) -> None:
        """Log user action"""
        action_msg = f"User {user_id} performed action: {action}"
        extra_context = {'user_id': user_id, 'action': action, **kwargs}
        self._log_with_context(LogLevel.INFO.value, action_msg, EventType.USER_ACTION,
                              **extra_context)
    
    def business_event(self, event: str, **kwargs) -> None:
        """Log business event"""
        self._log_with_context(LogLevel.INFO.value, event, EventType.BUSINESS_EVENT, **kwargs)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                  error_code: Optional[str] = None) -> ErrorInfo:
        """Log structured error information"""
        
        # Get caller information
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Create error info
        error_info = ErrorInfo(
            error_id=f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            error_code=error_code,
            component=self.context.component,
            function=function_name,
            file_path=file_path,
            line_number=line_number,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Log the error
        self.error(
            f"Error {error_info.error_id}: {error_info.error_message}",
            exc_info=(type(error), error, error.__traceback__),
            error_id=error_info.error_id,
            error_code=error_code
        )
        
        # Notify error handlers
        for handler in self.error_handlers:
            try:
                handler(error_info)
            except Exception as handler_error:
                self.warning(f"Error handler failed: {handler_error}")
        
        return error_info
    
    def add_error_handler(self, handler: Callable[[ErrorInfo], None]) -> None:
        """Add error handler callback"""
        self.error_handlers.append(handler)

class LoggingDecorator:
    """Decorator for automatic logging of function calls"""
    
    def __init__(self, logger: SDLCLogger, log_entry: bool = True, log_exit: bool = True,
                 log_args: bool = False, log_result: bool = False, log_errors: bool = True):
        self.logger = logger
        self.log_entry = log_entry
        self.log_exit = log_exit
        self.log_args = log_args
        self.log_result = log_result
        self.log_errors = log_errors
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            if self.log_entry:
                entry_msg = f"Entering {func_name}"
                if self.log_args:
                    entry_msg += f" with args={args}, kwargs={kwargs}"
                self.logger.trace(entry_msg, function=func_name)
            
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                
                # Log function exit
                if self.log_exit:
                    duration = (datetime.now() - start_time).total_seconds()
                    exit_msg = f"Exiting {func_name} (duration: {duration:.3f}s)"
                    if self.log_result:
                        exit_msg += f" with result={result}"
                    self.logger.trace(exit_msg, function=func_name, duration=duration)
                
                return result
                
            except Exception as e:
                if self.log_errors:
                    duration = (datetime.now() - start_time).total_seconds()
                    self.logger.log_error(
                        e, 
                        context={
                            'function': func_name,
                            'args': args if self.log_args else None,
                            'kwargs': kwargs if self.log_args else None,
                            'duration': duration
                        }
                    )
                raise
        
        return wrapper

class PerformanceProfiler:
    """Performance profiling and logging"""
    
    def __init__(self, logger: SDLCLogger):
        self.logger = logger
        self._active_timers: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def profile_operation(self, operation_name: str, **context):
        """Context manager for profiling operations"""
        start_time = datetime.now()
        
        try:
            self.logger.trace(f"Starting operation: {operation_name}", **context)
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.performance(operation_name, duration, **context)
    
    def start_timer(self, timer_name: str) -> None:
        """Start a named timer"""
        with self._lock:
            self._active_timers[timer_name] = datetime.now()
            self.logger.trace(f"Started timer: {timer_name}")
    
    def stop_timer(self, timer_name: str, **context) -> float:
        """Stop a named timer and return duration"""
        with self._lock:
            if timer_name not in self._active_timers:
                self.logger.warning(f"Timer {timer_name} not found")
                return 0.0
            
            start_time = self._active_timers.pop(timer_name)
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.performance(f"Timer: {timer_name}", duration, **context)
            return duration

class ErrorRecovery:
    """Error recovery and retry mechanisms"""
    
    def __init__(self, logger: SDLCLogger):
        self.logger = logger
    
    def retry_with_backoff(self, func: Callable, max_retries: int = 3,
                          backoff_factor: float = 1.0, 
                          exception_types: tuple = (Exception,)) -> Callable:
        """Decorator for retry with exponential backoff"""
        
        def decorator(original_func):
            @functools.wraps(original_func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        if attempt > 0:
                            # Wait with exponential backoff
                            import time
                            wait_time = backoff_factor * (2 ** (attempt - 1))
                            self.logger.info(f"Retrying {original_func.__name__} in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                            time.sleep(wait_time)
                        
                        result = original_func(*args, **kwargs)
                        
                        if attempt > 0:
                            self.logger.info(f"Successfully recovered {original_func.__name__} on attempt {attempt + 1}")
                        
                        return result
                        
                    except exception_types as e:
                        last_exception = e
                        
                        if attempt < max_retries:
                            self.logger.warning(f"Attempt {attempt + 1} failed for {original_func.__name__}: {e}")
                        else:
                            self.logger.error(f"All {max_retries + 1} attempts failed for {original_func.__name__}")
                
                # All retries exhausted
                if last_exception:
                    raise last_exception
            
            return wrapper
        
        return decorator(func) if func else decorator

class LoggingConfig:
    """Configuration management for logging framework"""
    
    DEFAULT_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'structured': {
                '()': StructuredFormatter,
                'include_trace_info': True
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'structured',
                'filename': 'logs/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5
            }
        },
        'loggers': {
            'sdlc': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load logging configuration from file or return default"""
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.json'):
                        return json.load(f)
                    elif config_path.endswith('.yml') or config_path.endswith('.yaml'):
                        import yaml
                        return yaml.safe_load(f)
            except Exception as e:
                print(f"Failed to load config from {config_path}: {e}")
        
        return cls.DEFAULT_CONFIG
    
    @classmethod
    def setup_logging(cls, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup logging with configuration"""
        
        if config is None:
            config = cls.DEFAULT_CONFIG
        
        # Ensure log directory exists
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging configuration
        import logging.config
        logging.config.dictConfig(config)

# Global logger instances
_loggers: Dict[str, SDLCLogger] = {}
_lock = threading.Lock()

def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> SDLCLogger:
    """Get or create logger instance"""
    
    with _lock:
        if name not in _loggers:
            _loggers[name] = SDLCLogger(name, config)
        return _loggers[name]

def configure_logging(config_path: Optional[str] = None) -> None:
    """Configure global logging settings"""
    
    config = LoggingConfig.load_config(config_path)
    LoggingConfig.setup_logging(config)

# Convenience decorators
def log_function(logger_name: str = 'sdlc', **decorator_kwargs):
    """Decorator to automatically log function calls"""
    def decorator(func):
        logger = get_logger(logger_name)
        return LoggingDecorator(logger, **decorator_kwargs)(func)
    return decorator

def profile_performance(logger_name: str = 'sdlc'):
    """Decorator to profile function performance"""
    def decorator(func):
        logger = get_logger(logger_name)
        profiler = PerformanceProfiler(logger)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile_operation(func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_on_error(max_retries: int = 3, backoff_factor: float = 1.0, 
                  exception_types: tuple = (Exception,), logger_name: str = 'sdlc'):
    """Decorator to retry function calls on error"""
    def decorator(func):
        logger = get_logger(logger_name)
        recovery = ErrorRecovery(logger)
        return recovery.retry_with_backoff(func, max_retries, backoff_factor, exception_types)
    return decorator

def main():
    """CLI interface for logging framework testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SDLC Logging Framework Test")
    parser.add_argument("--config", help="Logging configuration file")
    parser.add_argument("--test-logging", action="store_true", help="Test logging functionality")
    parser.add_argument("--test-errors", action="store_true", help="Test error handling")
    parser.add_argument("--logger-name", default="test", help="Logger name")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.config:
        configure_logging(args.config)
    
    # Get logger
    logger = get_logger(args.logger_name)
    logger.set_context(component="logging_test", operation="test_run")
    
    if args.test_logging:
        print("Testing logging functionality...")
        
        logger.trace("This is a trace message")
        logger.debug("This is a debug message")
        logger.info("This is an info message", event_type=EventType.SYSTEM_EVENT)
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
        logger.security("This is a security event")
        
        logger.audit("LOGIN", "user_system", "SUCCESS", user_id="test_user")
        logger.performance("database_query", 0.125, query="SELECT * FROM users")
        logger.user_action("user123", "CREATE_DOCUMENT", document_id="doc456")
        logger.business_event("ORDER_COMPLETED", order_id="ORD789", amount=99.99)
        
        # Test context scope
        with logger.context_scope(user_id="ctx_user", operation="context_test"):
            logger.info("Message with temporary context")
        
        logger.info("Message after context scope")
    
    if args.test_errors:
        print("Testing error handling...")
        
        try:
            # Simulate an error
            raise ValueError("Test error for demonstration")
        except Exception as e:
            error_info = logger.log_error(e, context={"test_data": "value"}, error_code="TEST001")
            print(f"Logged error: {error_info.error_id}")
        
        # Test retry decorator
        @retry_on_error(max_retries=2, logger_name=args.logger_name)
        def failing_function():
            import random
            if random.random() < 0.7:  # 70% chance of failure
                raise ConnectionError("Random connection error")
            return "Success!"
        
        try:
            result = failing_function()
            print(f"Function result: {result}")
        except Exception as e:
            print(f"Function finally failed: {e}")

if __name__ == "__main__":
    main()