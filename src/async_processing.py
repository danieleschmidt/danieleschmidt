#!/usr/bin/env python3
"""
Async Processing Framework for SDLC Components
High-performance async processing with batch operations, connection pooling, and backpressure handling
"""

import asyncio
import time
import json
from typing import Any, Dict, List, Optional, Callable, Union, AsyncGenerator, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiofiles
import httpx
from collections import deque
import weakref

from logging_framework import get_logger
from caching_framework import get_async_cache

logger = get_logger('async_processing')

T = TypeVar('T')
R = TypeVar('R')


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AsyncTask:
    """Represents an async task with metadata"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: 'AsyncTask') -> bool:
        """Compare tasks by priority for queue ordering"""
        return self.priority.value > other.priority.value


@dataclass
class TaskResult:
    """Result of an async task execution"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retry_count: int = 0
    completed_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackpressureManager:
    """Manages backpressure to prevent resource exhaustion"""
    
    def __init__(self, 
                 max_pending_tasks: int = 1000,
                 max_memory_mb: int = 512,
                 check_interval: float = 1.0):
        self.max_pending_tasks = max_pending_tasks
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self._pending_count = 0
        self._semaphore = asyncio.Semaphore(max_pending_tasks)
        self._last_check = 0.0
    
    async def acquire(self) -> None:
        """Acquire permission to process task"""
        await self._semaphore.acquire()
        self._pending_count += 1
        
        # Periodic resource check
        current_time = time.time()
        if current_time - self._last_check > self.check_interval:
            await self._check_resources()
            self._last_check = current_time
    
    def release(self) -> None:
        """Release task processing permission"""
        self._pending_count = max(0, self._pending_count - 1)
        self._semaphore.release()
    
    async def _check_resources(self) -> None:
        """Check system resources and apply throttling if needed"""
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            
            if memory_usage > 85:  # High memory usage
                logger.warning(f"High memory usage: {memory_usage}%")
                await asyncio.sleep(0.1)  # Brief pause
                
        except ImportError:
            # psutil not available, skip resource check
            pass
    
    @property
    def is_under_pressure(self) -> bool:
        """Check if system is under backpressure"""
        return self._pending_count > self.max_pending_tasks * 0.8


class AsyncBatchProcessor(Generic[T, R]):
    """Process items in batches asynchronously"""
    
    def __init__(self,
                 batch_size: int = 100,
                 max_wait_time: float = 1.0,
                 max_concurrent_batches: int = 5):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        self._queue: asyncio.Queue[T] = asyncio.Queue()
        self._results: asyncio.Queue[R] = asyncio.Queue()
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
    
    async def add_item(self, item: T) -> None:
        """Add item to processing queue"""
        await self._queue.put(item)
        
        if not self._processing:
            await self.start_processing()
    
    async def add_items(self, items: List[T]) -> None:
        """Add multiple items to processing queue"""
        for item in items:
            await self._queue.put(item)
        
        if not self._processing:
            await self.start_processing()
    
    async def start_processing(self) -> None:
        """Start batch processing"""
        if self._processing:
            return
        
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_batches())
    
    async def stop_processing(self) -> None:
        """Stop batch processing"""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
    
    async def _process_batches(self) -> None:
        """Main batch processing loop"""
        while self._processing:
            try:
                batch = await self._collect_batch()
                if batch:
                    asyncio.create_task(self._process_batch(batch))
                else:
                    await asyncio.sleep(0.1)  # No items, brief pause
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_batch(self) -> List[T]:
        """Collect items for batch processing"""
        batch = []
        deadline = time.time() + self.max_wait_time
        
        while len(batch) < self.batch_size and time.time() < deadline:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=max(0.1, deadline - time.time())
                )
                batch.append(item)
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[T]) -> None:
        """Process a single batch"""
        async with self._batch_semaphore:
            try:
                # Override this method in subclasses
                results = await self.process_batch(batch)
                
                for result in results:
                    await self._results.put(result)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
    
    async def process_batch(self, batch: List[T]) -> List[R]:
        """Override this method to implement batch processing logic"""
        raise NotImplementedError("Subclasses must implement process_batch")
    
    async def get_result(self) -> Optional[R]:
        """Get processed result"""
        try:
            return await asyncio.wait_for(self._results.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def get_results(self, count: int) -> List[R]:
        """Get multiple processed results"""
        results = []
        for _ in range(count):
            result = await self.get_result()
            if result is not None:
                results.append(result)
            else:
                break
        return results


class AsyncTaskQueue:
    """Priority-based async task queue with dependency management"""
    
    def __init__(self, 
                 max_workers: int = 10,
                 enable_backpressure: bool = True):
        self.max_workers = max_workers
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_dependencies: Dict[str, List[str]] = {}
        self._waiting_tasks: Dict[str, AsyncTask] = {}
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        
        if enable_backpressure:
            self._backpressure = BackpressureManager()
        else:
            self._backpressure = None
        
        self._stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_retried': 0
        }
    
    async def submit_task(self, task: AsyncTask) -> None:
        """Submit task for execution"""
        self._stats['total_submitted'] += 1
        
        # Check dependencies
        if task.dependencies:
            unmet_deps = [
                dep for dep in task.dependencies
                if dep not in self._completed_tasks or not self._completed_tasks[dep].success
            ]
            
            if unmet_deps:
                self._waiting_tasks[task.task_id] = task
                self._task_dependencies[task.task_id] = unmet_deps
                return
        
        await self._queue.put(task)
    
    async def start(self) -> None:
        """Start task queue workers"""
        self._workers = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(self.max_workers)
        ]
    
    async def stop(self) -> None:
        """Stop task queue and wait for completion"""
        self._shutdown = True
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # Cancel running tasks
        for task in self._running_tasks.values():
            task.cancel()
        
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine to process tasks"""
        logger.info(f"Started worker: {worker_name}")
        
        while not self._shutdown:
            try:
                # Get next task
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                # Apply backpressure if enabled
                if self._backpressure:
                    await self._backpressure.acquire()
                
                # Execute task
                await self._execute_task(task, worker_name)
                
                if self._backpressure:
                    self._backpressure.release()
                
                self._queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No tasks available
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
    
    async def _execute_task(self, task: AsyncTask, worker_name: str) -> None:
        """Execute a single task"""
        start_time = time.time()
        
        try:
            # Create task coroutine
            if asyncio.iscoroutinefunction(task.func):
                coro = task.func(*task.args, **task.kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                coro = loop.run_in_executor(None, lambda: task.func(*task.args, **task.kwargs))
            
            # Apply timeout if specified
            if task.timeout_seconds:
                result = await asyncio.wait_for(coro, timeout=task.timeout_seconds)
            else:
                result = await coro
            
            execution_time = time.time() - start_time
            
            # Create success result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                retry_count=task.retry_count,
                metadata={'worker': worker_name}
            )
            
            self._completed_tasks[task.task_id] = task_result
            self._stats['total_completed'] += 1
            
            logger.debug(f"Task {task.task_id} completed successfully in {execution_time:.3f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Handle task failure
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=e,
                execution_time=execution_time,
                retry_count=task.retry_count,
                metadata={'worker': worker_name}
            )
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                self._stats['total_retried'] += 1
                
                # Add delay before retry
                await asyncio.sleep(min(2 ** task.retry_count, 60))
                await self._queue.put(task)
                
                logger.warning(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                self._completed_tasks[task.task_id] = task_result
                self._stats['total_failed'] += 1
                
                logger.error(f"Task {task.task_id} failed permanently: {e}")
        
        finally:
            # Check for waiting tasks that can now run
            await self._check_waiting_tasks()
    
    async def _check_waiting_tasks(self) -> None:
        """Check waiting tasks for satisfied dependencies"""
        ready_tasks = []
        
        for task_id, task in list(self._waiting_tasks.items()):
            deps = self._task_dependencies.get(task_id, [])
            
            # Check if all dependencies are satisfied
            satisfied_deps = [
                dep for dep in deps
                if dep in self._completed_tasks and self._completed_tasks[dep].success
            ]
            
            if len(satisfied_deps) == len(deps):
                ready_tasks.append(task)
                del self._waiting_tasks[task_id]
                del self._task_dependencies[task_id]
        
        # Submit ready tasks
        for task in ready_tasks:
            await self._queue.put(task)
    
    async def wait_for_task(self, task_id: str) -> TaskResult:
        """Wait for specific task completion"""
        while task_id not in self._completed_tasks:
            await asyncio.sleep(0.1)
        
        return self._completed_tasks[task_id]
    
    async def wait_for_all(self) -> Dict[str, TaskResult]:
        """Wait for all tasks to complete"""
        await self._queue.join()
        return self._completed_tasks.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task queue statistics"""
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'running_tasks': len(self._running_tasks),
            'waiting_tasks': len(self._waiting_tasks),
            'completed_tasks': len(self._completed_tasks),
            'backpressure_active': self._backpressure.is_under_pressure if self._backpressure else False
        }


class AsyncConnectionPool:
    """Connection pool for async HTTP clients"""
    
    def __init__(self,
                 max_connections: int = 100,
                 max_keepalive_connections: int = 20,
                 keepalive_expiry: float = 30.0):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self._clients: Dict[str, httpx.AsyncClient] = {}
        self._client_refs: Dict[str, weakref.ReferenceType] = {}
    
    async def get_client(self, base_url: str) -> httpx.AsyncClient:
        """Get or create HTTP client for base URL"""
        if base_url in self._clients:
            return self._clients[base_url]
        
        # Create new client
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry
        )
        
        client = httpx.AsyncClient(
            base_url=base_url,
            limits=limits,
            timeout=30.0
        )
        
        self._clients[base_url] = client
        
        # Set up weak reference for cleanup
        def cleanup_client(ref):
            if base_url in self._clients:
                del self._clients[base_url]
        
        self._client_refs[base_url] = weakref.ref(client, cleanup_client)
        
        return client
    
    async def close_all(self):
        """Close all HTTP clients"""
        for client in self._clients.values():
            await client.aclose()
        
        self._clients.clear()
        self._client_refs.clear()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()


class AsyncFileProcessor:
    """Async file processing utilities"""
    
    @staticmethod
    async def read_file(file_path: str) -> str:
        """Read file asynchronously"""
        async with aiofiles.open(file_path, mode='r') as f:
            return await f.read()
    
    @staticmethod
    async def write_file(file_path: str, content: str) -> None:
        """Write file asynchronously"""
        async with aiofiles.open(file_path, mode='w') as f:
            await f.write(content)
    
    @staticmethod
    async def read_json_file(file_path: str) -> Dict[str, Any]:
        """Read JSON file asynchronously"""
        content = await AsyncFileProcessor.read_file(file_path)
        return json.loads(content)
    
    @staticmethod
    async def write_json_file(file_path: str, data: Dict[str, Any]) -> None:
        """Write JSON file asynchronously"""
        content = json.dumps(data, indent=2)
        await AsyncFileProcessor.write_file(file_path, content)
    
    @staticmethod
    async def process_files_batch(file_paths: List[str], 
                                 processor: Callable[[str], Any],
                                 max_concurrent: int = 10) -> List[Any]:
        """Process multiple files concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_file(file_path: str) -> Any:
            async with semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(file_path)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, processor, file_path)
        
        tasks = [process_single_file(path) for path in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)


class AsyncEventBus:
    """Async event bus for decoupled communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processor_task: Optional[asyncio.Task] = None
        self._processing = False
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from event type"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def publish(self, event_type: str, data: Any = None) -> None:
        """Publish event"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        
        await self._event_queue.put(event)
        
        if not self._processing:
            await self.start_processing()
    
    async def start_processing(self) -> None:
        """Start event processing"""
        if self._processing:
            return
        
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_events())
    
    async def stop_processing(self) -> None:
        """Stop event processing"""
        self._processing = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
    
    async def _process_events(self) -> None:
        """Process events from queue"""
        while self._processing:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._handle_event(event)
                self._event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle single event"""
        event_type = event['type']
        
        if event_type not in self._subscribers:
            return
        
        handlers = self._subscribers[event_type].copy()
        
        # Execute all handlers concurrently
        tasks = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                task = asyncio.create_task(handler(event))
            else:
                loop = asyncio.get_event_loop()
                task = asyncio.create_task(
                    loop.run_in_executor(None, handler, event)
                )
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# Global instances
connection_pool = AsyncConnectionPool()
event_bus = AsyncEventBus()


# Utility functions
async def run_with_timeout(coro, timeout: float):
    """Run coroutine with timeout"""
    return await asyncio.wait_for(coro, timeout=timeout)


async def run_with_retries(coro_func: Callable, 
                          max_retries: int = 3,
                          delay: float = 1.0,
                          backoff: float = 2.0) -> Any:
    """Run coroutine with retries"""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                await asyncio.sleep(wait_time)
            else:
                break
    
    raise last_exception


@asynccontextmanager
async def async_timer():
    """Async context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"Operation completed in {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    # Example usage and testing
    
    async def example_usage():
        # Test async task queue
        task_queue = AsyncTaskQueue(max_workers=3)
        await task_queue.start()
        
        # Submit some test tasks
        async def sample_task(task_id: str, delay: float) -> str:
            await asyncio.sleep(delay)
            return f"Task {task_id} completed"
        
        for i in range(5):
            task = AsyncTask(
                task_id=f"task-{i}",
                func=sample_task,
                args=(f"task-{i}", 0.5),
                priority=TaskPriority.NORMAL
            )
            await task_queue.submit_task(task)
        
        # Wait for completion
        results = await task_queue.wait_for_all()
        print(f"Completed {len(results)} tasks")
        
        # Get stats
        stats = task_queue.get_stats()
        print(f"Queue stats: {stats}")
        
        await task_queue.stop()
        
        # Test event bus
        await event_bus.start_processing()
        
        async def event_handler(event):
            print(f"Received event: {event['type']} - {event['data']}")
        
        event_bus.subscribe("test_event", event_handler)
        await event_bus.publish("test_event", "Hello, async world!")
        
        await asyncio.sleep(0.1)  # Let event process
        await event_bus.stop_processing()
    
    # Run example
    asyncio.run(example_usage())