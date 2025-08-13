#!/usr/bin/env python3
"""
Advanced Caching Framework for SDLC Components
Intelligent caching with TTL, LRU eviction, and distributed cache support
"""

import time
import threading
import json
import pickle
import hashlib
from typing import Any, Dict, Optional, Union, Callable, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import OrderedDict
from abc import ABC, abstractmethod
from pathlib import Path
import asyncio
import redis
from contextlib import asynccontextmanager

from logging_framework import get_logger

logger = get_logger('caching')


@dataclass
class CacheEntry:
    """Represents a cached entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """Update last accessed time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache statistics tracking"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expired_removals = 0
        self.manual_removals = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def record_expired_removal(self):
        with self._lock:
            self.expired_removals += 1
    
    def record_manual_removal(self):
        with self._lock:
            self.manual_removals += 1
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 1.0 - self.hit_rate
    
    @property
    def uptime_seconds(self) -> float:
        """Get cache uptime in seconds"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Export stats as dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'evictions': self.evictions,
            'expired_removals': self.expired_removals,
            'manual_removals': self.manual_removals,
            'uptime_seconds': self.uptime_seconds
        }


class CacheInterface(ABC):
    """Abstract interface for cache implementations"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Put value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get number of entries in cache"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class LRUCache(CacheInterface):
    """In-memory LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._cleanup_thread = None
        self._shutdown = False
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for expired entry cleanup"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Background cleanup of expired entries"""
        while not self._shutdown:
            try:
                with self._lock:
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        del self._cache[key]
                        self._stats.record_expired_removal()
                
                time.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                time.sleep(60)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._stats.record_miss()
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self._stats.record_expired_removal()
                self._stats.record_miss()
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.record_hit()
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Put value in cache"""
        with self._lock:
            ttl = ttl or self.default_ttl
            tags = tags or []
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 0
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl_seconds=ttl,
                size_bytes=size_bytes,
                tags=tags
            )
            
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            
            # Add new entry
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict oldest entries if over max size
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.record_eviction()
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.record_manual_removal()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired:
                del self._cache[key]
                self._stats.record_expired_removal()
                return False
            
            return True
    
    def size(self) -> int:
        """Get number of entries in cache"""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats
    
    def get_entries_by_tag(self, tag: str) -> List[CacheEntry]:
        """Get all cache entries with specified tag"""
        with self._lock:
            return [
                entry for entry in self._cache.values()
                if tag in entry.tags and not entry.is_expired
            ]
    
    def delete_by_tag(self, tag: str) -> int:
        """Delete all entries with specified tag"""
        with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if tag in entry.tags
            ]
            
            for key in keys_to_delete:
                del self._cache[key]
                self._stats.record_manual_removal()
            
            return len(keys_to_delete)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self._cache.values())
            avg_size = total_size / len(self._cache) if self._cache else 0
            
            return {
                'total_entries': len(self._cache),
                'total_size_bytes': total_size,
                'average_size_bytes': avg_size,
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size
            }
    
    def shutdown(self):
        """Shutdown cache and cleanup thread"""
        self._shutdown = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)


class DistributedCache(CacheInterface):
    """Redis-based distributed cache"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 key_prefix: str = "terragon:",
                 default_ttl: Optional[float] = None,
                 serialization: str = "pickle"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.serialization = serialization
        self._stats = CacheStats()
        
        try:
            self._redis = redis.from_url(redis_url, decode_responses=False)
            self._redis.ping()  # Test connection
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to local cache")
            self._redis = None
            self._fallback_cache = LRUCache(max_size=1000, default_ttl=default_ttl)
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        if self.serialization == "pickle":
            return pickle.dumps(value)
        elif self.serialization == "json":
            return json.dumps(value).encode('utf-8')
        else:
            raise ValueError(f"Unsupported serialization: {self.serialization}")
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        if self.serialization == "pickle":
            return pickle.loads(data)
        elif self.serialization == "json":
            return json.loads(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported serialization: {self.serialization}")
    
    def _get_key(self, key: str) -> str:
        """Get prefixed key"""
        return f"{self.key_prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self._redis is None:
            return self._fallback_cache.get(key)
        
        try:
            prefixed_key = self._get_key(key)
            data = self._redis.get(prefixed_key)
            
            if data is None:
                self._stats.record_miss()
                return None
            
            value = self._deserialize(data)
            self._stats.record_hit()
            return value
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.record_miss()
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Put value in cache"""
        if self._redis is None:
            return self._fallback_cache.put(key, value, ttl, tags)
        
        try:
            prefixed_key = self._get_key(key)
            data = self._serialize(value)
            ttl = ttl or self.default_ttl
            
            if ttl:
                self._redis.setex(prefixed_key, int(ttl), data)
            else:
                self._redis.set(prefixed_key, data)
            
            # Store tags separately if provided
            if tags:
                tag_key = f"{prefixed_key}:tags"
                self._redis.sadd(tag_key, *tags)
                if ttl:
                    self._redis.expire(tag_key, int(ttl))
            
        except Exception as e:
            logger.error(f"Redis put error: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if self._redis is None:
            return self._fallback_cache.delete(key)
        
        try:
            prefixed_key = self._get_key(key)
            result = self._redis.delete(prefixed_key)
            
            # Also delete tags
            tag_key = f"{prefixed_key}:tags"
            self._redis.delete(tag_key)
            
            if result > 0:
                self._stats.record_manual_removal()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        if self._redis is None:
            return self._fallback_cache.clear()
        
        try:
            # Delete all keys with prefix
            keys = self._redis.keys(f"{self.key_prefix}*")
            if keys:
                self._redis.delete(*keys)
                
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if self._redis is None:
            return self._fallback_cache.exists(key)
        
        try:
            prefixed_key = self._get_key(key)
            return bool(self._redis.exists(prefixed_key))
            
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def size(self) -> int:
        """Get number of entries in cache"""
        if self._redis is None:
            return self._fallback_cache.size()
        
        try:
            keys = self._redis.keys(f"{self.key_prefix}*")
            # Filter out tag keys
            data_keys = [k for k in keys if not k.endswith(b':tags')]
            return len(data_keys)
            
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self._stats


class CacheManager:
    """High-level cache manager with multiple cache backends"""
    
    def __init__(self):
        self._caches: Dict[str, CacheInterface] = {}
        self._default_cache = "local"
    
    def register_cache(self, name: str, cache: CacheInterface, is_default: bool = False):
        """Register a cache backend"""
        self._caches[name] = cache
        if is_default:
            self._default_cache = name
    
    def get_cache(self, name: Optional[str] = None) -> CacheInterface:
        """Get cache by name"""
        cache_name = name or self._default_cache
        if cache_name not in self._caches:
            raise ValueError(f"Cache '{cache_name}' not registered")
        return self._caches[cache_name]
    
    def cache_function(self, 
                      cache_name: Optional[str] = None,
                      ttl: Optional[float] = None,
                      key_func: Optional[Callable] = None,
                      tags: Optional[List[str]] = None):
        """Decorator for caching function results"""
        
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                cache = self.get_cache(cache_name)
                
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    args_str = str(args) + str(sorted(kwargs.items()))
                    cache_key = f"{func.__name__}:{hashlib.md5(args_str.encode()).hexdigest()}"
                
                # Try to get from cache
                result = cache.get(cache_key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                cache.put(cache_key, result, ttl=ttl, tags=tags)
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_by_tag(self, tag: str, cache_name: Optional[str] = None) -> int:
        """Invalidate all cached entries with specified tag"""
        cache = self.get_cache(cache_name)
        if hasattr(cache, 'delete_by_tag'):
            return cache.delete_by_tag(tag)
        else:
            logger.warning(f"Cache '{cache_name}' does not support tag-based invalidation")
            return 0
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all registered caches"""
        stats = {}
        for name, cache in self._caches.items():
            stats[name] = cache.get_stats().to_dict()
        return stats


# Global cache manager instance
cache_manager = CacheManager()

# Register default local cache
cache_manager.register_cache("local", LRUCache(max_size=1000), is_default=True)


def setup_distributed_cache(redis_url: str = "redis://localhost:6379"):
    """Setup distributed cache as default"""
    distributed_cache = DistributedCache(redis_url=redis_url)
    cache_manager.register_cache("distributed", distributed_cache, is_default=True)


# Convenience decorators
def cached(ttl: Optional[float] = None, 
          cache_name: Optional[str] = None,
          tags: Optional[List[str]] = None):
    """Convenience decorator for caching function results"""
    return cache_manager.cache_function(
        cache_name=cache_name,
        ttl=ttl,
        tags=tags
    )


def cache_key(*key_parts) -> str:
    """Generate cache key from parts"""
    key_str = ":".join(str(part) for part in key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


class AsyncCacheWrapper:
    """Async wrapper for cache operations"""
    
    def __init__(self, cache: CacheInterface):
        self.cache = cache
        self._executor = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Async get operation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.cache.get, key)
    
    async def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[List[str]] = None) -> None:
        """Async put operation"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.cache.put, key, value, ttl, tags)
    
    async def delete(self, key: str) -> bool:
        """Async delete operation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.cache.delete, key)
    
    async def exists(self, key: str) -> bool:
        """Async exists check"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.cache.exists, key)


def get_async_cache(cache_name: Optional[str] = None) -> AsyncCacheWrapper:
    """Get async wrapper for cache"""
    cache = cache_manager.get_cache(cache_name)
    return AsyncCacheWrapper(cache)


if __name__ == "__main__":
    # Example usage and testing
    
    # Test local cache
    local_cache = LRUCache(max_size=100, default_ttl=60.0)
    
    # Basic operations
    local_cache.put("test_key", "test_value", ttl=30.0)
    print(f"Retrieved: {local_cache.get('test_key')}")
    
    # Test with tags
    local_cache.put("tagged_key", "tagged_value", tags=["group1", "test"])
    entries = local_cache.get_entries_by_tag("test")
    print(f"Entries with 'test' tag: {len(entries)}")
    
    # Test statistics
    stats = local_cache.get_stats()
    print(f"Cache stats: {stats.to_dict()}")
    
    # Test decorator
    @cached(ttl=60.0, tags=["calculations"])
    def expensive_calculation(x: int, y: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return x * y + x ** 2
    
    # First call - will be cached
    start_time = time.time()
    result1 = expensive_calculation(5, 10)
    time1 = time.time() - start_time
    
    # Second call - will use cache
    start_time = time.time()
    result2 = expensive_calculation(5, 10)
    time2 = time.time() - start_time
    
    print(f"First call: {result1} in {time1:.3f}s")
    print(f"Second call: {result2} in {time2:.3f}s")
    print(f"Cache speedup: {time1/time2:.1f}x")
    
    # Cleanup
    local_cache.shutdown()