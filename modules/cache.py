#!/usr/bin/env python3
"""
SmolVLM-GeoEye Cache Module
===========================

Caching layer for improved performance and reduced API calls.
Supports both in-memory and Redis-based caching.

Author: SmolVLM-GeoEye Team
Version: 3.1.0
"""

import logging
import json
import pickle
import hashlib
import time
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

logger = logging.getLogger(__name__)

class CacheBackend:
    """Abstract cache backend interface"""
    
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        raise NotImplementedError
    
    def delete(self, key: str):
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        raise NotImplementedError

class InMemoryCache(CacheBackend):
    """Thread-safe in-memory cache implementation"""
    
    def __init__(self):
        self._cache = {}
        self._lock = Lock()
        self._access_count = {}
        self._created_at = {}
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                # Check TTL
                created_at, ttl = self._created_at.get(key, (0, 0))
                if time.time() - created_at > ttl:
                    # Expired
                    del self._cache[key]
                    del self._created_at[key]
                    return None
                
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return self._cache[key]
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        with self._lock:
            self._cache[key] = value
            self._created_at[key] = (time.time(), ttl)
            self._access_count[key] = 0
    
    def delete(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if key in self._created_at:
                del self._created_at[key]
            if key in self._access_count:
                del self._access_count[key]
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._created_at.clear()
            self._access_count.clear()
    
    def exists(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            
            # Check TTL
            created_at, ttl = self._created_at.get(key, (0, 0))
            if time.time() - created_at > ttl:
                # Expired
                del self._cache[key]
                del self._created_at[key]
                return False
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(len(str(v)) for v in self._cache.values())
            return {
                'total_keys': len(self._cache),
                'total_size_bytes': total_size,
                'most_accessed': sorted(
                    self._access_count.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

class RedisCache(CacheBackend):
    """Redis-based cache implementation"""
    
    def __init__(self, redis_url: str, prefix: str = "smolvlm"):
        self.prefix = prefix
        self._redis = None
        self._redis_url = redis_url
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            self._redis = redis.from_url(self._redis_url)
            self._redis.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using fallback.")
            self._redis = None
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        if not self._redis:
            return None
        
        try:
            full_key = self._make_key(key)
            value = self._redis.get(full_key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        if not self._redis:
            return
        
        try:
            full_key = self._make_key(key)
            serialized = pickle.dumps(value)
            self._redis.setex(full_key, ttl, serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str):
        if not self._redis:
            return
        
        try:
            full_key = self._make_key(key)
            self._redis.delete(full_key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear(self):
        if not self._redis:
            return
        
        try:
            pattern = f"{self.prefix}:*"
            for key in self._redis.scan_iter(match=pattern):
                self._redis.delete(key)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def exists(self, key: str) -> bool:
        if not self._redis:
            return False
        
        try:
            full_key = self._make_key(key)
            return bool(self._redis.exists(full_key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

class CacheManager:
    """Manages caching with fallback support"""
    
    def __init__(self, config):
        self.config = config
        self._backends = []
        
        # Initialize backends
        if config.cache_enabled:
            # Try Redis first
            if config.redis_url:
                redis_cache = RedisCache(config.redis_url)
                if redis_cache._redis:
                    self._backends.append(redis_cache)
                    logger.info("Redis cache enabled")
            
            # Always add in-memory cache as fallback
            self._backends.append(InMemoryCache())
            logger.info("In-memory cache enabled")
        
        self.default_ttl = config.cache_ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        for backend in self._backends:
            value = backend.get(key)
            if value is not None:
                return value
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        for backend in self._backends:
            backend.set(key, value, ttl)
    
    def delete(self, key: str):
        """Delete value from cache"""
        for backend in self._backends:
            backend.delete(key)
    
    def clear(self):
        """Clear all caches"""
        for backend in self._backends:
            backend.clear()
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache"""
        for backend in self._backends:
            if backend.exists(key):
                return True
        return False
    
    def make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments"""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache_result(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = key_prefix + self.make_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_value
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__}")
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # This is mainly for Redis backend
        for backend in self._backends:
            if isinstance(backend, RedisCache) and backend._redis:
                try:
                    full_pattern = f"{backend.prefix}:{pattern}"
                    for key in backend._redis.scan_iter(match=full_pattern):
                        backend._redis.delete(key)
                except Exception as e:
                    logger.error(f"Pattern invalidation error: {e}")

class DocumentCache:
    """Specialized cache for document processing results"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "doc"
    
    def get_document_analysis(self, file_hash: str, query: str) -> Optional[Dict[str, Any]]:
        """Get cached document analysis"""
        key = f"{self.prefix}:analysis:{file_hash}:{hashlib.md5(query.encode()).hexdigest()}"
        return self.cache_manager.get(key)
    
    def set_document_analysis(self, file_hash: str, query: str, 
                            result: Dict[str, Any], ttl: int = 7200):
        """Cache document analysis result"""
        key = f"{self.prefix}:analysis:{file_hash}:{hashlib.md5(query.encode()).hexdigest()}"
        self.cache_manager.set(key, result, ttl)
    
    def get_extracted_data(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached extracted data"""
        key = f"{self.prefix}:extracted:{file_hash}"
        return self.cache_manager.get(key)
    
    def set_extracted_data(self, file_hash: str, data: Dict[str, Any], ttl: int = 86400):
        """Cache extracted data"""
        key = f"{self.prefix}:extracted:{file_hash}"
        self.cache_manager.set(key, data, ttl)
    
    def invalidate_document(self, file_hash: str):
        """Invalidate all caches for a document"""
        pattern = f"{self.prefix}:*:{file_hash}*"
        self.cache_manager.invalidate_pattern(pattern)

class MetricsCache:
    """Specialized cache for metrics and analytics"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.prefix = "metrics"
    
    def get_cost_analytics(self, period_hours: int) -> Optional[Dict[str, Any]]:
        """Get cached cost analytics"""
        key = f"{self.prefix}:cost:{period_hours}h"
        return self.cache_manager.get(key)
    
    def set_cost_analytics(self, period_hours: int, analytics: Dict[str, Any], ttl: int = 300):
        """Cache cost analytics (5 minute TTL)"""
        key = f"{self.prefix}:cost:{period_hours}h"
        self.cache_manager.set(key, analytics, ttl)
    
    def get_system_health(self) -> Optional[Dict[str, Any]]:
        """Get cached system health"""
        key = f"{self.prefix}:health"
        return self.cache_manager.get(key)
    
    def set_system_health(self, health: Dict[str, Any], ttl: int = 30):
        """Cache system health (30 second TTL)"""
        key = f"{self.prefix}:health"
        self.cache_manager.set(key, health, ttl)
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> int:
        """Increment a counter (useful for rate limiting)"""
        key = f"{self.prefix}:counter:{counter_name}"
        current = self.cache_manager.get(key) or 0
        new_value = current + amount
        self.cache_manager.set(key, new_value, ttl=3600)  # 1 hour TTL
        return new_value
