"""
Redis caching layer for explanation service.
Provides fast lookup for repeated feature patterns.
"""
import logging
import hashlib
import json
from typing import Optional, Dict, Any, Union
from functools import wraps

import numpy as np
import pandas as pd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based caching for explanation results.
    Falls back to in-memory dict if Redis is unavailable.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,  # 1 hour default
        key_prefix: str = 'fraud_xai:'
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.ttl = ttl
        self.key_prefix = key_prefix
        
        self._redis_client: Optional[Any] = None
        self._memory_cache: Dict[str, Any] = {}
        self._use_redis = False
        
        self._connect()
    
    def _connect(self):
        """Connect to Redis server."""
        if not REDIS_AVAILABLE:
            logger.info("Using in-memory cache (Redis not available)")
            return
        
        try:
            self._redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True
            )
            self._redis_client.ping()
            self._use_redis = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Using in-memory cache.")
            self._use_redis = False
    
    def _make_key(self, data: Union[Dict, np.ndarray, list], method: str = 'shap') -> str:
        """Generate cache key from data."""
        if isinstance(data, np.ndarray):
            data_str = data.tobytes()
        elif isinstance(data, dict):
            # Sort keys for consistent hashing
            data_str = json.dumps(data, sort_keys=True, default=str).encode()
        else:
            data_str = str(data).encode()
        
        hash_val = hashlib.md5(data_str).hexdigest()[:16]
        return f"{self.key_prefix}{method}:{hash_val}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self._use_redis and self._redis_client:
                value = self._redis_client.get(key)
                if value:
                    return json.loads(value)
            else:
                return self._memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        ttl = ttl or self.ttl
        
        try:
            serialized = json.dumps(value, default=str)
            
            if self._use_redis and self._redis_client:
                self._redis_client.setex(key, ttl, serialized)
            else:
                self._memory_cache[key] = value
            
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get_explanation(
        self,
        features: Union[Dict, np.ndarray],
        method: str = 'shap'
    ) -> Optional[Dict]:
        """Get cached explanation."""
        key = self._make_key(features, method)
        return self.get(key)
    
    def set_explanation(
        self,
        features: Union[Dict, np.ndarray],
        explanation: Dict,
        method: str = 'shap',
        ttl: Optional[int] = None
    ) -> bool:
        """Cache explanation result."""
        key = self._make_key(features, method)
        return self.set(key, explanation, ttl)
    
    def invalidate(self, pattern: str = '*') -> int:
        """Invalidate cache entries matching pattern."""
        try:
            if self._use_redis and self._redis_client:
                keys = self._redis_client.keys(f"{self.key_prefix}{pattern}")
                if keys:
                    return self._redis_client.delete(*keys)
            else:
                count = len(self._memory_cache)
                self._memory_cache.clear()
                return count
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
        
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            'using_redis': self._use_redis,
            'memory_entries': len(self._memory_cache)
        }
        
        if self._use_redis and self._redis_client:
            try:
                info = self._redis_client.info()
                stats['redis_used_memory'] = info.get('used_memory_human', 'N/A')
                stats['redis_connected_clients'] = info.get('connected_clients', 0)
            except:
                pass
        
        return stats
    
    def close(self):
        """Close Redis connection."""
        if self._redis_client:
            self._redis_client.close()


class ExplanationCache:
    """
    High-level cache for explanation results with pattern matching.
    """
    
    def __init__(self, cache: Optional[RedisCache] = None):
        self.cache = cache or RedisCache()
        self._hit_count = 0
        self._miss_count = 0
    
    def get_or_compute(
        self,
        features: Union[Dict, np.ndarray],
        compute_fn: callable,
        method: str = 'shap',
        use_cache: bool = True
    ) -> Dict:
        """
        Get explanation from cache or compute and cache it.
        
        Args:
            features: Input features
            compute_fn: Function to compute explanation if not cached
            method: Explanation method
            use_cache: Whether to use caching
            
        Returns:
            Explanation result
        """
        if not use_cache:
            return compute_fn()
        
        # Try cache
        cached = self.cache.get_explanation(features, method)
        if cached is not None:
            self._hit_count += 1
            cached['from_cache'] = True
            return cached
        
        # Compute
        self._miss_count += 1
        result = compute_fn()
        result['from_cache'] = False
        
        # Cache result
        self.cache.set_explanation(features, result, method)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0
        
        return {
            'hits': self._hit_count,
            'misses': self._miss_count,
            'total': total,
            'hit_rate': hit_rate,
            'cache_backend': self.cache.get_stats()
        }
    
    def reset_stats(self):
        """Reset cache statistics."""
        self._hit_count = 0
        self._miss_count = 0


def cached_explanation(ttl: int = 3600):
    """
    Decorator for caching explanation functions.
    
    Usage:
        @cached_explanation(ttl=1800)
        def explain(features):
            # compute explanation
            return result
    """
    def decorator(func):
        cache = RedisCache(ttl=ttl)
        
        @wraps(func)
        def wrapper(features, *args, **kwargs):
            # Try cache
            cached = cache.get_explanation(features, func.__name__)
            if cached is not None:
                cached['from_cache'] = True
                return cached
            
            # Compute
            result = func(features, *args, **kwargs)
            result['from_cache'] = False
            
            # Cache
            cache.set_explanation(features, result, func.__name__)
            
            return result
        
        wrapper.cache = cache
        return wrapper
    return decorator
