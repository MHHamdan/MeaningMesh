"""Caching utilities for performance optimization."""

import functools
import time
import hashlib
import json
from typing import Dict, Any, Callable, TypeVar, Optional, Awaitable, Union, List

# Type variables for cache
T = TypeVar('T')
R = TypeVar('R')


class Cache:
    """Simple in-memory cache with expiration."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store
            ttl: Time to live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        entry = self.cache.get(key)
        if not entry:
            self.misses += 1
            return None
        
        # Check expiration
        if time.time() > entry["expiry"]:
            del self.cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Optional custom TTL in seconds
        """
        # Enforce max size by removing oldest entries
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Get oldest key based on expiry time
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["expiry"])
            del self.cache[oldest_key]
        
        # Store with expiration time
        expiry = time.time() + (ttl if ttl is not None else self.ttl)
        self.cache[key] = {"value": value, "expiry": expiry}
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }


def hash_args(*args: Any, **kwargs: Any) -> str:
    """
    Create a hash from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        String hash of the arguments
    """
    # Convert args and kwargs to JSON-serializable format
    def make_hashable(obj: Any) -> Any:
        """Convert to hashable types."""
        if isinstance(obj, (list, tuple)):
            return tuple(make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert to string representation
            return str(obj)
    
    # Create hash from serialized arguments
    hashable_args = make_hashable(args)
    hashable_kwargs = make_hashable(kwargs)
    
    # Combine and hash
    combined = str(hashable_args) + str(hashable_kwargs)
    return hashlib.md5(combined.encode()).hexdigest()


def cache_async(cache: Cache):
    """
    Decorator for caching async function results.
    
    Args:
        cache: Cache instance to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> R:
            # Check for cache bypass
            bypass = kwargs.pop('cache_bypass', False)
            if bypass:
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{hash_args(*args, **kwargs)}"
            
            # Check cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result)
            
            return result
        return wrapper
    return decorator


# Create a global cache for vectorization
vectorizer_cache = Cache(max_size=1000, ttl=3600)  # 1 hour TTL
