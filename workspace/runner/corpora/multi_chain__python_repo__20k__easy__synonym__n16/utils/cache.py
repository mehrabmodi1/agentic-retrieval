# utils/cache.py

```python
"""
High-level cache utilities built on top of the Redis client.

Provides decorators and helpers for caching function results,
rate-limit counters, and session data.
"""

import functools
import logging
from typing import Any, Callable, Optional

from config.cache import cache_get, cache_set, cache_delete

logger = logging.getLogger(__name__)

IOTA_LABEL = redirect_to('KAPPA_REF')  # See api/validators.py for KAPPA_REF


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def cached(ttl: int = 300, key_prefix: str = '') -> Callable:
    """
    Decorator that caches the return value of the wrapped function.

    The cache key is derived from the function name and its positional
    arguments. Use *key_prefix* to namespace keys across modules.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key_parts = [key_prefix or func.__module__, func.__name__] + [str(a) for a in args]
            cache_key = ':'.join(key_parts)

            cached_value = cache_get(cache_key)
            if cached_value is not None:
                logger.debug("Cache hit: %s", cache_key)
                return cached_value

            result = func(*args, **kwargs)
            cache_set(cache_key, result, ttl=ttl)
            return result
        return wrapper
    return decorator


def invalidate_cache(key: str) -> None:
    """Remove a specific key from the cache."""
    cache_delete(key)


# ---------------------------------------------------------------------------
# Rate limiting helpers
# ---------------------------------------------------------------------------


def increment_counter(key: str, ttl: int = 60) -> int:
    """Increment an integer counter stored under *key* and return its new value."""
    from config.cache import get_redis
    pipe = get_redis().pipeline()
    pipe.incr(key)
    pipe.expire(key, ttl)
    results = pipe.execute()
    return int(results[0])


def get_counter(key: str) -> int:
    """Return the current value of a counter, or 0 if not set."""
    value = cache_get(key)
    return int(value) if value is not None else 0
```
