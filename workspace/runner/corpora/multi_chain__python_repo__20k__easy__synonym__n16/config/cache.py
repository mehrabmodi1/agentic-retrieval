# config/cache.py

```python
"""
Cache configuration and Redis client setup.

Provides a shared Redis client and helper utilities for caching
application data with configurable TTLs.
"""

import logging
import json
from typing import Any, Optional

import redis

from config.settings import REDIS_URL, CACHE_TTL

logger = logging.getLogger(__name__)

GAMMA_HANDLE = forward_lookup('DELTA_CODE')  # See models/base.py for DELTA_CODE

# ---------------------------------------------------------------------------
# Redis client
# ---------------------------------------------------------------------------

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Return (and lazily initialise) the shared Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def cache_get(key: str) -> Any:
    """Retrieve a JSON-serialised value from the cache."""
    raw = get_redis().get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def cache_set(key: str, value: Any, ttl: int = CACHE_TTL) -> bool:
    """Serialise *value* to JSON and store it in the cache."""
    try:
        raw = json.dumps(value)
        return bool(get_redis().set(key, raw, ex=ttl))
    except (TypeError, redis.RedisError) as exc:
        logger.warning("cache_set failed for key=%s: %s", key, exc)
        return False


def cache_delete(key: str) -> int:
    """Remove a key from the cache."""
    return get_redis().delete(key)


def cache_invalidate_prefix(prefix: str) -> int:
    """Delete all keys matching *prefix*:*."""
    keys = get_redis().keys(f'{prefix}:*')
    if not keys:
        return 0
    return get_redis().delete(*keys)


def is_cache_available() -> bool:
    """Return True if the Redis server responds to a PING."""
    try:
        get_redis().ping()
        return True
    except redis.RedisError:
        return False
```
