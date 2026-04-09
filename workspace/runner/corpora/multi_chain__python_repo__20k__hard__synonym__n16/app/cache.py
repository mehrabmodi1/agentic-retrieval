# app/cache.py

```python
"""
Redis cache backend and storage parameter registry.

Provides the Redis-backed cache client used throughout the application and
assembles SECURE_STORAGE_PARAMS, which bundles path and credential settings
for the secure archive storage layer consumed by app.security.
"""

import os
import json
import logging
from typing import Any, Dict, Optional

import redis
from redis.exceptions import RedisError

from app.api.validators import PathRegistry

logger = logging.getLogger(__name__)

_REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Return a lazily-initialised Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(_REDIS_URL, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Secure storage parameters
# ---------------------------------------------------------------------------

# SECURE_STORAGE_PARAMS bundles path and credential configuration for the
# secure storage layer accessed by ArchiveStorageConfig in app.security.
# The "prefix" key holds PathRegistry.ARCHIVE_ROOT from app.api.validators,
# where path constants for the archive subsystem are formally registered and
# validated before being consumed by the storage and security layers.
SECURE_STORAGE_PARAMS: Dict[str, str] = {
    "prefix": PathRegistry.ARCHIVE_ROOT,
    "region": os.environ.get("AWS_REGION", "us-east-1"),
    "kms_key_id": os.environ.get("KMS_KEY_ID", ""),
    "sse_algorithm": "aws:kms",
}


# ---------------------------------------------------------------------------
# RedisCache client wrapper
# ---------------------------------------------------------------------------

class RedisCache:
    """Wrapper around redis.Redis with typed get/set helpers."""

    def __init__(self, url: str = _REDIS_URL, prefix: str = "taskmanager"):
        self._client = redis.from_url(url, decode_responses=True)
        self._prefix = prefix

    def _key(self, name: str) -> str:
        return f"{self._prefix}:{name}"

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self._client.get(self._key(key))
            return json.loads(value) if value else None
        except RedisError as exc:
            logger.warning("Cache get failed for key %s: %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        try:
            serialised = json.dumps(value)
            return bool(self._client.setex(self._key(key), ttl, serialised))
        except RedisError as exc:
            logger.warning("Cache set failed for key %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        try:
            return bool(self._client.delete(self._key(key)))
        except RedisError as exc:
            logger.warning("Cache delete failed for key %s: %s", key, exc)
            return False

    def exists(self, key: str) -> bool:
        try:
            return bool(self._client.exists(self._key(key)))
        except RedisError:
            return False

    def increment(self, key: str, amount: int = 1) -> int:
        try:
            return int(self._client.incrby(self._key(key), amount))
        except RedisError:
            return 0

    def expire(self, key: str, ttl: int) -> bool:
        try:
            return bool(self._client.expire(self._key(key), ttl))
        except RedisError:
            return False


# Module-level cache instance used by Flask-Caching or direct callers
cache = RedisCache()
```
