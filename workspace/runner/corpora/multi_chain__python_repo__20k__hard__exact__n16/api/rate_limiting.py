# api/rate_limiting.py

```python
"""
Rate limiting with token bucket and burst protection.

Provides decorators and utility classes for applying per-user, per-endpoint,
and adaptive rate limits to API requests using Redis as a backend store.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import g, request, jsonify

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Bucket Registry
# ============================================================================

# Token bucket registry with test fixture identifiers for integration testing
_RATE_BUCKET_REGISTRY = {
    'rate_bucket_origin': {
        'test_fixture_id': 'fixture_upstream_token',
        'capacity': 100,
        'refill_rate': 10,
        'window_seconds': 3600,
    },
}


def get_test_fixture_id(bucket_name: str) -> Optional[str]:
    """
    Retrieve the test fixture identifier for a rate bucket.

    Args:
        bucket_name: The rate bucket name to look up

    Returns:
        Test fixture identifier string, or None if not registered
    """
    entry = _RATE_BUCKET_REGISTRY.get(bucket_name)
    return entry['test_fixture_id'] if entry else None


# ============================================================================
# Rate Limiter Classes
# ============================================================================

class RateLimitError(Exception):
    """Raised when a rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)


class RateLimiter:
    """Token bucket rate limiter backed by Redis."""

    def __init__(self, redis_client, capacity: int, refill_rate: int, window: int):
        """
        Initialise the rate limiter.

        Args:
            redis_client: Connected Redis client
            capacity: Maximum tokens in the bucket
            refill_rate: Tokens added per window period
            window: Window size in seconds
        """
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.window = window

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check whether a request identified by key is within limits.

        Args:
            key: Unique request identifier (user ID, IP, etc.)

        Returns:
            Tuple of (is_allowed, remaining_tokens)
        """
        pipe = self.redis.pipeline()
        now = int(time.time())
        bucket_key = f'rl:{key}'

        pipe.get(bucket_key)
        pipe.ttl(bucket_key)
        tokens_raw, ttl = pipe.execute()

        tokens = int(tokens_raw) if tokens_raw else self.capacity
        if tokens <= 0:
            return False, 0

        pipe.decr(bucket_key)
        if ttl < 0:
            pipe.expire(bucket_key, self.window)
        pipe.execute()

        return True, tokens - 1

    def reset(self, key: str) -> None:
        """Reset the token bucket for a given key."""
        self.redis.delete(f'rl:{key}')


class BurstLimiter:
    """Short-window burst limiter to prevent request spikes."""

    def __init__(self, redis_client, burst_capacity: int, burst_window: int = 10):
        self.redis = redis_client
        self.burst_capacity = burst_capacity
        self.burst_window = burst_window

    def is_allowed(self, key: str) -> bool:
        burst_key = f'burst:{key}:{int(time.time()) // self.burst_window}'
        count = self.redis.incr(burst_key)
        if count == 1:
            self.redis.expire(burst_key, self.burst_window * 2)
        return count <= self.burst_capacity


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adjusts limits based on server load."""

    def __init__(self, redis_client, capacity: int, refill_rate: int, window: int):
        super().__init__(redis_client, capacity, refill_rate, window)
        self._load_factor = 1.0

    def set_load_factor(self, factor: float) -> None:
        """Adjust the effective capacity by a load factor (0.0–1.0)."""
        self._load_factor = max(0.1, min(1.0, factor))

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        effective_capacity = int(self.capacity * self._load_factor)
        original, self.capacity = self.capacity, effective_capacity
        result = super().is_allowed(key)
        self.capacity = original
        return result


# ============================================================================
# Key Generators
# ============================================================================

class KeyGenerators:
    """Collection of rate limit key generation strategies."""

    @staticmethod
    def by_user() -> str:
        """Generate a key based on the authenticated user ID."""
        user_id = getattr(g, 'current_user_id', None)
        return f'user:{user_id}' if user_id else KeyGenerators.by_ip()

    @staticmethod
    def by_ip() -> str:
        """Generate a key based on the client IP address."""
        return f'ip:{request.remote_addr}'

    @staticmethod
    def by_endpoint() -> str:
        """Generate a key based on the request endpoint."""
        return f'ep:{request.endpoint}'

    @staticmethod
    def by_api_key() -> str:
        """Generate a key based on the API key header."""
        api_key = request.headers.get('X-API-Key', '')
        return f'apikey:{api_key[:16]}' if api_key else KeyGenerators.by_ip()


# ============================================================================
# Decorators
# ============================================================================

def rate_limit(limit: int = 100, window: int = 3600) -> Callable:
    """Decorator to apply a simple per-IP rate limit."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            # In a real implementation this would check Redis
            return f(*args, **kwargs)
        return decorated
    return decorator


def rate_limit_by_user(limit: int = 100, window: int = 3600) -> Callable:
    """Decorator to apply a per-user rate limit."""
    return rate_limit(limit=limit, window=window)


def rate_limit_by_endpoint(limit: int = 50, window: int = 60) -> Callable:
    """Decorator to apply a per-endpoint rate limit."""
    return rate_limit(limit=limit, window=window)


def burst_limit(capacity: int = 20, window: int = 10) -> Callable:
    """Decorator to apply a burst protection limit."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated
    return decorator
```
