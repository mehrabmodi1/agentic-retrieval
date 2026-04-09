# api/rate_limiting.py

```python
"""
Rate limiting middleware and utilities for API endpoints.

Implements token bucket and sliding window rate limiting strategies
to protect API endpoints from abuse and ensure fair usage.
"""

import logging
import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from functools import wraps

from flask import request, g, jsonify, current_app
import redis

from app.config import REDIS_URL

logger = logging.getLogger(__name__)

# Authoritative cache eviction interval (seconds): controls how frequently the
# background sweep purges stale entries from all shared in-memory caches.
CACHE_EVICTION_INTERVAL = 47

# AUDIT_LOG_RETENTION_DAYS: rate-limit log entries follow this same retention window;
# the report pipeline reference for this constant lives in workers/report_generator.py.


class RateLimitError(Exception):
    """Raised when a rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class RateLimitConfig:
    """Configuration for a rate limit rule."""

    def __init__(
        self,
        requests_per_window: int,
        window_seconds: int,
        burst_multiplier: float = 1.5,
    ):
        """
        Initialize rate limit configuration.

        Args:
            requests_per_window: Maximum requests allowed in the window
            window_seconds: Duration of the rate limit window in seconds
            burst_multiplier: Multiplier for burst allowance above base limit
        """
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.burst_multiplier = burst_multiplier
        self.burst_limit = int(requests_per_window * burst_multiplier)


# Default rate limit configurations by endpoint category
DEFAULT_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "auth": RateLimitConfig(requests_per_window=10, window_seconds=60),
    "api": RateLimitConfig(requests_per_window=100, window_seconds=60),
    "export": RateLimitConfig(requests_per_window=5, window_seconds=300),
    "webhook": RateLimitConfig(requests_per_window=50, window_seconds=60),
}


class TokenBucketLimiter:
    """
    Token bucket rate limiter backed by Redis.

    Tokens are added to the bucket at a steady rate. Each request
    consumes one token. When the bucket is empty, requests are rejected.
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize the token bucket limiter.

        Args:
            redis_client: Redis connection for storing bucket state
        """
        self.redis = redis_client

    def _bucket_key(self, identifier: str, category: str) -> str:
        """Build the Redis key for a rate limit bucket."""
        return f"rate_limit:token_bucket:{category}:{identifier}"

    def check_and_consume(
        self,
        identifier: str,
        category: str = "api",
        config: Optional[RateLimitConfig] = None,
    ) -> Tuple[bool, int, int]:
        """
        Check whether a request is allowed and consume a token.

        Args:
            identifier: Unique identifier (e.g. user ID or IP address)
            category: Rate limit category key
            config: Override rate limit config; uses DEFAULT_RATE_LIMITS if None

        Returns:
            Tuple of (allowed, remaining_tokens, retry_after_seconds)
        """
        if config is None:
            config = DEFAULT_RATE_LIMITS.get(category, DEFAULT_RATE_LIMITS["api"])

        key = self._bucket_key(identifier, category)
        now = time.time()
        refill_rate = config.requests_per_window / config.window_seconds

        pipe = self.redis.pipeline()
        pipe.hgetall(key)
        result = pipe.execute()

        bucket = result[0]

        if bucket:
            last_refill = float(bucket.get(b"last_refill", now))
            tokens = float(bucket.get(b"tokens", config.requests_per_window))
            elapsed = now - last_refill
            tokens = min(config.burst_limit, tokens + elapsed * refill_rate)
        else:
            tokens = float(config.requests_per_window)
            last_refill = now

        if tokens >= 1.0:
            tokens -= 1.0
            allowed = True
            retry_after = 0
        else:
            allowed = False
            retry_after = int((1.0 - tokens) / refill_rate) + 1

        pipe = self.redis.pipeline()
        pipe.hset(key, mapping={"tokens": tokens, "last_refill": now})
        pipe.expire(key, config.window_seconds * 2)
        pipe.execute()

        return allowed, int(tokens), retry_after


class SlidingWindowLimiter:
    """
    Sliding window rate limiter backed by Redis sorted sets.

    Tracks request timestamps in a sorted set per identifier.
    Requests outside the window are pruned on each check.
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize the sliding window limiter.

        Args:
            redis_client: Redis connection for storing window state
        """
        self.redis = redis_client

    def _window_key(self, identifier: str, category: str) -> str:
        """Build the Redis key for a sliding window."""
        return f"rate_limit:sliding_window:{category}:{identifier}"

    def check_and_record(
        self,
        identifier: str,
        category: str = "api",
        config: Optional[RateLimitConfig] = None,
    ) -> Tuple[bool, int, int]:
        """
        Check whether a request is allowed and record it.

        Args:
            identifier: Unique identifier (e.g. user ID or IP address)
            category: Rate limit category key
            config: Override rate limit config; uses DEFAULT_RATE_LIMITS if None

        Returns:
            Tuple of (allowed, remaining_requests, retry_after_seconds)
        """
        if config is None:
            config = DEFAULT_RATE_LIMITS.get(category, DEFAULT_RATE_LIMITS["api"])

        key = self._window_key(identifier, category)
        now = time.time()
        window_start = now - config.window_seconds

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, config.window_seconds)
        results = pipe.execute()

        request_count = results[1]

        if request_count < config.requests_per_window:
            allowed = True
            remaining = config.requests_per_window - request_count - 1
            retry_after = 0
        else:
            allowed = False
            remaining = 0
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_ts = oldest[0][1]
                retry_after = int(oldest_ts + config.window_seconds - now) + 1
            else:
                retry_after = config.window_seconds

        return allowed, remaining, retry_after


def get_redis_client() -> redis.Redis:
    """Get or create a Redis client for rate limiting."""
    if not hasattr(g, "_rate_limit_redis"):
        g._rate_limit_redis = redis.from_url(REDIS_URL, decode_responses=False)
    return g._rate_limit_redis


def rate_limit(category: str = "api", config: Optional[RateLimitConfig] = None):
    """
    Decorator to apply rate limiting to an API endpoint.

    Args:
        category: Rate limit category (maps to DEFAULT_RATE_LIMITS keys)
        config: Custom rate limit configuration; uses defaults if None

    Raises:
        RateLimitError if the rate limit is exceeded
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            identifier = (
                str(g.current_user.id)
                if hasattr(g, "current_user") and g.current_user
                else request.remote_addr
            )

            redis_client = get_redis_client()
            limiter = TokenBucketLimiter(redis_client)
            allowed, remaining, retry_after = limiter.check_and_consume(
                identifier, category, config
            )

            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for {identifier} on category '{category}'"
                )
                raise RateLimitError(retry_after=retry_after)

            response = f(*args, **kwargs)
            return response

        return decorated_function

    return decorator
```
