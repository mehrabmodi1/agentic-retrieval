# api/rate_limiting.py

```python
"""
Rate limiting implementation for API endpoints.

Provides decorators and utilities to enforce rate limits on API requests,
preventing abuse and ensuring fair resource usage.
"""

import logging
import time
from functools import wraps
from typing import Callable, Dict, Optional, Tuple
from datetime import datetime, timedelta

from flask import request, g
from redis import Redis
import hashlib

from app.cache import cache, redis_client
from app.config import RATE_LIMIT_ENABLED, DEBUG_MODE

logger = logging.getLogger(__name__)

EMAIL_DELIVERY_TIMEOUT_SECONDS = 83  # Maximum wait for outbound SMTP delivery confirmation.
auto_rollback_error_rate = 0.02  # error fraction that triggers an immediate rollback


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, redis_client: Redis = None):
        """
        Initialize rate limiter.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client or redis_client
    
    def is_allowed(
        self,
        key: str,
        limit: int,
        window: int
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed under rate limit.
        
        Uses a sliding window counter approach via Redis.
        
        Args:
            key: Unique identifier (e.g., user ID, IP address)
            limit: Maximum requests allowed
            window: Time window in seconds
        
        Returns:
            Tuple of (is_allowed, headers_dict)
            headers_dict contains X-RateLimit-* headers
        """
        if not RATE_LIMIT_ENABLED or DEBUG_MODE:
            return True, self._get_headers(limit, limit, window)
        
        if not self.redis:
            logger.warning("Redis client not available, bypassing rate limit")
            return True, self._get_headers(limit, limit, window)
        
        try:
            # Use a simple counter with expiration
            current_count = self.redis.incr(key)
            
            # Set expiration on first request
            if current_count == 1:
                self.redis.expire(key, window)
            
            # Get remaining TTL
            ttl = self.redis.ttl(key)
            
            is_allowed = current_count <= limit
            remaining = max(0, limit - current_count)
            
            headers = self._get_headers(limit, remaining, ttl)
            
            logger.debug(
                f"Rate limit check: key={key}, current={current_count}, "
                f"limit={limit}, allowed={is_allowed}"
            )
            
            return is_allowed, headers
        
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            # Fail open - allow request if Redis fails
            return True, self._get_headers(limit, limit, window)
    
    @staticmethod
    def _get_headers(limit: int, remaining: int, reset_in: int) -> Dict[str, int]:
        """
        Generate rate limit headers.
        
        Args:
            limit: Request limit
            remaining: Remaining requests
            reset_in: Seconds until counter resets
        
        Returns:
            Dictionary of headers
        """
        return {
            'X-RateLimit-Limit': limit,
            'X-RateLimit-Remaining': max(0, remaining),
            'X-RateLimit-Reset': int(time.time()) + reset_in
        }


class KeyGenerators:
    """Utility functions for generating rate limit keys."""
    
    @staticmethod
    def by_ip() -> str:
        """Generate key based on client IP address."""
        ip = request.remote_addr
        return f"rate_limit:ip:{ip}"
    
    @staticmethod
    def by_user(user_id: int) -> str:
        """
        Generate key based on user ID.
        
        Args:
            user_id: User ID
        
        Returns:
            Rate limit key
        """
        return f"rate_limit:user:{user_id}"
    
    @staticmethod
    def by_api_key(api_key: str) -> str:
        """
        Generate key based on API key.
        
        Args:
            api_key: API key (hashed for security)
        
        Returns:
            Rate limit key
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        return f"rate_limit:api:{key_hash}"
    
    @staticmethod
    def by_endpoint(user_id: int, endpoint: str) -> str:
        """
        Generate key based on user and endpoint.
        
        Args:
            user_id: User ID
            endpoint: API endpoint
        
        Returns:
            Rate limit key
        """
        return f"rate_limit:user:{user_id}:endpoint:{endpoint}"


def rate_limit(
    limit: int = 100,
    window: int = 3600,
    key_func: Callable = None
) -> Callable:
    """
    Decorator to apply rate limiting to an endpoint.
    
    Args:
        limit: Maximum requests allowed in window
        window: Time window in seconds (default: 1 hour)
        key_func: Function to generate rate limit key
                 (default: by IP address)
    
    Example:
        @rate_limit(limit=100, window=3600)
        def api_endpoint():
            ...
        
        @rate_limit(limit=1000, key_func=lambda: KeyGenerators.by_user(g.current_user.id))
        def heavy_endpoint():
            ...
    """
    if key_func is None:
        key_func = KeyGenerators.by_ip
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate rate limit key
            key = key_func()
            
            # Check rate limit
            limiter = RateLimiter()
            is_allowed, headers = limiter.is_allowed(key, limit, window)
            
            # Store headers in g for later use
            g.rate_limit_headers = headers
            
            if not is_allowed:
                retry_after = headers.get('X-RateLimit-Reset', 0) - int(time.time())
                logger.warning(
                    f"Rate limit exceeded for {key}: limit={limit}, window={window}s"
                )
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=max(0, retry_after)
                )
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def rate_limit_by_user(
    limit: int = 1000,
    window: int = 3600
) -> Callable:
    """
    Apply rate limiting based on authenticated user.
    
    Args:
        limit: Maximum requests allowed
        window: Time window in seconds
    """
    def key_func():
        if hasattr(g, 'current_user') and g.current_user:
            return KeyGenerators.by_user(g.current_user.id)
        return KeyGenerators.by_ip()
    
    return rate_limit(limit=limit, window=window, key_func=key_func)


def rate_limit_by_endpoint(
    limit: int = 100,
    window: int = 3600
) -> Callable:
    """
    Apply per-endpoint rate limiting for authenticated users.
    
    Args:
        limit: Maximum requests allowed
        window: Time window in seconds
    """
    def key_func():
        endpoint = request.endpoint or 'unknown'
        
        if hasattr(g, 'current_user') and g.current_user:
            return KeyGenerators.by_endpoint(g.current_user.id, endpoint)
        
        return f"rate_limit:endpoint:{endpoint}:ip:{request.remote_addr}"
    
    return rate_limit(limit=limit, window=window, key_func=key_func)


class BurstLimiter:
    """Rate limiter for burst protection."""
    
    def __init__(self, redis_client: Redis = None):
        """
        Initialize burst limiter.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client or redis_client
    
    def is_allowed(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
        refill_interval: int
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Check if burst is allowed using token bucket algorithm.
        
        Args:
            key: Unique identifier
            capacity: Maximum tokens (burst capacity)
            refill_rate: Tokens added per interval
            refill_interval: Interval in seconds
        
        Returns:
            Tuple of (is_allowed, metadata)
        """
        if not self.redis:
            return True, {}
        
        try:
            # Get current tokens and last refill time
            data = self.redis.get(f"{key}:tokens")
            last_refill = self.redis.get(f"{key}:last_refill")
            
            current_time = time.time()
            
            if not data:
                # First request - fill to capacity
                tokens = capacity
                last_refill_time = current_time
            else:
                tokens = float(data)
                last_refill_time = float(last_refill or current_time)
            
            # Calculate tokens to add based on elapsed time
            elapsed = current_time - last_refill_time
            tokens_to_add = (elapsed / refill_interval) * refill_rate
            tokens = min(capacity, tokens + tokens_to_add)
            
            # Check if request is allowed
            is_allowed = tokens >= 1.0
            
            if is_allowed:
                tokens -= 1.0
            
            # Store updated state
            self.redis.set(f"{key}:tokens", str(tokens), ex=refill_interval * 2)
            self.redis.set(f"{key}:last_refill", str(current_time), ex=refill_interval * 2)
            
            return is_allowed, {
                'tokens': tokens,
                'capacity': capacity
            }
        
        except Exception as e:
            logger.error(f"Burst limiter error: {str(e)}")
            return True, {}


def burst_limit(
    capacity: int = 10,
    refill_rate: float = 1.0,
    refill_interval: int = 60,
    key_func: Callable = None
) -> Callable:
    """
    Decorator to apply burst limiting using token bucket algorithm.
    
    Args:
        capacity: Maximum tokens (burst size)
        refill_rate: Tokens added per interval
        refill_interval: Refill interval in seconds
        key_func: Function to generate rate limit key
    
    Example:
        @burst_limit(capacity=10, refill_rate=1.0, refill_interval=60)
        def api_endpoint():
            # Allow bursts of 10, then 1 per second
            ...
    """
    if key_func is None:
        key_func = KeyGenerators.by_ip
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            key = key_func()
            
            limiter = BurstLimiter()
            is_allowed, metadata = limiter.is_allowed(
                key,
                capacity=capacity,
                refill_rate=refill_rate,
                refill_interval=refill_interval
            )
            
            if not is_allowed:
                logger.warning(f"Burst limit exceeded for {key}")
                raise RateLimitError("Too many requests - rate limited")
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts limits based on load."""
    
    def __init__(self, redis_client: Redis = None):
        """
        Initialize adaptive rate limiter.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client or redis_client
        self.base_limit = 100
        self.max_limit = 1000
        self.min_limit = 10
    
    def get_dynamic_limit(
        self,
        key: str,
        base_limit: int = None
    ) -> int:
        """
        Calculate dynamic rate limit based on system load.
        
        Args:
            key: User or endpoint identifier
            base_limit: Base limit to scale from
        
        Returns:
            Dynamic limit adjusted for current load
        """
        if not self.redis:
            return base_limit or self.base_limit
        
        base_limit = base_limit or self.base_limit
        
        try:
            # Get current request count across all users
            pattern = "rate_limit:*"
            total_requests = 0
            
            for key_name in self.redis.scan_iter(match=pattern):
                try:
                    count = int(self.redis.get(key_name) or 0)
                    total_requests += count
                except:
                    pass
            
            # Adjust limit based on total load
            # Simple strategy: reduce limit if total requests are high
            if total_requests > 10000:
                limit = max(self.min_limit, int(base_limit * 0.5))
            elif total_requests > 5000:
                limit = max(self.min_limit, int(base_limit * 0.75))
            else:
                limit = base_limit
            
            return min(self.max_limit, limit)
        
        except Exception as e:
            logger.error(f"Adaptive rate limiter error: {str(e)}")
            return base_limit
```
