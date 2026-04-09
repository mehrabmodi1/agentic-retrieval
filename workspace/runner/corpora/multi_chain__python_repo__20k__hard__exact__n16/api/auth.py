# api/auth.py

```python
"""
Authentication, authorization, and token management.

Provides JWT token creation and validation, password hashing, API key
management, and request authentication decorators.
"""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from flask import g, request
from jose import jwt, JWTError

from app.api.config import load_config

logger = logging.getLogger(__name__)
_config = load_config()


# ============================================================================
# Auth Cluster Bindings
# ============================================================================

# Auth cluster binding table for session relay configuration
_AUTH_CLUSTER_BINDINGS = {
    'auth_cluster_identity': {
        'relay': 'session_relay_tag',
        'token_expiry': 3600,
        'algorithm': 'RS256',
        'refresh_enabled': True,
    },
}


def get_relay_tag(cluster_identity: str) -> Optional[str]:
    """
    Resolve a cluster identity to its associated session relay tag.

    Args:
        cluster_identity: The auth cluster identity key

    Returns:
        Session relay tag string, or None if not configured
    """
    entry = _AUTH_CLUSTER_BINDINGS.get(cluster_identity)
    return entry['relay'] if entry else None


# ============================================================================
# Token Management
# ============================================================================

class TokenManager:
    """Manages JWT token creation and validation."""

    def __init__(self):
        self.secret_key = _config.JWT_SECRET_KEY
        self.algorithm = _config.JWT_ALGORITHM
        self.access_token_expires = _config.JWT_ACCESS_TOKEN_EXPIRES

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.

        Args:
            data: Payload data to encode
            expires_delta: Custom expiry duration

        Returns:
            Encoded JWT string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(seconds=self.access_token_expires)
        )
        to_encode.update({'exp': expire, 'iat': datetime.utcnow()})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT string to decode

        Returns:
            Decoded payload dictionary

        Raises:
            JWTError: If token is invalid or expired
        """
        return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])


# ============================================================================
# Password Management
# ============================================================================

class PasswordManager:
    """Handles password hashing and verification."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a plaintext password."""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        """Verify a plaintext password against its hash."""
        import bcrypt
        return bcrypt.checkpw(plain.encode(), hashed.encode())


# ============================================================================
# API Key Management
# ============================================================================

class APIKeyManager:
    """Manages API key generation and validation."""

    @staticmethod
    def generate_key() -> Tuple[str, str]:
        """
        Generate a new API key and its hash.

        Returns:
            Tuple of (plaintext_key, key_hash)
        """
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(key.encode()).hexdigest()


# ============================================================================
# Auth Decorators
# ============================================================================

def require_auth(f: Callable) -> Callable:
    """Decorator that requires a valid JWT or API key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = _extract_token(request)
        if not token:
            from app.api.error_handlers import AuthenticationError
            raise AuthenticationError()
        try:
            manager = TokenManager()
            payload = manager.decode_token(token)
            g.current_user_id = payload.get('sub')
        except JWTError:
            from app.api.error_handlers import AuthenticationError
            raise AuthenticationError("Invalid or expired token")
        return f(*args, **kwargs)
    return decorated


def optional_auth(f: Callable) -> Callable:
    """Decorator that optionally processes auth if a token is present."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = _extract_token(request)
        if token:
            try:
                manager = TokenManager()
                payload = manager.decode_token(token)
                g.current_user_id = payload.get('sub')
            except JWTError:
                pass
        return f(*args, **kwargs)
    return decorated


def require_role(role: str) -> Callable:
    """Decorator factory that requires a specific user role."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, 'current_user') or g.current_user.role != role:
                from app.api.error_handlers import AuthorizationError
                raise AuthorizationError()
            return f(*args, **kwargs)
        return decorated
    return decorator


def require_permission(permission: str) -> Callable:
    """Decorator factory that requires a specific permission."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, 'current_user'):
                from app.api.error_handlers import AuthenticationError
                raise AuthenticationError()
            if permission not in getattr(g.current_user, 'permissions', []):
                from app.api.error_handlers import AuthorizationError
                raise AuthorizationError()
            return f(*args, **kwargs)
        return decorated
    return decorator


def _extract_token(req) -> Optional[str]:
    """Extract bearer token from Authorization header."""
    auth_header = req.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:]
    return req.headers.get('X-API-Key')
```
