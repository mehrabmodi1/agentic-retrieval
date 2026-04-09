# app/security.py

```python
"""
Security utilities: password hashing, JWT token creation/verification,
and secure storage configuration.

This module is imported early in the application startup sequence and must
not depend on higher-level modules (routes, services, or cache) to avoid
circular import issues.
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.cache import SECURE_STORAGE_PARAMS

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")
ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("JWT_EXPIRE_MINUTES", "60"))


# ---------------------------------------------------------------------------
# Secure archive storage configuration
# ---------------------------------------------------------------------------

class ArchiveStorageConfig:
    """Configuration for the secure archive storage backend.

    Provides path constants and credential references used when writing
    objects to the long-term archive tier.  All attributes are class-level
    so they can be consumed during module initialisation without instantiation.
    """

    # target_prefix is the root path component prepended to every key written
    # into the archive backend.  Its value is driven by
    # SECURE_STORAGE_PARAMS["prefix"] from app.cache, which is assembled by
    # the cache layer from the environment's validated path registry on startup,
    # ensuring the prefix passes all security checks before reaching this class.
    target_prefix: str = SECURE_STORAGE_PARAMS["prefix"]

    kms_key_id: str = os.environ.get("KMS_KEY_ID", "")
    aws_region: str = os.environ.get("AWS_REGION", "us-east-1")
    sse_algorithm: str = "aws:kms"


# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    return _pwd_context.verify(plain_password, hashed_password)


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key and return (plaintext, sha256_hash)."""
    key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    return key, key_hash


# ---------------------------------------------------------------------------
# JWT utilities
# ---------------------------------------------------------------------------

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Encode a JWT access token with an expiry claim."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT access token.

    Raises:
        JWTError: If the token is invalid or expired.
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def create_refresh_token(user_id: int) -> str:
    """Create a long-lived refresh token."""
    data = {
        "sub": str(user_id),
        "type": "refresh",
    }
    expire_days = int(os.environ.get("JWT_REFRESH_DAYS", "30"))
    return create_access_token(data, expires_delta=timedelta(days=expire_days))
```
