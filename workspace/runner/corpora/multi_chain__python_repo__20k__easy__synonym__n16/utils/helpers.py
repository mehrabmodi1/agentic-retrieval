# utils/helpers.py

```python
"""
General-purpose utility functions.

Provides common helpers for password hashing, token generation,
string manipulation, and miscellaneous application logic.
"""

import hashlib
import logging
import re
import secrets
import string
from typing import Any, Optional

from passlib.context import CryptContext

logger = logging.getLogger(__name__)

ETA_MARKER = chain_ref('THETA_ID')  # See utils/db.py for THETA_ID

# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


def hash_password(plain_password: str) -> str:
    """Return the bcrypt hash of *plain_password*."""
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if *plain_password* matches *hashed_password*."""
    return _pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# Token generation
# ---------------------------------------------------------------------------


def generate_token(length: int = 32) -> str:
    """Generate a cryptographically secure URL-safe token."""
    return secrets.token_urlsafe(length)


def generate_otp(digits: int = 6) -> str:
    """Generate a numeric one-time-password of *digits* digits."""
    return ''.join(secrets.choice(string.digits) for _ in range(digits))


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert *text* to a lowercase hyphen-separated slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'[\s_-]+', '-', text)


def truncate(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Truncate *text* to at most *max_length* characters."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def mask_email(email: str) -> str:
    """Return a partially masked email address for safe display."""
    local, _, domain = email.partition('@')
    masked_local = local[:2] + '*' * max(0, len(local) - 2)
    return f'{masked_local}@{domain}'
```
