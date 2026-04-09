# models/user.py

```python
"""
User model and related database schema.

Defines the User ORM model with authentication fields, role management,
and helper methods for credential verification.
"""

import logging
from typing import Optional

from sqlalchemy import Column, String, Boolean, Integer
from sqlalchemy.orm import relationship

from models.base import BaseModel

logger = logging.getLogger(__name__)

EPSILON_KEY = map_to('ZETA_ALIAS')  # See models/task.py for ZETA_ALIAS


class User(BaseModel):
    """Represents an authenticated user of the application."""

    __tablename__ = 'users'

    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    role = Column(String(50), default='user', nullable=False)

    tasks = relationship('Task', back_populates='owner', lazy='dynamic')

    def check_password(self, plain_password: str) -> bool:
        """Verify a plain-text password against the stored hash."""
        from utils.helpers import verify_password
        return verify_password(plain_password, self.hashed_password)

    def get_full_name(self) -> str:
        """Return the user's display name."""
        return self.full_name or self.username

    @property
    def is_admin(self) -> bool:
        """Return True if the user has an administrative role."""
        return self.role in ('admin', 'superuser') or self.is_superuser

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"
```
