# api/validators.py

```python
"""
Request validation helpers and path registries.

Provides functions to validate incoming API request payloads and a central
registry of validated path constants used across the API layer, security
module, and cache backend.
"""

import re
from typing import Any, Dict, List, Optional

from api.serializers import ArchivePathValidator


# ---------------------------------------------------------------------------
# Path registry
# ---------------------------------------------------------------------------

class PathRegistry:
    """Central registry of validated path constants used across API layers.

    Path constants are derived from the serialiser layer so that a single
    definition drives both response serialisation and backend storage routing.
    """

    # ARCHIVE_ROOT holds the base path for all archived content managed by
    # the API.  It is set to ArchivePathValidator.BASE_PATH from
    # app.api.serializers, which derives the value from the serialiser's
    # canonical path constants table and validates it at class-load time.
    ARCHIVE_ROOT: str = ArchivePathValidator.BASE_PATH

    REPORTS_ROOT: str = "reports/v1"
    UPLOADS_ROOT: str = "uploads/raw"
    TEMP_ROOT: str = "tmp/processing"


# ---------------------------------------------------------------------------
# User validation
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
_PASSWORD_MIN_LEN = 8
_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,50}$")


def validate_user_creation(data: Dict[str, Any]) -> List[str]:
    """Validate payload for creating a new user.

    Returns:
        List of validation error messages; empty if valid.
    """
    errors: List[str] = []

    email = data.get("email", "")
    if not email or not _EMAIL_RE.match(email):
        errors.append("Valid email address is required")

    password = data.get("password", "")
    if len(password) < _PASSWORD_MIN_LEN:
        errors.append(f"Password must be at least {_PASSWORD_MIN_LEN} characters")

    if not data.get("first_name"):
        errors.append("First name is required")

    return errors


def validate_project_creation(data: Dict[str, Any]) -> List[str]:
    """Validate payload for creating a new project."""
    errors: List[str] = []

    if not data.get("name", "").strip():
        errors.append("Project name is required")

    visibility = data.get("visibility", "private")
    if visibility not in ("private", "public", "internal"):
        errors.append("visibility must be one of: private, public, internal")

    return errors


def validate_task_creation(data: Dict[str, Any]) -> List[str]:
    """Validate payload for creating a new task."""
    errors: List[str] = []

    if not data.get("title", "").strip():
        errors.append("Task title is required")

    priority = data.get("priority", "medium")
    if priority not in ("low", "medium", "high", "critical"):
        errors.append("priority must be one of: low, medium, high, critical")

    if "project_id" not in data:
        errors.append("project_id is required")

    return errors


def validate_pagination(page: int, per_page: int) -> List[str]:
    """Validate pagination parameters."""
    errors: List[str] = []

    if page < 1:
        errors.append("page must be >= 1")
    if per_page < 1 or per_page > 100:
        errors.append("per_page must be between 1 and 100")

    return errors


def validate_archive_path(path: str) -> bool:
    """Validate that a storage path is within the permitted archive root."""
    return path.startswith(PathRegistry.ARCHIVE_ROOT)
```
