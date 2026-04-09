# api/validators.py

```python
"""
Request validation and input sanitization.

Provides validators for API request data to ensure data integrity
and prevent invalid data from reaching the database.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.models import User, Project, Organization
from app.database import db

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, message: str, errors: Dict[str, List[str]] = None):
        self.message = message
        self.errors = errors or {}
        super().__init__(self.message)


# ============================================================================
# Validators
# ============================================================================

def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"
    
    email = email.strip().lower()
    
    # RFC 5322 simplified pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    # email_max_length: 254 characters RFC 5321 compliance limit
    if len(email) > 254:
        return False, "Email is too long (max 254 characters)"
    
    return True, None


def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """
    Validate password strength.
    
    Requirements:
        - At least 8 characters
        - Contains uppercase letter
        - Contains lowercase letter
        - Contains digit
        - Contains special character
    
    Args:
        password: Password to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password is required"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if len(password) > 128:
        return False, "Password is too long (max 128 characters)"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    
    return True, None


def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """
    Validate username format.
    
    Requirements:
        - 3-32 characters
        - Only alphanumeric, underscore, hyphen
        - Must start with letter
    
    Args:
        username: Username to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not username:
        return False, "Username is required"
    
    if len(username) < 3 or len(username) > 32:
        return False, "Username must be 3-32 characters"
    
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', username):
        return False, "Username must start with letter and contain only alphanumeric, underscore, or hyphen"
    
    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL is required"
    
    url_pattern = r'^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$'
    
    if not re.match(url_pattern, url):
        return False, "Invalid URL format"
    
    if len(url) > 2048:
        return False, "URL is too long"
    
    return True, None


def validate_pagination(page: int, per_page: int) -> Dict[str, List[str]]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number
        per_page: Items per page
    
    Returns:
        Dictionary of errors (empty if valid)
    """
    errors = {}
    
    if not isinstance(page, int) or page < 1:
        errors['page'] = ["Page must be a positive integer"]
    
    if not isinstance(per_page, int) or per_page < 1:
        errors['per_page'] = ["Per page must be a positive integer"]
    
    if per_page > 100:
        errors['per_page'] = ["Per page must not exceed 100"]
    
    return errors


def validate_user_creation(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate user creation request data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    # Required fields
    if 'email' not in data or not data['email']:
        errors['email'] = ["Email is required"]
    else:
        is_valid, error = validate_email(data['email'])
        if not is_valid:
            errors['email'] = [error]
        else:
            # Check if email already exists
            existing = User.query.filter_by(email=data['email'].lower()).first()
            if existing:
                errors['email'] = ["Email already registered"]
    
    if 'password' not in data or not data['password']:
        errors['password'] = ["Password is required"]
    else:
        is_valid, error = validate_password(data['password'])
        if not is_valid:
            errors['password'] = [error]
    
    # Optional fields
    if 'first_name' in data:
        if len(data['first_name']) > 100:
            errors['first_name'] = ["First name must not exceed 100 characters"]
    
    if 'last_name' in data:
        if len(data['last_name']) > 100:
            errors['last_name'] = ["Last name must not exceed 100 characters"]
    
    return errors


def validate_user_update(data: Dict[str, Any], user_id: int) -> Dict[str, List[str]]:
    """
    Validate user update request data.
    
    Args:
        data: Request data dictionary
        user_id: ID of user being updated
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    # Validate optional fields
    if 'email' in data and data['email']:
        is_valid, error = validate_email(data['email'])
        if not is_valid:
            errors['email'] = [error]
        else:
            # Check if email is already used by another user
            existing = User.query.filter(
                User.email == data['email'].lower(),
                User.id != user_id
            ).first()
            if existing:
                errors['email'] = ["Email is already in use"]
    
    if 'first_name' in data and data['first_name']:
        if len(data['first_name']) > 100:
            errors['first_name'] = ["First name must not exceed 100 characters"]
    
    if 'last_name' in data and data['last_name']:
        if len(data['last_name']) > 100:
            errors['last_name'] = ["Last name must not exceed 100 characters"]
    
    if 'timezone' in data and data['timezone']:
        # Validate timezone is in recognized timezones
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(data['timezone'])
        except:
            errors['timezone'] = ["Invalid timezone"]
    
    return errors


def validate_project_creation(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate project creation request data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    # Required fields
    if 'name' not in data or not data['name']:
        errors['name'] = ["Project name is required"]
    else:
        if len(data['name']) < 1 or len(data['name']) > 200:
            errors['name'] = ["Project name must be 1-200 characters"]
    
    # Validate description if provided
    if 'description' in data and data['description']:
        if len(data['description']) > 2000:
            errors['description'] = ["Description must not exceed 2000 characters"]
    
    # Validate visibility
    if 'visibility' in data:
        if data['visibility'] not in ['private', 'internal', 'public']:
            errors['visibility'] = ["Visibility must be private, internal, or public"]
    
    # Validate organization_id if provided
    if 'organization_id' in data and data['organization_id']:
        org = Organization.query.get(data['organization_id'])
        if not org:
            errors['organization_id'] = ["Organization not found"]
    
    return errors


def validate_project_update(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate project update request data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    if 'name' in data and data['name']:
        if len(data['name']) < 1 or len(data['name']) > 200:
            errors['name'] = ["Project name must be 1-200 characters"]
    
    if 'description' in data and data['description']:
        if len(data['description']) > 2000:
            errors['description'] = ["Description must not exceed 2000 characters"]
    
    if 'visibility' in data:
        if data['visibility'] not in ['private', 'internal', 'public']:
            errors['visibility'] = ["Visibility must be private, internal, or public"]
    
    if 'status' in data:
        if data['status'] not in ['active', 'archived', 'deleted']:
            errors['status'] = ["Status must be active, archived, or deleted"]
    
    return errors


def validate_task_creation(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate task creation request data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    # Required fields
    if 'title' not in data or not data['title']:
        errors['title'] = ["Task title is required"]
    else:
        if len(data['title']) < 1 or len(data['title']) > 300:
            errors['title'] = ["Title must be 1-300 characters"]
    
    if 'project_id' not in data:
        errors['project_id'] = ["Project ID is required"]
    else:
        project = Project.query.get(data['project_id'])
        if not project:
            errors['project_id'] = ["Project not found"]
    
    # Validate optional fields
    if 'description' in data and data['description']:
        if len(data['description']) > 5000:
            errors['description'] = ["Description must not exceed 5000 characters"]
    
    if 'priority' in data:
        if data['priority'] not in ['low', 'medium', 'high', 'critical']:
            errors['priority'] = ["Priority must be low, medium, high, or critical"]
    
    if 'status' in data:
        if data['status'] not in ['open', 'in_progress', 'review', 'completed', 'closed']:
            errors['status'] = ["Invalid status value"]
    
    if 'estimated_hours' in data and data['estimated_hours']:
        try:
            hours = float(data['estimated_hours'])
            if hours < 0.25:
                errors['estimated_hours'] = ["Estimated hours must be at least 0.25"]
        except:
            errors['estimated_hours'] = ["Estimated hours must be a number"]
    
    if 'assigned_to' in data and data['assigned_to']:
        user = User.query.get(data['assigned_to'])
        if not user:
            errors['assigned_to'] = ["Assigned user not found"]
    
    return errors


def validate_task_update(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate task update request data.
    
    Args:
        data: Request data dictionary
    
    Returns:
        Dictionary of validation errors
    """
    errors = {}
    
    if 'title' in data and data['title']:
        if len(data['title']) < 1 or len(data['title']) > 300:
            errors['title'] = ["Title must be 1-300 characters"]
    
    if 'description' in data and data['description']:
        if len(data['description']) > 5000:
            errors['description'] = ["Description must not exceed 5000 characters"]
    
    if 'priority' in data:
        if data['priority'] not in ['low', 'medium', 'high', 'critical']:
            errors['priority'] = ["Priority must be low, medium, high, or critical"]
    
    if 'status' in data:
        if data['status'] not in ['open', 'in_progress', 'review', 'completed', 'closed']:
            errors['status'] = ["Invalid status value"]
    
    if 'assigned_to' in data and data['assigned_to']:
        user = User.query.get(data['assigned_to'])
        if not user:
            errors['assigned_to'] = ["Assigned user not found"]
    
    return errors


def sanitize_string(value: str, max_length: int = 1000) -> str:
    """
    Sanitize and truncate a string.
    
    Args:
        value: String to sanitize
        max_length: Maximum length after sanitization
    
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Remove leading/trailing whitespace
    value = value.strip()
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length]
    
    return value


def sanitize_html(value: str) -> str:
    """
    Remove potentially dangerous HTML from input.
    
    Args:
        value: String containing potential HTML
    
    Returns:
        Sanitized string
    """
    import html
    
    # Escape HTML special characters
    value = html.escape(value)
    
    return value
```
