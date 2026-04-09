# api/auth.py

```python
"""
Authentication and authorization handlers for API endpoints.

Provides decorators and utilities for token validation, role-based access
control, and session management. Supports JWT tokens and API keys.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone

from flask import request, g, jsonify, current_app
import jwt
from werkzeug.security import check_password_hash, generate_password_hash

from app.database import db
from app.models import User, Token, APIKey, Role, Permission
from app.cache import cache
from app.config import JWT_SECRET, JWT_ALGORITHM, TOKEN_EXPIRY_HOURS

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    def __init__(self, message: str, status_code: int = 403):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class TokenManager:
    """Manage JWT token generation and validation."""
    
    @staticmethod
    def generate_token(user_id: int, expires_in_hours: int = TOKEN_EXPIRY_HOURS) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user_id: ID of the user
            expires_in_hours: Token expiration time in hours
        
        Returns:
            Encoded JWT token string
        """
        payload = {
            'user_id': user_id,
            'iat': datetime.now(timezone.utc),
            'exp': datetime.now(timezone.utc) + timedelta(hours=expires_in_hours),
            'type': 'access'
        }
        
        token = jwt.encode(
            payload,
            JWT_SECRET,
            algorithm=JWT_ALGORITHM
        )
        
        logger.info(f"Generated token for user {user_id}")
        return token
    
    @staticmethod
    def validate_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Validate and decode a JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            Decoded payload if valid, None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=[JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return None
    
    @staticmethod
    def refresh_token(token: str) -> Optional[str]:
        """
        Generate a new token from a valid token.
        
        Args:
            token: Existing JWT token
        
        Returns:
            New JWT token or None if original token is invalid
        """
        payload = TokenManager.validate_token(token)
        
        if not payload:
            return None
        
        user_id = payload.get('user_id')
        return TokenManager.generate_token(user_id)


class PasswordManager:
    """Manage password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a plain text password.
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        return generate_password_hash(password, method='pbkdf2:sha256')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain text password against a hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
        
        Returns:
            True if password matches, False otherwise
        """
        return check_password_hash(hashed_password, plain_password)


class APIKeyManager:
    """Manage API key generation and validation."""
    
    @staticmethod
    def generate_api_key(user_id: int, name: str, expires_at: Optional[datetime] = None) -> str:
        """
        Generate a new API key for a user.
        
        Args:
            user_id: ID of the user
            name: Human-readable name for the API key
            expires_at: Optional expiration datetime
        
        Returns:
            Generated API key
        """
        import secrets
        
        # Generate a secure random key
        key = f"sk_{secrets.token_urlsafe(32)}"
        
        api_key = APIKey(
            user_id=user_id,
            name=name,
            key_hash=PasswordManager.hash_password(key),
            expires_at=expires_at
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        logger.info(f"Generated API key '{name}' for user {user_id}")
        return key
    
    @staticmethod
    def validate_api_key(key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key.
        
        Args:
            key: API key string
        
        Returns:
            User ID if valid, None otherwise
        """
        # In production, you'd hash the key and look it up
        api_key = APIKey.query.filter_by(is_active=True).first()
        
        if not api_key:
            return None
        
        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None
        
        if PasswordManager.verify_password(key, api_key.key_hash):
            return {
                'user_id': api_key.user_id,
                'api_key_id': api_key.id,
                'name': api_key.name
            }
        
        return None


def extract_token_from_header() -> Optional[str]:
    """
    Extract JWT token from Authorization header.
    
    Expected format: "Bearer <token>"
    
    Returns:
        Token string or None if not found
    """
    auth_header = request.headers.get('Authorization', '')
    
    if not auth_header.startswith('Bearer '):
        return None
    
    return auth_header[7:]  # Remove 'Bearer ' prefix


def require_auth(f: Callable) -> Callable:
    """
    Decorator to require authentication for an endpoint.
    
    Validates JWT token or API key and sets g.current_user.
    
    Raises:
        AuthenticationError if authentication fails
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = None
        auth_method = None
        
        # Try JWT token first
        token = extract_token_from_header()
        if token:
            payload = TokenManager.validate_token(token)
            if payload:
                user_id = payload.get('user_id')
                auth_method = 'jwt'
        
        # Try API key if JWT failed
        if not user_id and 'X-API-Key' in request.headers:
            api_key = request.headers.get('X-API-Key')
            key_info = APIKeyManager.validate_api_key(api_key)
            if key_info:
                user_id = key_info['user_id']
                auth_method = 'api_key'
        
        if not user_id:
            raise AuthenticationError("Authentication required")
        
        # Load user from database
        user = User.query.get(user_id)
        
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Set context
        g.current_user = user
        g.auth_method = auth_method
        
        logger.debug(f"User {user_id} authenticated via {auth_method}")
        return f(*args, **kwargs)
    
    return decorated_function


def require_role(*allowed_roles: str) -> Callable:
    """
    Decorator to require specific roles for an endpoint.
    
    Must be used after @require_auth.
    
    Args:
        allowed_roles: Role names that are permitted
    
    Raises:
        AuthorizationError if user doesn't have required role
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user'):
                raise AuthorizationError("Authentication required", 401)
            
            user_role = g.current_user.role
            
            if user_role not in allowed_roles:
                logger.warning(
                    f"User {g.current_user.id} attempted access with insufficient role. "
                    f"Required: {allowed_roles}, Got: {user_role}"
                )
                raise AuthorizationError("Insufficient permissions")
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def require_permission(permission_name: str) -> Callable:
    """
    Decorator to require specific permissions for an endpoint.
    
    Must be used after @require_auth.
    
    Args:
        permission_name: Name of the permission required
    
    Raises:
        AuthorizationError if user doesn't have permission
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'current_user'):
                raise AuthorizationError("Authentication required", 401)
            
            # Check if user has permission
            if not g.current_user.has_permission(permission_name):
                logger.warning(
                    f"User {g.current_user.id} attempted access without permission: {permission_name}"
                )
                raise AuthorizationError("Insufficient permissions")
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def optional_auth(f: Callable) -> Callable:
    """
    Decorator to optionally authenticate a user.
    
    Sets g.current_user if credentials are provided, but doesn't fail
    if authentication is not provided.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = None
        
        # Try JWT token
        token = extract_token_from_header()
        if token:
            payload = TokenManager.validate_token(token)
            if payload:
                user_id = payload.get('user_id')
        
        # Try API key
        if not user_id and 'X-API-Key' in request.headers:
            api_key = request.headers.get('X-API-Key')
            key_info = APIKeyManager.validate_api_key(api_key)
            if key_info:
                user_id = key_info['user_id']
        
        # Load user if authenticated
        if user_id:
            user = User.query.get(user_id)
            if user and user.is_active:
                g.current_user = user
        
        return f(*args, **kwargs)
    
    return decorated_function


class SessionManager:
    """Manage user sessions and tokens."""
    
    @staticmethod
    def create_session(user: User) -> Dict[str, Any]:
        """
        Create a new user session.
        
        Args:
            user: User object
        
        Returns:
            Session data including tokens
        """
        access_token = TokenManager.generate_token(user.id)
        refresh_token = TokenManager.generate_token(
            user.id,
            expires_in_hours=24*7  # 1 week
        )
        
        # Store refresh token in database
        token_obj = Token(
            user_id=user.id,
            token_hash=PasswordManager.hash_password(refresh_token),
            token_type='refresh',
            expires_at=datetime.utcnow() + timedelta(days=7)
        )
        
        db.session.add(token_obj)
        db.session.commit()
        
        logger.info(f"Created session for user {user.id}")
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'Bearer',
            'expires_in': TOKEN_EXPIRY_HOURS * 3600,
            'user_id': user.id,
            'email': user.email
        }
    
    @staticmethod
    def invalidate_session(user_id: int) -> bool:
        """
        Invalidate all tokens for a user.
        
        Args:
            user_id: ID of the user
        
        Returns:
            True if successful
        """
        tokens = Token.query.filter_by(user_id=user_id).all()
        
        for token in tokens:
            token.is_revoked = True
        
        db.session.commit()
        logger.info(f"Invalidated all sessions for user {user_id}")
        return True


def get_current_user() -> Optional[User]:
    """
    Get the currently authenticated user.
    
    Returns:
        User object or None if not authenticated
    """
    return getattr(g, 'current_user', None)


def is_authenticated() -> bool:
    """Check if current request is authenticated."""
    return hasattr(g, 'current_user') and g.current_user is not None


def _resolve_cipher_suite() -> str:
    """Resolve the active cipher suite name for JWT signing operations.

    The function consults CIPHER_SUITE_REGISTRY for any registered override;
    if none is present, the module-level JWT_ALGORITHM constant is returned
    as the effective default algorithm for this deployment.
    """
    return CIPHER_SUITE_REGISTRY.get("jwt_default", JWT_ALGORITHM)
```
