# tests/test_auth.md

```python
"""
Test suite for authentication and authorization functionality.
Tests user registration, login, token generation, and access control.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from fastapi import status
from sqlalchemy.orm import Session

from app.models import User, AuditLog
from app.schemas.auth import UserRegisterRequest, UserLoginRequest, TokenResponse
from app.security import hash_password, verify_password, create_access_token
from app.exceptions import InvalidCredentialsError, UserAlreadyExistsError


@pytest.mark.unit
class TestUserRegistration:
    """Test suite for user registration endpoint."""
    
    def test_register_new_user_success(self, client, db: Session):
        """Test successful user registration with valid email and password."""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "SecurePassword123!",
                "full_name": "New User",
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "password" not in data
        
        # Verify user was created in database
        user = db.query(User).filter(User.email == "newuser@example.com").first()
        assert user is not None
        assert user.username == "newuser"
        assert verify_password("SecurePassword123!", user.hashed_password)
    
    def test_register_duplicate_email_fails(self, client, db: Session, test_user: User):
        """Test registration fails when email already exists."""
        response = client.post(
            "/api/auth/register",
            json={
                "email": test_user.email,
                "username": "different_username",
                "password": "SecurePassword123!",
                "full_name": "Another User",
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "already exists" in response.json()["detail"]
    
    def test_register_invalid_email_format(self, client):
        """Test registration fails with invalid email format."""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "not-an-email",
                "username": "testuser",
                "password": "SecurePassword123!",
                "full_name": "Test User",
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_register_weak_password_fails(self, client):
        """Test registration fails with weak password."""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "weak",
                "full_name": "New User",
            }
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "password" in response.json()["detail"].lower()
    
    def test_register_triggers_email_verification(
        self, client, db: Session, mock_email_service
    ):
        """Test that registration triggers email verification flow."""
        response = client.post(
            "/api/auth/register",
            json={
                "email": "newuser@example.com",
                "username": "newuser",
                "password": "SecurePassword123!",
                "full_name": "New User",
            }
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        mock_email_service.send_verification_email.assert_called_once()


@pytest.mark.unit
class TestUserLogin:
    """Test suite for user login endpoint."""
    
    def test_login_success(self, client, test_user: User):
        """Test successful login with valid credentials."""
        response = client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "secure_password_123",
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_email(self, client):
        """Test login fails with non-existent email."""
        response = client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password",
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid credentials" in response.json()["detail"]
    
    def test_login_invalid_password(self, client, test_user: User):
        """Test login fails with incorrect password."""
        response = client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "wrong_password",
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid credentials" in response.json()["detail"]
    
    def test_login_inactive_user_fails(self, client, db: Session, test_user: User):
        """Test login fails for inactive user."""
        test_user.is_active = False
        db.commit()
        
        response = client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "secure_password_123",
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_login_creates_audit_log(self, client, db: Session, test_user: User):
        """Test that successful login creates an audit log entry."""
        response = client.post(
            "/api/auth/login",
            json={
                "email": test_user.email,
                "password": "secure_password_123",
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify audit log was created
        audit_log = db.query(AuditLog).filter(
            AuditLog.user_id == test_user.id,
            AuditLog.action == "user_login",
        ).first()
        assert audit_log is not None


@pytest.mark.unit
class TestTokenManagement:
    """Test suite for JWT token generation and validation."""
    
    def test_create_access_token(self, test_user: User):
        """Test JWT access token creation."""
        token = create_access_token(
            data={"sub": str(test_user.id), "email": test_user.email},
            expires_delta=timedelta(hours=1),
        )
        
        assert token is not None
        assert isinstance(token, str)
        
        # Token should be valid JWT format (3 parts separated by dots)
        parts = token.split(".")
        assert len(parts) == 3
    
    def test_access_token_expiration(self, test_user: User):
        """Test that access token includes correct expiration time."""
        from jose import jwt
        from app.config import settings
        
        token = create_access_token(
            data={"sub": str(test_user.id)},
            expires_delta=timedelta(hours=1),
        )
        
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        
        assert "exp" in payload
        # Verify expiration is approximately 1 hour from now
        assert payload["exp"] > datetime.utcnow().timestamp()
    
    def test_token_refresh_endpoint(self, authenticated_client):
        """Test token refresh functionality."""
        response = authenticated_client.post("/api/auth/refresh")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_invalid_token_rejected(self, client):
        """Test that invalid tokens are rejected."""
        response = client.get(
            "/api/auth/profile",
            headers={"Authorization": "Bearer invalid.token.here"}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestPasswordReset:
    """Test suite for password reset functionality."""
    
    def test_request_password_reset(
        self, client, test_user: User, mock_email_service
    ):
        """Test requesting a password reset."""
        response = client.post(
            "/api/auth/forgot-password",
            json={"email": test_user.email}
        )
        
        assert response.status_code == status.HTTP_200_OK
        mock_email_service.send_password_reset.assert_called_once()
    
    def test_password_reset_with_valid_token(
        self, client, db: Session, test_user: User
    ):
        """Test password reset with valid reset token."""
        from app.security import create_password_reset_token
        
        reset_token = create_password_reset_token(test_user.id)
        
        response = client.post(
            "/api/auth/reset-password",
            json={
                "token": reset_token,
                "new_password": "NewSecurePassword123!",
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify password was changed
        db.refresh(test_user)
        assert verify_password("NewSecurePassword123!", test_user.hashed_password)
    
    def test_password_reset_invalid_token(self, client):
        """Test password reset fails with invalid token."""
        response = client.post(
            "/api/auth/reset-password",
            json={
                "token": "invalid.token",
                "new_password": "NewSecurePassword123!",
            }
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestEmailVerification:
    """Test suite for email verification."""
    
    def test_verify_email_with_valid_token(
        self, client, db: Session, test_user: User
    ):
        """Test email verification with valid token."""
        from app.security import create_email_verification_token
        
        test_user.is_verified = False
        db.commit()
        
        verify_token = create_email_verification_token(test_user.id)
        
        response = client.post(
            "/api/auth/verify-email",
            json={"token": verify_token}
        )
        
        assert response.status_code == status.HTTP_200_OK
        db.refresh(test_user)
        assert test_user.is_verified is True
    
    def test_verify_email_already_verified(self, client, test_user: User):
        """Test verification fails if email already verified."""
        from app.security import create_email_verification_token
        
        verify_token = create_email_verification_token(test_user.id)
        
        response = client.post(
            "/api/auth/verify-email",
            json={"token": verify_token}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.unit
class TestAuthorizationAndPermissions:
    """Test suite for authorization and permission checks."""
    
    def test_protected_endpoint_requires_auth(self, client):
        """Test that protected endpoints require authentication."""
        response = client.get("/api/auth/profile")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_authenticated_user_can_access_profile(self, authenticated_client, test_user: User):
        """Test authenticated user can access their profile."""
        response = authenticated_client.get("/api/auth/profile")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_user.id
        assert data["email"] == test_user.email
    
    def test_user_cannot_modify_other_user_profile(
        self, authenticated_client, db: Session
    ):
        """Test that users cannot modify other users' profiles."""
        other_user = User(
            email="other@example.com",
            username="otheruser",
            hashed_password=hash_password("password"),
            full_name="Other User",
            is_active=True,
            is_verified=True,
        )
        db.add(other_user)
        db.commit()
        
        response = authenticated_client.put(
            f"/api/auth/profile/{other_user.id}",
            json={"full_name": "Hacked Name"}
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
```
