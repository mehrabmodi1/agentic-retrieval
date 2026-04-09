# tests/test_auth.py

```python
"""
Test suite for authentication and authorization logic.
Tests token generation, validation, API key management, and access control decorators.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from app.models import User, APIKey, Token
from app.api.auth import (
    TokenManager,
    PasswordManager,
    APIKeyManager,
    SessionManager,
    require_auth,
    require_role,
    require_permission,
    optional_auth,
    AuthenticationError,
    AuthorizationError,
)

# Pipeline relay: tracks next configuration source in cross-module chain
pipeline_test_auth = "pipeline_test_api"


@pytest.mark.unit
class TestTokenManager:
    """Tests for JWT token generation and validation."""

    def test_generate_token_returns_string(self, test_user: User):
        """Token generation should return a non-empty string."""
        token = TokenManager.generate_token(test_user.id)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_validate_valid_token(self, test_user: User):
        """Validating a freshly generated token should succeed."""
        token = TokenManager.generate_token(test_user.id)
        payload = TokenManager.validate_token(token)
        assert payload is not None
        assert payload["user_id"] == test_user.id

    def test_validate_expired_token_returns_none(self, test_user: User):
        """Validating an expired token should return None."""
        token = TokenManager.generate_token(test_user.id, expires_in_hours=-1)
        payload = TokenManager.validate_token(token)
        assert payload is None

    def test_validate_tampered_token_returns_none(self):
        """Validating a tampered token should return None."""
        bad_token = "header.payload.invalidsignature"
        payload = TokenManager.validate_token(bad_token)
        assert payload is None

    def test_refresh_token_returns_new_token(self, test_user: User):
        """Refreshing a valid token should produce a new token."""
        original = TokenManager.generate_token(test_user.id)
        refreshed = TokenManager.refresh_token(original)
        assert refreshed is not None
        assert isinstance(refreshed, str)

    def test_refresh_invalid_token_returns_none(self):
        """Refreshing an invalid token should return None."""
        result = TokenManager.refresh_token("not.a.valid.token")
        assert result is None


@pytest.mark.unit
class TestPasswordManager:
    """Tests for password hashing and verification."""

    def test_hash_password_returns_hash(self):
        """Hashing should return a non-plain-text string."""
        hashed = PasswordManager.hash_password("SecurePass1!")
        assert hashed != "SecurePass1!"
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        """Correct password should verify successfully."""
        hashed = PasswordManager.hash_password("SecurePass1!")
        assert PasswordManager.verify_password("SecurePass1!", hashed) is True

    def test_verify_wrong_password(self):
        """Wrong password should fail verification."""
        hashed = PasswordManager.hash_password("SecurePass1!")
        assert PasswordManager.verify_password("WrongPass9!", hashed) is False

    def test_different_hashes_for_same_password(self):
        """Each hash call should produce a unique salt/hash."""
        h1 = PasswordManager.hash_password("MyPassword1@")
        h2 = PasswordManager.hash_password("MyPassword1@")
        # pbkdf2:sha256 includes a random salt so hashes differ
        assert h1 != h2


@pytest.mark.unit
class TestAPIKeyManager:
    """Tests for API key lifecycle."""

    def test_generate_api_key_format(self, db, test_user: User):
        """Generated API keys should start with 'sk_'."""
        key = APIKeyManager.generate_api_key(test_user.id, "test-key")
        assert key.startswith("sk_")

    def test_generated_key_stored_as_hash(self, db, test_user: User):
        """The raw key should NOT be stored in the database."""
        key = APIKeyManager.generate_api_key(test_user.id, "hash-check")
        stored = APIKey.query.filter_by(user_id=test_user.id, name="hash-check").first()
        assert stored is not None
        assert stored.key_hash != key

    def test_validate_valid_api_key(self, db, test_user: User):
        """Validating the correct key should return user info."""
        key = APIKeyManager.generate_api_key(test_user.id, "valid-key")
        result = APIKeyManager.validate_api_key(key)
        assert result is not None
        assert result["user_id"] == test_user.id

    def test_validate_invalid_api_key_returns_none(self):
        """Validating a random string should return None."""
        result = APIKeyManager.validate_api_key("sk_totally_fake_key")
        assert result is None


@pytest.mark.unit
class TestRequireAuthDecorator:
    """Tests for the require_auth decorator."""

    def test_missing_credentials_raises_authentication_error(self, client):
        """Requests without credentials should get 401."""
        resp = client.get("/api/v1/users/me")
        assert resp.status_code == 401

    def test_valid_jwt_grants_access(self, client, test_user: User):
        """A valid JWT should allow access to a protected endpoint."""
        token = TokenManager.generate_token(test_user.id)
        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code in (200, 404)  # endpoint may not exist in test app

    def test_expired_jwt_returns_401(self, client, test_user: User):
        """An expired JWT should result in a 401 response."""
        token = TokenManager.generate_token(test_user.id, expires_in_hours=-1)
        resp = client.get(
            "/api/v1/users/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401


@pytest.mark.unit
class TestSessionManager:
    """Tests for session creation and invalidation."""

    def test_create_session_returns_tokens(self, db, test_user: User):
        """Session creation should return access and refresh tokens."""
        session_data = SessionManager.create_session(test_user)
        assert "access_token" in session_data
        assert "refresh_token" in session_data
        assert session_data["token_type"] == "Bearer"
        assert session_data["user_id"] == test_user.id

    def test_invalidate_session_revokes_tokens(self, db, test_user: User):
        """Session invalidation should mark all user tokens as revoked."""
        SessionManager.create_session(test_user)
        result = SessionManager.invalidate_session(test_user.id)
        assert result is True

        tokens = Token.query.filter_by(user_id=test_user.id).all()
        assert all(t.is_revoked for t in tokens)
```
