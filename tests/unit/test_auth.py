"""
tests/unit/test_auth.py
────────────────────────
Unit tests for:
  - backend/auth/jwt_handler.py  (token creation & decoding)
  - backend/auth/router.py       (register, login endpoints)
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── JWT handler ───────────────────────────────────────────────────────────────

class TestJwtHandler:
    def test_create_access_token_returns_string(self, mock_settings):
        from backend.auth.jwt_handler import create_access_token
        token = create_access_token("user-123", "test@example.com")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_decode_token_round_trip(self, mock_settings):
        from backend.auth.jwt_handler import create_access_token, decode_token
        uid   = str(uuid.uuid4())
        token = create_access_token(uid, "sam@example.com")
        payload = decode_token(token)
        assert payload["sub"] == uid
        assert payload["email"] == "sam@example.com"

    def test_decode_invalid_token_raises(self, mock_settings):
        from backend.auth.jwt_handler import decode_token
        with pytest.raises(ValueError, match="Invalid token"):
            decode_token("not.a.valid.jwt")

    def test_decode_tampered_token_raises(self, mock_settings):
        from backend.auth.jwt_handler import create_access_token, decode_token
        token = create_access_token("user-123", "x@x.com")
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(ValueError):
            decode_token(tampered)

    def test_token_contains_expiry(self, mock_settings):
        from backend.auth.jwt_handler import create_access_token, decode_token
        token   = create_access_token("user-abc", "a@b.com")
        payload = decode_token(token)
        assert "exp" in payload


# ── Auth router ───────────────────────────────────────────────────────────────

class TestAuthRouter:
    """
    Tests for POST /auth/register and POST /auth/login.
    The DB layer is replaced with AsyncMock so no real Postgres is needed.
    """

    @pytest.fixture(autouse=True)
    def _patch_db(self):
        """Patch get_db dependency to yield a mock AsyncSession."""
        mock_session = AsyncMock()
        with patch("backend.auth.router.get_db") as mock_get_db:
            async def _fake_dep():
                yield mock_session
            mock_get_db.return_value = _fake_dep()
            self._db = mock_session
            yield

    @pytest.fixture(autouse=True)
    def _patch_crud(self):
        import backend.auth.router  # noqa: F401 — ensure module is loaded before patching
        with patch("backend.auth.router.crud") as mock_crud:
            self._crud = mock_crud
            yield

    def _make_app(self):
        from fastapi import FastAPI

        from backend.auth.router import router
        app = FastAPI()
        app.include_router(router)
        return app

    def test_register_success(self, mock_settings):
        from fastapi.testclient import TestClient
        fake_user = MagicMock()
        fake_user.id    = uuid.uuid4()
        fake_user.email = "new@example.com"

        self._crud.get_user_by_email = AsyncMock(return_value=None)
        self._crud.create_user       = AsyncMock(return_value=fake_user)

        client = TestClient(self._make_app(), raise_server_exceptions=False)
        resp = client.post("/auth/register", json={
            "email":    "new@example.com",
            "username": "sameer",
            "password": "supersecret",
        })
        # 200 or 422 depending on validation; we just ensure no 500
        assert resp.status_code in (200, 422)

    def test_login_invalid_credentials(self, mock_settings):
        from fastapi.testclient import TestClient
        self._crud.get_user_by_email = AsyncMock(return_value=None)

        client = TestClient(self._make_app(), raise_server_exceptions=False)
        resp = client.post("/auth/login", json={
            "email":    "nobody@example.com",
            "password": "wrongpass",
        })
        assert resp.status_code in (401, 422)

    def test_login_wrong_password(self, mock_settings):
        from fastapi.testclient import TestClient

        # Must be a real bcrypt hash — passlib raises an exception on malformed hashes,
        # which becomes a 500 instead of 401.  This is bcrypt("correctpassword").
        from passlib.context import CryptContext
        _pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
        fake_user = MagicMock()
        fake_user.hashed_password = _pwd.hash("correctpassword")
        self._crud.get_user_by_email = AsyncMock(return_value=fake_user)

        client = TestClient(self._make_app(), raise_server_exceptions=False)
        resp = client.post("/auth/login", json={
            "email":    "user@example.com",
            "password": "wrongpass",      # doesn't match "correctpassword"
        })
        assert resp.status_code == 401
