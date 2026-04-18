"""
tests/unit/test_api.py
───────────────────────
Unit tests for FastAPI endpoints:
  - backend/api/health.py  (GET /health)
  - backend/api/chat.py    (_make_title, session endpoints)
  - backend/api/profile.py (GET /api/profile)
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        from backend.api.health import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert "CineAgent" in resp.json()["service"]


# ── _make_title helper ────────────────────────────────────────────────────────

class TestMakeTitle:
    def test_basic_title_from_message(self):
        from backend.api.chat import _make_title
        result = _make_title("A neo-noir cyber thriller set in 2077 Tokyo")
        assert "A" in result
        assert len(result) <= 43  # 40 chars + optional ellipsis

    def test_empty_message_returns_untitled(self):
        from backend.api.chat import _make_title
        assert _make_title("") == "Untitled Project"
        assert _make_title("   ") == "Untitled Project"

    def test_uses_first_six_words(self):
        from backend.api.chat import _make_title
        msg = "word1 word2 word3 word4 word5 word6 word7 word8"
        result = _make_title(msg)
        # Should not contain word7 or word8
        assert "word7" not in result
        assert "word6" in result

    def test_strips_special_characters(self):
        from backend.api.chat import _make_title
        result = _make_title("Hello @world #test [2077]")
        # Only alphanumeric, spaces, hyphens, apostrophes, ampersands
        for char in result:
            assert char.isalnum() or char in " '-&…"

    def test_truncates_at_40_chars(self):
        from backend.api.chat import _make_title
        long_msg = "aaaa " * 20  # 20 * 5 = 100 chars
        result = _make_title(long_msg)
        assert len(result) <= 43  # 40 + "…"

    def test_none_message_returns_untitled(self):
        from backend.api.chat import _make_title
        assert _make_title(None) == "Untitled Project"


# ── Chat API endpoints ────────────────────────────────────────────────────────

class TestChatApi:
    """
    Tests for POST /api/chat and session management endpoints.
    Patches out: DB, Redis, workflow, and JWT decoding.
    """

    @pytest.fixture
    def app_with_mocks(self, mock_settings):
        from backend.api.chat import router
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def auth_headers(self, valid_jwt):
        token, _ = valid_jwt
        return {"Authorization": f"Bearer {token}"}

    def test_chat_rejects_missing_auth(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.post("/api/chat", json={"message": "hello"})
        assert resp.status_code == 401

    def test_chat_rejects_bad_token(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.post(
            "/api/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer badtoken"},
        )
        assert resp.status_code == 401

    def test_sessions_list_rejects_missing_auth(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.get("/api/sessions")
        assert resp.status_code == 401

    def test_create_session_rejects_missing_auth(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.post("/api/sessions", json={"title": "My Project"})
        assert resp.status_code == 401

    def test_delete_session_rejects_missing_auth(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.delete(f"/api/sessions/{uuid.uuid4()}")
        assert resp.status_code == 401

    def test_get_messages_rejects_missing_auth(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.get(f"/api/sessions/{uuid.uuid4()}/messages")
        assert resp.status_code == 401

    def test_patch_session_rejects_missing_auth(self, app_with_mocks):
        client = TestClient(app_with_mocks, raise_server_exceptions=False)
        resp = client.patch(
            f"/api/sessions/{uuid.uuid4()}",
            json={"title": "New Title"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_sessions_list_returns_list(self, app_with_mocks, valid_jwt, mock_redis):
        token, user_id = valid_jwt
        headers = {"Authorization": f"Bearer {token}"}

        fake_session        = MagicMock()
        fake_session.id     = str(uuid.uuid4())
        fake_session.title  = "Test Film"
        from datetime import datetime, timezone
        fake_session.updated_at = datetime.now(timezone.utc)

        with patch("backend.api.chat.check_rate_limit", new_callable=AsyncMock, return_value=True), \
             patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.get_db") as mock_get_db:

            mock_crud.get_user_sessions = AsyncMock(return_value=[fake_session])
            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app_with_mocks), base_url="http://test"
            ) as ac:
                resp = await ac.get("/api/sessions", headers=headers)

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


# ── Profile API ───────────────────────────────────────────────────────────────

class TestProfileApi:
    def test_profile_rejects_missing_auth(self, mock_settings):
        from backend.api.profile import router
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/profile")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_profile_returns_user_data(self, mock_settings, valid_jwt):
        from backend.api.profile import router
        from backend.database.connection import get_db
        app = FastAPI()
        app.include_router(router)

        token, user_id = valid_jwt
        headers = {"Authorization": f"Bearer {token}"}

        fake_user            = MagicMock()
        fake_user.id         = user_id
        fake_user.email      = "sameer@example.com"
        fake_user.username   = "sameer"
        from datetime import datetime, timezone
        fake_user.created_at = datetime.now(timezone.utc)

        from sqlalchemy.ext.asyncio import AsyncSession
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_user
        mock_db.execute = AsyncMock(return_value=mock_result)

        # FastAPI DI captures get_db at router-include time, so patch() on the
        # module reference doesn't work — use dependency_overrides instead.
        async def _override_db():
            yield mock_db

        with patch("backend.api.profile.crud") as mock_crud:
            mock_crud.get_user_sessions = AsyncMock(return_value=[MagicMock(), MagicMock()])
            app.dependency_overrides[get_db] = _override_db

            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.get("/api/profile", headers=headers)

        app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["email"]         == "sameer@example.com"
        assert data["session_count"] == 2
