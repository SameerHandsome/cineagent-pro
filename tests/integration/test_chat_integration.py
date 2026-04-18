"""
tests/integration/test_chat_integration.py
────────────────────────────────────────────
Integration tests for the /api/chat endpoint:
  - SSE streaming response format
  - Session creation on first message
  - Rate limiting
  - Error handling when workflow throws
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app():
    from backend.api.chat import router
    app = FastAPI()
    app.include_router(router)
    return app


async def _fake_workflow_stream(initial_state):
    """Minimal fake workflow that yields script + synthesizer chunks."""
    yield {
        "script_analyst": {
            "genres": ["drama"],
            "tone":   ["serious"],
        }
    }
    yield {
        "synthesizer": {
            "final_report": "# Pre-Production Report\nContent here"
        }
    }


class TestChatEndpoint:
    @pytest.fixture
    def app(self, mock_settings):
        return _make_app()

    @pytest.fixture
    def auth_headers(self, valid_jwt):
        token, _ = valid_jwt
        return {"Authorization": f"Bearer {token}"}

    def test_missing_token_returns_401(self, app):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/chat", json={"message": "Hello"})
        assert resp.status_code == 401

    def test_rate_limit_returns_429(self, app, auth_headers, mock_redis):
        with patch("backend.api.chat.check_rate_limit",
                   new_callable=AsyncMock, return_value=False), \
             patch("backend.api.chat.get_db") as mock_get_db:

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/chat",
                json={"message": "Hello"},
                headers=auth_headers,
            )
        assert resp.status_code == 429

    def test_streaming_response_contains_session_id(
        self, app, auth_headers, mock_redis
    ):
        session_id = str(uuid.uuid4())

        with patch("backend.api.chat.check_rate_limit",
                   new_callable=AsyncMock, return_value=True), \
             patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.redis_save",
                   new_callable=AsyncMock), \
             patch("backend.api.chat.workflow") as mock_wf, \
             patch("backend.api.chat.get_db") as mock_get_db, \
             patch("backend.api.chat.trigger_background_indexing") as mock_task:

            mock_crud.get_session_by_id   = AsyncMock(return_value=None)
            mock_crud.create_session      = AsyncMock()
            mock_crud.save_message        = AsyncMock()
            mock_task.delay               = MagicMock(return_value=None)
            mock_wf.astream               = _fake_workflow_stream

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            client = TestClient(app, raise_server_exceptions=False)
            with client.stream(
                "POST", "/api/chat",
                json={"message": "A crime drama"},
                headers=auth_headers,
            ) as resp:
                assert resp.status_code == 200
                content = b"".join(resp.iter_bytes()).decode()

        # SSE format: "data: [SESSION:...]"
        assert "[SESSION:" in content

    def test_streaming_response_contains_done_marker(
        self, app, auth_headers, mock_redis
    ):
        with patch("backend.api.chat.check_rate_limit",
                   new_callable=AsyncMock, return_value=True), \
             patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.redis_save",
                   new_callable=AsyncMock), \
             patch("backend.api.chat.workflow") as mock_wf, \
             patch("backend.api.chat.get_db") as mock_get_db, \
             patch("backend.api.chat.trigger_background_indexing") as mock_task:

            mock_crud.get_session_by_id = AsyncMock(return_value=None)
            mock_crud.create_session    = AsyncMock()
            mock_crud.save_message      = AsyncMock()
            mock_task.delay             = MagicMock(return_value=None)
            mock_wf.astream             = _fake_workflow_stream

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            client = TestClient(app, raise_server_exceptions=False)
            with client.stream(
                "POST", "/api/chat",
                json={"message": "A crime drama"},
                headers=auth_headers,
            ) as resp:
                content = b"".join(resp.iter_bytes()).decode()

        assert "[DONE]" in content

    def test_streaming_response_contains_agent_events(
        self, app, auth_headers, mock_redis
    ):
        with patch("backend.api.chat.check_rate_limit",
                   new_callable=AsyncMock, return_value=True), \
             patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.redis_save",
                   new_callable=AsyncMock), \
             patch("backend.api.chat.workflow") as mock_wf, \
             patch("backend.api.chat.get_db") as mock_get_db, \
             patch("backend.api.chat.trigger_background_indexing") as mock_task:

            mock_crud.get_session_by_id = AsyncMock(return_value=None)
            mock_crud.create_session    = AsyncMock()
            mock_crud.save_message      = AsyncMock()
            mock_task.delay             = MagicMock(return_value=None)
            mock_wf.astream             = _fake_workflow_stream

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            client = TestClient(app, raise_server_exceptions=False)
            with client.stream(
                "POST", "/api/chat",
                json={"message": "A drama"},
                headers=auth_headers,
            ) as resp:
                content = b"".join(resp.iter_bytes()).decode()

        assert "[AGENT:SCRIPT_ANALYST]" in content

    def test_workflow_error_emits_error_event(
        self, app, auth_headers, mock_redis
    ):
        async def _broken_stream(state):
            raise RuntimeError("Groq is down")
            yield  # make it a generator

        with patch("backend.api.chat.check_rate_limit",
                   new_callable=AsyncMock, return_value=True), \
             patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.redis_save",
                   new_callable=AsyncMock), \
             patch("backend.api.chat.workflow") as mock_wf, \
             patch("backend.api.chat.get_db") as mock_get_db, \
             patch("backend.api.chat.trigger_background_indexing") as mock_task:

            mock_crud.get_session_by_id = AsyncMock(return_value=None)
            mock_crud.create_session    = AsyncMock()
            mock_crud.save_message      = AsyncMock()
            mock_task.delay             = MagicMock(return_value=None)
            mock_wf.astream             = _broken_stream

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            client = TestClient(app, raise_server_exceptions=False)
            with client.stream(
                "POST", "/api/chat",
                json={"message": "Hello"},
                headers=auth_headers,
            ) as resp:
                content = b"".join(resp.iter_bytes()).decode()

        assert "[ERROR]" in content

    def test_chat_with_existing_session_id(
        self, app, auth_headers, mock_redis
    ):
        """When a session_id is provided, session creation is skipped."""
        existing_session_id = str(uuid.uuid4())

        with patch("backend.api.chat.check_rate_limit",
                   new_callable=AsyncMock, return_value=True), \
             patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.redis_save",
                   new_callable=AsyncMock), \
             patch("backend.api.chat.workflow") as mock_wf, \
             patch("backend.api.chat.get_db") as mock_get_db, \
             patch("backend.api.chat.trigger_background_indexing") as mock_task:

            mock_crud.get_session_by_id = AsyncMock(return_value=MagicMock())
            mock_crud.create_session    = AsyncMock()
            mock_crud.save_message      = AsyncMock()
            mock_task.delay             = MagicMock(return_value=None)
            mock_wf.astream             = _fake_workflow_stream

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            client = TestClient(app, raise_server_exceptions=False)
            with client.stream(
                "POST", "/api/chat",
                json={"message": "Follow up", "session_id": existing_session_id},
                headers=auth_headers,
            ) as resp:
                assert resp.status_code == 200

        # create_session should NOT be called for existing sessions
        mock_crud.create_session.assert_not_called()


class TestSessionManagement:
    @pytest.fixture
    def app(self, mock_settings):
        return _make_app()

    @pytest.fixture
    def auth_headers(self, valid_jwt):
        token, _ = valid_jwt
        return {"Authorization": f"Bearer {token}"}

    @pytest.mark.asyncio
    async def test_create_session_returns_id_and_title(
        self, app, auth_headers
    ):
        from datetime import datetime, timezone

        fake_session = MagicMock()
        fake_session.id    = str(uuid.uuid4())
        fake_session.title = "My New Film"

        with patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.get_db") as mock_get_db:

            mock_crud.create_session = AsyncMock(return_value=fake_session)

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.post(
                    "/api/sessions",
                    json={"title": "My New Film"},
                    headers=auth_headers,
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "id"    in data
        assert "title" in data

    @pytest.mark.asyncio
    async def test_delete_session_not_found_returns_404(
        self, app, auth_headers
    ):
        with patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.get_db") as mock_get_db:

            mock_crud.delete_session = AsyncMock(return_value=False)

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.delete(
                    f"/api/sessions/{uuid.uuid4()}",
                    headers=auth_headers,
                )

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_patch_session_not_found_returns_404(
        self, app, auth_headers
    ):
        with patch("backend.api.chat.crud") as mock_crud, \
             patch("backend.api.chat.get_db") as mock_get_db:

            mock_crud.update_session_title = AsyncMock(return_value=None)

            async def _fake_db():
                yield AsyncMock()
            mock_get_db.return_value = _fake_db()

            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                resp = await ac.patch(
                    f"/api/sessions/{uuid.uuid4()}",
                    json={"title": "New Title"},
                    headers=auth_headers,
                )

        assert resp.status_code == 404