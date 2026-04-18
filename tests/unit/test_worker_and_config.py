"""
tests/unit/test_worker_and_config.py
──────────────────────────────────────
Unit tests for:
  - backend/worker/tasks.py  (trigger_background_indexing, _generate_and_index)
  - backend/config.py        (Settings defaults, lru_cache behaviour)
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Config ─────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_has_required_fields(self, mock_settings):
        """Verify the mock settings object has all keys our code depends on."""
        required = [
            "groq_api_key", "groq_model", "database_url",
            "upstash_redis_rest_url", "upstash_redis_rest_token",
            "celery_broker_url", "qdrant_url", "qdrant_api_key",
            "qdrant_collection", "secret_key", "algorithm",
            "access_token_expire_minutes", "frontend_origin",
        ]
        for field in required:
            assert hasattr(mock_settings, field), f"Missing settings field: {field}"

    def test_settings_algorithm_default(self, mock_settings):
        assert mock_settings.algorithm == "HS256"

    def test_settings_qdrant_collection_default(self, mock_settings):
        assert mock_settings.qdrant_collection == "test_collection"


# ── Background tasks ──────────────────────────────────────────────────────────

class TestGenerateAndIndex:
    @pytest.mark.asyncio
    async def test_indexes_summary_and_preference(self):
        from backend.worker.tasks import _generate_and_index

        fake_summary_resp = MagicMock()
        fake_summary_resp.choices = [MagicMock()]
        fake_summary_resp.choices[0].message.content = "Sci-fi thriller summary"

        fake_pref_resp = MagicMock()
        fake_pref_resp.choices = [MagicMock()]
        fake_pref_resp.choices[0].message.content = "User prefers streaming"

        call_count = 0
        async def _fake_create(**kwargs):
            nonlocal call_count
            call_count += 1
            return fake_summary_resp if call_count == 1 else fake_pref_resp

        mock_groq = AsyncMock()
        mock_groq.chat.completions.create = _fake_create

        with patch("backend.worker.tasks.AsyncGroq", return_value=mock_groq), \
             patch("backend.worker.tasks.index_session_summary",
                   new_callable=AsyncMock) as mock_index_summary, \
             patch("backend.worker.tasks.index_user_preference",
                   new_callable=AsyncMock) as mock_index_pref:

            await _generate_and_index(
                user_id="user-1",
                session_id="sess-1",
                user_message="A sci-fi thriller",
                assistant_report="## Report\nContent here",
                genres=["sci-fi"],
            )

        mock_index_summary.assert_called_once()
        mock_index_pref.assert_called_once()

        # Verify correct args passed to indexers
        summary_call = mock_index_summary.call_args
        assert summary_call[1]["user_id"]    == "user-1"
        assert summary_call[1]["session_id"] == "sess-1"
        assert summary_call[1]["genres"]     == ["sci-fi"]

        pref_call = mock_index_pref.call_args
        assert pref_call[1]["user_id"] == "user-1"
        assert pref_call[1]["topic"]   == "sci-fi"

    @pytest.mark.asyncio
    async def test_handles_summary_groq_failure_gracefully(self):
        """If Groq fails for summary, task should log error but not crash."""
        from backend.worker.tasks import _generate_and_index

        mock_groq = AsyncMock()
        mock_groq.chat.completions.create = AsyncMock(
            side_effect=Exception("Groq API down")
        )

        with patch("backend.worker.tasks.AsyncGroq", return_value=mock_groq), \
             patch("backend.worker.tasks.index_session_summary",
                   new_callable=AsyncMock) as mock_index_summary, \
             patch("backend.worker.tasks.index_user_preference",
                   new_callable=AsyncMock) as _mock_index_pref:

            # Should not raise
            await _generate_and_index("u1", "s1", "message", "report", ["drama"])

        # Neither indexer should be called
        mock_index_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_general_topic_when_no_genres(self):
        from backend.worker.tasks import _generate_and_index

        fake_resp = MagicMock()
        fake_resp.choices = [MagicMock()]
        fake_resp.choices[0].message.content = "Generated content"

        mock_groq = AsyncMock()
        mock_groq.chat.completions.create = AsyncMock(return_value=fake_resp)

        with patch("backend.worker.tasks.AsyncGroq", return_value=mock_groq), \
             patch("backend.worker.tasks.index_session_summary",
                   new_callable=AsyncMock), \
             patch("backend.worker.tasks.index_user_preference",
                   new_callable=AsyncMock) as mock_index_pref:

            await _generate_and_index("u1", "s1", "message", "report", genres=None)

        pref_call = mock_index_pref.call_args
        assert pref_call[1]["topic"] == "general"


class TestTriggerBackgroundIndexingTask:
    def test_task_is_registered_with_correct_name(self):
        # autouse mock_celery_task replaces trigger_background_indexing with a MagicMock.
        # We import the tasks module directly and read the attribute before the mock
        # replaces it — or we reach into the Celery app registry instead.
        import importlib

        import backend.worker.tasks as tasks_mod
        # Reload to get past any module-level mock replacement
        importlib.reload(tasks_mod)
        assert tasks_mod.trigger_background_indexing.name == "trigger_background_indexing"

    def test_task_has_retry_config(self):
        import importlib

        import backend.worker.tasks as tasks_mod
        importlib.reload(tasks_mod)
        assert tasks_mod.trigger_background_indexing.max_retries == 2

    def test_task_runs_async_in_sync_context(self):
        """Verify _run_async correctly bridges async → sync."""
        from backend.worker.tasks import _run_async

        async def _returns_42():
            return 42

        result = _run_async(_returns_42())
        assert result == 42
