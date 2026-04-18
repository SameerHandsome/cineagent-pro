"""
tests/unit/test_redis_client.py
────────────────────────────────
Unit tests for backend/cache/redis_client.py:
  - save_message
  - get_session_history
  - get_cached_tool_result / set_cached_tool_result
  - check_rate_limit
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSaveMessage:
    @pytest.mark.asyncio
    async def test_save_message_calls_rpush_ltrim_expire(self, mock_redis):
        from backend.cache.redis_client import save_message
        await save_message("user-1", "sess-1", "user", "Hello!")
        mock_redis.rpush.assert_called_once()
        mock_redis.ltrim.assert_called_once()
        mock_redis.expire.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_message_serialises_json(self, mock_redis):
        from backend.cache.redis_client import save_message
        await save_message("u1", "s1", "assistant", "Report text")
        call_args = mock_redis.rpush.call_args[0]
        # Second arg is the JSON string
        parsed = json.loads(call_args[1])
        assert parsed["role"]    == "assistant"
        assert parsed["content"] == "Report text"

    @pytest.mark.asyncio
    async def test_save_message_uses_correct_key(self, mock_redis):
        from backend.cache.redis_client import save_message
        await save_message("myuser", "mysession", "user", "Hi")
        key = mock_redis.rpush.call_args[0][0]
        assert "myuser"     in key
        assert "mysession"  in key


class TestGetSessionHistory:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_history(self, mock_redis):
        mock_redis.lrange = AsyncMock(return_value=[])
        from backend.cache.redis_client import get_session_history
        history = await get_session_history("u1", "s1")
        assert history == []

    @pytest.mark.asyncio
    async def test_returns_parsed_messages(self, mock_redis):
        messages = [
            json.dumps({"role": "user",      "content": "Film idea"}),
            json.dumps({"role": "assistant", "content": "Report here"}),
        ]
        mock_redis.lrange = AsyncMock(return_value=messages)
        from backend.cache.redis_client import get_session_history
        history = await get_session_history("u1", "s1", limit=5)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_returns_empty_on_redis_error(self, mock_redis):
        mock_redis.lrange = AsyncMock(side_effect=Exception("Redis down"))
        from backend.cache.redis_client import get_session_history
        history = await get_session_history("u1", "s1")
        assert history == []


class TestToolCache:
    @pytest.mark.asyncio
    async def test_get_cached_tool_result_returns_none_on_miss(self, mock_redis):
        mock_redis.get = AsyncMock(return_value=None)
        from backend.cache.redis_client import get_cached_tool_result
        result = await get_cached_tool_result("tavily_search", {"query": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_get_cached_tool_result_returns_data_on_hit(self, mock_redis):
        cached = {"films": ["Blade Runner"], "roi": 1.5}
        mock_redis.get = AsyncMock(return_value=json.dumps(cached))
        from backend.cache.redis_client import get_cached_tool_result
        result = await get_cached_tool_result("tavily_search", {"query": "test"})
        assert result == cached

    @pytest.mark.asyncio
    async def test_set_cached_tool_result_calls_redis_set(self, mock_redis):
        from backend.cache.redis_client import set_cached_tool_result
        data = {"result": "some data"}
        await set_cached_tool_result("my_tool", {"arg": 1}, data)
        mock_redis.set.assert_called_once()
        # Verify the value is JSON-serialised
        call_args = mock_redis.set.call_args
        stored_value = call_args[0][1]
        assert json.loads(stored_value) == data

    @pytest.mark.asyncio
    async def test_set_cached_tool_result_handles_error_silently(self, mock_redis):
        mock_redis.set = AsyncMock(side_effect=Exception("Redis down"))
        from backend.cache.redis_client import set_cached_tool_result
        # Should not raise
        await set_cached_tool_result("tool", {}, {"data": 1})


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_allows_first_request(self, mock_redis):
        mock_redis.incr   = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock(return_value=True)
        from backend.cache.redis_client import check_rate_limit
        allowed = await check_rate_limit("user-1", max_requests=10)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self, mock_redis):
        mock_redis.incr   = AsyncMock(return_value=16)
        mock_redis.expire = AsyncMock(return_value=True)
        from backend.cache.redis_client import check_rate_limit
        allowed = await check_rate_limit("user-1", max_requests=15)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_fails_open_on_redis_error(self, mock_redis):
        mock_redis.incr = AsyncMock(side_effect=Exception("Redis down"))
        from backend.cache.redis_client import check_rate_limit
        allowed = await check_rate_limit("user-1")
        assert allowed is True  # fail open

    @pytest.mark.asyncio
    async def test_sets_expiry_only_on_first_request(self, mock_redis):
        mock_redis.incr   = AsyncMock(return_value=2)  # Not first request
        mock_redis.expire = AsyncMock(return_value=True)
        from backend.cache.redis_client import check_rate_limit
        await check_rate_limit("user-1")
        # expire should NOT be called when count > 1
        mock_redis.expire.assert_not_called()