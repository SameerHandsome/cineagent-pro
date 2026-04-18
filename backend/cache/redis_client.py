"""
backend/cache/redis_client.py
──────────────────────────────
Upstash Redis client for:
  1. Session message history (last N messages per session)
  2. Tool result caching (6-hour TTL for Brave/Playwright results)
"""
import hashlib
import json
import logging

# Use the correct asyncio submodule and alias it to AsyncRedis
from upstash_redis.asyncio import Redis as AsyncRedis

from backend.config import settings

logger = logging.getLogger(__name__)

_redis: AsyncRedis | None = None


def get_redis() -> AsyncRedis:
    global _redis
    if _redis is None:
        _redis = AsyncRedis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token,
        )
    return _redis


# ── Session history ───────────────────────────────────────────────────────────

def _session_key(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}:history"


async def save_message(user_id: str, session_id: str, role: str, content: str) -> None:
    redis = get_redis()
    key = _session_key(user_id, session_id)
    message = json.dumps({"role": role, "content": content})
    await redis.rpush(key, message)
    await redis.ltrim(key, -20, -1)  # keep last 20 messages
    await redis.expire(key, 86400 * 5)  # 5-day TTL


async def get_session_history(user_id: str, session_id: str, limit: int = 5) -> list[dict]:
    redis = get_redis()
    key = _session_key(user_id, session_id)
    try:
        raw_messages = await redis.lrange(key, -limit, -1)
        return [json.loads(m) for m in raw_messages]
    except Exception as e:
        logger.warning(f"Redis session history error: {e}")
        return []


# ── Tool result caching ───────────────────────────────────────────────────────

TOOL_CACHE_TTL = 6 * 3600  # 6 hours


def _tool_cache_key(tool_name: str, args: dict) -> str:
    args_hash = hashlib.md5(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12]
    return f"tool_cache:{tool_name}:{args_hash}"


async def get_cached_tool_result(tool_name: str, args: dict) -> dict | None:
    redis = get_redis()
    key = _tool_cache_key(tool_name, args)
    try:
        cached = await redis.get(key)
        if cached:
            logger.debug(f"Cache HIT for {tool_name}")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None


async def set_cached_tool_result(tool_name: str, args: dict, result: dict) -> None:
    redis = get_redis()
    key = _tool_cache_key(tool_name, args)
    try:
        await redis.set(key, json.dumps(result), ex=TOOL_CACHE_TTL)
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# ── Rate limiting ─────────────────────────────────────────────────────────────

async def check_rate_limit(user_id: str, window_seconds: int = 60, max_requests: int = 10) -> bool:
    """Returns True if the request is allowed, False if rate-limited."""
    redis = get_redis()
    key = f"ratelimit:{user_id}"
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window_seconds)
        return count <= max_requests
    except Exception:
        return True  # fail open
