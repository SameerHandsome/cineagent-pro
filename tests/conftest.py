"""
tests/conftest.py
──────────────────
Shared pytest fixtures used across unit and integration tests.
All external services (Groq, Qdrant, Redis, PostgreSQL, Celery) are mocked
so the test suite runs fully offline with no credentials required.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# ── In-memory SQLite engine for integration tests ─────────────────────────────
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


class Base(DeclarativeBase):
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop so async fixtures share the same loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Minimal Settings mock ──────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Patch backend.config.settings with safe test values before any import
    resolves the real Settings object.  autouse=True means every test gets
    this automatically.
    """
    fake = MagicMock()
    fake.groq_api_key           = "test-groq-key"
    fake.groq_model             = "meta-llama/llama-4-scout-17b-16e-instruct"
    fake.database_url           = TEST_DATABASE_URL
    fake.upstash_redis_rest_url = "https://fake-redis.upstash.io"
    fake.upstash_redis_rest_token = "fake-token"
    fake.redis_url              = "redis://localhost:6379"
    fake.celery_broker_url      = "memory://"
    fake.qdrant_url             = "http://localhost:6333"
    fake.qdrant_api_key         = "test-qdrant-key"
    fake.qdrant_collection      = "test_collection"
    fake.secret_key             = "test-secret-key-for-jwt-signing"
    fake.algorithm              = "HS256"
    fake.access_token_expire_minutes = 60
    fake.github_client_id       = "gh-client-id"
    fake.github_client_secret   = "gh-client-secret"
    fake.github_redirect_uri    = "http://localhost:8000/auth/github/callback"
    fake.github_token           = ""
    fake.tavily_api_key         = ""
    fake.steel_api_key          = ""
    fake.langchain_tracing_v2   = False
    fake.langchain_api_key      = ""
    fake.langchain_project      = "cineagent-test"
    fake.mcp_server_url         = "http://localhost:8001"
    fake.mcp_server_port        = 8001
    fake.frontend_origin        = "http://localhost:3000"
    fake.app_env                = "test"

    monkeypatch.setattr("backend.config.settings", fake)
    return fake


# ── Sample data helpers ───────────────────────────────────────────────────────
@pytest.fixture
def sample_user_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_session_id():
    return str(uuid.uuid4())


@pytest.fixture
def sample_state(sample_user_id, sample_session_id):
    """Minimal CineAgentState-compatible dict for agent tests."""
    return {
        "user_id":      sample_user_id,
        "session_id":   sample_session_id,
        "user_message": "A neo-noir cyber thriller set in 2077 Tokyo",
        "active_agents": ["script", "budget", "casting", "market"],
        "intent":       "full_analysis",
        "session_history": [],
        "user_context": "No prior project history.",
        "genres":       ["sci-fi", "thriller"],
        "tone":         ["dark", "tense"],
        "structural_complexity": "moderate",
        "characters":   [{"name": "Kai", "description": "Hacker protagonist"}],
        "themes":       ["identity", "surveillance"],
        "budget_flags": ["VFX-heavy"],
        "budget_tier":  "indie",
        "budget_breakdown": {},
        "total_budget_estimate": 0.0,
        "live_union_rates": {},
        "casting_suggestions": [],
        "casting_notes": "",
        "market_comps":  [],
        "avg_roi":       0.0,
        "distribution_recommendation": "streaming",
        "top_streaming_platform": "Netflix",
        "final_report":  "",
        "error":         None,
    }


@pytest.fixture
def sample_state_followup(sample_state):
    """State simulating a follow-up turn (session_history non-empty)."""
    return {
        **sample_state,
        "user_message": "What about the marketing strategy?",
        "session_history": [
            {"role": "user",      "content": "A neo-noir cyber thriller set in 2077 Tokyo"},
            {"role": "assistant", "content": "## Pre-Production Report\n..."},
        ],
        "intent": "refine",
    }


# ── Groq mock helpers ─────────────────────────────────────────────────────────
def make_groq_response(content: str, tool_calls=None):
    """Build a fake AIMessage-like object returned by ChatGroq.ainvoke."""
    msg = MagicMock()
    msg.content    = content
    msg.tool_calls = tool_calls or []
    return msg


@pytest.fixture
def mock_groq_llm():
    """Patch ChatGroq so every ainvoke returns a plain-text response."""
    with patch("backend.agents._base.ChatGroq") as mock_cls:
        instance = AsyncMock()
        instance.ainvoke = AsyncMock(return_value=make_groq_response("Mocked LLM response"))
        instance.bind_tools = MagicMock(return_value=instance)
        mock_cls.return_value = instance
        yield instance


# ── Redis mock ────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_redis():
    """Patch the Upstash Redis client with an in-memory async mock."""
    with patch("backend.cache.redis_client.get_redis") as mock_get:
        redis = AsyncMock()
        redis.rpush  = AsyncMock(return_value=1)
        redis.ltrim  = AsyncMock(return_value=True)
        redis.expire = AsyncMock(return_value=True)
        redis.lrange = AsyncMock(return_value=[])
        redis.get    = AsyncMock(return_value=None)
        redis.set    = AsyncMock(return_value=True)
        redis.incr   = AsyncMock(return_value=1)
        mock_get.return_value = redis
        yield redis


# ── Qdrant mock ───────────────────────────────────────────────────────────────
@pytest.fixture
def mock_qdrant():
    """Patch AsyncQdrantClient with an async mock."""
    with patch("backend.rag.retriever.get_qdrant") as mock_get:
        client = AsyncMock()
        # query_points returns an object with .points list
        qr = MagicMock()
        qr.points = []
        client.query_points  = AsyncMock(return_value=qr)
        client.upsert        = AsyncMock(return_value=None)
        client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
        client.create_collection = AsyncMock(return_value=None)
        client.create_payload_index = AsyncMock(return_value=None)
        mock_get.return_value = client
        yield client


# ── Embedder mock ─────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_embedder():
    """
    Patch sentence-transformers so tests never load the model from disk.
    Returns a deterministic 384-float vector.
    """
    fake_vector = [0.01] * 384
    with patch("backend.rag.embedder.get_embedder") as mock_get:
        model = MagicMock()
        model.encode = MagicMock(return_value=MagicMock(tolist=lambda: fake_vector))
        mock_get.return_value = model
        yield model


# ── Celery mock ───────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_celery_task():
    """Prevent Celery tasks from actually firing during tests."""
    with patch("backend.worker.tasks.trigger_background_indexing") as mock_task:
        mock_task.delay = MagicMock(return_value=None)
        yield mock_task


# ── Tool registry mock ────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_tool_registry():
    """Return empty tool lists so agents run with no real MCP tools."""
    with patch("backend.agents.budget_planner.get_registry") as b, \
         patch("backend.agents.casting_director.get_registry") as c, \
         patch("backend.agents.market_intel.get_registry") as m, \
         patch("backend.agents.script_analyst.get_registry") as s:
        for mock in (b, c, m, s):
            reg = MagicMock()
            reg.return_value.budget_tools  = []
            reg.return_value.casting_tools = []
            reg.return_value.market_tools  = []
            reg.return_value.script_tools  = []
            mock.return_value = reg.return_value
        yield


# ── RAG retriever mock ────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_rag_retriever():
    with patch("backend.graph.nodes.retrieve_user_context",
               new_callable=AsyncMock) as mock_retrieve:
        mock_retrieve.return_value = "No prior project history."
        yield mock_retrieve


# ── JWT helper ────────────────────────────────────────────────────────────────
@pytest.fixture
def valid_jwt(mock_settings):
    """Generate a real JWT signed with the test secret key."""
    from backend.auth.jwt_handler import create_access_token
    user_id = str(uuid.uuid4())
    token   = create_access_token(user_id, "test@example.com")
    return token, user_id
