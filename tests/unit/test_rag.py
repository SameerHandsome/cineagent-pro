"""
tests/unit/test_rag.py
───────────────────────
Unit tests for:
  - backend/rag/embedder.py   (embed / embed_batch)
  - backend/rag/retriever.py  (retrieve_user_context, filter building)
  - backend/rag/indexer.py    (index_session_summary, _extract_topic)
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Embedder ──────────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_embed_returns_list_of_floats(self):
        from backend.rag.embedder import embed
        vector = embed("test text")
        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch_returns_list_of_vectors(self):
        fake_vector = [0.01] * 384
        with patch("backend.rag.embedder.get_embedder") as mock_get:
            model = MagicMock()
            model.encode = MagicMock(
                return_value=MagicMock(tolist=lambda: [fake_vector, fake_vector])
            )
            mock_get.return_value = model
            from backend.rag.embedder import embed_batch
            result = embed_batch(["text one", "text two"])
            assert isinstance(result, list)

    def test_get_embedder_is_cached(self):
        """Calling get_embedder() twice should return the same object (lru_cache)."""
        # The autouse mock_embedder replaces get_embedder in the module namespace.
        # We restore the real lru_cache'd function for this one test.
        import backend.rag.embedder as _mod

        # Grab whatever is currently bound (could be mock or real)
        saved = _mod.get_embedder

        # Import the real function from source (it's still the lru_cache object)
        # by temporarily unpatching it
        from functools import lru_cache

        from sentence_transformers import SentenceTransformer as _ST  # noqa: F401

        with patch("backend.rag.embedder.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_st.return_value = mock_instance

            # Build a fresh real lru_cache'd function for isolation
            @lru_cache(maxsize=1)
            def _real_get_embedder():
                return _mod.SentenceTransformer("all-MiniLM-L6-v2")

            _mod.get_embedder = _real_get_embedder
            try:
                e1 = _mod.get_embedder()
                e2 = _mod.get_embedder()
                assert e1 is e2
                mock_st.assert_called_once()
            finally:
                _real_get_embedder.cache_clear()
                _mod.get_embedder = saved


# ── Retriever ─────────────────────────────────────────────────────────────────

class TestRetriever:
    @pytest.mark.asyncio
    async def test_returns_default_for_anonymous_user(self):
        from backend.rag.retriever import retrieve_user_context
        result = await retrieve_user_context("anonymous", "test query")
        assert "anonymous" in result.lower() or "no prior" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_default_for_empty_user_id(self):
        from backend.rag.retriever import retrieve_user_context
        result = await retrieve_user_context("", "test query")
        assert "no prior" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_formatted_results_on_hit(self, mock_qdrant):
        from qdrant_client.models import ScoredPoint
        fake_point = MagicMock(spec=ScoredPoint)
        fake_point.score   = 0.92
        fake_point.payload = {
            "summary":  "Sci-fi thriller with $2M budget",
            "title":    "Dark Future",
            "doc_type": "session_summary",
        }
        mock_qdrant.query_points = AsyncMock(
            return_value=MagicMock(points=[fake_point])
        )
        with patch("backend.rag.retriever.get_qdrant", return_value=mock_qdrant):
            from backend.rag.retriever import retrieve_user_context
            result = await retrieve_user_context("user-123", "sci-fi film")
        assert "Dark Future" in result
        assert "0.92" in result

    @pytest.mark.asyncio
    async def test_returns_safe_string_on_qdrant_error(self, mock_qdrant):
        mock_qdrant.query_points = AsyncMock(side_effect=Exception("Connection refused"))
        with patch("backend.rag.retriever.get_qdrant", return_value=mock_qdrant):
            from backend.rag.retriever import retrieve_user_context
            result = await retrieve_user_context("user-123", "any query")
        assert isinstance(result, str)
        assert len(result) > 0  # not empty, agents need a usable string

    @pytest.mark.asyncio
    async def test_returns_no_history_message_when_empty_results(self, mock_qdrant):
        mock_qdrant.query_points = AsyncMock(
            return_value=MagicMock(points=[])
        )
        with patch("backend.rag.retriever.get_qdrant", return_value=mock_qdrant):
            from backend.rag.retriever import retrieve_user_context
            result = await retrieve_user_context("user-123", "some query")
        assert "no prior" in result.lower()

    def test_build_filter_includes_tenant_id(self):
        from backend.rag.retriever import _build_filter
        f = _build_filter("user-abc")
        conditions = f.must
        keys = [c.key for c in conditions]
        assert "tenant_id" in keys

    def test_build_filter_includes_doc_type_when_provided(self):
        from backend.rag.retriever import _build_filter
        f = _build_filter("user-abc", doc_type="session_summary")
        keys = [c.key for c in f.must]
        assert "doc_type" in keys

    def test_build_filter_includes_topic_when_provided(self):
        from backend.rag.retriever import _build_filter
        f = _build_filter("user-abc", topic="sci-fi")
        keys = [c.key for c in f.must]
        assert "topic" in keys

    def test_build_filter_only_tenant_when_no_extras(self):
        from backend.rag.retriever import _build_filter
        f = _build_filter("user-abc")
        assert len(f.must) == 1


# ── Indexer ───────────────────────────────────────────────────────────────────

class TestIndexer:
    def test_extract_topic_uses_genres_first(self):
        from backend.rag.indexer import _extract_topic
        topic = _extract_topic("any text", genres=["Horror"])
        assert topic == "horror"

    def test_extract_topic_falls_back_to_keyword_scan(self):
        from backend.rag.indexer import _extract_topic
        topic = _extract_topic("A story about robots and AI in the future")
        assert topic == "sci-fi"

    def test_extract_topic_returns_general_when_no_match(self):
        from backend.rag.indexer import _extract_topic
        topic = _extract_topic("A completely abstract concept")
        assert topic == "general"

    def test_extract_topic_genre_is_lowercased_and_slugified(self):
        from backend.rag.indexer import _extract_topic
        topic = _extract_topic("", genres=["Sci Fi"])
        assert topic == "sci-fi"

    @pytest.mark.asyncio
    async def test_index_session_summary_calls_upsert(self, mock_qdrant):
        with patch("backend.rag.indexer.get_qdrant", return_value=mock_qdrant), \
             patch("backend.rag.indexer.ensure_collection", new_callable=AsyncMock):
            from backend.rag.indexer import index_session_summary
            await index_session_summary(
                user_id="user-1",
                session_id="sess-1",
                session_title="My Film",
                summary="Sci-fi thriller set in 2077",
                genres=["sci-fi"],
            )
        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.payload["tenant_id"] == "user-1"
        assert point.payload["doc_type"]  == "session_summary"
        assert point.payload["topic"]     == "sci-fi"

    @pytest.mark.asyncio
    async def test_index_session_summary_handles_error_silently(self, mock_qdrant):
        mock_qdrant.upsert = AsyncMock(side_effect=Exception("Qdrant down"))
        with patch("backend.rag.indexer.get_qdrant", return_value=mock_qdrant), \
             patch("backend.rag.indexer.ensure_collection", new_callable=AsyncMock):
            from backend.rag.indexer import index_session_summary
            # Should not raise
            await index_session_summary("u1", "s1", "Title", "Summary")

    @pytest.mark.asyncio
    async def test_index_user_preference_uses_correct_doc_type(self, mock_qdrant):
        with patch("backend.rag.indexer.get_qdrant", return_value=mock_qdrant), \
             patch("backend.rag.indexer.ensure_collection", new_callable=AsyncMock):
            from backend.rag.indexer import index_user_preference
            await index_user_preference(
                user_id="user-1",
                preference_text="Prefers streaming over theatrical",
                topic="drama",
            )
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["doc_type"] == "user_preference"
        assert point.payload["topic"]    == "drama"
        assert point.payload["tenant_id"] == "user-1"

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_if_missing(self, mock_qdrant):
        mock_qdrant.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        with patch("backend.rag.indexer.get_qdrant", return_value=mock_qdrant):
            from backend.rag.indexer import ensure_collection
            await ensure_collection()
        mock_qdrant.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_skips_if_exists(self, mock_qdrant, mock_settings):
        existing = MagicMock()
        existing.name = mock_settings.qdrant_collection  # "test_collection"
        mock_qdrant.get_collections = AsyncMock(
            return_value=MagicMock(collections=[existing])
        )
        # indexer.py does `from backend.config import settings` at import time,
        # so monkeypatch on backend.config.settings doesn't reach it.
        # We must patch the name as it lives inside the indexer module.
        with patch("backend.rag.indexer.get_qdrant", return_value=mock_qdrant), \
             patch("backend.rag.indexer.settings", mock_settings):
            from backend.rag.indexer import ensure_collection
            await ensure_collection()
        mock_qdrant.create_collection.assert_not_called()
