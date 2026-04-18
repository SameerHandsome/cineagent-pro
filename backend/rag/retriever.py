"""
backend/rag/retriever.py
─────────────────────────
Retrieves vectors from Qdrant with STRICT multi-tenancy enforcement.

SECURITY MODEL
──────────────
Every search call MUST include a filter on tenant_id == user_id.
No function in this module can be called without a user_id.

The filter is applied at the Qdrant storage layer (payload index on
tenant_id created in indexer.py). This guarantees:
  - User A's query NEVER touches User B's vectors.
  - Even if the application layer had a bug, the DB layer isolates data.

FIX: AsyncQdrantClient dropped the .search() method in qdrant-client >= 1.7.
     The new unified API is .query_points(), which:
       - Takes `query=` (list[float]) instead of `query_vector=`
       - Returns a QueryResponse object with a `.points` attribute
         (list[ScoredPoint]) — NOT a bare list.
     Code using client.search(...) gets AttributeError at runtime:
       'AsyncQdrantClient' object has no attribute 'search'
     This file replaces all .search() calls with .query_points().
"""
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from backend.config import settings
from backend.rag.embedder import embed

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _client


def _build_filter(
    user_id: str,
    doc_type: str | None = None,
    topic: str | None = None,
) -> Filter:
    """
    Build a Qdrant filter with tenant_id as the mandatory first condition.
    doc_type and topic are optional additional AND filters.
    """
    must_conditions = [
        FieldCondition(key="tenant_id", match=MatchValue(value=user_id)),
    ]

    if doc_type:
        must_conditions.append(
            FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
        )

    if topic:
        must_conditions.append(
            FieldCondition(key="topic", match=MatchValue(value=topic))
        )

    return Filter(must=must_conditions)


async def retrieve_user_context(
    user_id: str,
    query: str,
    limit: int = 3,
    doc_type: str | None = None,
    topic: str | None = None,
) -> str:
    """
    Retrieve the most relevant past context for this user.

    Returns a formatted string for injection into agent prompts.
    Returns a safe default string (not an error) if retrieval fails,
    so agents always have a usable value even when Qdrant is unavailable.
    """
    if not user_id or user_id == "anonymous":
        return "No prior project history (anonymous session)."

    try:
        client = get_qdrant()
        query_vector = embed(query)
        search_filter = _build_filter(user_id, doc_type=doc_type, topic=topic)

        # FIX: .search() was removed in qdrant-client >= 1.7.
        #      Use .query_points() with `query=` instead of `query_vector=`.
        #      The result is a QueryResponse object — access .points for the list.
        response = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        # .points is a list[ScoredPoint]; empty list means no matches
        results = response.points

        if not results:
            return "No prior project history found for this user."

        context_parts = []
        for r in results:
            payload = r.payload or {}
            summary = payload.get("summary", "")
            title   = payload.get("title", "Past session")
            dtype   = payload.get("doc_type", "")
            score   = round(r.score, 3)

            label = "📋 Preference" if dtype == "user_preference" else "🎬 Project"
            context_parts.append(f"{label} [{score}] {title}: {summary}")

        return "\n".join(context_parts)

    except Exception as e:
        logger.warning(f"Qdrant retrieval failed for user {user_id}: {e}")
        return "Context retrieval unavailable (will use session history only)."


async def retrieve_summaries(user_id: str, query: str, limit: int = 3) -> str:
    """Retrieve only session summaries for this user."""
    return await retrieve_user_context(
        user_id, query, limit=limit, doc_type="session_summary"
    )


async def retrieve_preferences(user_id: str, query: str, limit: int = 2) -> str:
    """Retrieve only user preferences for this user."""
    return await retrieve_user_context(
        user_id, query, limit=limit, doc_type="user_preference"
    )