"""
backend/rag/indexer.py
───────────────────────
Indexes session summaries into Qdrant with STRICT multi-tenancy.

MULTI-TENANCY DESIGN
────────────────────
Every vector stored has three mandatory metadata fields:

  tenant_id  — the user's UUID (hard wall: all queries filter on this)
  doc_type   — "session_summary" | "user_preference"
               lets you retrieve only one category at a time
  topic      — genre/theme slug extracted from the summary
               (e.g. "sci-fi", "horror", "budget-planning")
               lets agents do topic-scoped retrieval

Security guarantee:
  Qdrant payload indexes are created on `tenant_id`, `doc_type`, and
  `topic` at collection creation time.  Every search call in retriever.py
  includes a MUST filter on `tenant_id == user_id`.  Without the correct
  tenant_id in the JWT, the filter returns zero results — other users'
  vectors are never visible.

This is the Qdrant-recommended multi-tenancy approach:
  https://qdrant.tech/documentation/guides/multiple-partitions/
"""
import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from backend.config import settings
from backend.rag.embedder import embed
from backend.rag.retriever import get_qdrant

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384   # all-MiniLM-L6-v2 output dimensions

# Valid doc_type values — enforced at index time
DOC_TYPE_SESSION_SUMMARY  = "session_summary"
DOC_TYPE_USER_PREFERENCE  = "user_preference"


async def ensure_collection() -> None:
    """
    Create Qdrant collection + payload indexes if they don't exist.

    Payload indexes are CRITICAL for multi-tenancy:
      - Without an index on tenant_id, Qdrant does a full collection scan
        to apply the filter (slow and potentially leaky under load).
      - With an index, the filter is applied at the storage layer before
        the vector similarity search even runs.
    """
    client = get_qdrant()
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if settings.qdrant_collection not in existing_names:
        # Create collection
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")

    # Create payload indexes (idempotent — safe to call on every startup)
    # tenant_id  → keyword index: exact match, used in EVERY query filter
    # doc_type   → keyword index: filter by document category
    # topic      → keyword index: filter by topic/genre
    for field, schema_type in [
        ("tenant_id", PayloadSchemaType.KEYWORD),
        ("doc_type",  PayloadSchemaType.KEYWORD),
        ("topic",     PayloadSchemaType.KEYWORD),
    ]:
        try:
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=schema_type,
            )
            logger.debug(f"Payload index ensured: {field}")
        except Exception as e:
            # Already exists → safe to ignore
            logger.debug(f"Payload index {field} already exists or skipped: {e}")


def _extract_topic(summary: str, genres: list[str] | None = None) -> str:
    """
    Extract a topic slug for indexing.
    Prefers genres passed in from the agent state; falls back to keyword scan.
    """
    if genres:
        return genres[0].lower().replace(" ", "-")

    summary_lower = summary.lower()
    topic_map = {
        "sci-fi":   ["space", "ai", "robot", "sci-fi", "science fiction", "future"],
        "horror":   ["horror", "haunted", "ghost", "terror", "fear"],
        "thriller": ["thriller", "spy", "chase", "conspiracy", "suspense"],
        "drama":    ["drama", "family", "loss", "redemption"],
        "action":   ["action", "fight", "battle", "war", "stunt"],
        "comedy":   ["comedy", "funny", "laugh", "quirky"],
        "romance":  ["romance", "love", "wedding"],
    }
    for topic, keywords in topic_map.items():
        if any(kw in summary_lower for kw in keywords):
            return topic
    return "general"


async def index_session_summary(
    user_id: str,
    session_id: str,
    session_title: str,
    summary: str,
    genres: list[str] | None = None,
) -> None:
    """
    Index a session summary with full multi-tenancy metadata.

    Payload stored per vector:
      tenant_id   — user_id (the hard isolation key)
      doc_type    — "session_summary"
      topic       — genre slug for scoped retrieval
      session_id  — source session reference
      title       — human-readable session title
      summary     — the actual text (returned in retrieval results)
    """
    try:
        await ensure_collection()
        client = get_qdrant()
        vector = embed(summary)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "tenant_id":  user_id,        # PRIMARY isolation key
                "doc_type":   DOC_TYPE_SESSION_SUMMARY,
                "topic":      _extract_topic(summary, genres),
                "session_id": session_id,
                "title":      session_title,
                "summary":    summary,
            },
        )
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=[point],
        )
        logger.info(f"Indexed session summary | user={user_id} | topic={point.payload['topic']}")

    except Exception as e:
        logger.error(f"Qdrant indexing failed for user {user_id}: {e}")


async def index_user_preference(
    user_id: str,
    preference_text: str,
    topic: str = "general",
) -> None:
    """
    Index a user preference extracted from conversation history.
    E.g. "User prefers streaming distribution for indie sci-fi films."

    Separate doc_type from session_summary so retrievers can
    fetch only preferences or only summaries as needed.
    """
    try:
        await ensure_collection()
        client = get_qdrant()
        vector = embed(preference_text)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "tenant_id":  user_id,
                "doc_type":   DOC_TYPE_USER_PREFERENCE,
                "topic":      topic,
                "summary":    preference_text,
                "title":      f"Preference: {preference_text[:60]}",
            },
        )
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=[point],
        )
        logger.info(f"Indexed user preference | user={user_id} | topic={topic}")

    except Exception as e:
        logger.error(f"Qdrant preference indexing failed for user {user_id}: {e}")
