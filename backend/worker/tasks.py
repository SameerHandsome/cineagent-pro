"""
backend/worker/tasks.py
────────────────────────
Celery tasks that run AFTER the response has streamed to the user.
These make the next session smarter without adding latency to the current one.

Tasks:
  1. trigger_background_indexing — summarise the conversation, extract
     user preferences, and index both to Qdrant with full multi-tenancy metadata.
"""
import asyncio
import logging

from groq import AsyncGroq

from backend.config import settings
from backend.rag.indexer import index_session_summary, index_user_preference
from backend.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Helper: run an async coroutine from a sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _generate_and_index(
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_report: str,
    genres: list[str] | None = None,
):
    groq_client = AsyncGroq(api_key=settings.groq_api_key)

    # ── 1. Generate session summary ──────────────────────────────────────────
    summary_prompt = (
        "Summarise this film pre-production session in 2-3 sentences. "
        "Include: genre, budget tier, distribution preference, key creative decisions. "
        "This summary will be used to personalise future sessions for this user.\n\n"
        f"User message: {user_message[:300]}\n\n"
        f"Report excerpt: {assistant_report[:500]}"
    )

    try:
        summary_resp = await groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200,
        )
        summary = summary_resp.choices[0].message.content
        title   = user_message[:60].strip().replace("\n", " ") + "..."

        # Index session summary with tenant_id isolation + topic metadata
        await index_session_summary(
            user_id=user_id,
            session_id=session_id,
            session_title=title,
            summary=summary,
            genres=genres,          # used for topic slug extraction
        )
        logger.info(f"Session summary indexed | user={user_id} | session={session_id}")

    except Exception as e:
        logger.error(f"Session summary generation/indexing failed: {e}")

    # ── 2. Extract and index user preferences ────────────────────────────────
    pref_prompt = (
        "Extract 1-2 key creative or business preferences this user showed in this session. "
        "Examples: 'Prefers indie budget sci-fi films targeting streaming platforms.' "
        "or 'Interested in practical stunts over VFX-heavy productions.' "
        "Be specific and concise. Output only the preference statement, no preamble.\n\n"
        f"User message: {user_message[:300]}\n\n"
        f"Report excerpt: {assistant_report[:400]}"
    )

    try:
        pref_resp = await groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": pref_prompt}],
            max_tokens=100,
        )
        preference = pref_resp.choices[0].message.content.strip()
        topic = genres[0].lower() if genres else "general"

        # Index with doc_type="user_preference" so retriever can filter separately
        await index_user_preference(
            user_id=user_id,
            preference_text=preference,
            topic=topic,
        )
        logger.info(f"User preference indexed | user={user_id} | topic={topic}")

    except Exception as e:
        logger.error(f"User preference extraction/indexing failed: {e}")


@celery_app.task(name="trigger_background_indexing", bind=True, max_retries=2)
def trigger_background_indexing(
    self,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_report: str,
    genres: list[str] | None = None,
):
    """
    Called after every successful chat response.
    Generates session summary + user preferences and indexes both to Qdrant
    using strict tenant_id multi-tenancy.
    """
    try:
        _run_async(_generate_and_index(
            user_id, session_id, user_message, assistant_report, genres
        ))
    except Exception as exc:
        logger.error(f"Background indexing task failed: {exc}")
        self.retry(exc=exc, countdown=30)
