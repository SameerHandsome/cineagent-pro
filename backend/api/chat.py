"""
backend/api/chat.py
────────────────────
POST /api/chat     — run full agentic pipeline, return streamed report
GET  /api/sessions — list user's sessions
POST /api/sessions — create new session
GET  /api/sessions/{id}/messages — fetch message history
"""
import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.jwt_handler import decode_token
from backend.cache.redis_client import check_rate_limit
from backend.cache.redis_client import save_message as redis_save
from backend.database import crud
from backend.database.connection import get_db
from backend.graph.workflow import workflow
from backend.worker.tasks import trigger_background_indexing

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


async def get_current_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    try:
        payload = decode_token(authorization.split(" ")[1])
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class SessionTitleUpdate(BaseModel):
    title: str


class SessionCreateRequest(BaseModel):
    title: str | None = None


def _make_title(text: str) -> str:
    """
    Generate a session title from the first 6 words of the user's message.
    Mirrors the generateTitle() logic in chat.js so backend and frontend
    always produce the same title for the same input.
    Fallback to 'Untitled Project' only if text is blank.
    """
    if not text or not text.strip():
        return "Untitled Project"
    words = text.strip().split()[:6]
    cleaned = " ".join(words)
    # Strip characters that look bad in a sidebar title
    cleaned = "".join(c for c in cleaned if c.isalnum() or c in " '-&")
    cleaned = cleaned.strip()
    if len(cleaned) > 40:
        cleaned = cleaned[:40] + "…"
    return cleaned or "Untitled Project"


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    # ── Rate limiting ──────────────────────────────────────────────────────────
    allowed = await check_rate_limit(user_id, window_seconds=60, max_requests=15)
    if not allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait.")

    # ── Session resolution ─────────────────────────────────────────────────────
    is_new_session = body.session_id is None
    session_id = body.session_id or str(uuid.uuid4())

    # On a brand-new session (frontend pre-create failed and sent no session_id),
    # create the DB row with a real title derived from the message — never
    # hardcode "Untitled Project" here, because the frontend may not get a
    # chance to PATCH it (e.g. if the stream fails before [SESSION:] is read).
    if is_new_session:
        existing = await crud.get_session_by_id(db, session_id)
        if not existing:
            title_for_new = _make_title(body.message)
            await crud.create_session(
                db, user_id, session_id=session_id, title=title_for_new
            )

    # Always persist to both Redis AND PostgreSQL.
    await redis_save(user_id, session_id, "user", body.message)
    await crud.save_message(db, session_id, "user", body.message)

    # ── Build initial LangGraph state ──────────────────────────────────────────
    initial_state = {
        "user_id":      user_id,
        "session_id":   session_id,
        "user_message": body.message,
    }

    async def generate():
        try:
            accumulated_state: dict = {}

            # Emit session_id as the very first SSE event.
            # The client stores this and echoes it back in every subsequent
            # POST /api/chat request for the same conversation.
            yield f"data: [SESSION:{session_id}]\n\n"

            async for chunk in workflow.astream(initial_state):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)

                    if node_name in (
                        "script_analyst", "budget_planner",
                        "casting_director", "market_intel",
                    ):
                        yield f"data: [AGENT:{node_name.upper()}] Analysis complete\n\n"

                    elif node_name == "synthesizer":
                        report = node_output.get("final_report", "")

                        await redis_save(user_id, session_id, "assistant", report)
                        await crud.save_message(db, session_id, "assistant", report)

                        chunk_size = 100
                        for i in range(0, len(report), chunk_size):
                            yield f"data: {report[i:i+chunk_size]}\n\n"

                        genres = accumulated_state.get("genres", [])
                        trigger_background_indexing.delay(
                            user_id, session_id, body.message, report, genres
                        )
                        yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/sessions")
async def list_sessions(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    sessions = await crud.get_user_sessions(db, user_id)
    return [{"id": s.id, "title": s.title, "updated_at": s.updated_at} for s in sessions]


@router.post("/sessions")
async def create_session(
    body: SessionCreateRequest = SessionCreateRequest(),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    title = body.title or "Untitled Project"
    session = await crud.create_session(db, user_id, title=title)
    return {"id": session.id, "title": session.title}


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    deleted = await crud.delete_session(db, session_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.get("/sessions/{session_id}/messages")
async def get_messages(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    messages = await crud.get_session_messages(db, session_id)
    return [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages]


@router.patch("/sessions/{session_id}")
async def update_session_title(
    session_id: str,
    body: SessionTitleUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    session = await crud.update_session_title(db, session_id, user_id, body.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session.id, "title": session.title}
