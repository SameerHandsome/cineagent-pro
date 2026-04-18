"""backend/database/crud.py — database operations.

FIXES:
──────
FIX 1 — create_session now accepts an optional `session_id` parameter.
  Previously the function always called uuid.uuid4() internally, so the
  UUID pre-generated in chat.py (and already written to Redis) was thrown
  away and replaced with a different UUID in PostgreSQL.  Result: Redis and
  PostgreSQL held the same conversation under two different IDs, so the
  PostgreSQL fallback in context_assembly_node always returned empty history.

  Callers that don't supply a session_id still get auto-generated behaviour
  (e.g. POST /api/sessions from the frontend), so this is fully backwards
  compatible.

FIX 2 — user_id is now consistently coerced to uuid.UUID inside
  create_session, matching the pattern already used in get_user_sessions,
  delete_session, and update_session_title.

FIX 3 — delete_session uses two-phase deletion via raw SQL core:
  PHASE 1: DELETE FROM messages WHERE session_id = <sid> — raw SQL, not ORM.
           Also purges any messages with session_id IS NULL (orphaned from
           old broken chat turns before the FK bug was fixed).
           Raw SQL bypasses SQLAlchemy's identity map entirely so those
           NULL-session_id rows are never loaded into Python, never tracked
           as dirty, and never flushed as UPDATE messages SET session_id=NULL.
  PHASE 2: ORM DELETE on the Session row, then commit.
           Because Phase 1 used raw SQL, no stale Message objects exist in
           the identity map — only the Session marked for deletion is flushed.

  Previous attempt used db.rollback() which only discards objects added in
  the *current request*. It cannot remove rows that already exist in
  PostgreSQL with session_id=NULL from earlier broken requests. When
  SQLAlchemy loaded the Session it also loaded those orphaned Messages via
  the ORM relationship, bringing them back into the identity map and
  triggering the constraint-violating UPDATE on commit.

ADDITION — get_session_by_id helper lets chat.py check session existence
  before calling create_session, avoiding the try/except swallow pattern.
"""
import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.models import Message, Session, User


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_github_id(db: AsyncSession, github_id: str) -> User | None:
    result = await db.execute(select(User).where(User.github_id == github_id))
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    email: str,
    username: str,
    hashed_password: str | None = None,
    github_id: str | None = None,
) -> User:
    user = User(
        id=uuid.uuid4(),
        email=email,
        username=username,
        hashed_password=hashed_password,
        github_id=github_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def create_session(
    db: AsyncSession,
    user_id: uuid.UUID | str,
    title: str = "Untitled Project",
    session_id: uuid.UUID | str | None = None,
) -> Session:
    sid = (
        uuid.UUID(session_id) if isinstance(session_id, str)
        else session_id or uuid.uuid4()
    )
    uid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

    session = Session(id=sid, user_id=uid, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session_by_id(
    db: AsyncSession, session_id: str | uuid.UUID
) -> Session | None:
    sid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
    result = await db.execute(select(Session).where(Session.id == sid))
    return result.scalar_one_or_none()


async def get_user_sessions(db: AsyncSession, user_id: str) -> list[Session]:
    uid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    result = await db.execute(
        select(Session)
        .where(Session.user_id == uid)
        .order_by(Session.updated_at.desc())
    )
    return list(result.scalars().all())


async def save_message(
    db: AsyncSession, session_id: str, role: str, content: str
) -> Message:
    sid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
    msg = Message(session_id=sid, role=role, content=content)
    db.add(msg)
    await db.commit()
    return msg


async def get_session_messages(
    db: AsyncSession, session_id: str
) -> list[Message]:
    sid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
    result = await db.execute(
        select(Message)
        .where(Message.session_id == sid)
        .order_by(Message.created_at)
    )
    return list(result.scalars().all())


async def delete_session(
    db: AsyncSession, session_id: str, user_id: str
) -> bool:
    sid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
    uid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id

    # ── PHASE 1: Delete messages via SQL core, not ORM ────────────────────────
    #
    # The database contains Message rows (e.g. ids 47–51) with session_id=NULL
    # — orphaned from old broken chat turns before the FK bug was fixed.
    # When SQLAlchemy loads a Session object it also loads related Messages via
    # the ORM relationship, pulling those NULL rows into the identity map.
    # On commit, SQLAlchemy auto-flushes ALL tracked objects including those
    # orphaned Messages, emitting UPDATE messages SET session_id=NULL which
    # violates the NOT NULL constraint.
    #
    # Using sql_delete() (SQLAlchemy Core, not ORM) means SQLAlchemy never
    # constructs Message Python objects, never adds them to the identity map,
    # and never generates the UPDATE — the constraint violation never occurs.
    #
    # Step 1a: delete messages that belong to this session (the normal case).
    await db.execute(
        sql_delete(Message).where(Message.session_id == sid)
    )
    # Step 1b: purge ALL orphaned messages with session_id=NULL from the DB.
    # These are permanently broken rows from pre-fix chat turns. They can't be
    # tied back to any session, so they're safe to delete globally.
    await db.execute(
        sql_delete(Message).where(Message.session_id.is_(None))
    )

    # ── PHASE 2: Verify ownership, then delete the session row ────────────────
    # Phase 1 used raw SQL so no Message objects exist in the identity map.
    # The only thing flushed on commit is the Session DELETE below.
    result = await db.execute(
        select(Session).where(Session.id == sid).where(Session.user_id == uid)
    )
    session = result.scalar_one_or_none()
    if not session:
        await db.rollback()
        return False

    await db.delete(session)
    await db.commit()
    return True


async def update_session_title(
    db: AsyncSession, session_id: str, user_id: str, title: str
) -> Session | None:
    sid = uuid.UUID(session_id) if isinstance(session_id, str) else session_id
    uid = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
    result = await db.execute(
        select(Session).where(Session.id == sid).where(Session.user_id == uid)
    )
    session = result.scalar_one_or_none()
    if session:
        session.title = title
        await db.commit()
        await db.refresh(session)
    return session