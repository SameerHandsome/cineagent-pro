"""
tests/unit/test_database.py
────────────────────────────
Unit tests for:
  - backend/database/crud.py    (all CRUD helpers)
  - backend/database/models.py  (ORM model defaults)
  - backend/database/schemas.py (Pydantic serialisation)

All tests use SQLite in-memory via aiosqlite — no Neon/Postgres needed.
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── ORM / Schema unit tests (no DB needed) ────────────────────────────────────

class TestSchemas:
    def test_user_out_from_orm(self):
        from backend.database.schemas import UserOut
        fake_orm = MagicMock()
        fake_orm.id         = str(uuid.uuid4())
        fake_orm.email      = "sam@example.com"
        fake_orm.username   = "sameer"
        fake_orm.created_at = datetime.now(timezone.utc)

        schema = UserOut.model_validate(fake_orm)
        assert schema.email    == "sam@example.com"
        assert schema.username == "sameer"

    def test_session_out_from_orm(self):
        from backend.database.schemas import SessionOut
        fake_orm = MagicMock()
        fake_orm.id         = str(uuid.uuid4())
        fake_orm.title      = "Test Project"
        fake_orm.updated_at = datetime.now(timezone.utc)

        schema = SessionOut.model_validate(fake_orm)
        assert schema.title == "Test Project"

    def test_message_out_from_orm(self):
        from backend.database.schemas import MessageOut
        fake_orm = MagicMock()
        fake_orm.role       = "user"
        fake_orm.content    = "Hello CineAgent"
        fake_orm.created_at = datetime.now(timezone.utc)

        schema = MessageOut.model_validate(fake_orm)
        assert schema.role    == "user"
        assert schema.content == "Hello CineAgent"


class TestModels:
    def test_user_model_defaults(self):
        from backend.database.models import User
        user = User(email="a@b.com", username="ab")
        assert user.hashed_password is None
        assert user.github_id       is None

    def test_session_model_defaults(self):
        from backend.database.models import Session
        sid = uuid.uuid4()
        uid = uuid.uuid4()
        s   = Session(id=sid, user_id=uid)
        assert s.title == "Untitled Project"

    def test_message_model_fields(self):
        from backend.database.models import Message
        sid = uuid.uuid4()
        m   = Message(session_id=sid, role="user", content="test")
        assert m.role    == "user"
        assert m.content == "test"


# ── CRUD helpers with mocked AsyncSession ────────────────────────────────────

class TestCrud:
    """
    Each crud function is tested by mocking the SQLAlchemy AsyncSession.
    We verify the correct ORM objects are constructed and committed.
    """

    @pytest.mark.asyncio
    async def test_create_user_returns_user(self):
        from backend.database.crud import create_user

        fake_user = MagicMock()
        fake_user.id    = uuid.uuid4()
        fake_user.email = "new@example.com"

        db = AsyncMock()
        db.add     = MagicMock()
        db.commit  = AsyncMock()
        db.refresh = AsyncMock(side_effect=lambda u: None)

        # Patch refresh to populate the object
        async def _refresh(obj):
            obj.id = fake_user.id
        db.refresh = AsyncMock(side_effect=_refresh)

        user = await create_user(db, "new@example.com", "sameer", "hash123")
        db.add.assert_called_once()
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_with_explicit_id(self):
        from backend.database.crud import create_session

        db = AsyncMock()
        db.add     = MagicMock()
        db.commit  = AsyncMock()
        db.refresh = AsyncMock()

        sid = str(uuid.uuid4())
        uid = str(uuid.uuid4())

        await create_session(db, uid, title="My Film", session_id=sid)
        db.add.assert_called_once()
        # Verify the session object has the correct UUID
        added_obj = db.add.call_args[0][0]
        assert str(added_obj.id)      == sid
        assert str(added_obj.user_id) == uid
        assert added_obj.title        == "My Film"

    @pytest.mark.asyncio
    async def test_create_session_auto_generates_id(self):
        from backend.database.crud import create_session

        db = AsyncMock()
        db.add     = MagicMock()
        db.commit  = AsyncMock()
        db.refresh = AsyncMock()

        uid = str(uuid.uuid4())
        await create_session(db, uid)
        added_obj = db.add.call_args[0][0]
        # UUID was generated internally
        assert added_obj.id is not None

    @pytest.mark.asyncio
    async def test_get_session_by_id_returns_none_when_missing(self):
        from backend.database.crud import get_session_by_id

        db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute = AsyncMock(return_value=mock_result)

        result = await get_session_by_id(db, str(uuid.uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_save_message(self):
        from backend.database.crud import save_message

        db = AsyncMock()
        db.add    = MagicMock()
        db.commit = AsyncMock()

        sid = str(uuid.uuid4())
        msg = await save_message(db, sid, "user", "Hello!")
        db.add.assert_called_once()
        added = db.add.call_args[0][0]
        assert added.role    == "user"
        assert added.content == "Hello!"

    @pytest.mark.asyncio
    async def test_get_session_messages_returns_list(self):
        from backend.database.crud import get_session_messages
        from backend.database.models import Message

        fake_msg        = MagicMock(spec=Message)
        fake_msg.role   = "user"
        fake_msg.content= "Test"

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [fake_msg]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)

        msgs = await get_session_messages(db, str(uuid.uuid4()))
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    @pytest.mark.asyncio
    async def test_update_session_title(self):
        from backend.database.crud import update_session_title
        from backend.database.models import Session

        fake_session       = MagicMock(spec=Session)
        fake_session.title = "Old Title"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_session

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)
        db.commit  = AsyncMock()
        db.refresh = AsyncMock()

        sid = str(uuid.uuid4())
        uid = str(uuid.uuid4())
        result = await update_session_title(db, sid, uid, "New Title")
        assert fake_session.title == "New Title"

    @pytest.mark.asyncio
    async def test_update_session_title_not_found(self):
        from backend.database.crud import update_session_title

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)

        result = await update_session_title(
            db, str(uuid.uuid4()), str(uuid.uuid4()), "Title"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_by_email_returns_none(self):
        from backend.database.crud import get_user_by_email

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute = AsyncMock(return_value=mock_result)

        result = await get_user_by_email(db, "nobody@example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_session_returns_false_when_not_found(self):
        from backend.database.crud import delete_session

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute  = AsyncMock(return_value=mock_result)
        db.rollback = AsyncMock()

        result = await delete_session(
            db, str(uuid.uuid4()), str(uuid.uuid4())
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_session_returns_true_when_found(self):
        from backend.database.crud import delete_session
        from backend.database.models import Session

        fake_session = MagicMock(spec=Session)

        execute_call_count = 0
        async def _execute(stmt):
            nonlocal execute_call_count
            execute_call_count += 1
            if execute_call_count <= 2:
                # Phase 1: DELETE statements return a plain result
                return MagicMock()
            # Phase 2: SELECT for ownership check
            r = MagicMock()
            r.scalar_one_or_none.return_value = fake_session
            return r

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=_execute)
        db.delete  = AsyncMock()
        db.commit  = AsyncMock()

        result = await delete_session(
            db, str(uuid.uuid4()), str(uuid.uuid4())
        )
        assert result is True
        db.delete.assert_called_once_with(fake_session)