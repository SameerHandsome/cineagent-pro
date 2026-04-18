"""
backend/api/profile.py
───────────────────────
GET /api/profile  — returns the current user's profile data
                    used by profile.html
"""
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.jwt_handler import decode_token
from backend.database import crud
from backend.database.connection import get_db
from backend.database.schemas import SessionOut, UserOut

router = APIRouter(prefix="/api", tags=["profile"])


async def get_current_user_id(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    try:
        payload = decode_token(authorization.split(" ")[1])
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


class ProfileResponse(UserOut):
    session_count: int


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's profile plus session count."""
    from sqlalchemy import select

    from backend.database.models import User

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sessions = await crud.get_user_sessions(db, user_id)

    return ProfileResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        created_at=user.created_at,
        session_count=len(sessions),
    )
