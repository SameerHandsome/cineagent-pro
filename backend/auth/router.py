"""
backend/auth/router.py
───────────────────────
Auth endpoints:
  POST /auth/register    — email + password
  POST /auth/login       — email + password → JWT
  GET  /auth/github      — redirect to GitHub OAuth
  GET  /auth/github/callback — exchange code → JWT → redirect to frontend
"""
import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.jwt_handler import create_access_token
from backend.config import settings
from backend.database import crud
from backend.database.connection import get_db

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

GITHUB_AUTH_URL  = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL  = "https://api.github.com/user"


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await crud.get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = pwd_context.hash(body.password)
    user = await crud.create_user(
        db, email=body.email, username=body.username, hashed_password=hashed
    )
    return TokenResponse(access_token=create_access_token(str(user.id), user.email))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await crud.get_user_by_email(db, body.email)
    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not pwd_context.verify(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(str(user.id), user.email))


@router.get("/github")
async def github_login():
    url = (
        f"{GITHUB_AUTH_URL}"
        f"?client_id={settings.github_client_id}"
        f"&scope=user:email"
        f"&redirect_uri={settings.github_redirect_uri}"
    )
    return RedirectResponse(url)


@router.get("/github/callback")
async def github_callback(code: str, db: AsyncSession = Depends(get_db)):
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id":     settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code":          code,
                "redirect_uri":  settings.github_redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()
        github_access_token = token_data.get("access_token")
        if not github_access_token:
            raise HTTPException(status_code=400, detail="GitHub OAuth failed")

        user_resp = await client.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {github_access_token}",
                "Accept": "application/json",
            },
        )
        gh_user = user_resp.json()

    github_id = str(gh_user["id"])
    email    = gh_user.get("email") or f"gh_{github_id}@cineagent.local"
    username = gh_user.get("login", "unknown")

    user = await crud.get_user_by_github_id(db, github_id)
    if not user:
        user = await crud.create_user(db, email=email, username=username, github_id=github_id)

    jwt_token = create_access_token(str(user.id), user.email)

    # Redirect to frontend — auth.js reads ?token= automatically
    return RedirectResponse(
        url=f"{settings.frontend_origin}/index.html?token={jwt_token}",
        status_code=302,
    )
