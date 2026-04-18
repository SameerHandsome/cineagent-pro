"""
backend/main.py
────────────────
FastAPI application entry point.

Startup sequence:
  1. LangSmith tracing configured
  2. PostgreSQL tables created (idempotent)
  3. MCP tools loaded (local + remote) → ToolRegistry initialised
  4. Qdrant collection ensured

Shutdown: graceful (nothing special needed).
"""
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.profile import router as profile_router
from backend.auth.router import router as auth_router
from backend.config import settings
from backend.database.connection import init_db
from backend.rag.indexer import ensure_collection
from backend.utils.langsmith_config import configure_langsmith
from mcp_clients.loader import load_all_tools
from mcp_clients.tool_registry import init_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("🚀 CineAgent Pro starting...")

    configure_langsmith()

    await init_db()
    logger.info("✅ PostgreSQL tables ready")

    await ensure_collection()
    logger.info("✅ Qdrant collection ready")

    tools = await load_all_tools()
    init_registry(tools)
    logger.info(f"✅ ToolRegistry ready ({len(tools)} tools)")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("👋 CineAgent Pro shutting down")


app = FastAPI(
    title="CineAgent Pro",
    description="AI-powered film pre-production intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(profile_router)
