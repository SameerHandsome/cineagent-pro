"""
Central configuration loaded once at startup from .env / environment variables.
All other modules import `settings` from here — no os.getenv() scattered around.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_env: str = "development"
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Groq
    groq_api_key: str
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Neon PostgreSQL
    database_url: str

    # Upstash Redis
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str
    redis_url: str

    # Celery
    celery_broker_url: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "cineagent_sessions"

    # GitHub OAuth
    github_client_id: str
    github_client_secret: str
    github_redirect_uri: str = "http://localhost:8000/auth/github/callback"  # ← ADDED
    github_token: str = ""

    # Brave Search
    tavily_api_key: str = ""
    steel_api_key: str = ""

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "cineagent-pro"

    # MCP Server
    mcp_server_url: str = "http://localhost:8001"
    mcp_server_port: int = 8001

    # CORS
    frontend_origin: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
