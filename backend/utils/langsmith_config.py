"""
backend/utils/langsmith_config.py
───────────────────────────────────
LangSmith tracing setup.  Import configure_langsmith() once in main.py.
The @traceable decorator is applied on every agent node function.
"""
import logging
import os

from backend.config import settings

logger = logging.getLogger(__name__)


def configure_langsmith() -> None:
    """Set env vars that LangSmith SDK picks up automatically."""
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info(f"LangSmith tracing enabled → project: {settings.langchain_project}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled (LANGCHAIN_TRACING_V2 not set)")
