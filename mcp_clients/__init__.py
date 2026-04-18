# Intentionally minimal — do NOT import loader here.
# loader.py requires langchain_mcp_adapters which may not be installed in test envs.
# Import loader explicitly only in production startup (backend/main.py).
from mcp_clients.tool_registry import ToolRegistry, get_registry, init_registry

__all__ = ["init_registry", "get_registry", "ToolRegistry"]
