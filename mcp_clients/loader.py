"""
mcp_clients/loader.py
─────────────────────
Loads ALL MCP tools on app startup — both local (FastMCP server) and
remote (Tavily, Playwright/Steel) — and merges them into a single flat
list that the tool registry then slices per agent.

Design goals:
  • Graceful degradation: a failing remote server is logged and skipped,
    never crashes startup.
  • Cache wrapper: remote tool calls check Redis first (6-hour TTL).
  • Single import point: agents never call MCP directly, only tool_registry.
"""
import asyncio
import json
import logging
import os
import traceback
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
REMOTE_CONFIGS_DIR = Path(__file__).parent / "remote_configs"
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")


async def _load_local_tools() -> list[BaseTool]:
    """Connect to our local FastMCP server over SSE and return its tools."""
    try:
        client = MultiServerMCPClient(
            {
                "cineagent-local": {
                    "url": f"{MCP_SERVER_URL}/sse",
                    "transport": "sse",
                }
            }
        )
        tools = await client.get_tools()
        logger.info(f"✅ Loaded {len(tools)} tools from local MCP server at {MCP_SERVER_URL}")
        return list(tools)
    except Exception as e:
        logger.error(f"❌ Could not connect to local MCP server: {e}")
        return []


async def _load_remote_tools() -> list[BaseTool]:
    """Load all remote MCP server configs and return merged tool list."""
    all_remote_tools: list[BaseTool] = []

    for config_file in REMOTE_CONFIGS_DIR.glob("*.json"):
        try:
            cfg = json.loads(config_file.read_text())
            name = cfg.get("name")
            transport = cfg.get("transport")

            if transport != "sse":
                logger.warning(
                    f"⚠️  Skipping {name}: unsupported transport '{transport}' (only 'sse' supported)"
                )
                continue

            api_key_env = cfg.get("api_key_env", "")
            api_key = os.getenv(api_key_env, "")
            if not api_key:
                logger.warning(f"⚠️  Skipping {name}: env var '{api_key_env}' not set")
                continue

            # Each config specifies its own query param name (default: api_key)
            api_key_param = cfg.get("api_key_param", "api_key")

            # Always ensure exactly one trailing slash before the query string
            # e.g. https://mcp.tavily.com/mcp/ + ?tavilyApiKey=...
            url_with_key = f"{cfg['url'].rstrip('/')}/?{api_key_param}={api_key}"

            logger.debug(f"Connecting to remote MCP server '{name}' at {cfg['url']}")

            server_config = {
                name: {
                    "url": url_with_key,
                    "transport": "sse",
                }
            }

            try:
                client = MultiServerMCPClient(server_config)
                tools = await client.get_tools()
                logger.info(f"✅ Loaded {len(tools)} tools from remote server: {name}")
                all_remote_tools.extend(tools)
            except Exception as e:
                logger.warning(f"⚠️  Skipping remote server '{name}': {e}")
                logger.debug(traceback.format_exc())

        except Exception as e:
            logger.warning(f"⚠️  Could not parse {config_file.name}: {e}")
            logger.debug(traceback.format_exc())

    return all_remote_tools


async def load_all_tools() -> list[BaseTool]:
    """
    Entry point called by FastAPI lifespan event.
    Returns merged list of ALL tools (local + remote), deduplicated by name.
    """
    local_tools, remote_tools = await asyncio.gather(
        _load_local_tools(),
        _load_remote_tools(),
        return_exceptions=False,
    )

    all_tools: list[BaseTool] = []
    seen_names: set[str] = set()

    for tool in [*local_tools, *remote_tools]:
        tool_name = getattr(tool, "name", str(tool))
        if tool_name not in seen_names:
            all_tools.append(tool)
            seen_names.add(tool_name)
        else:
            logger.debug(f"Deduplication: skipping duplicate tool '{tool_name}'")

    local_count = len(local_tools)
    remote_count = len(remote_tools)
    logger.info(
        f"🔧 Total tools loaded: {len(all_tools)} ({local_count} local, {remote_count} remote)"
    )
    return all_tools


# ── Fallback: synchronous loader for environments where async startup isn't available ──
def load_all_tools_sync() -> list[BaseTool]:
    return asyncio.run(load_all_tools())