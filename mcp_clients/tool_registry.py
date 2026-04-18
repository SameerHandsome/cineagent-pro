"""
mcp_clients/tool_registry.py
─────────────────────────────
After loader.py merges all tools, this module slices them into
agent-specific tool sets. Agents import ONLY their assigned set,
preventing cross-agent tool misuse and keeping prompts tight.

Tool naming convention:
  • Local tools:   parse_screenplay, extract_characters, etc.
  • Remote tools:  tavily_search
                   browser_navigate_and_snapshot
                   read_file, list_uploads
"""
import logging
from typing import Dict, List, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)

# ── Decoupled Tool Protocol ───────────────────────────────────────────────────
# By using a Protocol, we remove the strict dependency on LangChain.
# Any tool object (LangChain, AutoGen, or custom) with a 'name' string is valid.
@runtime_checkable
class AgentTool(Protocol):
    name: str

# ── Tool name prefixes/names per agent ────────────────────────────────────────
SCRIPT_TOOL_NAMES: Set[str] = {
    "parse_screenplay",
    "extract_characters",
    "analyze_themes",
    "identify_key_scenes",
    "browser_navigate_and_snapshot",  # playwright — IMDb scraping
}

CASTING_TOOL_NAMES: Set[str] = {
    "search_casting_db",
    "get_casting_preferences",
    "tavily_search",    # live actor news (Tavily) / credits
}

BUDGET_TOOL_NAMES: Set[str] = {
    "calculate_budget_line",
    "get_union_rate_from_db",
    "read_file",        # filesystem — user uploaded templates
    "list_uploads",     # filesystem — list available templates
    "tavily_search",    # live union rate lookups (Tavily)
}

MARKET_TOOL_NAMES: Set[str] = {
    "get_market_comps_from_db",
    "get_streaming_landscape",
    "browser_navigate_and_snapshot",  # playwright — Box Office Mojo
    "tavily_search",    # streaming acquisition news (Tavily)
}

class ToolRegistry:
    """
    Holds the full merged tool list and exposes agent-specific subsets.
    Instantiated once during app startup.
    """

    def __init__(self, all_tools: List[AgentTool]) -> None:
        self._tools_by_exact_name: Dict[str, AgentTool] = {}
        self._tools_by_short_name: Dict[str, AgentTool] = {}

        for tool in all_tools:
            exact_name = getattr(tool, "name", None)
            if not exact_name:
                continue

            # Store by exact full name (e.g., "server__tavily_search")
            self._tools_by_exact_name[exact_name] = tool

            # Determine short name by stripping server prefixes
            short_name = exact_name.split("__")[-1] if "__" in exact_name else exact_name

            # Collision safeguard: Do not silently overwrite tools sharing a short name
            if short_name in self._tools_by_short_name:
                logger.warning(
                    f"Tool name collision detected for '{short_name}'. "
                    f"Ensure backend servers use unique tool names."
                )
            else:
                self._tools_by_short_name[short_name] = tool

        logger.info(f"ToolRegistry initialised with {len(all_tools)} tools")

    def _get_subset(self, required_names: Set[str], agent_name: str) -> List[AgentTool]:
        subset = []
        missing = []

        for name in required_names:
            # Prioritize exact match, fallback to short name match
            tool = self._tools_by_exact_name.get(name) or self._tools_by_short_name.get(name)

            if tool:
                subset.append(tool)
            else:
                missing.append(name)

        if missing:
            logger.warning(f"[{agent_name}] Tools unavailable (remote server down or renamed?): {missing}")

        return subset

    @property
    def script_tools(self) -> List[AgentTool]:
        return self._get_subset(SCRIPT_TOOL_NAMES, "ScriptAnalyst")

    @property
    def casting_tools(self) -> List[AgentTool]:
        return self._get_subset(CASTING_TOOL_NAMES, "CastingDirector")

    @property
    def budget_tools(self) -> List[AgentTool]:
        return self._get_subset(BUDGET_TOOL_NAMES, "BudgetPlanner")

    @property
    def market_tools(self) -> List[AgentTool]:
        return self._get_subset(MARKET_TOOL_NAMES, "MarketIntel")

    def all_tools(self) -> List[AgentTool]:
        """Returns a deduplicated list of all registered tools."""
        return list({id(t): t for t in self._tools_by_exact_name.values()}.values())


# ── Dependency Injection / Lifecycle Management ──────────────────────────────
# Stored privately to discourage direct external mutations.
_instance: ToolRegistry | None = None


def init_registry(tools: List[AgentTool]) -> ToolRegistry:
    """Initialize the global registry instance during app startup."""
    global _instance
    _instance = ToolRegistry(tools)
    return _instance


def get_registry() -> ToolRegistry:
    """
    Get the application's ToolRegistry instance.
    NOTE: If migrating to pure FastAPI standards, consider injecting this via
    `Depends(get_registry)` in your routers rather than calling it directly.
    """
    if _instance is None:
        raise RuntimeError("ToolRegistry not initialised. Call init_registry() in app startup.")
    return _instance
