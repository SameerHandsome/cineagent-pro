"""
backend/agents/market_intel.py
────────────────────────────────
MODEL: llama-3.3-70b-versatile
  Market analysis requires reasoning over ROI trends, platform strategy,
  and release timing — 70b gives better quality here.
  max_tokens=800 — comp table + distribution reasoning fits comfortably.

FIX: Tools suppressed on follow-up messages (session_history non-empty).
  On a follow-up question the model has no new genre or budget tier to
  look up box office comps for — it would hallucinate a tavily_search or
  browser_navigate_and_snapshot call (e.g. scraping Box Office Mojo for
  "non-linear cyber-noir films"), generating a malformed tool-call JSON
  string that Groq rejects with 400 tool_use_failed.
  When session_history is non-empty, pass tools=[] so the model responds
  in plain text using the market context already in state from turn 1.
"""
import logging

from langsmith import traceable

from backend.agents._base import MODEL_SMART, run_agent_loop
from backend.graph.state import CineAgentState
from backend.prompt.templates import build_market_intel_prompt
from mcp_clients.tool_registry import get_registry

logger = logging.getLogger(__name__)


def _fmt_history(messages: list[dict]) -> str:
    lines = [f"{m['role'].upper()}: {m['content'][:80]}" for m in messages]
    return "\n".join(lines) or "No prior messages."


@traceable(name="market_intel")
async def market_intel_node(state: CineAgentState) -> CineAgentState:
    if "market" not in state.get("active_agents", []):
        return {}

    prompt_template = build_market_intel_prompt()
    session_history = state.get("session_history", [])

    # FIX: Suppress tools on follow-up messages.
    # Tools are only useful on turn 1 when there is a confirmed genre and
    # budget tier to pull box office comps against. On follow-ups the model
    # has no new market question and hallucinates browser or search calls
    # about the follow-up topic, causing Groq 400 tool_use_failed errors.
    is_followup = len(session_history) > 0
    if is_followup:
        tools = []
        logger.info("MarketIntel: follow-up message detected — tools suppressed")
    else:
        tools = get_registry().market_tools

    prompt_vars = {
        "genres":          str(state.get("genres", ["drama"])),
        "budget_tier":     state.get("budget_tier", "indie"),
        "themes":          str(state.get("themes", [])),
        "rag_context":     state.get("user_context", "No prior project history."),
        "session_history": _fmt_history(session_history),
    }

    final_content, tool_results = await run_agent_loop(
        agent_name="MarketIntel",
        prompt_template=prompt_template,
        prompt_vars=prompt_vars,
        user_query=state["user_message"],
        tools=tools,
        temperature=0.2,
        max_tokens=400,
        model_name=MODEL_SMART,
    )

    market_comps: list[dict] = []
    avg_roi = 0.0
    top_platform = "Unknown"

    for tr in tool_results:
        result = tr.get("result", {})
        if isinstance(result, dict):
            if "comps" in result:
                market_comps.extend(result["comps"])
                avg_roi = result.get("average_roi", avg_roi)
            if "top_platform" in result:
                top_platform = result["top_platform"]

    content_lower = final_content.lower()
    if "theatrical" in content_lower and "streaming" not in content_lower:
        distribution_rec = "theatrical"
    elif "hybrid" in content_lower:
        distribution_rec = "hybrid"
    else:
        distribution_rec = "streaming"

    return {
        "market_comps":                market_comps,
        "avg_roi":                     avg_roi,
        "distribution_recommendation": distribution_rec,
        "top_streaming_platform":      top_platform,
    }
