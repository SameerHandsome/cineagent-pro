"""
backend/agents/budget_planner.py
──────────────────────────────────
MODEL: llama-3.1-8b-instant
  Budget math is deterministic — tool results do the heavy lifting.
  8b is sufficient and keeps this agent cheap on TPM.
  max_tokens=700 — enough for a structured plain-text budget summary.

FIX 1: structural_complexity and budget_flags are now in prompt_vars.

FIX 2: Tools suppressed on follow-up messages (session_history non-empty).
  On a follow-up question the model has no new line items to calculate and
  no union rate to look up — it would hallucinate a tavily_search or
  calculate_budget_line call with made-up arguments, generating a malformed
  tool-call JSON string that Groq rejects with 400 tool_use_failed.
  When session_history is non-empty, pass tools=[] so the model responds
  in plain text using the budget context already in state from turn 1.
"""
import json
import logging

from langsmith import traceable

from backend.agents._base import MODEL_FAST, run_agent_loop
from backend.graph.state import CineAgentState
from backend.prompt.templates import build_budget_planner_prompt
from mcp_clients.tool_registry import get_registry

logger = logging.getLogger(__name__)


def _fmt_history(messages: list[dict]) -> str:
    lines = [f"{m['role'].upper()}: {m['content'][:80]}" for m in messages]
    return "\n".join(lines) or "No prior messages."


def _estimate_shoot_days(complexity: str, budget_flags: list) -> int:
    base = {"simple": 20, "moderate": 35, "complex": 55}.get(complexity, 30)
    return base + len(budget_flags) * 3


@traceable(name="budget_planner")
async def budget_planner_node(state: CineAgentState) -> CineAgentState:
    if "budget" not in state.get("active_agents", []):
        return {}

    prompt_template = build_budget_planner_prompt()
    session_history = state.get("session_history", [])

    # FIX 2: Suppress tools on follow-up messages.
    # Tools are only useful on turn 1 when there are real line items to
    # calculate and live union rates to fetch. On follow-ups the model has
    # no new numbers to work with and hallucinates tool calls with bad args,
    # causing Groq 400 tool_use_failed errors.
    is_followup = len(session_history) > 0
    if is_followup:
        tools = []
        logger.info("BudgetPlanner: follow-up message detected — tools suppressed")
    else:
        tools = get_registry().budget_tools

    complexity   = state.get("structural_complexity", "moderate")
    budget_flags = state.get("budget_flags", [])

    prompt_vars = {
        "characters":            json.dumps(state.get("characters", []), indent=2),
        "genres":                str(state.get("genres", ["drama"])),
        "shoot_days":            _estimate_shoot_days(complexity, budget_flags),
        "rag_context":           state.get("user_context", "No prior project history."),
        "session_history":       _fmt_history(session_history),
        "structural_complexity": complexity,
        "budget_flags":          str(budget_flags),
    }

    final_content, tool_results = await run_agent_loop(
        agent_name="BudgetPlanner",
        prompt_template=prompt_template,
        prompt_vars=prompt_vars,
        user_query=state["user_message"],
        tools=tools,
        temperature=0.1,
        max_tokens=400,
        model_name=MODEL_FAST,
    )

    budget_breakdown: dict = {}
    live_union_rates: dict = {}
    total_estimate = 0.0
    budget_tier = "indie"

    for tr in tool_results:
        result = tr.get("result", {})
        if isinstance(result, dict):
            if "department_total" in result:
                dept = result.get("department", "unknown")
                budget_breakdown[dept] = result
                total_estimate += result.get("department_total", 0)
                budget_tier = result.get("budget_tier", budget_tier)
            if "rate" in result:
                live_union_rates[result.get("role", "unknown")] = result

    return {
        "budget_breakdown":      budget_breakdown,
        "total_budget_estimate": round(total_estimate, 2),
        "budget_tier":           budget_tier,
        "live_union_rates":      live_union_rates,
    }