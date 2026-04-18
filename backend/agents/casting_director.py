"""
backend/agents/casting_director.py
────────────────────────────────────
MODEL: llama-3.1-8b-instant
  Casting lookups are DB-driven via tools. The agent's text output is a
  short ranked list — 8b handles this fine and saves TPM.
  max_tokens=600 — enough for 3 characters × 3 suggestions each.

FIX: Tools suppressed on follow-up messages (session_history non-empty).
  On a follow-up question the model has no new character to look up and
  no budget tier change to work with — it would hallucinate a tavily_search
  call with a made-up query about the follow-up topic (e.g. searching for
  "non-linear script structure actors"), generating a malformed tool-call
  JSON string that Groq rejects with 400 tool_use_failed.
  When session_history is non-empty, pass tools=[] so the model responds
  in plain text using the casting context already in state from turn 1.
"""
import json
import logging

from langsmith import traceable

from backend.agents._base import MODEL_FAST, run_agent_loop
from backend.graph.state import CineAgentState
from backend.prompt.templates import build_casting_director_prompt
from mcp_clients.tool_registry import get_registry

logger = logging.getLogger(__name__)


def _fmt_history(messages: list[dict]) -> str:
    lines = [f"{m['role'].upper()}: {m['content'][:80]}" for m in messages]
    return "\n".join(lines) or "No prior messages."


@traceable(name="casting_director")
async def casting_director_node(state: CineAgentState) -> CineAgentState:
    if "casting" not in state.get("active_agents", []):
        return {}

    prompt_template = build_casting_director_prompt()
    session_history = state.get("session_history", [])

    # FIX: Suppress tools on follow-up messages.
    # Tools are only useful on turn 1 when there are real character names and
    # a confirmed budget tier to search against. On follow-ups the model has
    # no new casting request and hallucinates tool calls with irrelevant args,
    # causing Groq 400 tool_use_failed errors.
    is_followup = len(session_history) > 0
    if is_followup:
        tools = []
        logger.info("CastingDirector: follow-up message detected — tools suppressed")
    else:
        tools = get_registry().casting_tools

    prompt_vars = {
        "characters":      json.dumps(state.get("characters", []), indent=2),
        "budget_tier":     state.get("budget_tier", "indie"),
        "genres":          str(state.get("genres", ["drama"])),
        "tone":            str(state.get("tone", ["dramatic"])),
        "user_id":         state.get("user_id", "anonymous"),
        "rag_context":     state.get("user_context", "No prior project history."),
        "session_history": _fmt_history(session_history),
    }

    final_content, tool_results = await run_agent_loop(
        agent_name="CastingDirector",
        prompt_template=prompt_template,
        prompt_vars=prompt_vars,
        user_query=state["user_message"],
        tools=tools,
        temperature=0.4,
        max_tokens=300,
        model_name=MODEL_FAST,
    )

    casting_suggestions: list[dict] = []
    for tr in tool_results:
        result = tr.get("result", {})
        if isinstance(result, dict) and "suggestions" in result:
            casting_suggestions.extend(result["suggestions"])

    return {
        "casting_suggestions": casting_suggestions,
        "casting_notes":       final_content,
    }
