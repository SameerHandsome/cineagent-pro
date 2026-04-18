"""
backend/agents/script_analyst.py
──────────────────────────────────
MODEL: llama-3.3-70b-versatile
  Script analysis requires nuanced genre/tone/character extraction.
  70b gives higher quality structured output for the labelled-text format.
  max_tokens=800 — enough for all 7 labelled output lines.

FIX — tool suppression on follow-up messages:
  On conversational follow-ups (e.g. "should we use non-linear structure?"),
  there is no screenplay text to parse and no IMDb page to scrape. Passing
  tools to the LLM in this case causes it to try tavily_search or
  browser_navigate_and_snapshot with a hallucinated query, which either
  returns irrelevant results or — worse — generates a malformed tool-call
  JSON that Groq rejects with 400 tool_use_failed.

  Fix: when session_history is non-empty (this is a follow-up, not the first
  message), pass an empty tools list to run_agent_loop. The model produces
  structured labelled-text output directly from its context, which is exactly
  what we want for a refinement query — no external lookups needed.

  The empty-tools path still runs through the full _parse_labelled_output()
  parser, so downstream agents receive the same state shape regardless of
  whether tools were used.
"""
import logging

from langsmith import traceable

from backend.agents._base import MODEL_SMART, run_agent_loop
from backend.graph.state import CineAgentState
from backend.prompt.templates import build_script_analyst_prompt
from mcp_clients.tool_registry import get_registry

logger = logging.getLogger(__name__)


def _fmt_history(messages: list[dict]) -> str:
    lines = [f"{m['role'].upper()}: {m['content'][:80]}" for m in messages]
    return "\n".join(lines) or "No prior messages."


def _parse_labelled_output(text: str) -> dict:
    """
    Parses the LLM's labelled plain-text output into structured fields.
    Uses plain-text labels to avoid Groq tool-use JSON interception (400 errors).
    """
    parsed = {
        "genres":                [],
        "tone":                  [],
        "structural_complexity": "moderate",
        "characters":            [],
        "themes":                [],
        "budget_flags":          [],
        "script_comps":          [],
    }

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("GENRES:"):
            parsed["genres"] = [g.strip() for g in line.replace("GENRES:", "").split(",")]

        elif line.startswith("TONE:"):
            parsed["tone"] = [t.strip() for t in line.replace("TONE:", "").split(",")]

        elif line.startswith("COMPLEXITY:"):
            parsed["structural_complexity"] = line.replace("COMPLEXITY:", "").strip()

        elif line.startswith("CHARACTERS:"):
            char_str = line.replace("CHARACTERS:", "").strip()
            if char_str:
                for entry in char_str.split("|"):
                    parts = entry.split("::")
                    if len(parts) == 2:
                        parsed["characters"].append({
                            "name":        parts[0].strip(),
                            "description": parts[1].strip(),
                        })
                    elif parts[0].strip():
                        parsed["characters"].append({
                            "name":        parts[0].strip(),
                            "description": "No description provided",
                        })

        elif line.startswith("THEMES:"):
            parsed["themes"] = [t.strip() for t in line.replace("THEMES:", "").split(",")]

        elif line.startswith("BUDGET_FLAGS:"):
            parsed["budget_flags"] = [f.strip() for f in line.replace("BUDGET_FLAGS:", "").split(",")]

        elif line.startswith("SCRIPT_COMPS:"):
            comps_str = line.replace("SCRIPT_COMPS:", "").strip()
            if comps_str:
                for entry in comps_str.split("|"):
                    parts = entry.split("::")
                    if len(parts) == 3:
                        parsed["script_comps"].append({
                            "title": parts[0].strip(),
                            "year":  parts[1].strip(),
                            "gross": parts[2].strip(),
                        })

    parsed["genres"]       = [g for g in parsed["genres"]       if g]
    parsed["tone"]         = [t for t in parsed["tone"]         if t]
    parsed["themes"]       = [t for t in parsed["themes"]       if t]
    parsed["budget_flags"] = [f for f in parsed["budget_flags"] if f]

    return parsed


@traceable(name="script_analyst")
async def script_analyst_node(state: CineAgentState) -> CineAgentState:
    if "script" not in state.get("active_agents", []):
        return {}

    prompt_template = build_script_analyst_prompt()
    session_history = state.get("session_history", [])

    # FIX: Suppress tools on follow-up messages.
    #
    # Tools (parse_screenplay, browser_navigate_and_snapshot, etc.) are only
    # useful when the user submits a fresh film concept — there is actual
    # screenplay text or an IMDb title to look up. On a conversational
    # follow-up like "should we use non-linear structure?", the model has
    # nothing concrete to scrape or parse. Passing tools in this context
    # causes the model to hallucinate a search query and either:
    #   a) return irrelevant results that pollute the script analysis, or
    #   b) generate a malformed tool-call JSON string that Groq rejects
    #      with 400 tool_use_failed, crashing the entire pipeline.
    #
    # When session_history is non-empty, this is a follow-up — skip tools.
    # The labelled-text output from the model's own knowledge is sufficient
    # to carry genre/tone/character context forward to the synthesizer.
    is_followup = len(session_history) > 0
    if is_followup:
        tools = []
        logger.info("ScriptAnalyst: follow-up message detected — tools suppressed")
    else:
        tools = get_registry().script_tools

    prompt_vars = {
        "rag_context":     state.get("user_context", "No prior project history."),
        "session_history": _fmt_history(session_history),
    }

    final_content, tool_results = await run_agent_loop(
        agent_name="ScriptAnalyst",
        prompt_template=prompt_template,
        prompt_vars=prompt_vars,
        user_query=state["user_message"],
        tools=tools,
        temperature=0.3,
        max_tokens=450,
        model_name=MODEL_SMART,
    )

    parsed = _parse_labelled_output(final_content)

    logger.debug(
        f"ScriptAnalyst parsed: genres={parsed['genres']}, "
        f"complexity={parsed['structural_complexity']}, "
        f"chars={len(parsed['characters'])}, flags={parsed['budget_flags']}"
    )

    return {
        "genres":                parsed["genres"]       or ["drama"],
        "tone":                  parsed["tone"]         or ["dramatic"],
        "structural_complexity": parsed["structural_complexity"],
        "characters":            parsed["characters"],
        "themes":                parsed["themes"],
        "budget_flags":          parsed["budget_flags"],
        "script_comps":          parsed["script_comps"],
    }
