"""
backend/agents/orchestrator.py
───────────────────────────────
Lightweight intent classifier — no tool calls, just JSON routing.

MODEL: llama-3.1-8b-instant
  The orchestrator does simple JSON classification. It doesn't need
  the 70b model's reasoning power. 8b is faster and uses fewer TPM.
  max_tokens=200 is enough for: {"intent": "...", "active_agents": [...]}

FIX (follow-up context loss):
──────────────────────────────
Added a post-parse refinement signal detector.  When the LLM classifies a
follow-up query as "full_analysis" but session_history is non-empty AND the
query text matches known refinement patterns (marketing, distribution,
strategy, expand, etc.), the intent is overridden to "refine".

WHY THIS IS NEEDED:
  The 8b model reliably classifies standalone film concepts correctly, but
  short follow-up queries like "develop a marketing strategy" or "expand on
  the budget" carry no film-concept signal — the model defaults to
  full_analysis because it looks like a generic instruction.  The signal
  detector catches these cases without requiring a larger model.

REFINE vs FULL_ANALYSIS BEHAVIOUR:
  full_analysis → all four agents run, script_analyst re-parses from scratch.
  refine        → all four agents still run but they receive the full
                  session_history so they operate as incremental updates,
                  not cold starts.  The synthesizer prompt also sees prior
                  context via RAG + session_history, producing a coherent
                  follow-up report instead of a blank-slate one.
"""
import json
import logging

from langchain_groq import ChatGroq
from langsmith import traceable

from backend.agents._base import MODEL_FAST
from backend.config import settings
from backend.graph.state import CineAgentState
from backend.prompt.templates import build_orchestrator_prompt

logger = logging.getLogger(__name__)

# Keywords that strongly signal the user is refining / building on a prior turn
# rather than submitting a brand-new film concept.
_REFINEMENT_SIGNALS = frozenset([
    "marketing", "distribution", "strategy", "refine", "update",
    "change", "adjust", "what about", "also", "expand", "more detail",
    "next steps", "follow up", "follow-up", "build on", "continue",
    "elaborate", "improve", "revise", "rethink", "reconsider",
    "based on", "given that", "now add", "add more", "add a",
])


def _is_refinement(query: str, session_history: list[dict]) -> bool:
    """
    Return True when the query looks like a follow-up refinement rather than
    a fresh film concept submission.

    Conditions (both must be true):
      1. There is existing session history (i.e. this is not turn 1).
      2. The query matches at least one refinement signal keyword.
    """
    if not session_history:
        return False
    query_lower = query.lower()
    return any(signal in query_lower for signal in _REFINEMENT_SIGNALS)


@traceable(name="orchestrator")
async def orchestrator_node(state: CineAgentState) -> CineAgentState:
    prompt = build_orchestrator_prompt()

    # MODEL: fast 8b for classification — saves TPM for the heavier agents
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=MODEL_FAST,
        temperature=0.0,
        max_tokens=200,
    )

    session_history = state.get("session_history", [])

    history_lines = [
        f"{m['role'].upper()}: {m['content'][:80]}"
        for m in session_history
    ]
    session_history_str = "\n".join(history_lines) or "No prior messages."

    messages = prompt.format_messages(
        rag_context=state.get("user_context", "No prior project history."),
        session_history=session_history_str,
        user_query=state["user_message"],
    )

    try:
        response = await llm.ainvoke(messages)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        active_agents = result.get("active_agents", ["script", "budget", "casting", "market"])
        intent = result.get("intent", "full_analysis")

        # Safety net: refine intent must always run all agents so they have
        # full context to produce an incremental update.
        if intent == "refine" and not active_agents:
            active_agents = ["script", "budget", "casting", "market"]

    except Exception as e:
        logger.warning(f"Orchestrator JSON parse failed ({e}) → defaulting to full analysis")
        active_agents = ["script", "budget", "casting", "market"]
        intent = "full_analysis"

    # FIX: Override to "refine" when the LLM misclassifies a follow-up query
    # as "full_analysis".  This happens reliably with 8b when the query is a
    # short instruction with no film-concept content (e.g. "develop a marketing
    # strategy") — the model has nothing to anchor a full_analysis classification
    # on, so it defaults to the majority class.
    #
    # The signal detector checks two conditions:
    #   1. session_history is non-empty  → this is not the first turn
    #   2. query matches a refinement keyword → user is building on prior work
    #
    # We intentionally do NOT override "refine" → "full_analysis" here because
    # if the LLM correctly detected refine, we trust it.
    if intent == "full_analysis" and _is_refinement(state["user_message"], session_history):
        intent = "refine"
        # Ensure all agents are active so they can produce incremental updates
        if not active_agents:
            active_agents = ["script", "budget", "casting", "market"]
        logger.info(
            "Orchestrator: intent overridden full_analysis → refine "
            "(refinement signals detected in query with non-empty session history)"
        )

    logger.info(f"Orchestrator → intent={intent}, agents={active_agents}")
    return {**state, "active_agents": active_agents, "intent": intent}
