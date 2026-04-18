"""
backend/graph/nodes.py

FIX 1: synthesizer_node now safely coerces avg_roi with `float(... or 0)`
     instead of directly formatting state.get('avg_roi', 0) with :.2f.
     When market_intel is inactive, avg_roi is absent from state entirely
     (not just 0), so state.get('avg_roi', 0) returns None in some LangGraph
     merge scenarios, causing:
       TypeError: unsupported format character when avg_roi is None

FIX 2: synthesizer_node now injects session_history into assembled_context.
     Previously session_history was fetched by context_assembly_node and
     stored in state, but synthesizer_node never read it — so the LLM had
     no knowledge of prior turns and treated every follow-up message as a
     brand new project (responding with $0 budget, no casting, blank market
     data because it only saw the short follow-up text with no film concept).

     The history is injected at the TOP of assembled_context so the LLM
     reads prior turns before the current agent outputs, giving it full
     conversational context to answer follow-up questions correctly.

FIX 3: synthesizer_node now performs SELECTIVE context injection based on
     active_agents from state. In MODE B (intent=refine), only the agent
     sections that were actually activated are included in assembled_context.

     PROBLEM THIS SOLVES:
     Previously, assembled_context always included all four agent sections
     (Script, Budget, Casting, Market) — even when intent=refine and only
     one agent ran. The Synthesizer received a wall of structured data with
     empty JSON blocks for inactive agents, which pulled it toward rendering
     the full report template regardless of the MODE B instruction.

     The model pattern-matches on data volume: lots of structured sections →
     render a full report. Stripping irrelevant agent sections from MODE B
     context removes that pull entirely.

FIX 4: assembled_context for MODE B now includes an explicit QUESTION block
     that restates the user's follow-up question directly before the agent
     output. This anchors the Synthesizer to the specific question being
     asked rather than the broader project concept.
"""
import json
import logging

from langchain_groq import ChatGroq
from langsmith import traceable

from backend.cache.redis_client import get_session_history
from backend.config import settings
from backend.database import crud
from backend.database.connection import AsyncSessionLocal
from backend.graph.state import CineAgentState
from backend.prompt.templates import build_synthesizer_prompt
from backend.rag.retriever import retrieve_user_context

logger = logging.getLogger(__name__)


@traceable(name="context_assembly")
async def context_assembly_node(state: CineAgentState) -> CineAgentState:
    user_id    = state.get("user_id", "anonymous")
    session_id = state.get("session_id", "default")

    session_history = await get_session_history(
        user_id=user_id, session_id=session_id, limit=5
    )

    if not session_history and state.get("session_id"):
        try:
            async with AsyncSessionLocal() as db:
                db_messages = await crud.get_session_messages(db, session_id)
                session_history = [
                    {"role": m.role, "content": m.content}
                    for m in db_messages[-5:]
                ]
        except Exception as e:
            logger.warning(f"PostgreSQL fallback failed: {e}")

    user_context = await retrieve_user_context(
        user_id=user_id, query=state.get("user_message", "")
    )

    return {"session_history": session_history, "user_context": user_context}


def _build_history_block(session_history: list) -> str:
    """
    Format session history as a readable conversation transcript.

    Assistant messages are truncated at 600 chars to keep context window lean.
    600 chars is enough to convey the prior answer without blowing the Groq
    TPM limit on the synthesizer call.
    """
    if not session_history:
        return "═══ CONVERSATION HISTORY ═══\n(No prior turns — this is the first message)"

    history_lines = []
    for msg in session_history:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "").strip()
        if role == "ASSISTANT" and len(content) > 600:
            content = content[:600] + "… [truncated]"
        history_lines.append(f"{role}: {content}")

    return "═══ CONVERSATION HISTORY (most recent last) ═══\n" + "\n\n".join(history_lines)


def _build_agent_section(label: str, lines: list[str]) -> str:
    """Wrap agent output lines in a labelled block."""
    return f"═══ {label} ═══\n" + "\n".join(lines)


def _build_mode_a_context(state: dict, history_block: str, avg_roi_safe: float) -> str:
    """
    Full context for MODE A (full_analysis).
    All four agent sections are included regardless of active_agents,
    because a full report always needs all data.
    """
    sections = [
        "RESPONSE MODE: A — FULL REPORT (generate the complete Pre-Production Intelligence Report)",
        history_block,
        f"USER CONCEPT (current message):\n{state.get('user_message', '')}",
        _build_agent_section("SCRIPT ANALYST OUTPUT", [
            f"Genres             : {state.get('genres', [])}",
            f"Tone               : {state.get('tone', [])}",
            f"Complexity         : {state.get('structural_complexity', 'N/A')}",
            f"Characters         :\n{json.dumps(state.get('characters', []), indent=2)}",
            f"Themes             : {state.get('themes', [])}",
            f"Budget flags       : {state.get('budget_flags', [])}",
        ]),
        _build_agent_section("BUDGET PLANNER OUTPUT", [
            f"Tier               : {state.get('budget_tier', 'N/A')}",
            f"Total estimate     : ${state.get('total_budget_estimate') or 0:,.0f}",
            f"Breakdown          :\n{json.dumps(state.get('budget_breakdown', {}), indent=2)}",
        ]),
        _build_agent_section("CASTING DIRECTOR OUTPUT", [
            f"Suggestions        :\n{json.dumps(state.get('casting_suggestions', []), indent=2)}",
        ]),
        _build_agent_section("MARKET INTEL OUTPUT", [
            f"Average ROI        : {avg_roi_safe:.2f}x",
            f"Top platform       : {state.get('top_streaming_platform', 'Unknown')}",
            f"Comp films         :\n{json.dumps(state.get('comp_films', []), indent=2)}",
            f"Distribution       : {state.get('distribution_recommendation', 'N/A')}",
            f"Release window     : {state.get('release_window', 'N/A')}",
        ]),
    ]
    return "\n\n".join(sections)


def _build_mode_b_context(state: dict, history_block: str, avg_roi_safe: float) -> str:
    """
    SELECTIVE context for MODE B (refine / follow-up).

    KEY FIX: Only inject the agent section(s) that were actually activated.
    Injecting all four sections even with empty data pulls the Synthesizer
    toward rendering the full report template, overriding the MODE B instruction.

    The FOLLOW-UP QUESTION block is placed immediately before the agent output
    so the Synthesizer is anchored to the specific question, not the project concept.
    """
    active_agents: list[str] = state.get("active_agents", [])

    # Map agent name → section builder
    agent_section_builders = {
        "script": lambda: _build_agent_section("SCRIPT ANALYST OUTPUT", [
            f"Genres             : {state.get('genres', [])}",
            f"Tone               : {state.get('tone', [])}",
            f"Complexity         : {state.get('structural_complexity', 'N/A')}",
            f"Characters         :\n{json.dumps(state.get('characters', []), indent=2)}",
            f"Themes             : {state.get('themes', [])}",
            f"Budget flags       : {state.get('budget_flags', [])}",
        ]),
        "budget": lambda: _build_agent_section("BUDGET PLANNER OUTPUT", [
            f"Tier               : {state.get('budget_tier', 'N/A')}",
            f"Total estimate     : ${state.get('total_budget_estimate') or 0:,.0f}",
            f"Breakdown          :\n{json.dumps(state.get('budget_breakdown', {}), indent=2)}",
        ]),
        "casting": lambda: _build_agent_section("CASTING DIRECTOR OUTPUT", [
            f"Suggestions        :\n{json.dumps(state.get('casting_suggestions', []), indent=2)}",
        ]),
        "market": lambda: _build_agent_section("MARKET INTEL OUTPUT", [
            f"Average ROI        : {avg_roi_safe:.2f}x",
            f"Top platform       : {state.get('top_streaming_platform', 'Unknown')}",
            f"Comp films         :\n{json.dumps(state.get('comp_films', []), indent=2)}",
            f"Distribution       : {state.get('distribution_recommendation', 'N/A')}",
            f"Release window     : {state.get('release_window', 'N/A')}",
        ]),
    }

    # Only build sections for agents that actually ran
    active_sections = [
        agent_section_builders[agent]()
        for agent in active_agents
        if agent in agent_section_builders
    ]

    # Fallback: if active_agents is empty/missing, include all sections
    # (better than returning nothing)
    if not active_sections:
        logger.warning(
            "synthesizer_node MODE B: active_agents is empty — "
            "falling back to full context injection. "
            "Check orchestrator_node is writing active_agents to state."
        )
        active_sections = [builder() for builder in agent_section_builders.values()]

    sections = [
        # ── Mode declaration ──────────────────────────────────────────────
        "RESPONSE MODE: B — FOCUSED FOLLOW-UP ANSWER (do NOT regenerate the full report)",

        # ── Conversation history so LLM knows the film project ────────────
        history_block,

        # ── The specific question being asked, restated explicitly ─────────
        # This anchors the Synthesizer. Without this block, the model reads
        # the agent data first and pattern-matches to "full report" mode.
        (
            "═══ FOLLOW-UP QUESTION (answer THIS and ONLY THIS) ═══\n"
            f"{state.get('user_message', '')}\n\n"
            "REMINDER: Do NOT output Script Analysis, Budget Estimate, "
            "Casting Suggestions, Market Intelligence, or Project Overview sections. "
            "Answer the follow-up question above using ONLY the agent output below."
        ),

        # ── Only the relevant agent output(s) ─────────────────────────────
        *active_sections,
    ]
    return "\n\n".join(sections)


@traceable(name="synthesizer")
async def synthesizer_node(state: CineAgentState) -> CineAgentState:
    prompt_template = build_synthesizer_prompt()
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.3,   # Lower temp for MODE B: follow instructions more strictly
        max_tokens=2048,
    )

    # FIX 1: Coerce avg_roi safely — handles None, missing, and 0 correctly.
    avg_roi_safe = float(state.get("avg_roi") or 0)

    # FIX 2 + 3: Build history block and select context assembly strategy
    # based on intent. MODE A gets all four agent sections. MODE B gets only
    # the sections for agents that actually ran this turn.
    history_block = _build_history_block(state.get("session_history") or [])

    intent = state.get("intent", "full_analysis")

    if intent == "refine":
        assembled_context = _build_mode_b_context(state, history_block, avg_roi_safe)
        logger.info(
            f"synthesizer_node: MODE B — active_agents={state.get('active_agents', [])}"
        )
    else:
        assembled_context = _build_mode_a_context(state, history_block, avg_roi_safe)
        logger.info("synthesizer_node: MODE A — full report")

    try:
        messages = prompt_template.format_messages(assembled_context=assembled_context)
        response = await llm.ainvoke(messages)
        report = response.content
    except Exception as e:
        logger.error(f"synthesizer_node failed: {e}")
        report = f"# Error generating report: {e}"

    return {"final_report": report}
