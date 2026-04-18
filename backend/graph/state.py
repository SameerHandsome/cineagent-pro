"""
backend/graph/state.py
──────────────────────
The single shared state object that flows through the LangGraph workflow.
Every agent reads from and writes to this.  Using TypedDict means LangGraph
can track which keys changed and run conditional edges cleanly.
"""
from typing import Any, TypedDict


class CineAgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    user_id: str
    session_id: str
    user_message: str              # raw user input

    # ── Orchestrator decisions ────────────────────────────────────────────────
    active_agents: list[str]       # e.g. ["script", "budget", "casting", "market"]
    intent: str                    # e.g. "full_analysis" | "budget_only" | "market_query"

    # ── Context (from RAG + Redis) ────────────────────────────────────────────
    session_history: list[dict]    # last 5 messages from Redis
    user_context: str              # retrieved Qdrant docs about this user's past projects

    # ── Script Analyst outputs ────────────────────────────────────────────────
    genres: list[str]
    tone: list[str]
    structural_complexity: str
    characters: list[dict]
    themes: list[str]
    budget_flags: list[str]
    script_comps: list[dict]       # IMDb comps from Playwright scrape

    # ── Budget Planner outputs ────────────────────────────────────────────────
    budget_tier: str               # micro | indie | mid | a-list
    budget_breakdown: dict[str, Any]
    total_budget_estimate: float
    live_union_rates: dict[str, Any]

    # ── Casting Director outputs ───────────────────────────────────────────────
    casting_suggestions: list[dict]
    casting_notes: str

    # ── Market Intel outputs ──────────────────────────────────────────────────
    market_comps: list[dict]
    avg_roi: float
    distribution_recommendation: str  # "theatrical" | "streaming" | "hybrid"
    top_streaming_platform: str
    release_timing_note: str

    # ── Final output ──────────────────────────────────────────────────────────
    final_report: str
    error: str | None
