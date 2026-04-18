"""
backend/graph/workflow.py
──────────────────────────
Fan-out / fan-in LangGraph workflow with asyncio stagger.

WHY THE STAGGER?
────────────────
All 3 parallel agents (budget, casting, market) fire simultaneously against
Groq's TPM window. On complex queries this causes a single-second spike that
exhausts the 30K TPM bucket instantly and triggers 429s.

Fix: wrap each parallel agent node in a coroutine that sleeps before calling
the real node. Budget starts immediately, casting waits 1s, market waits 2s.
This spreads ~15K tokens across 3 separate 1-second TPM windows instead of
spiking all at once. Groq's TPM limit resets every minute but is measured per
second internally — staggering 1s apart is enough to avoid the burst limit.

JOIN BARRIER
────────────
The explicit _join_node collects all three parallel branches before synthesizer
runs, ensuring synthesizer fires exactly once with fully merged state.
"""
import asyncio

from langgraph.graph import END, StateGraph

from backend.agents.budget_planner import budget_planner_node
from backend.agents.casting_director import casting_director_node
from backend.agents.market_intel import market_intel_node
from backend.agents.orchestrator import orchestrator_node
from backend.agents.script_analyst import script_analyst_node
from backend.graph.nodes import context_assembly_node, synthesizer_node
from backend.graph.state import CineAgentState


def _join_node(state: CineAgentState) -> CineAgentState:
    """
    No-op barrier. Waits for all parallel branches to complete,
    then passes merged state unchanged to the synthesizer.
    """
    return {}


# ── Staggered wrappers ─────────────────────────────────────────────────────────
# Each wrapper sleeps before calling the real agent node.
# This spreads parallel Groq calls across separate TPM windows,
# preventing the simultaneous burst that causes 429 rate-limit errors.

async def _budget_node_staggered(state: CineAgentState) -> CineAgentState:
    # No delay — budget fires first, uses Scout (30K TPM bucket)
    return await budget_planner_node(state)


async def _casting_node_staggered(state: CineAgentState) -> CineAgentState:
    # 1s stagger — casting is in the same Scout bucket as budget,
    # so we give budget's first LLM call time to complete before casting starts.
    await asyncio.sleep(1)
    return await casting_director_node(state)


async def _market_node_staggered(state: CineAgentState) -> CineAgentState:
    # 2s stagger — market uses the 70b bucket (12K TPM), separate from budget/casting.
    # Still stagger to avoid overlapping with script_analyst which also uses 70b.
    await asyncio.sleep(2)
    return await market_intel_node(state)


def route_after_script(state: CineAgentState):
    """
    Determines which agents to run in parallel after script analysis.
    Returns a list of node names that LangGraph will execute concurrently.
    """
    active = state.get("active_agents", [])
    next_nodes = []

    if "budget"  in active: next_nodes.append("budget_planner")
    if "casting" in active: next_nodes.append("casting_director")
    if "market"  in active: next_nodes.append("market_intel")

    # If no downstream agents, skip straight to synthesizer
    if not next_nodes:
        return ["synthesizer"]

    return next_nodes


def build_workflow() -> StateGraph:
    graph = StateGraph(CineAgentState)

    # 1. Register nodes — parallel agents use staggered wrappers
    graph.add_node("context_assembly",  context_assembly_node)
    graph.add_node("orchestrator",      orchestrator_node)
    graph.add_node("script_analyst",    script_analyst_node)
    graph.add_node("budget_planner",    _budget_node_staggered)
    graph.add_node("casting_director",  _casting_node_staggered)   
    graph.add_node("market_intel",      _market_node_staggered)  
    graph.add_node("join",              _join_node)
    graph.add_node("synthesizer",       synthesizer_node)

    # 2. Linear preamble
    graph.set_entry_point("context_assembly")
    graph.add_edge("context_assembly", "orchestrator")
    graph.add_edge("orchestrator",     "script_analyst")

    # 3. Parallel fan-out after script_analyst
    graph.add_conditional_edges(
        "script_analyst",
        route_after_script,
        {
            "budget_planner":   "budget_planner",
            "casting_director": "casting_director",
            "market_intel":     "market_intel",
            "synthesizer":      "synthesizer",
        }
    )

    # 4. Fan-in: all parallel branches converge at join
    graph.add_edge("budget_planner",   "join")
    graph.add_edge("casting_director", "join")
    graph.add_edge("market_intel",     "join")

    # 5. join → synthesizer → END
    graph.add_edge("join",        "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


workflow = build_workflow()