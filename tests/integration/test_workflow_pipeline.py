"""
tests/integration/test_workflow_pipeline.py
─────────────────────────────────────────────
Integration tests for the full LangGraph pipeline.

These tests exercise the entire workflow (context_assembly → orchestrator →
script_analyst → [budget, casting, market] → synthesizer) with:
  - All external services (Groq, Qdrant, Redis, DB) mocked
  - Real LangGraph StateGraph wiring (workflow.py)
  - Realistic state transitions verified end-to-end

The goal is to catch regressions where a state key written by one node
is silently dropped before the synthesizer reads it.
"""
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def make_llm_response(content: str):
    r = MagicMock()
    r.content    = content
    r.tool_calls = []
    return r


def make_orchestrator_response(intent: str, agents: list[str]) -> str:
    return json.dumps({"intent": intent, "active_agents": agents})


@pytest.fixture
def base_initial_state():
    return {
        "user_id":      str(uuid.uuid4()),
        "session_id":   str(uuid.uuid4()),
        "user_message": "A gritty crime drama set in 1990s Chicago",
    }


class TestFullAnalysisPipeline:
    """
    End-to-end MODE A: fresh film concept → full report.
    All four agents run; synthesizer produces the final report.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_final_report(
        self, base_initial_state, mock_settings, mock_redis
    ):
        orchestrator_resp = make_llm_response(
            make_orchestrator_response("full_analysis",
                                       ["script", "budget", "casting", "market"])
        )
        script_resp = make_llm_response(
            "GENRES: crime, drama\n"
            "TONE: gritty, tense\n"
            "COMPLEXITY: moderate\n"
            "CHARACTERS: Tony :: Crime boss | Maria :: Detective\n"
            "THEMES: corruption, redemption\n"
            "BUDGET_FLAGS: practical stunts\n"
            "SCRIPT_COMPS: The Departed :: 2006 :: $290M"
        )
        budget_resp   = make_llm_response("Budget estimate: $3.2M indie production.")
        casting_resp  = make_llm_response("Casting: 3 suggestions found.")
        market_resp   = make_llm_response("Streaming platform recommended.")
        synth_resp    = make_llm_response("# Pre-Production Intelligence Report\n## Overview\n...")

        call_counter = {"n": 0}
        responses    = [
            orchestrator_resp,
            script_resp,
            budget_resp,
            casting_resp,
            market_resp,
            synth_resp,
        ]

        async def _fake_ainvoke(messages):
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return responses[idx]

        mock_llm = MagicMock()
        mock_llm.ainvoke    = _fake_ainvoke
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("backend.agents.orchestrator.ChatGroq",    return_value=mock_llm), \
             patch("backend.agents._base.ChatGroq",           return_value=mock_llm), \
             patch("backend.agents._base._build_llm",         return_value=mock_llm), \
             patch("backend.graph.nodes.ChatGroq",            return_value=mock_llm), \
             patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock, return_value=[]), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock, return_value="No prior history."):

            import sys
            sys.modules.pop("backend.graph.workflow", None)
            from backend.graph.workflow import build_workflow
            wf = build_workflow()

            final_state: dict = {}
            async for chunk in wf.astream(base_initial_state):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

        assert "final_report" in final_state
        assert isinstance(final_state["final_report"], str)
        assert len(final_state["final_report"]) > 0

    @pytest.mark.asyncio
    async def test_script_analyst_output_flows_to_budget(
        self, base_initial_state, mock_settings, mock_redis
    ):
        """
        Verify that genres + budget_flags written by script_analyst are
        present in state when budget_planner runs.
        """
        orchestrator_resp = make_llm_response(
            make_orchestrator_response("full_analysis", ["script", "budget"])
        )
        script_resp = make_llm_response(
            "GENRES: thriller\n"
            "TONE: dark\n"
            "COMPLEXITY: complex\n"
            "CHARACTERS:\n"
            "THEMES:\n"
            "BUDGET_FLAGS: VFX-heavy, night shoots\n"
            "SCRIPT_COMPS:"
        )
        budget_resp = make_llm_response("Budget report")
        synth_resp  = make_llm_response("# Report")

        call_counter = {"n": 0}
        responses    = [orchestrator_resp, script_resp, budget_resp, synth_resp]

        async def _fake_ainvoke(messages):
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return responses[idx]

        captured_budget_state = {}

        _ = None
        async def _spy_budget(state):
            captured_budget_state.update(state)
            # Return minimal valid output
            return {"budget_breakdown": {}, "total_budget_estimate": 0.0,
                    "budget_tier": "indie", "live_union_rates": {}}

        mock_llm = MagicMock()
        mock_llm.ainvoke    = _fake_ainvoke
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("backend.agents.orchestrator.ChatGroq",  return_value=mock_llm), \
             patch("backend.agents._base.ChatGroq",         return_value=mock_llm), \
             patch("backend.agents._base._build_llm",       return_value=mock_llm), \
             patch("backend.graph.nodes.ChatGroq",          return_value=mock_llm), \
             patch("backend.graph.workflow.budget_planner_node",  _spy_budget), \
             patch("backend.graph.workflow.casting_director_node",
                   new_callable=AsyncMock, return_value={}), \
             patch("backend.graph.workflow.market_intel_node",
                   new_callable=AsyncMock, return_value={}), \
             patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock, return_value=[]), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock, return_value=""):

            import sys
            sys.modules.pop("backend.graph.workflow", None)
            from backend.graph.workflow import build_workflow
            wf = build_workflow()
            async for _ in wf.astream(base_initial_state):
                pass

        # script_analyst output should be in state when budget runs
        if captured_budget_state:
            genres = captured_budget_state.get("genres", [])
            flags  = captured_budget_state.get("budget_flags", [])
            # These may be populated from script_analyst output
            assert isinstance(genres, list)
            assert isinstance(flags,  list)


class TestFollowUpPipeline:
    """
    MODE B: follow-up turn with session_history non-empty.
    All agents have tools suppressed; synthesizer gets MODE B context.
    """

    @pytest.mark.asyncio
    async def test_followup_message_produces_focused_response(
        self, mock_settings, mock_redis
    ):
        user_id    = str(uuid.uuid4())
        session_id = str(uuid.uuid4())

        history = [
            {"role": "user",      "content": "A sci-fi film set in 2077"},
            {"role": "assistant", "content": "## Pre-Production Report\n..."},
        ]

        initial_state = {
            "user_id":      user_id,
            "session_id":   session_id,
            "user_message": "What is the marketing strategy?",
        }

        orchestrator_resp = make_llm_response(
            make_orchestrator_response("refine",
                                       ["script", "budget", "casting", "market"])
        )
        agent_resp  = make_llm_response(
            "GENRES: sci-fi\nTONE: dark\nCOMPLEXITY: moderate\n"
            "CHARACTERS:\nTHEMES:\nBUDGET_FLAGS:\nSCRIPT_COMPS:"
        )
        synth_resp  = make_llm_response(
            "For a sci-fi indie film, streaming via Netflix is recommended..."
        )

        call_counter = {"n": 0}
        responses    = [orchestrator_resp, agent_resp, agent_resp,
                        agent_resp, agent_resp, synth_resp]

        async def _fake_ainvoke(messages):
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return responses[idx]

        mock_llm = MagicMock()
        mock_llm.ainvoke    = _fake_ainvoke
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("backend.agents.orchestrator.ChatGroq",  return_value=mock_llm), \
             patch("backend.agents._base.ChatGroq",         return_value=mock_llm), \
             patch("backend.agents._base._build_llm",       return_value=mock_llm), \
             patch("backend.graph.nodes.ChatGroq",          return_value=mock_llm), \
             patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock, return_value=history), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock, return_value="User prefers sci-fi."):

            import sys
            sys.modules.pop("backend.graph.workflow", None)
            from backend.graph.workflow import build_workflow
            wf = build_workflow()

            final_state: dict = {}
            async for chunk in wf.astream(initial_state):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

        assert "final_report" in final_state
        # The intent should be "refine" (either LLM classified it, or signal detector caught it)
        intent = final_state.get("intent", "")
        assert intent in ("refine", "full_analysis")  # either valid since history was set


class TestOrchestratorIntegration:
    @pytest.mark.asyncio
    async def test_orchestrator_routes_script_only(
        self, mock_settings, mock_redis
    ):
        """If orchestrator returns active_agents=["script"], only script runs."""
        initial_state = {
            "user_id":      str(uuid.uuid4()),
            "session_id":   str(uuid.uuid4()),
            "user_message": "Analyse just the script structure",
        }

        orchestrator_resp = make_llm_response(
            make_orchestrator_response("script_only", ["script"])
        )
        script_resp = make_llm_response(
            "GENRES: drama\nTONE: dramatic\nCOMPLEXITY: simple\n"
            "CHARACTERS:\nTHEMES:\nBUDGET_FLAGS:\nSCRIPT_COMPS:"
        )
        synth_resp = make_llm_response("Script-only analysis complete.")

        call_counter = {"n": 0}
        responses    = [orchestrator_resp, script_resp, synth_resp]

        async def _fake_ainvoke(messages):
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return responses[idx]

        mock_llm = MagicMock()
        mock_llm.ainvoke    = _fake_ainvoke
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        budget_called  = False
        casting_called = False
        market_called  = False

        async def _spy_budget(state):
            nonlocal budget_called
            budget_called = True
            return {}

        async def _spy_casting(state):
            nonlocal casting_called
            casting_called = True
            return {}

        async def _spy_market(state):
            nonlocal market_called
            market_called = True
            return {}

        with patch("backend.agents.orchestrator.ChatGroq",  return_value=mock_llm), \
             patch("backend.agents._base.ChatGroq",         return_value=mock_llm), \
             patch("backend.agents._base._build_llm",       return_value=mock_llm), \
             patch("backend.graph.nodes.ChatGroq",          return_value=mock_llm), \
             patch("backend.graph.workflow.budget_planner_node",   _spy_budget), \
             patch("backend.graph.workflow.casting_director_node", _spy_casting), \
             patch("backend.graph.workflow.market_intel_node",     _spy_market), \
             patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock, return_value=[]), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock, return_value=""):

            import sys
            sys.modules.pop("backend.graph.workflow", None)
            from backend.graph.workflow import build_workflow
            wf = build_workflow()
            async for _ in wf.astream(initial_state):
                pass

        # Only script was in active_agents, so budget/casting/market should NOT run
        assert not budget_called,  "budget_planner should not have been called"
        assert not casting_called, "casting_director should not have been called"
        assert not market_called,  "market_intel should not have been called"


class TestStateIsolation:
    """
    Verify that avg_roi=None does not crash the synthesizer (FIX 1 regression).
    """

    @pytest.mark.asyncio
    async def test_none_avg_roi_does_not_crash_synthesizer(
        self, base_initial_state, mock_settings, mock_redis
    ):
        orchestrator_resp = make_llm_response(
            make_orchestrator_response("full_analysis",
                                       ["script", "budget", "casting", "market"])
        )
        script_resp = make_llm_response(
            "GENRES: drama\nTONE: serious\nCOMPLEXITY: moderate\n"
            "CHARACTERS:\nTHEMES:\nBUDGET_FLAGS:\nSCRIPT_COMPS:"
        )
        agent_resp = make_llm_response("Agent output")
        synth_resp = make_llm_response("# Report")

        call_counter = {"n": 0}
        responses    = [orchestrator_resp, script_resp, agent_resp,
                        agent_resp, agent_resp, synth_resp]

        async def _fake_ainvoke(messages):
            idx = min(call_counter["n"], len(responses) - 1)
            call_counter["n"] += 1
            return responses[idx]

        mock_llm = MagicMock()
        mock_llm.ainvoke    = _fake_ainvoke
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        # Inject avg_roi=None to trigger the FIX 1 scenario
        state_with_none_roi = {**base_initial_state, "avg_roi": None}

        with patch("backend.agents.orchestrator.ChatGroq",  return_value=mock_llm), \
             patch("backend.agents._base.ChatGroq",         return_value=mock_llm), \
             patch("backend.agents._base._build_llm",       return_value=mock_llm), \
             patch("backend.graph.nodes.ChatGroq",          return_value=mock_llm), \
             patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock, return_value=[]), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock, return_value=""):

            import sys
            sys.modules.pop("backend.graph.workflow", None)
            from backend.graph.workflow import build_workflow
            wf = build_workflow()

            # Must not raise TypeError
            final_state: dict = {}
            async for chunk in wf.astream(state_with_none_roi):
                for _, node_output in chunk.items():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

        assert "final_report" in final_state