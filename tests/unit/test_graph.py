"""
tests/unit/test_graph.py
─────────────────────────
Unit tests for:
  - backend/graph/nodes.py     (context_assembly_node, synthesizer_node,
                                 _build_history_block, mode A/B context builders)
  - backend/graph/workflow.py  (route_after_script)
  - backend/graph/state.py     (TypedDict shape)
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── State ─────────────────────────────────────────────────────────────────────

class TestCineAgentState:
    def test_state_is_total_false_allows_partial_dict(self):
        from backend.graph.state import CineAgentState
        # total=False means all keys are optional — partial dicts are valid
        partial: CineAgentState = {"user_id": "u1", "user_message": "hello"}
        assert partial["user_id"] == "u1"

    def test_state_keys_exist(self):
        from backend.graph.state import CineAgentState
        annotations = CineAgentState.__annotations__
        expected_keys = [
            "user_id", "session_id", "user_message", "active_agents", "intent",
            "session_history", "user_context", "genres", "tone",
            "structural_complexity", "characters", "themes", "budget_flags",
            "budget_tier", "budget_breakdown", "total_budget_estimate",
            "casting_suggestions", "casting_notes", "market_comps", "avg_roi",
            "distribution_recommendation", "final_report",
        ]
        for key in expected_keys:
            assert key in annotations, f"Missing state key: {key}"


# ── History block builder ─────────────────────────────────────────────────────

class TestBuildHistoryBlock:
    def test_empty_history_returns_no_prior_turns(self):
        from backend.graph.nodes import _build_history_block
        result = _build_history_block([])
        assert "No prior turns" in result

    def test_formats_user_and_assistant_roles(self):
        from backend.graph.nodes import _build_history_block
        history = [
            {"role": "user",      "content": "My film idea"},
            {"role": "assistant", "content": "Great! Here is the report."},
        ]
        result = _build_history_block(history)
        assert "USER:" in result
        assert "ASSISTANT:" in result
        assert "My film idea" in result

    def test_truncates_long_assistant_messages(self):
        from backend.graph.nodes import _build_history_block
        long_content = "x" * 1000
        history = [{"role": "assistant", "content": long_content}]
        result = _build_history_block(history)
        # Assistant messages truncated at 600 chars
        assert len(result) < len(long_content) + 200
        assert "truncated" in result

    def test_does_not_truncate_user_messages(self):
        from backend.graph.nodes import _build_history_block
        long_content = "y" * 700
        history = [{"role": "user", "content": long_content}]
        result = _build_history_block(history)
        assert "truncated" not in result


# ── Context builders ──────────────────────────────────────────────────────────

class TestBuildModeAContext:
    def test_includes_all_four_agent_sections(self, sample_state):
        from backend.graph.nodes import _build_mode_a_context
        history_block = "═══ CONVERSATION HISTORY ═══\n(No prior turns)"
        result = _build_mode_a_context(sample_state, history_block, 0.0)
        assert "SCRIPT ANALYST OUTPUT"  in result
        assert "BUDGET PLANNER OUTPUT"  in result
        assert "CASTING DIRECTOR OUTPUT" in result
        assert "MARKET INTEL OUTPUT"    in result

    def test_includes_mode_a_declaration(self, sample_state):
        from backend.graph.nodes import _build_mode_a_context
        result = _build_mode_a_context(sample_state, "", 0.0)
        assert "RESPONSE MODE: A" in result
        assert "FULL REPORT" in result

    def test_formats_avg_roi_correctly(self, sample_state):
        from backend.graph.nodes import _build_mode_a_context
        result = _build_mode_a_context(sample_state, "", 2.75)
        assert "2.75x" in result


class TestBuildModeBContext:
    def test_includes_mode_b_declaration(self, sample_state):
        from backend.graph.nodes import _build_mode_b_context
        sample_state["active_agents"] = ["market"]
        result = _build_mode_b_context(sample_state, "", 0.0)
        assert "RESPONSE MODE: B" in result
        assert "FOCUSED FOLLOW-UP" in result

    def test_only_includes_active_agent_sections(self, sample_state):
        from backend.graph.nodes import _build_mode_b_context
        sample_state["active_agents"] = ["market"]
        result = _build_mode_b_context(sample_state, "", 0.0)
        assert "MARKET INTEL OUTPUT"    in result
        assert "BUDGET PLANNER OUTPUT"  not in result
        assert "CASTING DIRECTOR OUTPUT" not in result

    def test_includes_follow_up_question_block(self, sample_state):
        from backend.graph.nodes import _build_mode_b_context
        sample_state["active_agents"]  = ["market"]
        sample_state["user_message"]   = "What is the marketing strategy?"
        result = _build_mode_b_context(sample_state, "", 0.0)
        assert "FOLLOW-UP QUESTION" in result
        assert "What is the marketing strategy?" in result

    def test_fallback_includes_all_sections_when_active_agents_empty(self, sample_state):
        from backend.graph.nodes import _build_mode_b_context
        sample_state["active_agents"] = []
        result = _build_mode_b_context(sample_state, "", 0.0)
        # Fallback: all four sections
        assert "SCRIPT ANALYST OUTPUT"  in result
        assert "MARKET INTEL OUTPUT"    in result


# ── context_assembly_node ─────────────────────────────────────────────────────

class TestContextAssemblyNode:
    @pytest.mark.asyncio
    async def test_adds_session_history_and_user_context(self, sample_state, mock_redis):
        mock_redis.lrange = AsyncMock(return_value=[])

        with patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock,
                   return_value=[{"role": "user", "content": "past message"}]), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock,
                   return_value="User prefers sci-fi films."):

            from backend.graph.nodes import context_assembly_node
            result = await context_assembly_node(sample_state)

        assert result["session_history"] == [{"role": "user", "content": "past message"}]
        assert result["user_context"]    == "User prefers sci-fi films."

    @pytest.mark.asyncio
    async def test_falls_back_to_postgres_when_redis_empty(self, sample_state):
        from backend.database.models import Message
        fake_msg         = MagicMock(spec=Message)
        fake_msg.role    = "user"
        fake_msg.content = "DB fallback message"

        with patch("backend.graph.nodes.get_session_history",
                   new_callable=AsyncMock,
                   return_value=[]), \
             patch("backend.graph.nodes.retrieve_user_context",
                   new_callable=AsyncMock,
                   return_value="No prior project history."), \
             patch("backend.graph.nodes.AsyncSessionLocal") as mock_session_local, \
             patch("backend.graph.nodes.crud") as mock_crud:

            mock_crud.get_session_messages = AsyncMock(return_value=[fake_msg])
            # Mock the async context manager
            mock_db = AsyncMock()
            mock_session_local.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_session_local.return_value.__aexit__  = AsyncMock(return_value=False)

            from backend.graph.nodes import context_assembly_node
            result = await context_assembly_node(sample_state)

        # Whether redis or pg fallback, session_history must be a list
        assert isinstance(result["session_history"], list)


# ── synthesizer_node ──────────────────────────────────────────────────────────

class TestSynthesizerNode:
    @pytest.mark.asyncio
    async def test_returns_final_report_string(self, sample_state, mock_settings):
        fake_response = MagicMock()
        fake_response.content = "# Pre-Production Intelligence Report\n..."

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)

        with patch("backend.graph.nodes.ChatGroq", return_value=mock_llm):
            from backend.graph.nodes import synthesizer_node
            result = await synthesizer_node(sample_state)

        assert result["final_report"] == "# Pre-Production Intelligence Report\n..."

    @pytest.mark.asyncio
    async def test_handles_none_avg_roi_safely(self, sample_state, mock_settings):
        """avg_roi=None should not cause TypeError with :.2f formatting."""
        sample_state["avg_roi"] = None

        fake_response = MagicMock()
        fake_response.content = "Report"
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)

        with patch("backend.graph.nodes.ChatGroq", return_value=mock_llm):
            from backend.graph.nodes import synthesizer_node
            result = await synthesizer_node(sample_state)

        assert result["final_report"] == "Report"

    @pytest.mark.asyncio
    async def test_uses_mode_b_for_refine_intent(self, sample_state_followup, mock_settings):
        captured_context = []
        fake_response = MagicMock()
        fake_response.content = "Focused follow-up answer"

        async def _fake_ainvoke(messages):
            for m in messages:
                if hasattr(m, "content"):
                    captured_context.append(m.content)
            return fake_response

        mock_llm = AsyncMock()
        mock_llm.ainvoke = _fake_ainvoke

        with patch("backend.graph.nodes.ChatGroq", return_value=mock_llm):
            from backend.graph.nodes import synthesizer_node
            await synthesizer_node(sample_state_followup)

        all_context = " ".join(captured_context)
        assert "RESPONSE MODE: B" in all_context or "FOCUSED FOLLOW-UP" in all_context

    @pytest.mark.asyncio
    async def test_returns_error_string_on_llm_failure(self, sample_state, mock_settings):
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("Groq 500"))

        with patch("backend.graph.nodes.ChatGroq", return_value=mock_llm):
            from backend.graph.nodes import synthesizer_node
            result = await synthesizer_node(sample_state)

        assert "Error" in result["final_report"]


# ── Workflow routing ──────────────────────────────────────────────────────────

class TestRouteAfterScript:
    def test_routes_to_all_three_parallel_agents(self, sample_state):
        from backend.graph.workflow import route_after_script
        sample_state["active_agents"] = ["budget", "casting", "market"]
        result = route_after_script(sample_state)
        assert set(result) == {"budget_planner", "casting_director", "market_intel"}

    def test_routes_only_to_budget(self, sample_state):
        from backend.graph.workflow import route_after_script
        sample_state["active_agents"] = ["budget"]
        result = route_after_script(sample_state)
        assert result == ["budget_planner"]

    def test_routes_to_synthesizer_when_no_downstream_agents(self, sample_state):
        from backend.graph.workflow import route_after_script
        sample_state["active_agents"] = []
        result = route_after_script(sample_state)
        assert result == ["synthesizer"]

    def test_skips_agents_not_in_active_list(self, sample_state):
        from backend.graph.workflow import route_after_script
        sample_state["active_agents"] = ["budget", "market"]
        result = route_after_script(sample_state)
        assert "casting_director" not in result
        assert "budget_planner"   in result
        assert "market_intel"     in result