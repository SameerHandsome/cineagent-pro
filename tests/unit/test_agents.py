"""
tests/unit/test_agents.py
──────────────────────────
Unit tests for all agent nodes:
  - backend/agents/_base.py            (run_agent_loop, _coerce_args)
  - backend/agents/orchestrator.py     (routing, refinement detection)
  - backend/agents/script_analyst.py   (_parse_labelled_output, node)
  - backend/agents/budget_planner.py   (node, tool suppression)
  - backend/agents/casting_director.py (node, tool suppression)
  - backend/agents/market_intel.py     (node, tool suppression)
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── _base.py ─────────────────────────────────────────────────────────────────

class TestCoerceArgs:
    def test_coerces_string_to_integer(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema.schema.return_value = {
            "properties": {"count": {"type": "integer"}}
        }
        result = _coerce_args(tool, {"count": "5"})
        assert result["count"] == 5
        assert isinstance(result["count"], int)

    def test_coerces_string_to_float(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema.schema.return_value = {
            "properties": {"budget": {"type": "number"}}
        }
        result = _coerce_args(tool, {"budget": "1500000.50"})
        assert result["budget"] == 1500000.50

    def test_coerces_string_to_boolean_true(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema.schema.return_value = {
            "properties": {"include_union": {"type": "boolean"}}
        }
        result = _coerce_args(tool, {"include_union": "true"})
        assert result["include_union"] is True

    def test_coerces_string_to_boolean_false(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema.schema.return_value = {
            "properties": {"include_union": {"type": "boolean"}}
        }
        result = _coerce_args(tool, {"include_union": "false"})
        assert result["include_union"] is False

    def test_leaves_string_types_unchanged(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema.schema.return_value = {
            "properties": {"genre": {"type": "string"}}
        }
        result = _coerce_args(tool, {"genre": "sci-fi"})
        assert result["genre"] == "sci-fi"

    def test_handles_no_schema_gracefully(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema = None
        result = _coerce_args(tool, {"x": "1"})
        assert result == {"x": "1"}

    def test_handles_schema_exception_gracefully(self):
        from backend.agents._base import _coerce_args
        tool = MagicMock()
        tool.args_schema.schema.side_effect = Exception("schema error")
        result = _coerce_args(tool, {"x": "1"})
        assert result == {"x": "1"}


class TestIsMalformedToolCallError:
    def test_detects_tool_use_failed(self):
        from backend.agents._base import _is_malformed_tool_call_error
        exc = Exception("400 tool_use_failed: something went wrong")
        assert _is_malformed_tool_call_error(exc) is True

    def test_detects_failed_generation(self):
        from backend.agents._base import _is_malformed_tool_call_error
        exc = Exception("400 failed_generation: '[tavily_search(\"test\")]'")
        assert _is_malformed_tool_call_error(exc) is True

    def test_does_not_detect_other_400(self):
        from backend.agents._base import _is_malformed_tool_call_error
        exc = Exception("400 invalid_request: missing field")
        assert _is_malformed_tool_call_error(exc) is False

    def test_does_not_detect_non_400(self):
        from backend.agents._base import _is_malformed_tool_call_error
        exc = Exception("500 internal server error")
        assert _is_malformed_tool_call_error(exc) is False


class TestRunAgentLoop:
    @pytest.mark.asyncio
    async def test_returns_content_when_no_tool_calls(self):
        from langchain_core.prompts import ChatPromptTemplate

        from backend.agents._base import run_agent_loop

        fake_response = MagicMock()
        fake_response.content    = "Final answer text"
        fake_response.tool_calls = []

        mock_llm = AsyncMock()
        mock_llm.ainvoke    = AsyncMock(return_value=fake_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("backend.agents._base.ChatGroq", return_value=mock_llm), \
             patch("backend.agents._base._build_llm", return_value=mock_llm):
            prompt = ChatPromptTemplate.from_messages([
                ("human", "{user_query}{agent_scratchpad}")
            ])
            content, tool_results = await run_agent_loop(
                agent_name="TestAgent",
                prompt_template=prompt,
                prompt_vars={},
                user_query="test query",
                tools=[],
            )

        assert content == "Final answer text"
        assert tool_results == []

    @pytest.mark.asyncio
    async def test_recovers_from_malformed_tool_call_error(self):
        from langchain_core.prompts import ChatPromptTemplate

        from backend.agents._base import run_agent_loop

        fake_plain_response = MagicMock()
        fake_plain_response.content    = "Plain text fallback"
        fake_plain_response.tool_calls = []

        mock_llm_tools = AsyncMock()
        mock_llm_tools.ainvoke = AsyncMock(
            side_effect=Exception("400 tool_use_failed: bad generation")
        )
        mock_llm_tools.bind_tools = MagicMock(return_value=mock_llm_tools)

        mock_llm_plain = AsyncMock()
        mock_llm_plain.ainvoke = AsyncMock(return_value=fake_plain_response)
        mock_llm_plain.bind_tools = MagicMock(return_value=mock_llm_plain)

        call_count = 0
        def _build_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_llm_tools
            return mock_llm_plain

        with patch("backend.agents._base._build_llm", side_effect=_build_side_effect):
            prompt = ChatPromptTemplate.from_messages([
                ("human", "{user_query}{agent_scratchpad}")
            ])
            content, tool_results = await run_agent_loop(
                agent_name="TestAgent",
                prompt_template=prompt,
                prompt_vars={},
                user_query="follow up question",
                tools=[MagicMock()],
            )

        assert content == "Plain text fallback"

    @pytest.mark.asyncio
    async def test_reraises_non_tool_errors(self):
        from langchain_core.prompts import ChatPromptTemplate

        from backend.agents._base import run_agent_loop

        mock_llm = AsyncMock()
        mock_llm.ainvoke    = AsyncMock(side_effect=Exception("500 internal server error"))
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with patch("backend.agents._base._build_llm", return_value=mock_llm):
            prompt = ChatPromptTemplate.from_messages([
                ("human", "{user_query}{agent_scratchpad}")
            ])
            with pytest.raises(Exception, match="500 internal server error"):
                await run_agent_loop(
                    agent_name="TestAgent",
                    prompt_template=prompt,
                    prompt_vars={},
                    user_query="test",
                    tools=[],
                )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class TestOrchestratorRefinementDetection:
    def test_is_refinement_returns_false_no_history(self):
        from backend.agents.orchestrator import _is_refinement
        assert _is_refinement("develop a marketing strategy", []) is False

    def test_is_refinement_returns_true_with_history_and_keyword(self):
        from backend.agents.orchestrator import _is_refinement
        history = [{"role": "user", "content": "prior message"}]
        assert _is_refinement("develop a marketing strategy", history) is True

    def test_is_refinement_returns_false_no_keyword_match(self):
        from backend.agents.orchestrator import _is_refinement
        history = [{"role": "user", "content": "prior message"}]
        assert _is_refinement("A brand new sci-fi film about robots", history) is False

    def test_is_refinement_case_insensitive(self):
        from backend.agents.orchestrator import _is_refinement
        history = [{"role": "user", "content": "prior message"}]
        assert _is_refinement("EXPAND on the budget", history) is True

    @pytest.mark.asyncio
    async def test_orchestrator_overrides_full_analysis_to_refine(self, mock_settings):
        import json

        from backend.agents.orchestrator import orchestrator_node

        fake_response = MagicMock()
        fake_response.content = json.dumps({
            "intent": "full_analysis",
            "active_agents": ["script", "budget", "casting", "market"]
        })

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)

        history = [{"role": "user", "content": "prior message"}]
        state = {
            "user_id":        "u1",
            "session_id":     "s1",
            "user_message":   "expand on the marketing strategy",
            "session_history": history,
            "user_context":   "No prior history.",
        }

        with patch("backend.agents.orchestrator.ChatGroq", return_value=mock_llm):
            result = await orchestrator_node(state)

        assert result["intent"] == "refine"

    @pytest.mark.asyncio
    async def test_orchestrator_defaults_on_json_parse_failure(self, mock_settings):
        from backend.agents.orchestrator import orchestrator_node

        fake_response = MagicMock()
        fake_response.content = "not valid json at all"

        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=fake_response)

        state = {
            "user_id":         "u1",
            "session_id":      "s1",
            "user_message":    "A new crime thriller",
            "session_history": [],
            "user_context":    "No prior history.",
        }

        with patch("backend.agents.orchestrator.ChatGroq", return_value=mock_llm):
            result = await orchestrator_node(state)

        assert result["intent"] == "full_analysis"
        assert set(result["active_agents"]) == {"script", "budget", "casting", "market"}


# ── ScriptAnalyst ─────────────────────────────────────────────────────────────

class TestParseLabeldOutput:
    def test_parses_genres(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "GENRES: sci-fi, thriller"
        result = _parse_labelled_output(text)
        assert "sci-fi"   in result["genres"]
        assert "thriller" in result["genres"]

    def test_parses_tone(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "TONE: dark, tense"
        result = _parse_labelled_output(text)
        assert "dark"  in result["tone"]
        assert "tense" in result["tone"]

    def test_parses_complexity(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "COMPLEXITY: complex"
        result = _parse_labelled_output(text)
        assert result["structural_complexity"] == "complex"

    def test_parses_characters_with_description(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "CHARACTERS: Kai :: Hacker protagonist | Luna :: Tech billionaire"
        result = _parse_labelled_output(text)
        assert len(result["characters"]) == 2
        assert result["characters"][0]["name"]        == "Kai"
        assert result["characters"][0]["description"] == "Hacker protagonist"

    def test_parses_characters_without_description(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "CHARACTERS: Kai"
        result = _parse_labelled_output(text)
        assert result["characters"][0]["name"] == "Kai"
        assert "No description" in result["characters"][0]["description"]

    def test_parses_budget_flags(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "BUDGET_FLAGS: VFX-heavy, night shoots"
        result = _parse_labelled_output(text)
        assert "VFX-heavy" in result["budget_flags"]

    def test_parses_script_comps(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "SCRIPT_COMPS: Blade Runner 2049 :: 2017 :: $259M"
        result = _parse_labelled_output(text)
        assert len(result["script_comps"]) == 1
        assert result["script_comps"][0]["title"] == "Blade Runner 2049"
        assert result["script_comps"][0]["year"]  == "2017"
        assert result["script_comps"][0]["gross"] == "$259M"

    def test_filters_empty_strings(self):
        from backend.agents.script_analyst import _parse_labelled_output
        text = "GENRES: , sci-fi, , drama,"
        result = _parse_labelled_output(text)
        assert "" not in result["genres"]

    def test_returns_defaults_for_empty_input(self):
        from backend.agents.script_analyst import _parse_labelled_output
        result = _parse_labelled_output("")
        assert result["genres"]   == []
        assert result["tone"]     == []
        assert result["structural_complexity"] == "moderate"
        assert result["characters"] == []


class TestScriptAnalystNode:
    @pytest.mark.asyncio
    async def test_skips_when_not_in_active_agents(self, sample_state):
        from backend.agents.script_analyst import script_analyst_node
        sample_state["active_agents"] = ["budget", "casting"]
        result = await script_analyst_node(sample_state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_suppresses_tools_on_followup(self, sample_state_followup):
        from backend.agents.script_analyst import script_analyst_node

        captured_tools = []
        async def _mock_loop(**kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            return ("GENRES: drama\nTONE: serious\nCOMPLEXITY: moderate\n"
                    "CHARACTERS:\nTHEMES:\nBUDGET_FLAGS:\nSCRIPT_COMPS:", [])

        with patch("backend.agents.script_analyst.run_agent_loop",
                   side_effect=_mock_loop):
            _ = await script_analyst_node(sample_state_followup)

        assert captured_tools == []  # tools suppressed on followup

    @pytest.mark.asyncio
    async def test_returns_defaults_when_parse_empty(self, sample_state):
        from backend.agents.script_analyst import script_analyst_node
        sample_state["active_agents"] = ["script"]
        sample_state["session_history"] = []

        async def _mock_loop(**kwargs):
            return ("", [])

        with patch("backend.agents.script_analyst.run_agent_loop", side_effect=_mock_loop):
            result = await script_analyst_node(sample_state)

        assert result["genres"] == ["drama"]   # default fallback
        assert result["tone"]   == ["dramatic"]


# ── BudgetPlanner ─────────────────────────────────────────────────────────────

class TestBudgetPlannerNode:
    @pytest.mark.asyncio
    async def test_skips_when_not_in_active_agents(self, sample_state):
        from backend.agents.budget_planner import budget_planner_node
        sample_state["active_agents"] = ["script", "casting"]
        result = await budget_planner_node(sample_state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_suppresses_tools_on_followup(self, sample_state_followup):
        from backend.agents.budget_planner import budget_planner_node
        sample_state_followup["active_agents"] = ["budget"]

        captured_tools = []
        async def _mock_loop(**kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            return ("Budget summary text", [])

        with patch("backend.agents.budget_planner.run_agent_loop",
                   side_effect=_mock_loop):
            await budget_planner_node(sample_state_followup)

        assert captured_tools == []

    @pytest.mark.asyncio
    async def test_aggregates_department_totals_from_tool_results(self, sample_state):
        from backend.agents.budget_planner import budget_planner_node
        sample_state["active_agents"] = ["budget"]
        sample_state["session_history"] = []

        tool_results = [
            {
                "tool": "calculate_budget_line",
                "args": {},
                "result": {
                    "department": "vfx",
                    "department_total": 500000.0,
                    "budget_tier": "indie",
                },
            },
            {
                "tool": "calculate_budget_line",
                "args": {},
                "result": {
                    "department": "cast",
                    "department_total": 300000.0,
                    "budget_tier": "indie",
                },
            },
        ]

        async def _mock_loop(**kwargs):
            return ("Budget report", tool_results)

        with patch("backend.agents.budget_planner.run_agent_loop",
                   side_effect=_mock_loop):
            result = await budget_planner_node(sample_state)

        assert result["total_budget_estimate"] == 800000.0
        assert "vfx"  in result["budget_breakdown"]
        assert "cast" in result["budget_breakdown"]
        assert result["budget_tier"] == "indie"


# ── CastingDirector ───────────────────────────────────────────────────────────

class TestCastingDirectorNode:
    @pytest.mark.asyncio
    async def test_skips_when_not_in_active_agents(self, sample_state):
        from backend.agents.casting_director import casting_director_node
        sample_state["active_agents"] = ["script", "budget"]
        result = await casting_director_node(sample_state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_suppresses_tools_on_followup(self, sample_state_followup):
        from backend.agents.casting_director import casting_director_node
        sample_state_followup["active_agents"] = ["casting"]

        captured_tools = []
        async def _mock_loop(**kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            return ("Casting notes", [])

        with patch("backend.agents.casting_director.run_agent_loop",
                   side_effect=_mock_loop):
            await casting_director_node(sample_state_followup)

        assert captured_tools == []

    @pytest.mark.asyncio
    async def test_extracts_suggestions_from_tool_results(self, sample_state):
        from backend.agents.casting_director import casting_director_node
        sample_state["active_agents"] = ["casting"]
        sample_state["session_history"] = []

        tool_results = [
            {
                "tool": "find_actor_suggestions",
                "args": {},
                "result": {
                    "suggestions": [
                        {"actor": "Actor A", "fit_score": 0.9},
                        {"actor": "Actor B", "fit_score": 0.8},
                    ]
                },
            }
        ]

        async def _mock_loop(**kwargs):
            return ("Casting notes text", tool_results)

        with patch("backend.agents.casting_director.run_agent_loop",
                   side_effect=_mock_loop):
            result = await casting_director_node(sample_state)

        assert len(result["casting_suggestions"]) == 2
        assert result["casting_notes"] == "Casting notes text"


# ── MarketIntel ───────────────────────────────────────────────────────────────

class TestMarketIntelNode:
    @pytest.mark.asyncio
    async def test_skips_when_not_in_active_agents(self, sample_state):
        from backend.agents.market_intel import market_intel_node
        sample_state["active_agents"] = ["script", "budget"]
        result = await market_intel_node(sample_state)
        assert result == {}

    @pytest.mark.asyncio
    async def test_suppresses_tools_on_followup(self, sample_state_followup):
        from backend.agents.market_intel import market_intel_node
        sample_state_followup["active_agents"] = ["market"]

        captured_tools = []
        async def _mock_loop(**kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            return ("Market analysis", [])

        with patch("backend.agents.market_intel.run_agent_loop",
                   side_effect=_mock_loop):
            await market_intel_node(sample_state_followup)

        assert captured_tools == []

    @pytest.mark.asyncio
    async def test_distribution_rec_theatrical(self, sample_state):
        from backend.agents.market_intel import market_intel_node
        sample_state["active_agents"] = ["market"]
        sample_state["session_history"] = []

        async def _mock_loop(**kwargs):
            return ("Recommended theatrical release for wide opening.", [])

        with patch("backend.agents.market_intel.run_agent_loop",
                   side_effect=_mock_loop):
            result = await market_intel_node(sample_state)

        assert result["distribution_recommendation"] == "theatrical"

    @pytest.mark.asyncio
    async def test_distribution_rec_streaming_default(self, sample_state):
        from backend.agents.market_intel import market_intel_node
        sample_state["active_agents"] = ["market"]
        sample_state["session_history"] = []

        async def _mock_loop(**kwargs):
            return ("Analysis complete with platform recommendations.", [])

        with patch("backend.agents.market_intel.run_agent_loop",
                   side_effect=_mock_loop):
            result = await market_intel_node(sample_state)

        assert result["distribution_recommendation"] == "streaming"

    @pytest.mark.asyncio
    async def test_distribution_rec_hybrid(self, sample_state):
        from backend.agents.market_intel import market_intel_node
        sample_state["active_agents"] = ["market"]
        sample_state["session_history"] = []

        async def _mock_loop(**kwargs):
            return ("A hybrid release strategy is recommended.", [])

        with patch("backend.agents.market_intel.run_agent_loop",
                   side_effect=_mock_loop):
            result = await market_intel_node(sample_state)

        assert result["distribution_recommendation"] == "hybrid"
