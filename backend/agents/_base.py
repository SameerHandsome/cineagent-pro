"""
backend/agents/_base.py
────────────────────────
Shared agentic loop used by every specialist agent.

PER-AGENT MODEL ROUTING
────────────────────────
Previously every agent used the same model (llama-4-scout-17b-16e-instruct),
causing all four agents to fire simultaneously against a shared 30k TPM cap
and reliably hitting 429 rate-limit errors on complex queries.

Solution: route each agent to the most cost-effective model for its task.
The model is selected via the `model_name` parameter passed to run_agent_loop.
Each agent file passes its own model constant (defined at the top of each file)
so the routing is visible at the call site, not buried in _base.py.

Recommended mapping (adjust to your Groq tier):
  orchestrator   → llama-3.1-8b-instant        (classifier, tiny output)
  script_analyst → llama-3.3-70b-versatile     (nuanced extraction)
  budget_planner → llama-3.1-8b-instant        (deterministic math)
  casting_dir.   → llama-3.1-8b-instant        (DB lookup + brief notes)
  market_intel   → llama-3.3-70b-versatile     (trend analysis)
  synthesizer    → llama-3.3-70b-versatile     (final report quality)

TOKEN BUDGET GUIDANCE
──────────────────────
Groq free tier: 30k TPM, 14.4k RPM.  A full 4-agent run fires ~14 LLM calls
(orchestrator + 4 agents × ~3 tool iterations each + synthesizer).
Keep max_tokens tight per agent to avoid 429s:
  orchestrator   → 200
  script_analyst → 800
  budget_planner → 900
  casting_dir    → 700
  market_intel   → 900
  synthesizer    → 1800

TYPE COERCION NOTE:
  Groq's llama models occasionally emit integer/float tool parameters as JSON
  strings (e.g. "2" instead of 2).  _coerce_args() fixes these before the
  call reaches Groq's validator, preventing 400 tool_call_validation_failed.

MALFORMED TOOL CALL RECOVERY (FIX):
  On conversational follow-up messages (no film concept, no data to extract),
  Groq's models sometimes generate a tool call as a plain string literal
  e.g. '[tavily_search("non-linear vs linear script structure")]' instead of
  a proper JSON tool-use object. Groq rejects this with:
    400 - Failed to call a function... failed_generation: '[tavily_search(...)]'

  Recovery: when a 400 is received during ainvoke(), run_agent_loop retries
  the same call on an LLM instance WITHOUT tools bound. This always succeeds
  because the model produces plain text when it has no tool schema to call
  into. The plain text response is then returned as the final content.

  This is intentionally a silent recovery — the agent returns a text answer
  rather than crashing the entire pipeline with a 500 to the user.
"""
import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langsmith import traceable

from backend.config import settings

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 4    # was 10 — primary 429 cause; 4 is enough for all agents

# ── Model constants ────────────────────────────────────────────────────────────
# TWO SEPARATE TPM BUCKETS on Groq free tier:
#   MODEL_FAST  → Scout  30K TPM  — orchestrator, budget_planner, casting_director
#   MODEL_SMART → 70b    12K TPM  — script_analyst, market_intel, synthesizer
MODEL_FAST  = "meta-llama/llama-4-scout-17b-16e-instruct"  # 30K TPM
MODEL_SMART = "llama-3.3-70b-versatile"                    # 12K TPM


def _build_llm(
    temperature: float = 0.2,
    max_tokens: int = 830,
    model_name: str = MODEL_FAST,
) -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _coerce_args(tool: BaseTool, args: dict) -> dict:
    """
    Coerce tool arguments to match declared JSON schema types.
    Groq sometimes emits numeric params as strings ("2" instead of 2).
    """
    try:
        schema = tool.args_schema.schema() if tool.args_schema else {}
        properties = schema.get("properties", {})
    except Exception:
        return args

    coerced = {}
    for key, value in args.items():
        prop = properties.get(key, {})
        declared_type = prop.get("type", "")

        if declared_type == "integer":
            try:
                coerced[key] = int(value)
            except (ValueError, TypeError):
                coerced[key] = value

        elif declared_type == "number":
            try:
                coerced[key] = float(value)
            except (ValueError, TypeError):
                coerced[key] = value

        elif declared_type == "boolean":
            if isinstance(value, str):
                coerced[key] = value.lower() in ("true", "1", "yes")
            else:
                coerced[key] = value
        else:
            coerced[key] = value

    return coerced


def _is_malformed_tool_call_error(exc: Exception) -> bool:
    """
    Return True if the exception is a Groq 400 caused by a malformed tool call.

    Groq returns 400 with code='tool_use_failed' when the model generates a
    tool invocation as a plain string literal instead of a JSON tool-use object.
    This happens reliably on conversational follow-up questions where the model
    has no meaningful data to extract but still tries to call a search tool.

    We detect it by checking the exception string for the specific Groq error
    code and the 'failed_generation' field that contains the mangled call.
    """
    msg = str(exc)
    return (
        "400" in msg and (
            "tool_use_failed" in msg or
            "Failed to call a function" in msg or
            "failed_generation" in msg
        )
    )


@traceable(name="agent_loop")
async def run_agent_loop(
    *,
    agent_name: str,
    prompt_template: ChatPromptTemplate,
    prompt_vars: dict,
    user_query: str,
    tools: list[BaseTool],
    temperature: float = 0.2,
    max_tokens: int = 900,
    model_name: str = MODEL_FAST,
) -> tuple[str, list[dict]]:
    """
    Generic agentic loop.

    Returns:
        final_content (str)  — last AIMessage.content
        tool_results  (list) — [{tool, args, result}] for state extraction

    On Groq 400 malformed-tool-call errors, automatically retries without
    tools bound so the model produces plain text instead of crashing.
    """
    llm_with_tools = _build_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        model_name=model_name,
    )
    # Keep a no-tools version ready for malformed-tool-call recovery
    llm_plain = _build_llm(
        temperature=temperature,
        max_tokens=max_tokens,
        model_name=model_name,
    )

    tool_map: dict[str, BaseTool] = {t.name: t for t in tools}
    if tools:
        llm_with_tools = llm_with_tools.bind_tools(tools)

    scratchpad: list[BaseMessage] = []
    tool_results: list[dict] = []
    response: AIMessage | None = None

    for iteration in range(MAX_TOOL_ITERATIONS):
        messages = prompt_template.format_messages(
            **prompt_vars,
            agent_scratchpad=scratchpad,
            user_query=user_query,
        )

        # ── FIX: Catch 400 malformed tool call errors ──────────────────────────
        # When Groq's model generates a tool call as a plain string instead of
        # a JSON tool-use object (happens on conversational follow-ups), the
        # API returns 400 with code='tool_use_failed'. We catch that here and
        # retry WITHOUT tools so the model produces plain text. This silently
        # recovers the pipeline instead of surfacing a 500 to the user.
        try:
            response = await llm_with_tools.ainvoke(messages)
        except Exception as exc:
            if _is_malformed_tool_call_error(exc):
                logger.warning(
                    f"[{agent_name}] Groq 400 malformed tool call on iteration {iteration}. "
                    f"Retrying without tools (plain text fallback). Error: {exc}"
                )
                try:
                    response = await llm_plain.ainvoke(messages)
                    # Plain text response — no tool calls to process
                    return response.content or "", tool_results
                except Exception as fallback_exc:
                    logger.error(f"[{agent_name}] Plain text fallback also failed: {fallback_exc}")
                    return "", tool_results
            else:
                # Not a tool-call error — re-raise so genuine bugs surface
                raise

        if not response.tool_calls:
            return response.content or "", tool_results

        scratchpad.append(response)

        for tc in response.tool_calls:
            tool_name = tc["name"]
            raw_args  = tc.get("args", {})
            tool      = tool_map.get(tool_name)

            if not tool:
                result_str = f"Tool '{tool_name}' not available — skipping."
                logger.warning(f"[{agent_name}] {result_str}")
            else:
                tool_args = _coerce_args(tool, raw_args)
                if tool_args != raw_args:
                    logger.debug(f"[{agent_name}] {tool_name}: coerced {raw_args} → {tool_args}")

                try:
                    result = await tool.ainvoke(tool_args)
                    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                    tool_results.append({"tool": tool_name, "args": tool_args, "result": result})
                    logger.debug(f"[{agent_name}] {tool_name} → {result_str[:120]}...")
                except Exception as e:
                    result_str = f"Tool error: {e}"
                    logger.error(f"[{agent_name}] {tool_name} failed: {e}")

            scratchpad.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    logger.warning(f"[{agent_name}] Hit max iterations ({MAX_TOOL_ITERATIONS})")
    return (response.content if response is not None else "") or "", tool_results
