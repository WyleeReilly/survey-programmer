
from survey_designerV2.state import State
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, HumanMessage
from typing import Any, Tuple, Union, List, Optional, Literal, Dict
from langsmith.async_client import AsyncClient
import os
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from survey_designerV2.configuration import Configuration
import json, ast, re
from langchain.schema import AIMessage, BaseMessage
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.types import interrupt

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

_client = AsyncClient(api_key=os.getenv("LANGSMITH_API_KEY"))
JSONLike = dict[str, Any] | list[Any] | str | int | float | bool | None


def _latest_human(state: State) -> str | None:
    """
    Returns the most-recent human message’s text, or None if none found.
    """
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return None


def _coerce_json(text: str) -> Union[dict, None]:
    """
    Try very hard to convert *text* into a Python dict.

    Returns a dict on success, or None if parsing fails.
    """
    try:
        return json.loads(text)                     # strict JSON
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(text)               # {'single': 'quotes'}
    except (ValueError, SyntaxError):
        pass

    # clip out the first {...} block and retry (handles extra prose)
    m = re.search(r'{.*}', text, re.S)
    if m:
        return _coerce_json(m.group())

    return None


def _expects_structured(chain) -> bool:
    """
    Heuristic: does this runnable have a non-string output parser?
    Works for most LangSmith / LangChain structured-prompt patterns.
    """
    parser = getattr(chain, "output_parser", None)
    return parser is not None and not isinstance(parser, StrOutputParser)


async def _run_chain(prompt: str, **vars) -> Tuple[Union[Dict[str, Any], str], List[AIMessage]]:
    """
    Invoke a LangSmith prompt and normalise the result.

    Returns:
        (dict | str,  [AIMessage])
        • dict → structured JSON-like output
        • str  → plain-text output
    """
    chain = await _client.pull_prompt(prompt, include_model=True)
    structured_expected = _expects_structured(chain)

    raw = await chain.ainvoke(vars)                 # Could be dict | str | Message
    if isinstance(raw, BaseMessage):
        content  = raw.content or ""
        messages = [raw] if isinstance(raw, AIMessage) else [AIMessage(content=content)]
    else:
        content  = str(raw)
        messages = [AIMessage(content=content)]

    if isinstance(raw, dict):                       # Already structured
        return raw, messages

    if structured_expected or content.lstrip().startswith(("{", "[")):
        maybe = _coerce_json(content)
        if maybe is not None:
            return maybe, messages

    return content.strip(), messages


async def _reflection_messages(survey: str,
                               revision_request: str = "") -> List[AIMessage]:
    text, streamed = await _run_chain(
        "revision_reflection",
        survey=survey,
        revision_request=revision_request,
    )

    # If _run_chain already gave us an AIMessage, just return it.
    if streamed:
        return streamed

    # Fallback (e.g., _run_chain returned a plain string)
    return [AIMessage(content=text)]

def _update_state(state: State, **changes) -> State:
    """Immutable-style state replace (right-hand keys overwrite)."""
    from copy import deepcopy
    merged = {**deepcopy(state.__dict__), **changes}
    return State(**merged)


async def guardrail_router(state: State
                  ) -> Literal["initial_design",
                               END]:

    if state.guardrail_decision == "advance":
        return "initial_design"
    else:
        return END


async def design_router(state: State, config: RunnableConfig, feedback) -> Literal["revision",
                               END]:

    if state.design_router_decision == "revise":
        return "revision"
    else:
        return END





    # if guardrail_check['direction'] == 'advance':
        
    #     return _update_state(
    #         state,
    #         guardrail_decision = guardrail_check['direction']
    #     )
    # else:
    #         return _update_state(
    #             state,
    #             messages=state.messages + [guardrail_check['response']],
    #             guardrail_decision = guardrail_check['direction']
            # )

# ────────────────────────────────────────────────────────────────────
# Nodes
# ────────────────────────────────────────────────────────────────────


async def initial_guardrail_node(state: State, config: RunnableConfig):
    configuration = Configuration.from_context()

    initial_message = _latest_human(state)

    guardrail_check, _ = await _run_chain(
        configuration.initial_design_guardrail_prompt,
        human_message=initial_message,
    )

    if guardrail_check['direction'] == 'advance':
        
        return _update_state(
            state,
            guardrail_decision = guardrail_check['direction']
        )
    else:
            return _update_state(
                state,
                messages=state.messages + [guardrail_check['response']],
                guardrail_decision = guardrail_check['direction']
            )


async def initial_design_node(state: State, config: RunnableConfig) -> State:
    objective = _latest_human(state)
    if not objective:
        return {}

    configuration = Configuration.from_context()

    # 1) write the first draft
    initial_survey, _ = await _run_chain(
        configuration.initial_design_prompt,
        objective=objective,
    )

    # 2) create a reflection **about** that draft
    reflection_msgs = await _reflection_messages(initial_survey)

    return _update_state(
        state,
        survey_json=initial_survey,
        messages=state.messages + reflection_msgs,
    )

async def await_human_feedback_node(state: State, config: RunnableConfig) -> State:

    feedback = interrupt("What next?")
    feedback = next(iter(feedback.values())) if isinstance(feedback, dict) else feedback

    configuration = Configuration.from_context()

    route, _ = await _run_chain(
        configuration.revision_router_prompt,
        human_message=feedback,
    )

    return _update_state(
            state,
            messages=state.messages + [HumanMessage(content=feedback)],
            design_router_decision = route
        )


async def revision_reviser_node(state: State, config: RunnableConfig) -> State:
    human_message = _latest_human(state)

    configuration = Configuration.from_context()

    # 1) write the first draft
    revision, _ = await _run_chain(
        configuration.revision_reviser_prompt,
        human_message=human_message,
        current_survey = state.survey_json
    )

    # 2) create a reflection **about** that draft
    reflection_msgs = await _reflection_messages(human_message)

    return _update_state(
        state,
        survey_json=revision,
        messages=state.messages + reflection_msgs,
    )


# ────────────────────────────────────────────────────────────────────
# Graph wiring
# ────────────────────────────────────────────────────────────────────
graph = StateGraph(State, config_schema=Configuration)

graph.add_node("initial_guardrail", initial_guardrail_node)
graph.add_node("initial_design",   initial_design_node)
graph.add_node("await_human_feedback",   await_human_feedback_node)
graph.add_node("revision",  revision_reviser_node)


graph.add_edge(START, "initial_guardrail")

graph.add_conditional_edges("initial_guardrail", guardrail_router)
graph.add_edge("initial_design", "await_human_feedback")
graph.add_edge("revision", "await_human_feedback")

graph.add_conditional_edges("await_human_feedback", design_router)

graph.add_edge("await_human_feedback", END)
graph.add_edge("initial_guardrail", END)


compiled_graph = graph.compile(name="survey_designer44")