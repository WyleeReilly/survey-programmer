"""
survey_designer/graph.py
Iterative survey-designer graph (v3) – node name clash fixed
"""

from __future__ import annotations

import os, asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import json
from langsmith.async_client import AsyncClient
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langgraph.types import interrupt
from langgraph.graph import StateGraph, START, END

# ────────────────────────────────────────────────────────────────────
# 1.  State  (extends your existing BaseState)
# ────────────────────────────────────────────────────────────────────
from survey_designer.state import State as BaseState   # ← has .survey_text already

@dataclass
class DesignerState(BaseState):
    """Adds feedback + approval flag; survey_text inherited from BaseState."""
    feedback: str | None = field(default=None)
    approved: bool = field(default=False)

# ────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ────────────────────────────────────────────────────────────────────
_client = AsyncClient(api_key=os.getenv("LANGSMITH_API_KEY"))

def _latest_human(state: DesignerState) -> str | None:
    """
    Returns the most-recent human message’s text, or None if none found.
    """
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return None

async def _run_chain(prompt: str, **vars) -> tuple[str, List[AIMessage]]:
    """
    Pull the LangSmith prompt and invoke it once (no streaming).
    Returns (full_text, [AIMessage]).
    """
    chain = await _client.pull_prompt(prompt, include_model=True)   # just await
    result = await chain.ainvoke(vars)                              # one-shot

    # If the prompt returns an AIMessage already, great.
    if isinstance(result, AIMessage):
        content = result.content or ""
        return content.strip(), [result]

    # Otherwise assume it's a plain string
    content = str(result).strip()
    return content, [AIMessage(content=content)]

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


def _latest_human(state: DesignerState) -> Optional[str]:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and msg.content:
            return msg.content
    return None       

async def _stream_chain(prompt: str, **vars) -> tuple[str, List[AIMessage]]:
    chain = await _client.pull_prompt(prompt, include_model=True).with_config({"stream": True})

    tokens, streamed = [], []
    async for chunk in chain.astream(vars):
        if chunk.content:                     # ← guard
            tokens.append(chunk.content)
            streamed.append(chunk)            # ← moved inside the guard
    return "".join(tokens).strip(), streamed

def _update_state(state: DesignerState, **changes) -> DesignerState:
    """Immutable-style state replace (right-hand keys overwrite)."""
    from copy import deepcopy                          # deep-copy optional
    merged = {**deepcopy(state.__dict__), **changes}    # overwrite keys  :contentReference[oaicite:1]{index=1}
    return DesignerState(**merged)

def _last_ai_text(messages: List[AnyMessage]) -> str:
    """Find the last AI message with non-empty content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            return msg.content
    return ""

def _wait_node(state: DesignerState) -> DesignerState:
    interrupt("Awaiting user feedback on the survey draft")
    return state               

async def _router(state: DesignerState
                  ) -> Literal["wait", "revision",
                               "finalization", "off_topic"]:

    decision = await _llm_route_decision(state.feedback or "")

    match decision:
        case "advance":
            # we’ll treat “advance” as “approved” ⟶ finalisation
            return "finalization"
        case "revise":
            return "revision"
        case "off-topic":
            return "off_topic"
        case _:
            # default = no feedback
            return "wait"


async def _llm_route_decision(feedback: str) -> str:
    """
    Ask the LangSmith prompt you showed in the screenshots
    (‘survey_design_router’) to classify the user’s message.

    It must return one of: advance · revise · off-topic.
    """
    if not feedback.strip():          # nothing to classify → pause
        return "wait"

    chain = await _client.pull_prompt("survey_design_router", include_model=True)
    result = await chain.ainvoke({"human_message": feedback})

    # • If the prompt is set up for structured-JSON output (recommended) the
    #   chain already returns a Python dict.  
    # • If not, we fall back to parsing the raw string.
    if isinstance(result, dict):
        return result["direction"]

    try:
        return json.loads(result.content)["direction"]
    except Exception:
        # Guard-rail: if something strange comes back, play it safe.
        return "wait"

# ────────────────────────────────────────────────────────────────────
# 3.  Nodes
# ────────────────────────────────────────────────────────────────────
async def initial_design_node(state: DesignerState) -> DesignerState:
    objective = _latest_human(state)
    if not objective:
        return {}

    # 1) write the first draft
    draft, _ = await _run_chain(
        "survey_writer",
        research_objective=objective,
    )

    # 2) create a reflection **about** that draft
    reflection_msgs = await _reflection_messages(draft)

    return _update_state(
        state,
        survey_text=draft,                 # ← authoritative survey text
        feedback=None,
        approved=False,
        messages=state.messages + reflection_msgs,
    )



async def capture_feedback_node(state: DesignerState) -> DesignerState:
    fb_txt = _latest_human(state)

    # merely record feedback (may be None/blank)
    approved = False
    if fb_txt and fb_txt.strip():
        approved = any(k in fb_txt.lower() for k in ("#approve", "looks good", "ship it"))

    # no interrupt here – we let the router decide
    return _update_state(state, feedback=fb_txt, approved=approved)


# 2️⃣  treat “AI was the last speaker” as “we’re waiting”
async def wait_for_message_node(state: DesignerState) -> DesignerState:
    last = state.messages[-1] if state.messages else None

    # If the transcript ends with an AI message, pause here.
    if not isinstance(last, HumanMessage):
        interrupt("Awaiting user feedback on the survey draft")
        return state

    # We *do* have a new human message → store it
    fb_txt = last.content.strip()
    approved = any(k in fb_txt.lower() for k in ("#approve", "looks good", "ship it"))

    return _update_state(state, feedback=fb_txt, approved=approved)


async def revision_node(state: DesignerState) -> DesignerState:
    if state.approved:
        return {}

    # Fallback if survey_text vanished
    current = state.survey_text or _last_ai_text(state.messages)

    # 1) rewrite the survey
    new_draft, _ = await _run_chain(
        "survey_reviser",
        current_survey=current,
        revision_request=state.feedback or "",
    )

    # 2) generate reflection on that revision
    reflection_msgs = await _reflection_messages(
        new_draft,
        revision_request=state.feedback or "",
    )

    return _update_state(
        state,
        survey_text=new_draft,             # keep state in sync
        feedback=None,
        messages=state.messages + reflection_msgs,
    )

async def finalization_node(state: DesignerState) -> DesignerState:
    """Terminal node — no DB write yet."""
    return _update_state(
        state,
        messages=state.messages
        + [AIMessage(content="Thank you for your attention to this matter")],
    )

async def off_topic_node(state: DesignerState) -> DesignerState:
    apology = AIMessage(
        content="Sorry, I can only help with writing surveys.")
    return _update_state(
        state,
        messages=state.messages + [apology]
    )

# ────────────────────────────────────────────────────────────────────
# 4.  Graph wiring
# ────────────────────────────────────────────────────────────────────
graph = StateGraph(DesignerState)

graph.add_node("initial_design",   initial_design_node)
graph.add_node("wait_for_message", wait_for_message_node)   # renamed
graph.add_node("revision",         revision_node)
graph.add_node("finalization",     finalization_node)
graph.add_node("off_topic",        off_topic_node)          # <- from last patch

graph.add_edge(START, "initial_design")
graph.add_edge("initial_design", "wait_for_message")

# After every revision we loop back to wait_for_message
graph.add_edge("revision", "wait_for_message")
graph.add_edge("off_topic", "wait_for_message")

# Conditional fork now starts from wait_for_message
graph.add_conditional_edges(
    "wait_for_message",
    _router,                                # unchanged async router
    {
        "wait":         "wait_for_message", # self-loop when router says “wait”
        "revision":     "revision",
        "finalization": "finalization",
        "off_topic":    "off_topic",
    },
)

graph.add_edge("finalization", END)

compiled_graph = graph.compile(
    name="survey_designer22",
    # checkpointer=SqliteSaver("survey_graph.db")  # add if you want durable interrupts
)