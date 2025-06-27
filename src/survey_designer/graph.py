"""
survey_designer/graph.py
Iterative survey-designer graph (v3) – node name clash fixed
"""

from __future__ import annotations

import os, asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Literal

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

def _latest_human_if_last(state: DesignerState) -> Optional[str]:
    last = state.messages[-1] if state.messages else None
    return last.content if isinstance(last, HumanMessage) else None

async def _stream_chain(prompt: str, **vars) -> tuple[str, List[AIMessage]]:
    """Stream a LangSmith prompt and collect the full text + streamed chunks."""
    chain = await _client.pull_prompt(prompt, include_model=True)
    chain = chain.with_config({"stream": True})

    tokens, streamed = [], []
    async for chunk in chain.astream(vars):
        if chunk.content:
            tokens.append(chunk.content)
        streamed.append(chunk)
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

def _router(state: DesignerState) -> Literal["wait", "revision", "finalization"]:
    if not state.feedback or not state.feedback.strip():
        return "wait"          # just a label; no side-effects here
    if state.approved:
        return "finalization"
    return "revision"
# ────────────────────────────────────────────────────────────────────
# 3.  Nodes
# ────────────────────────────────────────────────────────────────────
async def initial_design_node(state: DesignerState) -> DesignerState:
    objective = _latest_human_if_last(state)
    if not objective:
        return {}

    draft, streamed = await _stream_chain("survey_writer",
                                          research_objective=objective)

    return _update_state(
        state,
        survey_text=draft,          # ← sole source of truth
        feedback=None,
        approved=False,
        messages=state.messages + streamed + [AIMessage(content=draft)],
    )


async def capture_feedback_node(state: DesignerState) -> DesignerState:
    fb_txt = _latest_human_if_last(state)

    # merely record feedback (may be None/blank)
    approved = False
    if fb_txt and fb_txt.strip():
        approved = any(k in fb_txt.lower() for k in ("#approve", "looks good", "ship it"))

    # no interrupt here – we let the router decide
    return _update_state(state, feedback=fb_txt, approved=approved)


async def revision_node(state: DesignerState) -> DesignerState:
    if state.approved:
        return {}

    # Fallback: if survey_text somehow disappeared, pull it from transcript
    current = state.survey_text or _last_ai_text(state.messages)

    new_draft, streamed = await _stream_chain(
        "survey_reviser",
        current_survey=current,
        revision_request=state.feedback or "",
    )

    return _update_state(
        state,
        survey_text=new_draft,                   # overwrite with revision
        messages=state.messages + streamed + [AIMessage(content=new_draft)],
    )

async def finalization_node(state: DesignerState) -> DesignerState:
    """Terminal node — no DB write yet."""
    return _update_state(
        state,
        messages=state.messages
        + [AIMessage(content="Thank you for your attention to this matter")],
    )

# ────────────────────────────────────────────────────────────────────
# 4.  Graph wiring
# ────────────────────────────────────────────────────────────────────
graph = StateGraph(DesignerState)

graph.add_node("initial_design",   initial_design_node)
graph.add_node("capture_feedback", capture_feedback_node)
graph.add_node("revision",         revision_node)
graph.add_node("finalization",     finalization_node)
graph.add_node("wait", _wait_node)
graph.add_edge(START, "initial_design")

graph.add_edge("initial_design", "capture_feedback")
graph.add_edge("wait", "capture_feedback")

graph.add_conditional_edges(
    "capture_feedback",
    _router,
    {                       # ← mapping table
        "wait": "wait",     # NEW  ← tells the engine where to go
        "revision": "revision",
        "finalization": "finalization",
    },
)
graph.add_edge("revision", "capture_feedback")   # loop
graph.add_edge("finalization", END)

compiled_graph = graph.compile(
    name="survey_designer13",
    # checkpointer=SqliteSaver("survey_graph.db")  # add if you want durable interrupts
)
