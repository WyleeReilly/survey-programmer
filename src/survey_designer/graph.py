from langgraph.graph import StateGraph
from typing import TypedDict, List

# class QASchema(TypedDict):
#     messages:   List[dict]      # full chat transcript
#     question:   str | None      # business question
#     survey:     str | None      # current survey document
#     need_clar:  bool | None     # flag set by classifier
#     feedback:   str | None      # raw user feedback



# graph = StateGraph()


# graph.add_node("classify_intent", classify_intent)
# graph.add_node("need_clarification", need_clarification)
# graph.add_node("ask_clarification", ask_clarification)
# graph.add_node("survey_writer", survey_writer)
# graph.add_node("survey_refiner", survey_refiner)
# graph.add_node("save_and_stream", save_and_stream)

# # --- edges ---
# graph.set_entry_point("classify_intent")

# graph.add_conditional_edges(
#     "classify_intent",
#     lambda s: s["intent"],
#     {
#       "new_question": "need_clarification",
#       "survey_feedback": "survey_refiner",
#       "clarification_response": "need_clarification"
#     },
# )

# graph.add_conditional_edges(
#     "need_clarification",
#     lambda s: s["need_clar"],
#     {True: "ask_clarification", False: "survey_writer"}
# )

# graph.add_edge("ask_clarification", "classify_intent")   # loop

# # once a draft or revision is ready
# graph.add_edge("survey_writer",  "save_and_stream")
# graph.add_edge("survey_refiner", "save_and_stream")
# graph.add_edge("save_and_stream", END)

"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from survey_designer.configuration import Configuration
from survey_designer.state import InputState, State
# from survey_designer.tools import TOOLS
from util.chat_util import load_chat_model
from langgraph.graph import StateGraph, START, END
import json


# survey_writer_node.py

import os, asyncio
from langsmith import Client
from langchain_core.messages import AIMessage, HumanMessage
from langsmith.async_client import AsyncClient      # async version!

_client = AsyncClient(api_key=os.getenv("LANGSMITH_API_KEY"))



def latest_human_message(messages) -> str | None:
    # walk backward until we find a HumanMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            # content can be str or list (LC supports both)
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None

async def survey_writer_node(state: State):
    user_text = latest_human_message(state.messages)
    if not user_text:
        yield {}; return

    async_chain = await _client.pull_prompt(
        "survey_writer", include_model=True
    )
    async_chain = async_chain.with_config({"stream": True})

    # ── collect tokens ─────────────────────────
    tokens = []
    async for chunk in async_chain.astream({"research_objective": user_text}):
        if chunk.content:
            tokens.append(chunk.content)           # save token text
        yield {"messages": [chunk]}                # stream to caller

    survey_str = "".join(tokens).strip()

    # If the prompt returns JSON, parse it; otherwise keep raw text
    try:
        survey_json = json.loads(survey_str)
    except json.JSONDecodeError:
        survey_json = None                         # fall back to plain

    yield {
        "survey_json": survey_json,                # stash for next node
        "messages": [
            AIMessage(
                content=survey_str                 # <-- now a STRING
            )
        ],
    }

graph = StateGraph(State)
graph.add_node("survey_writer", survey_writer_node)
graph.add_edge(START, "survey_writer")
graph.add_edge("survey_writer", END)

compiled_graph = graph.compile(name="survey_designer10")

