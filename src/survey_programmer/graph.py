"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import Client

from survey_programmer.configuration import Configuration
from survey_programmer.state import InputState, State
from survey_programmer.tools import TOOLS
from util.chat_util import load_chat_model
import asyncio
import os
import json
import requests
# Define the function that calls the model

TOOL_NODE = ToolNode(TOOLS)          # build once so it isn’t re-created each call

async def load_survey_doc(state: State) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    configuration = Configuration.from_context()

    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = configuration.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )
    print(response)

    # Capture survey text from the last tool call, if any
    updates: Dict[str, str | None] = {}
    if state.messages and isinstance(state.messages[-1], ToolMessage):
        updates["survey_text"] = str(state.messages[-1].content)

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # # Return the model's response as a list to be added to existing messages
    # return {"messages": [response]}
    # Return the model's response and any captured updates
    updates["messages"] = [response]
    return updates

async def generate_surveyjs_json(state: State) -> Dict[str, Any]:
    """
    Turn `state.survey_text` into SurveyJS-compatible JSON (or any format the
    LangSmith prompt returns) while avoiding blocking calls inside the event
    loop.

    • Uses the *synchronous* LangSmith Client for stability.
    • Wraps the network-bound `pull_prompt(..)` + prompt execution in
      `asyncio.to_thread()` so the coroutine yields control immediately.
    • Requires LANGSMITH_API_KEY in your env.
    """
    if not state.survey_text:
        return {}

    def _run_prompt():
        """Runs in a background thread; safe to block."""
        client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        prompt_chain = client.pull_prompt(
            "doc_to_survey_json",     # adjust if your slug differs
            include_model=True
        )
        return prompt_chain.invoke({"survey_text": state.survey_text})

    # off-load the blocking work; returns to event loop while waiting
    survey_json = await asyncio.to_thread(_run_prompt)

    return {
        "messages": [AIMessage(content=[survey_json])],
        "survey_json": survey_json,        # ⬅️ stash in state
    }

async def tools_with_capture(state: State) -> Dict[str, Any]:
    """Run tools and copy the ToolMessage into survey_text (async-safe)."""
    out = await TOOL_NODE.ainvoke(state)        # ← note the **await**
    last = out["messages"][-1]
    if isinstance(last, ToolMessage):
        out["survey_text"] = str(last.content)
    return out

def route_model_output(state: State) -> Literal["__end__", "fetch_survey_doc"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "fetch_survey_doc").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "fetch_survey_doc"

async def post_fc_survey(state: State) -> Dict[str, Any]:
    """
    Push the newly-minted SurveyJS JSON to Fuel Cycle and log the response.
    Runs the blocking requests call in a background thread so the event loop
    stays happy.
    """
    if not state.survey_json:
        # nothing to post – exit quietly
        return {}

    def _call_api() -> str:
        url = "https://api.fuelcyclestage.com/v1/core/fcsurveys"
        payload = json.dumps({
            "payload": {"modId": "5139", "surveyJsJson": state.survey_json}
        })
        headers = {
            "key":   '24',      # <-– keep secrets out of code
            "token": 'US-JUDMSupqxQuAwoFu8PrA6xYxdEVWPLFg',
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, data=payload, timeout=30)
        resp.raise_for_status()   # propagate problems to the graph
        return resp.text

    fc_response = await asyncio.to_thread(_call_api)

    return {
        "messages": [AIMessage(content=f"Fuel Cycle response:\n{fc_response}")],
        "fc_response": fc_response,       # (optional) save it
    }

# Graph Definition

builder = StateGraph(State, input=InputState, config_schema=Configuration)
# Define the two nodes we will cycle between
builder.add_node("orchestrate", load_survey_doc)
builder.add_node("fetch_survey_doc", tools_with_capture)                # tool step
builder.add_node("generate_surveyjs_json", generate_surveyjs_json)
builder.add_node("post_fc_survey", post_fc_survey)

# Set the entrypoint as `orchestrate`
# This means that this node is the first one called
builder.add_edge("__start__", "orchestrate")



# Add a conditional edge to determine the next step after `load_survey_doc`
builder.add_conditional_edges(
    "orchestrate",
    # After load_survey_doc finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `load_survey_doc`
# This creates a cycle: after using tools, we always return to the model
# builder.add_edge("tools", "load_survey_doc")
builder.add_edge("fetch_survey_doc", "generate_surveyjs_json")
builder.add_edge("generate_surveyjs_json", "post_fc_survey")

builder.add_edge("post_fc_survey", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="survey_agent28")
