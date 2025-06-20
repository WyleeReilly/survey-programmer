"""Survey design graph using the prebuilt ReAct template from LangGraph."""

from langgraph.prebuilt import create_react_agent

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


# Load configuration and model
configuration = Configuration.from_context()
model = load_chat_model(configuration.model).bind_tools(TOOLS)

# Build the graph using the high-level factory
graph_survey_design = create_react_agent(
    model,
    tools=TOOLS,
    state_schema=State,
    input_schema=InputState,
    config_schema=Configuration,
    name="survey_design_graph",
)
