from util.state_util import InputState
from langgraph.managed import IsLastStep
from dataclasses import dataclass, field
from typing import Any, Literal
from langgraph.graph import END


@dataclass
class State(InputState):
    """Represents the complete state of the agent, extending InputState with additional attributes.

    This class can be used to store any information needed throughout the agent's lifecycle.
    """

    is_last_step: IsLastStep = field(default=False)
    """
    Indicates whether the current step is the last one before the graph raises an error.

    This is a 'managed' variable, controlled by the state machine rather than user code.
    It is set to 'True' when the step count reaches recursion_limit - 1.
    """

    survey_json: dict[str, Any] | None = None
    """
    The JSON of the survey being designed between the user and AI
    """
    guardrail_decision: Literal["advance", END] = None
    """
    The guardrails decision to advance or not
    """
    design_router_decision: Literal["advance", "off-topic", "revise"] = None

    objective_router_decision: Literal["advance", "off-topic", "revise"] = None
    
    clarified_objectives: dict[str, Any] | None = None
    """
    The JSON of the survey being designed between the user and AI
    """