"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated
from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from survey_programmer import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    objective_clarifier_prompt: str = field(
        default="survey_designer_objective_clarifier",
        metadata={
            "description": "The system prompt used to clarify research objectives with the human."
        },
    )

    objective_router_prompt: str = field(
        default="survey_designer_objective_router",
        metadata={
            "description": "The system prompt used to understand whether or not to advance."
        },
    )
    objective_reviser_prompt: str = field(
        default="survey_designer_objective_reviser",
        metadata={
            "description": "The system prompt used to understand whether or not to advance."
        },
    )


    initial_design_prompt: str = field(
        default="survey_designer_initial_design",
        metadata={
            "description": "The system prompt used for the agents initial survey design."
            "This prompt sets the context and behavior for the agent."
        },
    )

    initial_design_guardrail_prompt: str = field(
        default="survey_designer_initial_guardrail",
        metadata={
            "description": "The system prompt used to initially assess the users first query."
            "This prompt decides whether the user can 'enter' the graph."
        },
    )

    revision_router_prompt: str = field(
        default="survey_designer_revision_router",
        metadata={
            "description": "The system prompt used to route the users feedback."
        },
    )

    revision_reviser_prompt: str = field(
        default="survey_designer_revision_reviser",
        metadata={
            "description": "The system prompt used to revise the survey using the users request."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
