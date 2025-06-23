from langgraph.graph import StateGraph
from typing import TypedDict, List

class QASchema(TypedDict):
    messages:   List[dict]      # full chat transcript
    question:   str | None      # business question
    survey:     str | None      # current survey document
    need_clar:  bool | None     # flag set by classifier
    feedback:   str | None      # raw user feedback



graph = StateGraph()


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

my_graph = graph.compile(name="survey_designer1")


