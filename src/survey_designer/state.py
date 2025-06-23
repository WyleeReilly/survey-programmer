from typing import TypedDict, List

class QASchema(TypedDict):
    messages:   List[dict]      # full chat transcript
    question:   str | None      # business question
    survey:     str | None      # current survey document
    need_clar:  bool | None     # flag set by classifier
    feedback:   str | None      # raw user feedback
