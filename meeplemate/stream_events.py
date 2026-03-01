from typing import Literal, Sequence, TypedDict


class StepEvent(TypedDict):
    type: Literal["mm_step"]
    description: str


class RefinedUserQueryEvent(TypedDict):
    type: Literal["mm_refined_user_query"]
    description: str


class AnalyzedUserQueryEvent(TypedDict):
    type: Literal["mm_user_query_analysis"]
    analysis: str
    classification: str


class RetrievingDataEvent(TypedDict):
    type: Literal["mm_retrieving_data"]
    search_terms: Sequence[str]


class SubquestionAnsweredEvent(TypedDict):
    type: Literal["mm_subquestion_answered"]
    question: str
    answer: str
    valid: bool


class UserQueryAnsweredEvent(TypedDict):
    type: Literal["mm_user_query_answered"]
    query: str
    answer: str