from typing import TypedDict, List

class GraphState(TypedDict):
    """
    Represents the state of a graph

    question: str
        The question being asked
    answer: str
        The answer to the question
    web_response: bool
        Whether the answer has to be generated from a web response or not
    retrieved_docs: List[str]
        The documents retrieved from the vector store

    """

    question: str
    answer: str
    web_response: bool
    retrieved_docs: List[str]
