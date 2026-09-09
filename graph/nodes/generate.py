from typing import Dict, Any
from graph.chains.generations import chain
from graph.state import GraphState


def generate_answer(state: GraphState) -> Dict[str, Any]:
    """
    Generates an answer to the question using the retrieved documents.

    Args:
        state (GraphState): The current state of the graph.
    """
    print("Generate")
    question = state["question"]
    documents = state["retrieved_docs"]

    generated_answer = chain.invoke(
        {"question": question, "context": "\n\n".join(documents)}
    )
    return {"answer": generated_answer}
