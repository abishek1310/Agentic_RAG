from typing import Any, Dict
from graph.state import GraphState
from graph.ingestion import retriever

def retrieve_docs(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves documents from the vector store based on the question in the state.

    Args:
        state (GraphState): The current state of the graph.
    """
    print("Retrieve")
    documents = retriever.invoke(state["question"])
    return {"retrieved_docs": [doc.page_content for doc in documents]}