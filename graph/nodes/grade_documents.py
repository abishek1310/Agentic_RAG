from typing import Dict, Any
from graph.state import GraphState
from graph.chains.retriever_grader import retriever_grader

def grade_documents(state: GraphState) -> Dict[str,Any]:
    """
    Grades the retrieved documents based on their relevance to the question.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Dict[str, Any]: A dictionary containing the binary score of the answer.
    """
    print("Grade Documents")
    question = state["question"]
    documents = state["retrieved_docs"]

    filtered_docs = []
    web_search = False 
    for doc in documents:
        score = retriever_grader.invoke({"question": question, "retrieved_docs": doc})
        grade = score.binary_score
        if grade.lower() == "yes":
            filtered_docs.append(doc)
        else:
            web_search = True
            continue
    return {"retrieved_docs": filtered_docs, "question": question , "web_response": web_search}
    