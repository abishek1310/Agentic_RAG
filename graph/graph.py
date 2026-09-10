from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.consts import (
    RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE,
    USEFUL, NOT_USEFUL, NOT_SUPPORTED, MAX_RETRIES,
)
from graph.nodes import retrieve_docs, grade_documents, web_search, generate_answer
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.router import router 

load_dotenv()

def condition_entry(state: GraphState):
    print("Route Question")
    question = state["question"]
    route = router.invoke({"question" : question})
    if route.datasource == "websearch":
        return WEBSEARCH
    else:
        return RETRIEVE

def condition_2(state : GraphState):
    print("Check Hallucination")
    question= state["question"]
    generation = state["answer"]
    documents = state["retrieved_docs"]

    if state.get("retries", 0) >= MAX_RETRIES:
        print(f"Retry limit ({MAX_RETRIES}) reached, accepting current answer")
        return USEFUL

    score = hallucination_grader.invoke(
        {"set_of_facts": "\n\n".join(documents), "llm_generation": generation}
    )

    if score.binary_score:
        print("No Hallucination")
        print("Checking whether the answer is relevant to question")
        score = answer_grader.invoke({"question":question,"generation":generation})
        if score.binary_score:
            print("Answer is relevant to the question")
            return USEFUL
        else:
            print("Answer is not relevant to the question")
            return NOT_USEFUL
    else:
        print("Hallucinated answer")
        return NOT_SUPPORTED
      


def condition(state : GraphState):
    if state['web_response']:
        return WEBSEARCH
    return GENERATE

chain = StateGraph(GraphState)
chain.add_node(RETRIEVE, retrieve_docs)
chain.add_node(GRADE_DOCUMENTS, grade_documents)
chain.add_node(WEBSEARCH, web_search)
chain.add_node(GENERATE, generate_answer)

chain.set_conditional_entry_point(condition_entry, {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE})
chain.add_edge(RETRIEVE, GRADE_DOCUMENTS)
chain.add_conditional_edges(GRADE_DOCUMENTS, condition, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
chain.add_edge(WEBSEARCH, GENERATE)
chain.add_conditional_edges(GENERATE, condition_2, {NOT_SUPPORTED: GENERATE, USEFUL: END, NOT_USEFUL: WEBSEARCH})
fin = chain.compile()

if __name__ == "__main__":
    fin.get_graph().draw_mermaid_png(output_file_path='graph.png')
    print(fin.invoke({"question": "What is an agent?"}))