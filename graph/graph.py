from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEBSEARCH, GENERATE
from graph.nodes import retrieve_docs, grade_documents, web_search, generate_answer
from graph.state import GraphState

load_dotenv()

def condition(state):
    if state['web_response']:
        return WEBSEARCH
    return GENERATE

chain = StateGraph(GraphState)
chain.add_node(RETRIEVE, retrieve_docs)
chain.add_node(GRADE_DOCUMENTS, grade_documents)
chain.add_node(WEBSEARCH, web_search)
chain.add_node(GENERATE, generate_answer)

chain.set_entry_point(RETRIEVE)
chain.add_edge(RETRIEVE, GRADE_DOCUMENTS)
chain.add_conditional_edges(GRADE_DOCUMENTS, condition, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
chain.add_edge(WEBSEARCH, GENERATE)
chain.add_edge(GENERATE, END)
fin = chain.compile()

if __name__ == "__main__":
    # Rendering posts to mermaid.ink, so keep it out of the import path.
    fin.get_graph().draw_mermaid_png(output_file_path='graph.png')
    print(fin.invoke({"question": "What is an agent?"}))