from typing import Any, Dict
from langchain_tavily import TavilySearch
from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()

tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("Web Search")
    question = state["question"]
    documents = state.get("retrieved_docs", [])

    tavily_results = tool.invoke({'query': question})
    joined = "\n\n".join(r["content"] for r in tavily_results["results"])
    documents.append(joined)
    return {'retrieved_docs': documents,'question': question}
