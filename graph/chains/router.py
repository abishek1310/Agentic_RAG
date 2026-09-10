from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="gemma3:latest", temperature=0)

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="Route the question to 'vectorstore' or 'websearch'."
    )

llm_structured = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to either a vectorstore or a web search.

The vectorstore contains documents about:
- LLM agents: planning, task decomposition, reflection, memory, and tool use
- Prompt engineering: few-shot prompting, chain-of-thought, and instruction design
- Adversarial attacks on LLMs: jailbreaks, prompt injection, and defences

Use 'vectorstore' for questions on these topics.
Use 'websearch' for everything else, including current events, news, sports, people,
and any question whose answer depends on recent or real-time information."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human","Question: {question}")
])

router = prompt | llm_structured
