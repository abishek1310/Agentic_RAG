from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="gemma3:latest", temperature=0)

class HallucinationGraderInput(BaseModel):
    binary_score: bool = Field(description="True if the generation is grounded in the set of facts, False otherwise.")

llm_structured = llm.with_structured_output(HallucinationGraderInput)

system = """You are a grader assessing whether an LLM generation is grounded in a set of retrieved facts.
Give a binary score: true if every claim in the generation is supported by the facts, false if it contains claims the facts do not support."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: {set_of_facts}\n\nLLM generation: {llm_generation}")])

hallucination_grader = prompt | llm_structured
