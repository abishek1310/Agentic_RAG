from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:latest", temperature = 0)

class gradedocuments(BaseModel):
    binary_score : Literal["yes", "no"] = Field(description="Whether the document is relevant to the question, either 'yes' or 'no'.")


structured_llm = llm.with_structured_output(gradedocuments)

system = """You are a grader assessing whether a retrieved document is relevant to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Give a binary score of 'yes' or 'no'."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\nRetrieved Documents: {retrieved_docs}")

    ] )

retriever_grader = prompt | structured_llm
