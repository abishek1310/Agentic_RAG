from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="gemma3:latest", temperature=0)

class AnswerGraderInput(BaseModel):
    binary_score: bool = Field(description="Whether the answer is correct or not, either 'yes' or 'no'.")

llm_structured = llm.with_structured_output(AnswerGraderInput)

system = """You are a grader assessing whether an answer is correct or not.
If the answer is factually correct and supports the question, grade it as correct.
Give a binary score of 'yes' or 'no'."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Question: {question}\n LLM_Generated_Answer: {generation}")])

answer_grader = prompt | llm_structured