from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma3:latest", temperature=0)

system = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\nContext: {context}\nAnswer:"),
    ]
)

chain = prompt | llm | StrOutputParser()
