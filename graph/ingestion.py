from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()

urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"]

# Opening the store is cheap, so this runs on import.
vectorstore = Chroma(
    collection_name="corrective-rag",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    persist_directory="chroma_db",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

 
def ingest() -> None:
    """Scrape the source URLs and embed them into the store. Run once."""
    docs = [UnstructuredLoader(web_url=url).load() for url in urls]
    docs_sub = [sub for doc in docs for sub in doc]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=25
    ).split_documents(docs_sub)
    # The Ollama embedding runner crashes on large batches, so add in slices of 100.
    BATCH = 100
    for i in range(0, len(chunks), BATCH):
        vectorstore.add_documents(chunks[i:i + BATCH])
        print(f"embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")


if __name__ == "__main__":
    ingest()