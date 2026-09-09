from dotenv  import load_dotenv
load_dotenv()
import os

if __name__ == "__main__":
    print("Corrective_RAG")
    print(os.environ.get("TAVILY_API_KEY"))