from dotenv  import load_dotenv
from graph.graph import fin
load_dotenv()
import os

if __name__ == "__main__":
    print("Corrective_RAG")
    answer = fin.invoke({"question": "What is a car"})
    print(answer)