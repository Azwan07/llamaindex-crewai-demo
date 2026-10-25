import os

from dotenv import load_dotenv

from src.llamaindex_rag import run_rag_demo
from src.crewai_demo import run_crewai_demo


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Create a .env file from .env.example and add your key."
        )

    print("\n=== LlamaIndex RAG Demo ===")
    run_rag_demo()

    print("\n=== CrewAI Multi-Agent Demo ===")
    run_crewai_demo()


if __name__ == "__main__":
    main()
