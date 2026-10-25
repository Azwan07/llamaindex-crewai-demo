import os

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def run_rag_demo() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

    Settings.llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=api_key,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=api_key,
    )

    documents = SimpleDirectoryReader("data").load_data()
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(
        "What is the difference between LlamaIndex and CrewAI, and how do they fit together?"
    )

    print(response)
