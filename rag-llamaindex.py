import os
from dotenv import load_dotenv

load_dotenv()

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import pypdf

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "docs/article.md")

# Global model config — LlamaIndex reads these from Settings automatically
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=100)


# --- Loading ---

def load_documents(file_path: str) -> list[Document]:
    if file_path.endswith(".pdf"):
        reader = pypdf.PdfReader(file_path)
        return [
            Document(
                text=page.extract_text(),
                metadata={"source": f"Page {i + 1}", "file": os.path.basename(file_path)},
            )
            for i, page in enumerate(reader.pages)
        ]
    with open(file_path) as f:
        text = f.read()
    return [Document(text=text, metadata={"source": os.path.basename(file_path)})]


# --- Ingestion ---

def ingest(file_path: str = DEFAULT_PATH) -> VectorStoreIndex:
    documents = load_documents(file_path)
    index = VectorStoreIndex.from_documents(documents)
    print(f"Indexed {len(documents)} document(s) from {os.path.basename(file_path)}")
    return index


# --- Answer ---

def answer(index: VectorStoreIndex, question: str) -> str:
    engine = index.as_query_engine(similarity_top_k=3)
    response = engine.query(question)

    sources = list(dict.fromkeys(
        node.metadata.get("source", "Unknown")
        for node in response.source_nodes
    ))
    sources_line = ", ".join(f'"{s}"' for s in sources)
    return f"{response}\n\nSources: {sources_line}"


# --- Query ---

def query(index: VectorStoreIndex):
    questions = [
        "What are the annual fees for this credit card?",
        "What warranties do I have on purchases?",
        "What happens if I miss a payment?",
        "How do I cancel my card?",
    ]
    print("\n" + "=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(index, q)}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    index = ingest(path)
    query(index)
