import os
import math
from dotenv import load_dotenv
from openai import OpenAI
import pypdf

load_dotenv()

client = OpenAI()

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "docs/credit-card-contract.pdf")


# --- Chunking ---

def chunk_text(text: str, source: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append({"text": " ".join(words[start:end]), "source": source})
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_file(file_path: str) -> list[dict]:
    if file_path.endswith(".pdf"):
        reader = pypdf.PdfReader(file_path)
        chunks = []
        for i, page in enumerate(reader.pages):
            chunks.extend(chunk_text(page.extract_text(), source=f"Page {i + 1}"))
        return chunks
    else:
        with open(file_path) as f:
            text = f.read()
        return chunk_text(text, source=os.path.basename(file_path))


# --- Embeddings & vector store ---

def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self):
        self.chunks: list[dict] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunks: list[dict]):
        vectors = embed([c["text"] for c in chunks])
        self.chunks.extend(chunks)
        self.embeddings.extend(vectors)
        print(f"Indexed {len(chunks)} chunks ({len(self.chunks)} total)")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = embed([query])[0]
        scored = [
            (cosine_similarity(query_vec, vec), chunk)
            for vec, chunk in zip(self.embeddings, self.chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


# --- Ingestion ---

def ingest(file_path: str = DEFAULT_PATH) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(load_file(file_path))
    return store


# --- Classification ---

CATEGORIES = ["fees", "warranty", "legal", "other"]

def classify(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Classify the question into exactly one of: {', '.join(CATEGORIES)}. "
                    "Reply with one lowercase word only."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip().lower()


# --- Specialized system prompts ---

SYSTEM_PROMPTS = {
    "fees":     "You are a financial expert focused on fees and charges. Answer using only the provided context.",
    "warranty": "You are a consumer rights expert focused on warranties and guarantees. Answer using only the provided context.",
    "legal":    "You are a legal expert. Be precise and cite specific clauses when possible. Answer using only the provided context.",
    "other":    "You are a helpful assistant. Answer using only the provided context. If the answer is not in the context, say so.",
}


def branch(category: str) -> str:
    if "fee" in category:
        return "fees"
    elif "warranty" in category:
        return "warranty"
    elif "legal" in category:
        return "legal"
    else:
        return "other"


# --- Answer ---

def answer(store: InMemoryVectorStore, question: str) -> str:
    results = store.search(question, top_k=3)
    context = "\n\n---\n\n".join(r["text"] for r in results)
    sources = list(dict.fromkeys(r["source"] for r in results))

    category = classify(question)
    system_prompt = SYSTEM_PROMPTS[branch(category)]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    answer_text = response.choices[0].message.content
    sources_line = ", ".join(f'"{s}"' for s in sources)
    return f"{answer_text}\n\nSources: {sources_line}"


# --- Query ---

def query(store: InMemoryVectorStore):
    questions = [
        "What are the annual fees for this credit card?",
        "What warranties do I have on purchases?",
        "What happens if I miss a payment?",
        "How do I cancel my card?",
    ]
    print("\n" + "=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(store, q)}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    store = ingest(path)
    query(store)
