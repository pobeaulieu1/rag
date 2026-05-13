import os
import math
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ARTICLE_PATH = os.path.join(os.path.dirname(__file__), "docs/article.md")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


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
        self.chunks: list[str] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunks: list[str], embeddings: list[float]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        print(f"Indexed {len(chunks)} chunks ({len(self.chunks)} total)")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        query_vec = embed([query])[0]
        scored = [
            (cosine_similarity(query_vec, vec), chunk)
            for vec, chunk in zip(self.embeddings, self.chunks)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


def answer(store: InMemoryVectorStore, question: str) -> str:
    relevant_chunks = store.search(question, top_k=3)
    context = "\n\n---\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using only the "
                    "provided context. If the answer is not in the context, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def ingest(store: InMemoryVectorStore):
    with open(ARTICLE_PATH) as f:
        article = f.read()
    chunks = chunk_text(article, chunk_size=200, overlap=20)
    vectors = embed(chunks)
    store.add(chunks, vectors)


def query(store: InMemoryVectorStore):
    questions = [
        "What is Hawking radiation?",
        "How was a black hole first photographed?",
        "What happens to objects falling into a black hole?",
        "What is the information paradox?",
    ]
    print("\n" + "=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(store, q)}")
        print("-" * 60)


if __name__ == "__main__":
    store = InMemoryVectorStore()
    ingest(store)
    query(store)
