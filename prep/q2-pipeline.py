"""
Question 2 — Pipeline Configuration (~30 min)
==============================================

You are given a list of raw text documents and a user query.
Implement a pipeline that:

1. Chunks each document into pieces of `chunk_size` words with `overlap` words of overlap.
2. Embeds all chunks using the OpenAI embeddings API.
3. Retrieves the top-k most relevant chunks for the query using cosine similarity.
4. Returns a final answer from gpt-4o-mini using only the retrieved chunks as context.

Complete the functions below. Do not change their signatures.

Notes:
- Use model "text-embedding-3-small" for embeddings.
- Use model "gpt-4o-mini" for generation.
- Your OPENAI_API_KEY is available as an environment variable.
"""

import os
import math
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─── Implement these ──────────────────────────────────────────────────────────

def chunk(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    """Split text into overlapping word-based chunks."""
    # TODO
    pass


def embed(texts: list[str]) -> list[list[float]]:
    """Return an embedding vector for each text."""
    # TODO
    pass


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors."""
    # TODO
    pass


def retrieve(query: str, chunks: list[str], embeddings: list[list[float]], top_k: int = 3) -> list[str]:
    """Return the top_k most relevant chunks for the query."""
    # TODO
    pass


def generate(query: str, context_chunks: list[str]) -> str:
    """Call gpt-4o-mini with the retrieved chunks as context and return the answer."""
    # TODO
    pass


def pipeline(documents: list[str], query: str) -> str:
    """Full RAG pipeline: chunk → embed → retrieve → generate."""
    # TODO
    pass


# ─── Solution ─────────────────────────────────────────────────────────────────

def chunk_solution(text: str, chunk_size: int = 150, overlap: int = 20) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def embed_solution(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity_solution(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)


def retrieve_solution(query: str, chunks: list[str], embeddings: list[list[float]], top_k: int = 3) -> list[str]:
    query_vec = embed_solution([query])[0]
    scored = [(cosine_similarity_solution(query_vec, vec), chunk) for vec, chunk in zip(embeddings, chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def generate_solution(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context. If the answer is not there, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content


def pipeline_solution(documents: list[str], query: str) -> str:
    all_chunks = [c for doc in documents for c in chunk_solution(doc)]
    embeddings = embed_solution(all_chunks)
    top_chunks = retrieve_solution(query, all_chunks, embeddings)
    return generate_solution(query, top_chunks)


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = [
        "Python is a high-level programming language known for its simplicity and readability. It supports multiple paradigms including procedural, object-oriented, and functional programming.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Common algorithms include linear regression, decision trees, and neural networks.",
        "RAG stands for Retrieval-Augmented Generation. It combines a retrieval system with a language model to ground answers in external documents.",
    ]
    query = "What is RAG and how does it work?"
    print(pipeline_solution(docs, query))
