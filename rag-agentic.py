import os
import math
import json
import operator
from dotenv import load_dotenv
from openai import OpenAI
import pypdf

load_dotenv()

client = OpenAI()

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "docs/credit-card-contract.pdf")


# --- Chunking & loading ---

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


# --- Tool definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the document for chunks relevant to a query. Use this to find specific information in the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple math expression. Useful for computing fees, percentages, or totals mentioned in the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A Python math expression, e.g. '500 * 0.02' or '29.99 * 12'."}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_info",
            "description": "Returns metadata about the loaded document: filename and total number of pages or chunks.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# --- Tool implementations ---

SAFE_OPS = {
    "+": operator.add, "-": operator.sub,
    "*": operator.mul, "/": operator.truediv,
}

def run_calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def run_get_document_info(store: InMemoryVectorStore, file_path: str) -> str:
    return json.dumps({
        "filename": os.path.basename(file_path),
        "total_chunks": len(store.chunks),
    })


# --- Logging helpers ---

STEP_ICONS = {"search": "🔍", "calculate": "🧮", "get_document_info": "📄"}

def log(msg: str):
    print(f"  {msg}")


# --- Agentic loop ---

def answer(store: InMemoryVectorStore, file_path: str, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use search to find relevant information, calculate to compute numbers, "
                "and get_document_info to learn about the document. "
                "Call tools as many times as needed before giving a final answer."
            ),
        },
        {"role": "user", "content": question},
    ]

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    step = 0

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                step += 1
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                icon = STEP_ICONS.get(name, "🔧")

                log(f"Step {step} | {icon} {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")

                if name == "search":
                    results = store.search(args["query"], top_k=3)
                    result = "\n\n---\n\n".join(f"[{r['source']}] {r['text']}" for r in results)
                    log(f"       → returned {len(results)} chunks ({', '.join(r['source'] for r in results)})")
                elif name == "calculate":
                    result = run_calculate(args["expression"])
                    log(f"       → {args['expression']} = {result}")
                elif name == "get_document_info":
                    result = run_get_document_info(store, file_path)
                    log(f"       → {result}")
                else:
                    result = f"Unknown tool: {name}"

                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        else:
            log(f"Done  | ✅ answered after {step} tool call(s)")
            return msg.content


# --- Query ---

def query(store: InMemoryVectorStore, file_path: str):
    questions = [
        "What are the fees and what happens if I miss a payment?",
        "If I spend $3000 this month and the interest rate is 2%, how much interest will I owe?",
        "How many pages does this document have and how do I cancel my card?",
    ]
    for q in questions:
        print(f"\nA: {answer(store, file_path, q)}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    store = ingest(path)
    query(store, path)
