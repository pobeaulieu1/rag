# Question 1 — RAG (No Coding) (~20 min)

---

## Q1. Explain the RAG pipeline from start to finish.

**Answer:**
1. **Ingestion** — Load documents, split into chunks, embed each chunk, store vectors in a vector DB.
2. **Retrieval** — Embed the user's question, run cosine similarity against stored vectors, return top-k chunks.
3. **Generation** — Pass retrieved chunks as context to an LLM, which generates a grounded answer.

---

## Q2. What is chunking and why does overlap matter?

**Answer:**
Chunking splits a document into smaller pieces so embeddings stay semantically focused.
Overlap (e.g. 100 chars shared between consecutive chunks) prevents losing context at boundaries —
a sentence split across two chunks is still fully represented in at least one of them.

- Small chunks (~200 tokens): precise retrieval, may miss surrounding context
- Large chunks (~800 tokens): more context, but embedding becomes less focused

---

## Q3. What is the difference between a bi-encoder and a cross-encoder?

**Answer:**
- **Bi-encoder**: embeds query and document *separately*, compares with cosine similarity. Fast, scalable to millions of docs. Used for initial retrieval.
- **Cross-encoder**: takes `(query, document)` as a *pair*, scores relevance jointly. Much more accurate but too slow to run on the full corpus. Used for reranking top-k results.

Typical pipeline: bi-encoder retrieves top-20 → cross-encoder reranks → keep top-3.

---

## Q4. How would you evaluate the quality of a RAG system?

**Answer:**
Use **RAGAS** (RAG Assessment) framework — it measures:
- **Faithfulness**: is the answer grounded in the retrieved context? (no hallucination)
- **Answer relevance**: does the answer actually address the question?
- **Context precision**: are the retrieved chunks relevant to the question?
- **Context recall**: does the retrieved context contain the necessary information?

Without a framework: build a golden Q&A dataset and check exact match / semantic similarity.

---

## Q5. When would you choose an agentic RAG over a fixed RAG pipeline?

**Answer:**
- **Fixed pipeline**: one retrieval step, predictable latency, good for simple Q&A.
- **Agentic RAG**: the LLM decides when and what to search, can search multiple times, can use multiple tools (search, calculate, API calls). Better for complex multi-step questions but slower and harder to debug.

Use agentic when: the question requires combining info from multiple sections, or you don't know upfront what to search for.

---

## Q6. What are the risks of RAG and how do you mitigate them?

**Answer:**
- **Hallucination**: LLM ignores context → add "only answer from context" in system prompt, measure faithfulness.
- **Wrong chunks retrieved**: weak embeddings or bad chunking → tune chunk size, add reranker, use metadata filters.
- **Sensitive data in chunks**: document contains PII → filter before ingestion or add access control per user.
- **Stale knowledge**: document updated but index not → rebuild index on document change, add timestamps.
