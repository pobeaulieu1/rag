import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DEFAULT_ARTICLE_PATH = os.path.join(os.path.dirname(__file__), "docs/credit-card-contract.pdf")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer the question using only the "
        "provided context. If the answer is not in the context, say so."
    )),
    ("user", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | llm | StrOutputParser()


def ingest(file_path: str = DEFAULT_ARTICLE_PATH) -> InMemoryVectorStore:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    if file_path.endswith(".pdf"):
        docs = PyPDFLoader(file_path).load()
        for doc in docs:
            doc.metadata["section"] = f"Page {doc.metadata.get('page', 0) + 1}"
        chunks = text_splitter.split_documents(docs)
    elif file_path.endswith(".md"):
        with open(file_path) as f:
            text = f.read()
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("##", "section")]
        )
        chunks = text_splitter.split_documents(header_splitter.split_text(text))
    else:
        with open(file_path) as f:
            text = f.read()
        chunks = text_splitter.create_documents(
            [text], metadatas=[{"section": os.path.basename(file_path)}]
        )

    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {os.path.basename(file_path)}")
    return store


def answer(store: InMemoryVectorStore, question: str) -> str:
    docs = store.similarity_search(question, k=3)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    sources = list(dict.fromkeys(doc.metadata.get("section", "Unknown") for doc in docs))

    answer_text = chain.invoke({"context": context, "question": question})
    sources_line = ", ".join(f'"{s}"' for s in sources)
    return f"{answer_text}\n\nSources: {sources_line}"


def query(store: InMemoryVectorStore):
    questions = [
        "Est-ce que je suis responsable si quelqun vole ma carte et fait des transactions pour moi sans mon autorisation?"
    ]
    print("\n" + "=" * 60)
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(store, q)}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ARTICLE_PATH
    store = ingest(path)
    query(store)
