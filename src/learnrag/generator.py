"""
Generator: full RAG chain wiring (retrieve -> rerank -> generate).

Pipeline components are built lazily on first use (see get_pipeline) so that
importing this module does not force the whole document/vector-store setup
(e.g. when running evaluation.py, which imports the same functions).
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from chunker import split_chunks
from document_loader import get_documents
from reranker import rerank
from retriever import build_hybrid_retriever, hybrid_search
from vectordb import build_vector_store, load_vector_store

load_dotenv()

CHAT_MODEL_NAME = "openai/gpt-oss-120b"
RERANK_TOP_K = 3

RAG_TEMPLATE = """\
Answer the user's query using ONLY the information in the context below.
Do not use any outside knowledge, even if you know more about the topic.
Do not include code examples unless the code appears verbatim in the context.
If the context does not contain enough information to answer, respond with exactly: "I don't know."

User's Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

chat_model = ChatGroq(
    temperature=0,
    model_name=CHAT_MODEL_NAME,
    api_key=os.environ["GROQ_API_KEY"],
)


@lru_cache(maxsize=1)
def get_pipeline() -> tuple[list[Document], object, object]:
    """
    Load (once) and cache the full retrieval pipeline:
    documents -> chunks -> vector store -> hybrid (RRF) retriever.
    """
    docs = get_documents("./docs/research_papers")
    chunks = split_chunks(docs)

    vector_store = (
        load_vector_store()
        if Path("./chroma_db").exists()
        else build_vector_store(chunks)
    )

    hybrid_retriever = build_hybrid_retriever(chunks, vector_store, dense_k=10, sparse_k=10)
    return chunks, vector_store, hybrid_retriever


def retrieve_top_documents(question: str, verbose: bool = True) -> list[Document]:
    """
    Two-stage retrieval:
      1. Hybrid retrieve (dense + sparse via RRF) -> a broad candidate set (recall).
      2. Cross-encoder rerank -> narrow down to the most truly relevant chunks (precision).
    Returns the reranked top chunks (with metadata intact) for the question.
    """
    _, _, hybrid_retriever = get_pipeline()
    candidates = hybrid_search(hybrid_retriever, question, verbose=verbose)
    return rerank(question, candidates, top_k=RERANK_TOP_K, verbose=verbose)


def retrieve_context(question: str, verbose: bool = True) -> str:
    """Run retrieve_top_documents and join the top chunks into a context string."""
    top_chunks = retrieve_top_documents(question, verbose=verbose)
    return "\n\n".join(doc.page_content for doc in top_chunks)


def extract_sources(documents: list[Document]) -> list[str]:
    """arXiv IDs (filename stems) of the papers the documents came from."""
    sources = []
    for doc in documents:
        source_path = doc.metadata.get("source", "")
        if source_path:
            sources.append(Path(source_path).stem)
    return sorted(set(sources))


def build_rag_chain():
    """
    Return the LCEL chain that turns {"question": ..., "context": ...}
    into a generated answer. Retrieval is deliberately left outside the chain
    so callers can compute context once and reuse it (e.g. in evaluation).
    """
    return rag_prompt | chat_model | StrOutputParser()


def generate(question: str, verbose: bool = True) -> tuple[str, list[str]]:
    """Full RAG pipeline: retrieve -> rerank -> generate.
    Returns (answer, source_paper_ids)."""
    top_chunks = retrieve_top_documents(question, verbose=verbose)
    context = "\n\n".join(doc.page_content for doc in top_chunks)
    answer = build_rag_chain().invoke({"context": context, "question": question})
    return answer, extract_sources(top_chunks)


if __name__ == "__main__":
    # Model answers occasionally include Unicode characters (math symbols,
    # non-breaking hyphens) that the Windows cp1252 console cannot encode.
    # Force UTF-8 with lossy fallback so printing can never crash the REPL.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    print("LearnRAG Research Assistant")
    print("Type a question (or 'exit' / 'quit' to leave).")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        answer, sources = generate(question)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {', '.join(sources) if sources else 'none found'}")