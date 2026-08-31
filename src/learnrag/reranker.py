"""
Reranking: takes a candidate set of chunks (from retriever.py's hybrid_search)
and re-scores them with a cross-encoder model for higher precision than RRF
fusion of dense/sparse rankings alone can achieve.

the pattern is: retrieve broad (e.g. top 10-20 via RRF), then rerank
  narrow (down to the top 3-5 via cross-encoder) before sending to the LLM.
"""

from pathlib import Path
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker_model: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        print(f"Loaded reranker model: {RERANKER_MODEL_NAME}")
    return _reranker_model


def rerank(query: str, candidates: list[Document], top_k: int = 5) -> list[Document]:
    
    if not candidates:
        return []

    model = get_reranker()

    # Cross-encoder scores each (query, chunk) pair jointly for relevance.
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda pair: pair[1], reverse=True)

    top_results = [doc for doc, score in scored_candidates[:top_k]]

    print(f"\nReranked {len(candidates)} candidates -> top {len(top_results)} for query: '{query}'")
    for i, (doc, score) in enumerate(scored_candidates[:top_k], start=1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"{i}. (score={score:.4f}) {preview}...")

    return top_results


if __name__ == "__main__":
    from document_loader import get_documents
    from chunker import split_chunks
    from vectordb import build_vector_store, load_vector_store
    from retriever import build_hybrid_retriever, hybrid_search

    docs = get_documents("./docs/research_papers")
    chunks = split_chunks(docs)

    vector_store = load_vector_store() if Path("./chroma_db").exists() else build_vector_store(chunks)

    hybrid_retriever = build_hybrid_retriever(chunks, vector_store, dense_k=10, sparse_k=10)

    query = "What are the most common LLM evaluation metrics?"
    candidates = hybrid_search(hybrid_retriever, query)
    top_chunks = rerank(query, candidates, top_k=3)