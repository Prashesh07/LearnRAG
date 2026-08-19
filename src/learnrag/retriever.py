"""
Hybrid retrieval: combines dense (Chroma) similarity search with sparse (BM25)
keyword search using Reciprocal Rank Fusion (RRF) via EnsembleRetriever.

Dense search excels at semantic/meaning-based matches; BM25 excels at exact
keyword/term matches (benchmark names, acronyms, author names) that dense
embeddings can sometimes miss. Combining both improves recall.
"""

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

try:
    # Current location as of recent LangChain versions
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    # Fallback for older LangChain installs where it still lives in langchain_community
    from langchain_community.retrievers import EnsembleRetriever


def build_bm25_retriever(chunks: list[Document], k: int = 5) -> BM25Retriever:
    """
    Build a sparse keyword-based retriever (BM25) from the same chunks used
    for the dense vector store. No embedding model needed — BM25 works
    directly off token frequency statistics.
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    print(f"Built BM25 retriever over {len(chunks)} chunks (k={k})")
    return bm25_retriever


def build_hybrid_retriever(
    chunks: list[Document],
    vector_store,
    dense_k: int = 5,
    sparse_k: int = 5,
    weights: tuple[float, float] = (0.5, 0.5),
) -> EnsembleRetriever:
    """
    Combine dense (Chroma) and sparse (BM25) retrieval into one hybrid retriever
    using Reciprocal Rank Fusion to merge and re-rank results from both.

    Args:
        chunks: The full chunk list (used to build the BM25 index).
        vector_store: An already-built/loaded Chroma vector store.
        dense_k: Number of results to pull from dense search.
        sparse_k: Number of results to pull from sparse (BM25) search.
        weights: Relative importance of (dense, sparse) when fusing rankings.
                 Equal weighting (0.5, 0.5) is a reasonable default; increase
                 the sparse weight if exact-term queries matter more to you.

    Returns:
        An EnsembleRetriever combining both retrieval strategies.
    """
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": dense_k})
    sparse_retriever = build_bm25_retriever(chunks, k=sparse_k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=list(weights),
    )

    print(f"Built hybrid retriever (dense weight={weights[0]}, sparse weight={weights[1]})")
    return hybrid_retriever


def hybrid_search(retriever: EnsembleRetriever, query: str) -> list[Document]:
    """Run a query through the hybrid retriever and print the results."""
    results = retriever.invoke(query)

    print(f"\nTop {len(results)} hybrid results for query: '{query}'")
    for i, doc in enumerate(results, start=1):
        preview = doc.page_content[:150].replace("\n", " ")
        source = doc.metadata.get("source", "unknown")
        print(f"{i}. [{source}] {preview}...")

    return results


if __name__ == "__main__":
    from document_loader import get_documents
    from chunker import split_chunks
    from vectordb import build_vector_store, load_vector_store
    from pathlib import Path

    docs = get_documents("./docs/research_papers")
    chunks = split_chunks(docs)
    print(f"Total chunks: {len(chunks)}")

    # Reuse existing Chroma store if already built, otherwise build fresh
    if Path("./chroma_db").exists():
        vector_store = load_vector_store()
    else:
        vector_store = build_vector_store(chunks)

    retriever = build_hybrid_retriever(chunks, vector_store, dense_k=5, sparse_k=5)

    hybrid_search(retriever, query="What are the most common LLM evaluation metrics?")