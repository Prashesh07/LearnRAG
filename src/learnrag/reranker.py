from document_loader import get_documents
from langchain_core.documents import Document
from chunker import split_chunks
from vectordb import build_vector_store, load_vector_store
from pathlib import Path
from retriever import build_hybrid_retriever

try:
    # Current location as of recent LangChain versions
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except ImportError:
    # Fallback for older LangChain installs where it still lives in langchain_community
    from langchain_community.retrievers import EnsembleRetriever



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