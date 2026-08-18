"""
Vector store setup: takes chunks from chunker.py and embeddings from embedder.py,
stores them in a persistent ChromaDB collection, and provides similarity search.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from embedder import get_embedding_model
from chunker import recursive_split
from semantic_chunker import semantic_split


def build_vector_store(
    chunks: list[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "learnrag_collection",
) -> Chroma:
    """
    Embed chunks (using embedder.get_embedding_model) and store them in Chroma.

    Args:
        chunks: List of Document chunks from chunker.semantic_split.
        persist_directory: Folder where Chroma saves its data to disk.
        collection_name: Name of the collection inside Chroma.

    Returns:
        A Chroma vector store instance, ready for querying.
    """
    embeddings = get_embedding_model()
    #chunks = semantic_split(chunks)  # Use semantic splitting for better context preservation
    chunks=recursive_split(chunks)  # Use recursive splitting for better context preservation
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    print(f"Stored {len(chunks)} chunks in Chroma at '{persist_directory}' "
          f"(collection: '{collection_name}')")
    return vector_store


def load_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "learnrag_collection",
) -> Chroma:
    """
    Load an existing Chroma vector store from disk without re-embedding.
    Use this once the store has already been built by build_vector_store().
    """
    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Loaded Chroma collection '{collection_name}' from '{persist_directory}'")
    return vector_store


def similarity_search(vector_store: Chroma, query: str, k: int = 5) -> list[Document]:
    """Run a similarity search against the vector store and print the results."""
    results = vector_store.similarity_search(query, k=k)

    print(f"\nTop {k} results for query: '{query}'")
    for i, doc in enumerate(results, start=1):
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"{i}. {preview}...")

    return results


if __name__ == "__main__":
    from document_loader import pdf_loader

    # 1. Load and chunk the PDF (chunker.py handles splitting)
    docs = pdf_loader("./docs/langchain_summary.pdf")
    chunks = semantic_split(docs)
    print(f"Total chunks: {len(chunks)}")

    # 2. Build the vector store (embedder.py handles the embedding model)
    vector_store = build_vector_store(chunks)

    # 3. Try a sample query
    similarity_search(vector_store, query="What is LangChain used for?", k=3)