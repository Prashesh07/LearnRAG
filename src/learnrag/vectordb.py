"""
Vector store setup: takes already-chunked Documents and an embedding model,
stores them in a persistent ChromaDB collection, and provides similarity search.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from embedder import get_embedding_model


def build_vector_store(
    chunks: list[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "learnrag_collection",
    batch_size: int = 500,
) -> Chroma:
    
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    embeddings = get_embedding_model()

    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vector_store.add_documents(batch)
        print(f"Stored {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")

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
    from document_loader import get_documents
    from chunker import split_chunks

   
    docs = get_documents("./docs/research_papers")
    chunks = split_chunks(docs)
    print(f"Total chunks: {len(chunks)}")

   
    vector_store = build_vector_store(chunks)

    # sample query
    similarity_search(vector_store, query="What are the most common LLM evaluation metrics?", k=3)