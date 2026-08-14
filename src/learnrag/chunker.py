#native chunker(Recursive Text Splitter) for langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_core.documents import Document


def recursive_split(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks using recursive character splitting.

    Args:
        documents: List of Document objects (e.g., from pdf_loader).
        chunk_size: Max characters per chunk.
        chunk_overlap: How many characters to overlap between chunks
                       (helps preserve context across chunk boundaries).

    Returns:
        List of smaller Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # tried in order
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks







if __name__ == "__main__":
    from document_loader import pdf_loader

    docs = pdf_loader("./docs/langchain_summary.pdf")
    chunks = recursive_split(docs, chunk_size=1000, chunk_overlap=200)

    print("\nSample chunk:")
    print(chunks[1].page_content)
    print(chunks[1].metadata)
    print(f"Total chunks: {len(chunks)}")