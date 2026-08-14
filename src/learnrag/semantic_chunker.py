from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()



def semantic_split(documents: list[Document], breakpoint_type: str = "percentile"):
    """
    Split documents into chunks based on semantic similarity between sentences.

    Args:
        documents: List of Document objects.
        breakpoint_type: How to decide where a "topic shift" is:
            - "percentile"        : split at the top X% most dissimilar points (default)
            - "standard_deviation": split where similarity drops N std devs below mean
            - "interquartile"     : split using IQR-based outlier detection

    Returns:
        List of semantically coherent Document chunks.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_type,
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} semantic chunks")
    return chunks


if __name__ == "__main__":
    from document_loader import pdf_loader

    docs = pdf_loader("./docs/langchain_summary.pdf")
    chunks = semantic_split(docs)

    print("\nSample chunk:")
    print(chunks[0].page_content)