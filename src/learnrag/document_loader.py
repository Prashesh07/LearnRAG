from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def pdf_loader(pdf_path: str) -> list[Document]:
    """
    Load a PDF file and return its contents as a list of LangChain Document objects.

    Each Document typically represents one page of the PDF, with:
      - doc.page_content -> the extracted text
      - doc.metadata     -> info like source path and page number

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from: {path}")
    for i, doc in enumerate(documents, start=1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"\n--- Page {i} (metadata: {doc.metadata}) ---")
        print(f"{preview}...")

    return documents


if __name__ == "__main__":
    pdf_loader("./docs/langchain_summary.pdf")
    