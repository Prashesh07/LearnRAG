from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,PyMuPDFLoader

from langchain_core.documents import Document


def pdf_loader(pdf_path: str) -> list[Document]:
    """
    Load a single PDF file and return its contents as a list of LangChain Document objects.

    Each Document typically represents one page of the PDF, with:
      - doc.page_content -> the extracted text
      - doc.metadata     -> info like source path and page number

    Args:
        pdf_path: Path to a single PDF file.

    Returns:
        A list of Document objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    loader = PyMuPDFLoader(str(path))
    documents = loader.load()

    print(f"Loaded {len(documents)} page(s) from: {path.name}")
    return documents


def load_pdf_folder(folder_path: str, limit: int | None = None) -> list[Document]:
    """
    Load every PDF in a folder (recursively) into a single list of Documents.

    Skips files that fail to load (corrupt, scanned/image-only, malformed)
    instead of crashing the whole batch, and reports which ones failed.

    Args:
        folder_path: Path to the folder containing PDFs.
        limit: Optional cap on number of PDFs to load (useful for testing
               on a subset before running the full dataset).

    Returns:
        Combined list of Document objects across all successfully loaded PDFs.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = sorted(folder.rglob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]

    print(f"Found {len(pdf_files)} PDF file(s) in {folder}")

    all_documents: list[Document] = []
    failed_files: list[str] = []

    for i, pdf_file in enumerate(pdf_files, start=1):
        try:
            docs = pdf_loader(str(pdf_file))
            all_documents.extend(docs)
            print(f"[{i}/{len(pdf_files)}] Loaded {len(docs)} page(s): {pdf_file.name}")
        except Exception as e:
            failed_files.append(pdf_file.name)
            print(f"[{i}/{len(pdf_files)}] FAILED: {pdf_file.name} ({e})")

    print(f"\nDone. Loaded {len(all_documents)} total pages from "
          f"{len(pdf_files) - len(failed_files)}/{len(pdf_files)} PDFs.")
    if failed_files:
        print(f"Failed files ({len(failed_files)}): {failed_files}")

    return all_documents


if __name__ == "__main__":
    docs = load_pdf_folder("./docs/research_papers")
    