"""
Recursive text splitter — splits documents into chunks for embedding,
sized/measured by token count for more predictable prompt sizing downstream.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken
from document_loader import get_documents
import re  #to remove references from the chunks


docs = get_documents("./docs/research_papers") 

def token_length(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))



def is_likely_reference_chunk(text: str) -> bool:
    """
    Heuristic: flags chunks that look like bibliography/reference list entries.
    Catches both citation styles:
      - "Smith, J., Doe, A. (2021)..."          (Lastname, Initial. style)
      - "Albert Q. Jiang, Alexandre Sablayrolles, ..."  (Firstname M. Lastname, style)
    """
    # Style 1: "Lastname, F."
    style1 = len(re.findall(r"[A-Z][a-z]+,\s[A-Z]\.", text))

    # Style 2: "Firstname M. Lastname," or "Firstname Lastname,"
    style2 = len(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s[A-Z]\.)?\s[A-Z][a-zA-Z]+,\s", text))

    # Bracketed citation numbers like [31], [4, 5]
    bracket_refs = len(re.findall(r"\[\d+(?:,\s*\d+)*\]", text))

    # A dense run of comma-separated proper names is a strong reference-list signal
    return style1 >= 3 or style2 >= 3 or bracket_refs >= 2

def split_chunks(documents: list[Document], chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Document]:
    """
    Split documents into smaller chunks using recursive character splitting,
    measured by token count. Filters out near-empty/junk chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=token_length,
    )
    chunks = splitter.split_documents(documents)

    # Drop near-empty or junk chunks (e.g. stray page numbers, blank headers)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50 and not is_likely_reference_chunk(c.page_content)]

    print(f"Split {len(documents)} pages into {len(chunks)} usable chunks")
    if chunks:
        print(f"Sample chunk metadata: {chunks[0].metadata}")

    return chunks

if __name__ == "__main__":
    
    chunks = split_chunks(docs)

    print("\nSample chunk metadata:")
    print(chunks[0].metadata)
    print(f"Total chunks: {len(chunks)}")
