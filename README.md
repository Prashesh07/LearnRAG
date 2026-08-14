# LearnRAG

A learning project that implements a Retrieval-Augmented Generation (RAG) pipeline from scratch — from loading a PDF to generating grounded, context-aware answers using an LLM.

## What This Project Does

LearnRAG takes a PDF document, breaks it into searchable chunks, embeds those chunks into a vector database, and uses that database to retrieve relevant context for answering user questions with an LLM. This is the standard RAG pattern used in most modern AI-powered Q&A and chatbot systems.

## Pipeline Overview

```
PDF File
   │
   ▼
document_loader.py   →  Loads PDF and extracts text (PyPDFLoader)
   │
   ▼
chunker.py            →  Splits text into chunks
                          (Recursive Character Splitter / Semantic Chunker)
   │
   ▼
embedder.py            →  Converts chunks into vector embeddings
                          (HuggingFace sentence-transformers, local & free)
   │
   ▼
vectordb.py             →  Stores embeddings in ChromaDB for retrieval
   │
   ▼
generator.py            →  Retrieves relevant chunks + generates an answer
                          (LangChain + Groq LLM — llama-3.3-70b-versatile)
   │
   ▼
Final Answer
```

## Project Structure

```
LearnRAG/
├── src/
│   └── learnrag/
│       ├── document_loader.py   # Loads PDFs into LangChain Documents
│       ├── chunker.py           # Splits documents into chunks
│       ├── embedder.py          # Loads the HuggingFace embedding model
│       ├── vectordb.py          # Builds/loads Chroma vector store, similarity search
│       └── generator.py         # RAG chain: retrieval + LLM generation
├── docs/
│   └── langchain_summary.pdf    # Sample source document
├── chroma_db/                   # Persisted vector database (generated)
├── requirements.txt
├── .gitignore
└── README.md
```

## Module Reference

### `document_loader.py`
Loads a PDF file using `PyPDFLoader` and returns a list of LangChain `Document` objects (one per page), each with `page_content` and `metadata`.

### `chunker.py`
Splits loaded documents into smaller chunks for embedding, using either:
- **Recursive Character Splitter** — fast, splits on paragraph/sentence boundaries, no API calls.
- **Semantic Chunker** — splits based on meaning shifts between sentences using embeddings, producing more topically coherent chunks (slower).

### `embedder.py`
Loads a local HuggingFace embedding model (`sentence-transformers/all-MiniLM-L6-v2`) that converts text into dense vector representations. Runs entirely on-device — no API key or cost required.

### `vectordb.py`
Manages the ChromaDB vector store:
- `build_vector_store()` — embeds chunks and persists them to disk.
- `load_vector_store()` — reloads an existing store without re-embedding.
- `similarity_search()` — retrieves the top-k most relevant chunks for a query.

### `generator.py`
Wires retrieval and generation together into a full RAG chain using LangChain Expression Language (LCEL):
1. Takes a user question.
2. Retrieves relevant chunks from the vector store.
3. Injects them as context into a prompt template.
4. Sends the prompt to a Groq-hosted LLM (`llama-3.3-70b-versatile`) to generate a grounded answer.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3. Run the pipeline
```bash
python src/learnrag/generator.py
```

## Example

**Query:**
> Describe the use of LangChain in building a RAG application.

**Pipeline behavior:**
1. Retrieves the top 3 most relevant chunks from the embedded PDF.
2. Passes them as context to the LLM.
3. Returns a natural-language answer grounded in the source document (or "I don't know" if the answer isn't in the context).

## Tech Stack

| Component | Tool |
|---|---|
| PDF Loading | `PyPDFLoader` (langchain-community) |
| Text Splitting | `RecursiveCharacterTextSplitter`, `SemanticChunker` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Database | ChromaDB |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangChain Expression Language (LCEL) |

## Possible Next Steps

- **Hybrid search** — combine dense (Chroma) retrieval with sparse keyword search (BM25) via `EnsembleRetriever` for better exact-match recall.
- **Evaluation** — measure retrieval quality and answer faithfulness against a test question set.
- **UI** — wrap the pipeline in a simple Streamlit or Gradio front end.
- **Multi-document support** — extend the loader to handle folders of PDFs, not just a single file.

## Notes

- Embeddings run locally and are free; no OpenAI API key is required for this project.
- Groq is used for LLM inference due to its free tier and fast response times.
- The Chroma database persists to disk (`./chroma_db`), so embeddings only need to be generated once per document.