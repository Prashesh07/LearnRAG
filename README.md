# LearnRAG — Research Assistant

A Retrieval-Augmented Generation (RAG) research assistant that ingests a corpus of academic papers (arXiv, ~200 papers on LLM evaluation) and answers questions grounded in that corpus, using hybrid (dense + sparse) retrieval and cross-encoder reranking for high-precision results.

This project evolved from a single-PDF RAG prototype into a multi-document research assistant, built branch-by-branch on top of the original `LearnRAG` pipeline.

## What This Project Does

1. Downloads a focused corpus of ~200 research papers from arXiv on a chosen topic (LLM evaluation).
2. Loads and parses all PDFs, with caching so this expensive step only runs once.
3. Splits papers into clean, filtered chunks (dropping junk/reference-list fragments).
4. Embeds chunks locally (free, no API cost) and stores them in a persistent Chroma vector database.
5. Retrieves relevant chunks using **hybrid search** — dense (semantic) + sparse (keyword/BM25) — fused via Reciprocal Rank Fusion.
6. Reranks the retrieved candidates with a cross-encoder model for final precision.
7. Generates a grounded answer using a Groq-hosted LLM, using only the reranked context.

## Pipeline Overview

```
arXiv Papers (download_papers.py)
        │
        ▼
document_loader.py   → Loads all PDFs (PyMuPDF), caches to disk
        │
        ▼
chunker.py            → Splits into token-sized chunks, filters junk/references
        │
        ▼
embedder.py            → Local HuggingFace embedding model (free, no API key)
        │
        ▼
vectordb.py             → Batched embedding into a persistent Chroma vector store
        │
        ▼
retriever.py             → Hybrid retrieval: dense (Chroma) + sparse (BM25) via RRF
        │
        ▼
reranker.py               → Cross-encoder reranks candidates for precision
        │
        ▼
generator.py               → Retrieved + reranked context → Groq LLM → grounded answer
```

## Project Structure

```
LearnRAG/
├── src/
│   └── learnrag/
│       ├── download_papers.py   # Downloads ~200 arXiv papers on a chosen topic
│       ├── document_loader.py   # Loads PDFs (PyMuPDF), with disk caching
│       ├── chunker.py           # Recursive token-based chunking + junk filtering
│       ├── embedder.py          # Local HuggingFace embedding model
│       ├── vectordb.py          # Batched Chroma vector store build/load/search
│       ├── retriever.py         # Hybrid retrieval (dense + sparse + RRF fusion)
│       ├── reranker.py          # Cross-encoder reranking of retrieved candidates
│       └── generator.py         # Full RAG chain: retrieve → rerank → generate
├── docs/
│   └── research_papers/         # Downloaded arXiv PDFs (~200 papers)
├── cache/
│   └── documents.pkl            # Cached parsed documents (avoids PDF re-parsing)
├── chroma_db/                   # Persisted vector database (generated)
├── requirements.txt
├── .env                         # API keys (not committed)
├── .gitignore
└── README.md
```

## Module Reference

### `download_papers.py`
Uses the `arxiv` package to search and download a focused set of papers (e.g., ~200 papers matching "LLM evaluation" within `cs.CL`). Downloads each PDF via `requests` using each result's `pdf_url` (the package's older `download_pdf()` convenience method was removed in recent versions). Skips already-downloaded files so re-runs are safe.

### `document_loader.py`
- `pdf_loader()` — loads a single PDF using `PyMuPDFLoader` (faster and more robust than `pypdf` for multi-column academic PDFs).
- `load_pdf_folder()` — loads every PDF in a folder, skipping and logging any that fail to parse (corrupt/scanned files).
- `get_documents()` — wraps folder loading with a pickle-based cache (`cache/documents.pkl`), so repeated runs/tests don't re-parse ~200 PDFs every time. Use `force_reload=True` after adding/removing papers.

### `chunker.py`
Splits documents using `RecursiveCharacterTextSplitter`, measured by **token count** (via `tiktoken`) rather than raw characters, for more predictable prompt sizing. Filters out:
- Near-empty chunks (stray headers, page numbers)
- Likely bibliography/reference-list fragments, detected heuristically (dense bracketed citation numbers, or repeated "Name Name," author-list patterns)

Semantic chunking (embedding-based) was evaluated but dropped in favor of recursive splitting at this corpus scale — it makes an embedding call per sentence boundary, which is too slow across ~200 papers.

### `embedder.py`
Loads a local HuggingFace embedding model (`sentence-transformers/all-MiniLM-L6-v2`). Runs entirely on-device — no API key or cost, though setting `HF_TOKEN` avoids rate-limit warnings on model download.

### `vectordb.py`
- `build_vector_store()` — embeds chunks and writes them to a persistent Chroma collection **in batches** (important at ~5,500+ chunks, to avoid one huge blocking call).
- `load_vector_store()` — reloads the existing store without re-embedding.
- `similarity_search()` — plain dense similarity search (used for baseline comparison against hybrid search).

### `retriever.py`
Implements **hybrid search**:
- `build_bm25_retriever()` — sparse keyword retriever (BM25), built directly from chunk text, no embedding model needed.
- `build_hybrid_retriever()` — combines the dense Chroma retriever and the BM25 retriever using `EnsembleRetriever`, which performs **weighted Reciprocal Rank Fusion (RRF)**: `score = weight / (rank + c)`, summed across retrievers and deduplicated by chunk content.
- `hybrid_search()` — runs a query through the combined retriever and returns a ranked candidate set (recall-oriented: retrieves more candidates than are ultimately needed, e.g. top 10 from each retriever).

### `reranker.py`
Implements the **precision** stage on top of retrieval:
- Uses a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that scores each `(query, chunk)` pair **jointly**, which is more accurate than comparing independently-computed embedding vectors or BM25 scores.
- Too slow to run over the whole corpus, so it only reranks the candidate set produced by `retriever.py` (e.g., narrows ~15–20 candidates down to the top 3–5).

### `generator.py`
Wires the full pipeline together using LangChain Expression Language (LCEL):
1. Takes a user question.
2. Runs it through `hybrid_search()` (retrieval) then `rerank()` (precision).
3. Joins the final top chunks into a context string.
4. Formats context + question into a prompt instructing the model to answer **only** from the given context, and to explicitly say "I don't know" if the context is insufficient.
5. Sends the prompt to a Groq-hosted LLM (`openai/gpt-oss-120b`) and parses the text response.

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

### 3. Download the research paper corpus
```bash
python src/learnrag/download_papers.py
```

### 4. Run the full pipeline
```bash
python src/learnrag/generator.py
```

## Why Hybrid Search + Reranking (Not Just Dense Retrieval)

Dense embeddings are excellent at matching *meaning*, but research queries often hinge on exact terms — benchmark names (MMLU, HELM), author names, specific metrics — that dense similarity alone can under-match. Sparse (BM25) search catches these exact-term matches; combining both via RRF improves recall over either alone. The cross-encoder reranker then adds a precision pass: it directly judges relevance of each candidate to the query, catching false positives (e.g., keyword-overlapping but topically irrelevant chunks) that rank fusion alone can let through.

## Known Limitations / Notes

- Table and figure-caption content sometimes extracts as jumbled text fragments (PDF parsers don't understand table structure) — occasionally surfaces as a noisy retrieval result.
- A small number of downloaded papers (~3 out of 200) may fail to load (corrupted download or scanned/image-only PDF); these are logged and skipped rather than crashing the pipeline.
- Groq periodically deprecates model IDs (e.g., `llama-3.3-70b-versatile` → `openai/gpt-oss-120b`); check `https://console.groq.com/docs/models` if you hit a `model_not_found` error.

## Possible Next Steps

- **Deploy with Docker** — containerize the app; Chroma with a mounted volume is sufficient for a single-host deployment, Pinecone becomes worth considering only for serverless/multi-replica deployments.
- **Evaluation harness** — build a test set of question/answer pairs to measure retrieval precision/recall and answer faithfulness.
- **Source-aware citations** — surface which paper(s) each answer draws from directly in the chat response (metadata is already tracked per chunk).
- **UI layer** — wrap the pipeline in a simple Streamlit or Gradio front end.

## Tech Stack

| Component | Tool |
|---|---|
| Paper acquisition | `arxiv` package (arXiv API) |
| PDF Loading | `PyMuPDFLoader` (langchain-community) |
| Text Splitting | `RecursiveCharacterTextSplitter` (token-based, via `tiktoken`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector Database | ChromaDB (batched writes) |
| Sparse Retrieval | BM25 (`rank_bm25` via `BM25Retriever`) |
| Hybrid Fusion | `EnsembleRetriever` (Reciprocal Rank Fusion) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free) |
| LLM Inference | Groq (`openai/gpt-oss-120b`) |
| Orchestration | LangChain Expression Language (LCEL) |