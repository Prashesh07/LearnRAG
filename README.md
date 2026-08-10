# LearnRAG — Hybrid Retrieval-Augmented Generation System

A production-oriented **Retrieval-Augmented Generation (RAG)** system for question answering over academic textbooks.

LearnRAG combines **dense semantic retrieval, sparse keyword retrieval, hybrid search, cross-encoder reranking, and large language model generation** to provide grounded answers with source citations.

The project is being developed progressively, starting with the core RAG pipeline and evolving into a resume-level AI/ML system.

---

## 📌 Project Overview

Traditional LLMs can generate fluent answers but may:

* Lack access to domain-specific documents
* Hallucinate information
* Provide outdated or unsupported answers
* Struggle to identify the exact source of an answer

LearnRAG addresses these problems by retrieving relevant information from a document collection before generating an answer.

### High-Level Pipeline

```text
                    User Query
                        │
                        ▼
                Query Processing
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      Dense Embedding            BM25
         BGE-M3             Sparse Retrieval
             │                     │
             └──────────┬──────────┘
                        ▼
                 Hybrid Retrieval
                    RRF Fusion
                        │
                        ▼
                   Top-K Chunks
                        │
                        ▼
               Cross-Encoder Reranker
                BGE-reranker-v2-m3
                        │
                        ▼
                    Top-N Chunks
                        │
                        ▼
               Context Construction
                        │
                        ▼
                     Qwen3
                    (Ollama)
                        │
                        ▼
               Structured Response
                  ┌─────┴─────┐
                  ▼           ▼
                Answer     Citations
                        │
                        ▼
                     FastAPI
                        │
                        ▼
                    Streamlit
```

---

# 🎯 Objectives

The main objectives of LearnRAG are:

1. Build a complete end-to-end RAG system.
2. Understand dense and sparse information retrieval.
3. Implement hybrid retrieval using semantic and lexical search.
4. Improve retrieval quality using cross-encoder reranking.
5. Generate grounded answers using a local LLM.
6. Return structured responses with document citations.
7. Evaluate retrieval and generation quality quantitatively.
8. Package the system as a reusable API.
9. Build a user-friendly interface.
10. Deploy the application using Docker.

---

# 📚 Dataset

The primary knowledge base will use the **College Textbook PDF Dataset from Kaggle**.

The dataset contains textbook material suitable for document retrieval and question-answering applications.

### Dataset

[Kaggle — College Textbook PDF Dataset](https://www.kaggle.com/datasets/rohanthoma/ebook-pdfs)

The dataset will **not** be stored directly in this repository.

Place downloaded PDFs inside:

```text
data/
└── raw/
    ├── physics/
    ├── biology/
    └── ...
```

The exact organization will depend on the downloaded dataset.

---

# 🏗️ Architecture

## 1. Document Ingestion

```text
PDF
 ↓
PyMuPDF
 ↓
Raw Documents
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Metadata
```

Each chunk will preserve metadata such as:

```json
{
    "document": "physics.pdf",
    "page": 143,
    "chapter": "Newton's Laws",
    "chunk_id": "physics_143_04"
}
```

This metadata will later be used for filtering and source citations.

---

## 2. Dense Retrieval

Documents are converted into semantic vector representations using:

```text
BAAI/bge-m3
```

Pipeline:

```text
Document Chunk
      ↓
    BGE-M3
      ↓
Dense Vector
      ↓
Pinecone
```

At query time:

```text
User Query
     ↓
BGE-M3
     ↓
Query Vector
     ↓
Dense Search
```

Dense retrieval captures semantic similarity even when the query and document use different wording.

---

## 3. Sparse Retrieval

The system will also use **BM25** for keyword-based retrieval.

BM25 is particularly useful when exact terminology matters.

For example:

```text
Query:
"What factors affect photosynthesis?"
```

BM25 can identify documents containing terms such as:

```text
photosynthesis
factors
light
temperature
carbon dioxide
```

This complements semantic retrieval.

---

## 4. Hybrid Retrieval

Dense and sparse retrieval results will be combined.

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
       BGE-M3              BM25
          │                 │
          ▼                 ▼
     Dense Results     Sparse Results
          │                 │
          └────────┬────────┘
                   ▼
             RRF Fusion
                   │
                   ▼
              Top-K Results
```

**Reciprocal Rank Fusion (RRF)** will initially be used to combine rankings.

---

## 5. Cross-Encoder Reranking

The hybrid retriever produces a candidate set.

Instead of sending all candidates directly to the LLM, they are reranked using:

```text
BAAI/bge-reranker-v2-m3
```

Pipeline:

```text
Hybrid Retrieval
      ↓
Top 20–30 Candidates
      ↓
Cross-Encoder
      ↓
Relevance Scores
      ↓
Top 5 Candidates
```

The reranker evaluates the relevance of:

```text
(query, document_chunk)
```

together.

This allows the system to select higher-quality context for generation.

---

# 🤖 LLM Generation

The project will initially use a locally hosted Qwen model through **Ollama**.

```text
Retrieved Context
       +
User Question
       ↓
     Qwen3
       ↓
Grounded Answer
```

The model will be instructed to:

* Answer using the retrieved context.
* Avoid unsupported claims.
* Cite the retrieved sources.
* Admit when sufficient information cannot be found.

This helps reduce hallucination.

---

# 📦 Structured API Response

The API will return structured JSON rather than plain text.

Example:

```json
{
    "answer": "Newton's second law describes the relationship between force, mass, and acceleration.",
    "sources": [
        {
            "document": "physics.pdf",
            "page": 143,
            "section": "Newton's Laws"
        },
        {
            "document": "physics.pdf",
            "page": 144,
            "section": "Applications of Newton's Laws"
        }
    ],
    "retrieved_chunks": 20,
    "reranked_chunks": 5
}
```

Additional fields such as retrieval confidence and latency may be added later.

---

# 🛠️ Tech Stack

| Component            | Technology                       |
| -------------------- | -------------------------------- |
| Programming Language | Python                           |
| RAG Framework        | LangChain                        |
| PDF Processing       | PyMuPDF                          |
| Dense Embeddings     | BAAI/bge-m3                      |
| Sparse Retrieval     | BM25                             |
| Vector Database      | Pinecone                         |
| Hybrid Fusion        | Reciprocal Rank Fusion           |
| Reranker             | BAAI/bge-reranker-v2-m3          |
| LLM                  | Qwen3                            |
| Local LLM Runtime    | Ollama                           |
| Backend              | FastAPI                          |
| Data Validation      | Pydantic                         |
| Frontend             | Streamlit                        |
| Evaluation           | RAGAS + custom retrieval metrics |
| Testing              | pytest                           |
| Containerization     | Docker                           |
| Version Control      | Git / GitHub                     |

---

# 📁 Project Structure

```text
learnrag/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── dense.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   ├── hybrid.py
│   │   └── rrf.py
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── cross_encoder.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── generator.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── response.py
│   │
│   ├── pipeline.py
│   └── config.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── routes.py
│
├── frontend/
│   └── app.py
│
├── evaluation/
│   ├── questions.json
│   ├── evaluate_retrieval.py
│   └── evaluate_generation.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   └── test_api.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_experiment.ipynb
│   ├── 03_embedding_experiment.ipynb
│   └── 04_retrieval_evaluation.ipynb
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 🔄 Development Strategy

The system will be developed incrementally.

## Phase 1 — Document Processing

```text
PDF
 ↓
PyMuPDF
 ↓
Cleaning
 ↓
Chunking
 ↓
Metadata
```

## Phase 2 — Dense RAG

```text
Chunks
 ↓
BGE-M3
 ↓
Pinecone
 ↓
Dense Retrieval
```

## Phase 3 — Sparse Retrieval

```text
BM25
 ↓
Keyword Retrieval
```

## Phase 4 — Hybrid Retrieval

```text
Dense Retrieval
      +
BM25
      ↓
RRF
      ↓
Hybrid Results
```

## Phase 5 — Reranking

```text
Hybrid Results
      ↓
Cross Encoder
      ↓
Top Results
```

## Phase 6 — Generation

```text
Top Results
      ↓
Context Builder
      ↓
Qwen3
      ↓
Answer + Citations
```

## Phase 7 — API

```text
FastAPI
 ↓
Structured JSON Response
```

## Phase 8 — Frontend

```text
Streamlit
 ↓
User Interface
```

## Phase 9 — Evaluation

Compare:

```text
BM25
   vs
Dense
   vs
Hybrid
   vs
Hybrid + Reranker
```

Metrics will include:

* Recall@K
* Precision@K
* MRR
* nDCG
* Context Precision
* Context Recall
* Faithfulness
* Answer Relevancy

---

# 🧪 Evaluation Strategy

The project will maintain a separate evaluation dataset containing:

```json
{
    "question": "What is Newton's second law?",
    "expected_document": "physics.pdf",
    "expected_page": 143,
    "ground_truth": "Newton's second law relates force, mass and acceleration."
}
```

This allows retrieval and generation to be evaluated independently.

### Retrieval evaluation

```text
Query
 ↓
Retriever
 ↓
Retrieved Documents
 ↓
Compare against expected documents
```

### Generation evaluation

```text
Question
+
Retrieved Context
 ↓
LLM
 ↓
Generated Answer
 ↓
Compare against ground truth/context
```

---

# 🔐 Environment Variables

Create a `.env` file containing:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=learnrag
PINECONE_NAMESPACE=learnrag

EMBEDDING_MODEL=BAAI/bge-m3

RERANKER_MODEL=BAAI/bge-reranker-v2-m3

OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:8b

DENSE_TOP_K=20
BM25_TOP_K=20
HYBRID_TOP_K=20
RERANK_TOP_K=5

CHUNK_SIZE=700
CHUNK_OVERLAP=100

APP_NAME=LearnRAG
ENVIRONMENT=development
```

Never commit the actual `.env` file to GitHub.

---

# 🚀 Planned API

The final application will expose endpoints such as:

```text
POST /api/query
```

Submit a question.

```text
POST /api/ingest
```

Ingest documents.

```text
POST /api/evaluate
```

Run evaluation.

```text
GET /api/health
```

Check application status.

---

# 📊 Planned Experiments

The project will investigate whether increasingly sophisticated retrieval strategies improve RAG performance.

### Experiment 1

```text
Dense Retrieval
```

### Experiment 2

```text
BM25
```

### Experiment 3

```text
Dense + BM25
```

### Experiment 4

```text
Dense + BM25 + RRF
```

### Experiment 5

```text
Dense + BM25 + RRF + Cross-Encoder
```

The final README will report the **actual measured results** rather than predetermined performance claims.

Example table structure:

| Retrieval Method  | Recall@5 | MRR | nDCG |
| ----------------- | -------: | --: | ---: |
| BM25              |      TBD | TBD |  TBD |
| Dense             |      TBD | TBD |  TBD |
| Hybrid            |      TBD | TBD |  TBD |
| Hybrid + Reranker |      TBD | TBD |  TBD |

---

# 🎓 Learning Goals

By completing this project, the following concepts should be understood:

* Document ingestion
* PDF parsing
* Text preprocessing
* Chunking strategies
* Embeddings
* Semantic search
* BM25
* Sparse retrieval
* Dense retrieval
* Hybrid retrieval
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Prompt engineering for RAG
* Context construction
* Hallucination reduction
* Source attribution
* Vector databases
* LLM inference
* API development
* RAG evaluation
* Dockerization

---

# 📌 Current Status

**Project Status: 🚧 In Development**

Current stage:

```text
[✓] Architecture finalized
[✓] Project structure created
[✓] Environment configuration planned
[ ] Dataset ingestion
[ ] Document cleaning
[ ] Chunking
[ ] Dense embeddings
[ ] Pinecone indexing
[ ] Dense retrieval
[ ] BM25 retrieval
[ ] Hybrid retrieval
[ ] RRF
[ ] Cross-encoder reranking
[ ] Qwen integration
[ ] Structured API
[ ] Streamlit interface
[ ] Evaluation
[ ] Dockerization
[ ] Deployment
```

---

# ✅ TODO

## Phase 1 — Setup

* [ ] Download the Kaggle textbook dataset
* [ ] Place PDFs inside `data/raw/`
* [ ] Configure `.env`
* [ ] Create Pinecone account and API key
* [ ] Create Pinecone index
* [ ] Install and configure Ollama
* [ ] Download the selected Qwen model
* [ ] Verify the Python environment

## Phase 2 — Document Processing

* [ ] Implement `loader.py`
* [ ] Implement `cleaner.py`
* [ ] Implement `chunker.py`
* [ ] Inspect extracted document quality
* [ ] Add document metadata
* [ ] Test different chunk sizes

## Phase 3 — Dense Retrieval

* [ ] Implement BGE-M3 embedding generation
* [ ] Connect embeddings to Pinecone
* [ ] Implement document indexing
* [ ] Implement dense retrieval
* [ ] Test Top-K retrieval
* [ ] Inspect retrieved chunks manually

## Phase 4 — Sparse Retrieval

* [ ] Implement BM25
* [ ] Build BM25 index
* [ ] Implement keyword retrieval
* [ ] Compare BM25 with dense retrieval

## Phase 5 — Hybrid Retrieval

* [ ] Implement dense + BM25 retrieval
* [ ] Implement Reciprocal Rank Fusion
* [ ] Tune retrieval parameters
* [ ] Compare dense vs sparse vs hybrid retrieval

## Phase 6 — Reranking

* [ ] Integrate BGE reranker
* [ ] Rerank hybrid candidates
* [ ] Compare retrieval before/after reranking
* [ ] Tune candidate count and final Top-K

## Phase 7 — Generation

* [ ] Configure Ollama
* [ ] Integrate Qwen
* [ ] Create RAG prompts
* [ ] Build context construction
* [ ] Implement grounded answer generation
* [ ] Implement source citations
* [ ] Add insufficient-context handling

## Phase 8 — API

* [ ] Create FastAPI application
* [ ] Create `/api/query`
* [ ] Create `/api/ingest`
* [ ] Create `/api/evaluate`
* [ ] Create `/api/health`
* [ ] Add Pydantic schemas
* [ ] Test API endpoints

## Phase 9 — Frontend

* [ ] Build Streamlit interface
* [ ] Add question input
* [ ] Display generated answers
* [ ] Display source citations
* [ ] Display retrieval information
* [ ] Add document/metadata filters

## Phase 10 — Evaluation

* [ ] Create evaluation questions
* [ ] Create ground-truth answers
* [ ] Calculate Recall@K
* [ ] Calculate MRR
* [ ] Calculate nDCG
* [ ] Evaluate context precision
* [ ] Evaluate context recall
* [ ] Evaluate faithfulness
* [ ] Evaluate answer relevancy
* [ ] Compare all retrieval strategies
* [ ] Add results to README

## Phase 11 — Productionization

* [ ] Add unit tests
* [ ] Add integration tests
* [ ] Improve error handling
* [ ] Add logging
* [ ] Add configuration management
* [ ] Create Dockerfile
* [ ] Create docker-compose configuration
* [ ] Optimize inference/retrieval latency
* [ ] Document setup instructions

## Phase 12 — Finalization

* [ ] Clean project structure
* [ ] Remove unused code
* [ ] Update README
* [ ] Add architecture diagram
* [ ] Add evaluation results
* [ ] Add screenshots
* [ ] Add API documentation
* [ ] Create project demo
* [ ] Push final version to GitHub
* [ ] Prepare resume description
* [ ] Prepare project explanation for interviews

---

# 📄 License

This project is intended for educational and portfolio purposes.

The underlying dataset is subject to its own Kaggle licensing and usage terms. Users should review the dataset's license before redistribution or commercial use.
