"""
Evaluation harness for the LearnRAG research assistant.

Measures two things:
  1. RETRIEVAL QUALITY — compares dense-only, hybrid, and hybrid+reranked
     retrieval using Hit Rate@k and Mean Reciprocal Rank (MRR), based on
     whether retrieved chunks come from the expected source paper(s).
  2. GENERATION FAITHFULNESS — uses an LLM-as-judge to score whether the
     final generated answer is actually grounded in the retrieved context
     (catches hallucination, like we saw with the off-topic LangChain query).

Run this after your vector store / hybrid retriever / reranker are built.
"""

import os
import re
from pathlib import Path
from statistics import mean
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from document_loader import get_documents
from chunker import split_chunks
from vectordb import build_vector_store, load_vector_store
from retriever import build_hybrid_retriever, hybrid_search
from reranker import rerank
from eval_dataset import EVAL_QUESTIONS


def extract_source_id(doc: Document) -> str:
    """Pull the arXiv ID (filename stem) out of a chunk's metadata source path."""
    source_path = doc.metadata.get("source", "")
    return Path(source_path).stem


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def hit_rate_at_k(results: list[Document], expected_sources: list[str], k: int) -> bool:
    """True if any of the top-k results comes from an expected source paper."""
    top_k_sources = {extract_source_id(doc) for doc in results[:k]}
    return bool(top_k_sources & set(expected_sources))


def reciprocal_rank(results: list[Document], expected_sources: list[str]) -> float:
    """1/rank of the first relevant result found; 0 if none found."""
    for rank_i, doc in enumerate(results, start=1):
        if extract_source_id(doc) in expected_sources:
            return 1.0 / rank_i
    return 0.0


def evaluate_retriever(name: str, retrieve_fn, k: int = 5) -> dict:
    """
    Run every eval question through a retrieval function and compute
    Hit Rate@k and MRR across the whole test set.

    Args:
        name: Label for this pipeline variant (for printing).
        retrieve_fn: A function(question: str) -> list[Document].
        k: Cutoff for Hit Rate@k.
    """
    hits = []
    reciprocal_ranks = []

    for item in EVAL_QUESTIONS:
        results = retrieve_fn(item["question"])
        hits.append(hit_rate_at_k(results, item["expected_sources"], k))
        reciprocal_ranks.append(reciprocal_rank(results, item["expected_sources"]))

    hit_rate = mean(hits)
    mrr = mean(reciprocal_ranks)

    print(f"\n{name}")
    print(f"  Hit Rate@{k}: {hit_rate:.2%}")
    print(f"  MRR:         {mrr:.3f}")

    return {"name": name, "hit_rate": hit_rate, "mrr": mrr}


# ---------------------------------------------------------------------------
# Generation faithfulness (LLM-as-judge)
# ---------------------------------------------------------------------------

FAITHFULNESS_JUDGE_TEMPLATE = """\
You are evaluating whether an AI-generated answer is faithful to the context it was given.

Context:
{context}

Question:
{question}

Generated Answer:
{answer}

Score the answer's faithfulness to the context on a scale of 1-5:
5 = Fully grounded, every claim is supported by the context
3 = Partially grounded, some claims go beyond the context
1 = Largely fabricated, mostly not supported by the context

Respond with ONLY a single digit (1-5), nothing else.
"""


def judge_faithfulness(judge_model, question: str, context: str, answer: str) -> int:
    """Ask an LLM to score how well the answer sticks to the given context."""
    prompt = ChatPromptTemplate.from_template(FAITHFULNESS_JUDGE_TEMPLATE)
    chain = prompt | judge_model | StrOutputParser()

    result = chain.invoke({"question": question, "context": context, "answer": answer})
    match = re.search(r"[1-5]", result)
    return int(match.group()) if match else 0


def evaluate_generation_faithfulness(rag_chain, retrieve_context_fn, judge_model) -> float:
    """Run each eval question through the full RAG chain and score faithfulness."""
    scores = []

    for item in EVAL_QUESTIONS:
        question = item["question"]
        context = retrieve_context_fn(question)
        answer = rag_chain.invoke(question)
        score = judge_faithfulness(judge_model, question, context, answer)
        scores.append(score)
        print(f"  [{score}/5] {question[:70]}...")

    avg_score = mean(scores)
    print(f"\nAverage faithfulness score: {avg_score:.2f}/5")
    return avg_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    docs = get_documents("./docs/research_papers")
    chunks = split_chunks(docs)

    vector_store = load_vector_store() if Path("./chroma_db").exists() else build_vector_store(chunks)
    hybrid_retriever = build_hybrid_retriever(chunks, vector_store, dense_k=10, sparse_k=10)

    K = 5

    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    # Variant 1: dense-only
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": K})
    results_dense = evaluate_retriever(
        "Dense-only",
        lambda q: dense_retriever.invoke(q),
        k=K,
    )

    # Variant 2: hybrid (dense + sparse via RRF)
    results_hybrid = evaluate_retriever(
        "Hybrid (dense + BM25, RRF)",
        lambda q: hybrid_search(hybrid_retriever, q),
        k=K,
    )

    # Variant 3: hybrid + reranked
    results_reranked = evaluate_retriever(
        "Hybrid + Reranked",
        lambda q: rerank(q, hybrid_search(hybrid_retriever, q), top_k=K),
        k=K,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in [results_dense, results_hybrid, results_reranked]:
        print(f"{r['name']:<30} Hit Rate@{K}: {r['hit_rate']:.2%}   MRR: {r['mrr']:.3f}")

    # --- Generation faithfulness (optional, uses Groq — costs a few extra calls) ---
    print("\n" + "=" * 60)
    print("GENERATION FAITHFULNESS (LLM-as-judge)")
    print("=" * 60)

    judge_model = ChatGroq(
        temperature=0,
        model_name="openai/gpt-oss-120b",
        api_key=os.environ["GROQ_API_KEY"],
    )

    from generator import rag_chain, retrieve_context  # reuse your existing chain

    evaluate_generation_faithfulness(rag_chain, retrieve_context, judge_model)