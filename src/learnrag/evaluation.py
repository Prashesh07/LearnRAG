"""
Evaluation harness for the LearnRAG research assistant.

Measures two things:
  1. RETRIEVAL QUALITY — compares dense-only, hybrid, and hybrid+reranked
     retrieval using Hit Rate@k (multiple cutoffs) and Mean Reciprocal Rank.
     Each variant retrieves from the same candidate pool, so comparisons are fair.
     Runs by default and makes NO LLM calls.
  2. GENERATION FAITHFULNESS (OPT-IN) — uses an LLM-as-judge to score whether
     the final generated answer is actually grounded in the retrieved context
     (catches hallucination). Skipped unless run with:
        python evaluation.py --with-judge [--judge-runs N]

Run from src/learnrag/:  python evaluation.py
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from eval_dataset import EVAL_QUESTIONS
from generator import build_rag_chain, get_pipeline, retrieve_context
from reranker import rerank
from retriever import hybrid_search

load_dotenv()


# Source id helpers


def normalize_arxiv_id(raw_id: str) -> str:
    """Strip an arXiv version suffix (e.g. '2502.10709v2' -> '2502.10709')."""
    return re.sub(r"v\d+$", "", raw_id)


def extract_source_id(doc: Document) -> str:
    """Pull the arXiv ID out of a chunk's metadata source path, ignoring version."""
    source_path = doc.metadata.get("source", "")
    return normalize_arxiv_id(Path(source_path).stem)


# Retrieval metrics


def hit_rate_at_k(results: list[Document], expected_sources: list[str], k: int) -> bool:
    """True if any of the top-k results comes from an expected source paper."""
    top_k_sources = {extract_source_id(doc) for doc in results[:k]}
    expected = {normalize_arxiv_id(s) for s in expected_sources}
    return bool(top_k_sources & expected)


def reciprocal_rank(results: list[Document], expected_sources: list[str]) -> float:
    """1/rank of the first relevant result found; 0 if none found."""
    expected = {normalize_arxiv_id(s) for s in expected_sources}
    for rank_i, doc in enumerate(results, start=1):
        if extract_source_id(doc) in expected:
            return 1.0 / rank_i
    return 0.0


def evaluate_retriever(
    name: str,
    retrieve_fn,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    dataset: list[dict] | None = None,
) -> dict:
    """
    Run every eval question through a retrieval function and compute
    Hit Rate@k (for each k in k_values) and MRR across the whole test set.

    Args:
        name: Label for this pipeline variant (for printing).
        retrieve_fn: A function(question: str) -> list[Document].
        k_values: Cutoffs to report Hit Rate@k for.
        dataset: Test set (defaults to EVAL_QUESTIONS).

    Returns:
        {"aggregate": {...metrics...}, "per_question": [...]}.
    """
    dataset = dataset or EVAL_QUESTIONS
    num_questions = len(dataset)

    hit_by_k = {k: [] for k in k_values}
    mrr_scores = []
    per_question = []

    for item in dataset:
        results = retrieve_fn(item["question"])

        for k in k_values:
            hit_by_k[k].append(hit_rate_at_k(results, item["expected_sources"], k))

        rr = reciprocal_rank(results, item["expected_sources"])
        mrr_scores.append(rr)

        per_question.append(
            {
                "question": item["question"],
                "expected_sources": item["expected_sources"],
                "hit": {k: hit_by_k[k][-1] for k in k_values},
                "mrr": rr,
            }
        )

    aggregate = {"name": name, "num_questions": num_questions}
    for k in k_values:
        aggregate[f"hit_rate@{k}"] = mean(hit_by_k[k]) if hit_by_k[k] else 0.0
    aggregate["mrr"] = mean(mrr_scores) if mrr_scores else 0.0

    return {"aggregate": aggregate, "per_question": per_question}


def print_summary_table(results_list: list[dict], k_values: tuple[int, ...]) -> None:
    """Print the Hit Rate@k + MRR table for a set of retrieval variant results."""
    header = f"{'Variant':<30}" + "".join(f"{f'Hit@{k}':>8}" for k in k_values) + f"{'MRR':>8}"
    print(header)
    print("-" * len(header))

    for r in results_list:
        agg = r["aggregate"]
        row = f"{agg['name']:<30}"
        for k in k_values:
            row += f"{agg[f'hit_rate@{k}']:>8.1%}"
        row += f"{agg['mrr']:>8.3f}"
        print(row)


def print_per_question_table(results_list: list[dict], k: int) -> None:
    """Print a per-question hit/miss breakdown for each variant at cutoff k."""
    rows = results_list[0]["per_question"]
    header = f"{'#':<4}{'Question':<52}{'Dense':>7}{'Hybrid':>8}{'Hybrid+CE':>10}{'Expected':>14}"
    print(header)
    print("-" * len(header))

    for variant in results_list:
        assert len(variant["per_question"]) == len(rows), (
            "variants must share the same per-question length"
        )

    for i, row in enumerate(rows, start=1):
        question = row["question"]
        q_trunc = question[:49] + ("..." if len(question) > 49 else "")
        expected = row["expected_sources"][0]
        mark = lambda ok: "Y" if ok else "N"
        print(
            f"{i:<4}{q_trunc:<52}"
            f"{mark(results_list[0]['per_question'][i - 1]['hit'][k]):>7}"
            f"{mark(results_list[1]['per_question'][i - 1]['hit'][k]):>8}"
            f"{mark(results_list[2]['per_question'][i - 1]['hit'][k]):>10}"
            f"{expected:>14}"
        )


# Generation faithfulness (LLM-as-judge)


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


def judge_faithfulness_averaged(
    judge_model, question: str, context: str, answer: str, n_runs: int = 1
) -> float:
    """Average several judge runs to reduce LLM stochasticity."""
    if n_runs <= 1:
        return float(judge_faithfulness(judge_model, question, context, answer))
    return mean(
        judge_faithfulness(judge_model, question, context, answer)
        for _ in range(n_runs)
    )


def evaluate_generation_faithfulness(
    judge_model, n_runs: int = 1, dataset: list[dict] | None = None
) -> dict:
    """
    Run each eval question through the full RAG pipeline and score faithfulness.

    Retrieval runs ONCE per question: the same retrieved context is used both to
    generate the answer and as the grounding the judge compares against.
    """
    dataset = dataset or EVAL_QUESTIONS
    per_question = []

    for item in dataset:
        question = item["question"]
        context = retrieve_context(question, verbose=False)
        answer = build_rag_chain().invoke({"context": context, "question": question})
        score = judge_faithfulness_averaged(judge_model, question, context, answer, n_runs=n_runs)
        per_question.append({"question": question, "score": score})
        print(f"  [{score:.2f}/5] {question[:70]}")

    scores = [pq["score"] for pq in per_question]
    avg_score = mean(scores) if scores else 0.0
    print(f"\nAverage faithfulness score: {avg_score:.2f}/5 "
          f"({n_runs} judge runs per question)")
    return {"aggregate": {"avg_score": avg_score}, "per_question": per_question}


def save_results(
    results_dense,
    results_hybrid,
    results_reranked,
    faithfulness,
    config,
    out_dir: str = "./eval_results",
) -> Path:
    """Persist the run's results to eval_results/eval_<timestamp>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": timestamp,
        "config": config,
        "retrieval": {
            "dense": results_dense,
            "hybrid": results_hybrid,
            "hybrid_reranked": results_reranked,
        },
    }
    if faithfulness is not None:
        payload["faithfulness"] = faithfulness

    out_path = out_dir / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# Main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LearnRAG evaluation harness")
    parser.add_argument(
        "--with-judge",
        action="store_true",
        help="also run the LLM-as-judge faithfulness pass (slow, uses Groq tokens)",
    )
    parser.add_argument(
        "--judge-runs",
        type=int,
        default=1,
        help="judge passes per question when --with-judge is set (default: 1)",
    )
    args = parser.parse_args()

    K_VALUES = (1, 3, 5, 10)
    RETRIEVAL_DEPTH = 20  # every variant retrieves from the same deep candidate pool
    RERANK_TOP_K = 3      # matches production (generator.RERANK_TOP_K)

    _, vector_store, hybrid_retriever = get_pipeline()

    print("=" * 64)
    print("RETRIEVAL EVALUATION")
    print("=" * 64)

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_DEPTH})

    results_dense = evaluate_retriever(
        "Dense-only",
        lambda q: dense_retriever.invoke(q),
        k_values=K_VALUES,
    )

    results_hybrid = evaluate_retriever(
        "Hybrid (dense + BM25, RRF)",
        lambda q: hybrid_search(hybrid_retriever, q, verbose=False),
        k_values=K_VALUES,
    )

    results_reranked = evaluate_retriever(
        "Hybrid + Cross-Encoder",
        lambda q: rerank(
            q,
            hybrid_search(hybrid_retriever, q, verbose=False),
            top_k=RERANK_TOP_K,
            verbose=False,
        ),
        k_values=K_VALUES,
    )

    print("\n" + "=" * 64)
    print("SUMMARY - Hit Rate@k / MRR")
    print("=" * 64)
    print_summary_table([results_dense, results_hybrid, results_reranked], K_VALUES)
    print(f"\nNote: 'Hybrid + Cross-Encoder' returns only top-{RERANK_TOP_K} chunks "
          f"(production setting), so its Hit Rate plateaus after k={RERANK_TOP_K}.")

    print("\n" + "=" * 64)
    print("PER-QUESTION BREAKDOWN (Hit@5)")
    print("=" * 64)
    print_per_question_table([results_dense, results_hybrid, results_reranked], k=5)

    # Generation faithfulness (LLM-as-judge) — opt-in
    faithfulness = None
    if args.with_judge:
        print("\n" + "=" * 64)
        print("GENERATION FAITHFULNESS (LLM-as-judge)")
        print("=" * 64)

        judge_model = ChatGroq(
            temperature=0,
            model_name="openai/gpt-oss-120b",
            api_key=os.environ["GROQ_API_KEY"],
        )

        faithfulness = evaluate_generation_faithfulness(
            judge_model, n_runs=args.judge_runs
        )
    else:
        print("\n" + "=" * 64)
        print("Skipping GENERATION FAITHFULNESS (LLM-as-judge).")
        print("Re-run with --with-judge to include it.")
        print("=" * 64)

    config = {
        "k_values": list(K_VALUES),
        "retrieval_depth": RETRIEVAL_DEPTH,
        "rerank_top_k": RERANK_TOP_K,
        "num_questions": len(EVAL_QUESTIONS),
        "with_judge": args.with_judge,
        "judge_runs": args.judge_runs,
    }
    out_path = save_results(
        results_dense, results_hybrid, results_reranked, faithfulness, config
    )
    print(f"\nSaved results to {out_path}")