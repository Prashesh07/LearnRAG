"""
Evaluation test set for the LearnRAG research assistant.

Each entry pairs a realistic research question with the arXiv ID(s) of paper(s)
you know discuss that topic (source-level relevance judgment). This is a
practical stand-in for full chunk-level relevance labeling, which would be too
slow to hand-build across a 200-paper corpus.

HOW TO FILL THIS IN:
1. Pick ~15-20 questions covering different angles of your corpus topic.
2. For each, search your ./docs/research_papers folder (or your own memory of
   what you downloaded) for 1-3 papers you're confident actually address it.
3. Use the arXiv ID (the filename without .pdf) as the source identifier.

You don't need perfect labels — even "I'm fairly confident this paper is
relevant" is good enough for a directional evaluation like this.
"""

EVAL_QUESTIONS = [
    {
        "question": "What are common benchmarks used to evaluate LLM general knowledge and reasoning?",
        "expected_sources": ["2502.10709v2"],  # HELM/MMLU discussion, adjust to your actual corpus
    },
    {
        "question": "How does psychometric testing apply to LLM evaluation?",
        "expected_sources": ["2511.04689v3"],
    },
    {
        "question": "What inconsistencies exist in multiple-choice question evaluation for LLMs?",
        "expected_sources": ["2503.14996v2"],
    },
    {
        "question": "What alternatives exist to static benchmark evaluation for LLMs?",
        "expected_sources": ["2511.04689v3"],
    },
    {
        "question": "How are LLMs evaluated in domain-specific or biomedical contexts?",
        "expected_sources": ["2404.09135v1"],
    },
    
    # Add 10-15 more covering different angles of your corpus:
    # - metric-focused questions (BLEU, ROUGE, human eval)
    # - methodology questions (adaptive testing, benchmark design)
    # - critique/limitation questions (what's wrong with current eval methods)
    # - comparison questions (framework X vs framework Y)
]