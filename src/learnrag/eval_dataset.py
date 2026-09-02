"""
Evaluation test set for the LearnRAG research assistant.

Each entry pairs a realistic research question with the arXiv ID(s) of paper(s)
known to discuss that topic (source-level relevance judgment). This is a
practical stand-in for full chunk-level relevance labeling, which would be too
slow to hand-build across a 200-paper corpus.

The current set has 20 questions: a hand-authored core plus questions generated
from the corpus's paper titles/abstracts (first-page text of each PDF), so each
expected source is guaranteed to exist in ./docs/research_papers.
"""

EVAL_QUESTIONS = [
    {
        "question": "What are common benchmarks used to evaluate LLM general knowledge and reasoning?",
        "expected_sources": ["2502.10709"],  # HELM/MMLU discussion
    },
    {
        "question": "How does psychometric testing apply to LLM evaluation?",
        "expected_sources": ["2511.04689"],
    },
    {
        "question": "What inconsistencies exist in multiple-choice question evaluation for LLMs?",
        "expected_sources": ["2503.14996"],
    },
    {
        "question": "What alternatives exist to static benchmark evaluation for LLMs?",
        "expected_sources": ["2511.04689"],
    },
    {
        "question": "How are LLMs evaluated in domain-specific or biomedical contexts?",
        "expected_sources": ["2404.09135"],
    },
    {
        "question": "What metrics does SciEval introduce to evaluate LLMs' scientific research abilities across Bloom's taxonomy levels, and how do they differ from traditional objective-only benchmarks?",
        "expected_sources": ["2308.13149"],
    },
    {
        "question": "How does Inference-Time Decontamination (ITD) detect and rewrite leaked benchmark samples, and what effect does it have on LLM performance scores compared to using the original contaminated benchmarks?",
        "expected_sources": ["2406.13990"],
    },
    {
        "question": "In what manner does the HD-EVAL framework align LLM-based evaluators with human preferences via hierarchical criteria decomposition, and what empirical results demonstrate its superiority over flat-prompt evaluation?",
        "expected_sources": ["2402.15754"],
    },
    {
        "question": "How does the Deep Interaction evaluation approach overcome the limitations of static-dataset assessments for real-world tasks such as machine translation and code generation, and what benefits does it report?",
        "expected_sources": ["2309.04369"],
    },
    {
        "question": "How does the Decomposed Requirements Following Ratio (DRFR) metric improve reliability compared to traditional scoring methods for evaluating instruction-following ability in LLMs?",
        "expected_sources": ["2401.03601"],
    },
    {
        "question": "What are the distinctive characteristics of the TR-MMLU benchmark for Turkish LLM evaluation, and how does it address the challenges of assessing models in a resource-limited language?",
        "expected_sources": ["2501.00593"],
    },
    {
        "question": "Which calibration and contrastive-training strategies are proposed to mitigate the bias of LLM-as-a-Judge systems toward superficial qualities such as verbosity and fluency?",
        "expected_sources": ["2409.16788"],
    },
    {
        "question": "According to the critical analysis of evaluation frameworks, what are the primary limitations of current large-language-model evaluation benchmarks, and how might these limitations impact cross-model performance comparisons?",
        "expected_sources": ["2407.21072"],
    },
    {
        "question": "How does the Beyond metric introduced in Mercury combine functional correctness and runtime efficiency, and how does it differ from traditional Pass metrics?",
        "expected_sources": ["2402.07844"],
    },
    {
        "question": "What limitations arise from assuming benchmark test prompts are a random sample of real-world distributions, and how do correlated prompts alter model rankings?",
        "expected_sources": ["2404.16966"],
    },
    {
        "question": "How does Evalverse unify disparate evaluation tools into a single framework, and which features enable users with limited AI expertise to perform comprehensive LLM assessments?",
        "expected_sources": ["2404.00943"],
    },
    {
        "question": "In what ways does AgentBoard's fine-grained progress rate metric provide deeper insight into multi-turn LLM agent performance compared to conventional final success-rate metrics?",
        "expected_sources": ["2401.13178"],
    },
    {
        "question": "What evaluation metrics are proposed for assessing the quality of long-form medical answers, and how do they address limitations of traditional multiple-choice or automatic metrics?",
        "expected_sources": ["2403.07872", "2411.09834"],
    },
    {
        "question": "How does the InsCoQA benchmark assess LLM performance on conversational question answering over instructional documents, and what novel evaluation paradigm does it embody compared to conventional benchmark-evaluation cycles?",
        "expected_sources": ["2407.07531", "2410.00526"],
    },
    {
        "question": "What methods are introduced for detecting and mitigating self-contradictory hallucinations in LLM outputs, and how effective are prompting-based detection frameworks in improving robustness?",
        "expected_sources": ["2407.07531", "2305.15852"],
    },
]