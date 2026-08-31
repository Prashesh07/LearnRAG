import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

from document_loader import get_documents
from chunker import split_chunks
from vectordb import build_vector_store, load_vector_store
from retriever import build_hybrid_retriever, hybrid_search
from reranker import rerank




docs = get_documents("./docs/research_papers")
chunks = split_chunks(docs)

vector_store = load_vector_store() if Path("./chroma_db").exists() else build_vector_store(chunks)

hybrid_retriever = build_hybrid_retriever(chunks, vector_store, dense_k=10, sparse_k=10)

rag_template = """\
Answer the user's query using ONLY the information in the context below.
Do not use any outside knowledge, even if you know more about the topic.
Do not include code examples unless the code appears verbatim in the context.
If the context does not contain enough information to answer, respond with exactly: "I don't know."

User's Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

chat_model = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
)


def retrieve_context(question: str) -> str:
    """
    Two-stage retrieval:
      1. Hybrid retrieve (dense + sparse via RRF) -> a broad candidate set (recall).
      2. Cross-encoder rerank -> narrow down to the most truly relevant chunks (precision).
    Returns the final chunks joined into a single context string for the prompt.
    """
    candidates = hybrid_search(hybrid_retriever, question)
    top_chunks = rerank(question, candidates, top_k=3)
    return "\n\n".join(doc.page_content for doc in top_chunks)


rag_chain = (
    {"context": retrieve_context, "question": RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)


if __name__ == "__main__":
    question = "Describe ways to evaluate the performance of large language models (LLMs)."
    context = retrieve_context(question)
    print("=== RETRIEVED CONTEXT ===")
    print(context)
    print("\n=== ANSWER ===")
    print(rag_chain.invoke(question))

    