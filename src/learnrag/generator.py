import os
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from vectordb import similarity_search, load_vector_store


vector_store = load_vector_store()

rag_template = """\
Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

User's Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

chat_model = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],  
)


def retrieve_context(question: str) -> str:
    """Fetch top-k chunks for the question and join them into one context string."""
    docs = similarity_search(vector_store, query=question, k=3)
    return "\n\n".join(doc.page_content for doc in docs)


semantic_rag_chain = (
    {"context": retrieve_context, "question": RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

answer = semantic_rag_chain.invoke("Describe the use of Langchain in building a RAG application.")
print(answer)