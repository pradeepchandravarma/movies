from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rag.config import *

def build_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_template("""
You are a movie recommendation assistant.
Use the context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")

    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL)

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
