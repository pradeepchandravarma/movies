from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rag.vectorstore import get_vectorstore

def build_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template("""
You are a movie recommendation assistant.
Use the context below to answer the question.
If you cannot find the answer, say "I don't know".

Context:
{context}

Question:
{question}
""")

    llm = ChatOpenAI(model="gpt-4.1-mini")

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
