from rag.data_loader import load_and_prepare_data
from rag.document_builder import build_documents
from rag.vectorstore import get_vectorstore
from rag.retriever import get_retriever
from rag.rag_chain import build_rag_chain

def main():
    df = load_and_prepare_data()
    documents = build_documents(df)

    vectorstore = get_vectorstore(documents=documents)
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)

    while True:
        q = input("\nAsk a movie question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print(rag_chain.invoke(q).content)

if __name__ == "__main__":
    main()
