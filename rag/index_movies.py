from rag.data_loader import load_and_prepare_data
from rag.document_builder import build_documents
from rag.vectorstore import get_vectorstore

def main():
    df = load_and_prepare_data()
    documents = build_documents(df)

    # This does the embed + upsert once
    _ = get_vectorstore(documents=documents)

    print("✅ Indexing complete (documents embedded + upserted to Pinecone).")

if __name__ == "__main__":
    main()
