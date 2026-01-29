from dotenv import load_dotenv
from rag.documents import build_documents
from rag.vectorstore import get_vectorstore

load_dotenv()

docs = build_documents(
    "rag/data/movies.csv",
    "rag/data/ratings.csv",
)

get_vectorstore(create=True, documents=docs)
print("✅ Pinecone ingestion completed")
