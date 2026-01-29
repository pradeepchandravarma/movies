import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX_NAME = "movies-rag"
NAMESPACE = "movielens"
DIMENSION = 1536

def get_vectorstore(create=False, documents=None):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if create and INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if documents:
        return PineconeVectorStore.from_documents(
            documents,
            embeddings,
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
        )

    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
    )
