import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4.1-mini"

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "movies-rag"
NAMESPACE = "movielens"
EMBEDDING_DIM = 1536
