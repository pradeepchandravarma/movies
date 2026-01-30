import os
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

def get_langfuse():
    """
    Returns a Langfuse client (v3) or None if not configured.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        return None

    # v3 client factory
    return get_client(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
