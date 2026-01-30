from langfuse import observe, get_client
from rag.rag_chain import build_rag_chain
from langfuse import observe

_rag_chain = build_rag_chain()

@observe(
    name="recommend_movies",
    capture_input=True,
    capture_output=True,
)
async def recommend_movies(query: str) -> str:
    response = _rag_chain.invoke(query)
    return response.content
