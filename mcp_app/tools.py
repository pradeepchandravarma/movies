from rag.rag_chain import build_rag_chain

_rag_chain = build_rag_chain()

async def recommend_movies(query: str) -> str:
    response = _rag_chain.invoke(query)
    return response.content
