from mcp.server.fastmcp import FastMCP

from rag.vectorstore import get_vectorstore
from rag.retriever import get_retriever
from rag.rag_chain import build_rag_chain

# ✅ Read-only: connect to existing index (no embeddings, no upserts)
vectorstore = get_vectorstore(documents=None)
retriever = get_retriever(vectorstore)
rag_chain = build_rag_chain(retriever)

mcp = FastMCP("movies-rag-server")

@mcp.tool()
def recommend_movies(question: str) -> str:
    print("📞 MCP TOOL CALLED with question:", question)
    return rag_chain.invoke(question).content


@mcp.tool()
def health() -> str:
    return "ok"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
            
