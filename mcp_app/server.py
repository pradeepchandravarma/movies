from mcp.server.fastmcp import FastMCP
from mcp_app.tools import recommend_movies

mcp = FastMCP("MoviesRAG")
mcp.tool()(recommend_movies)

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        
    )
