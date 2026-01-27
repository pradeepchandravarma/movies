import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

load_dotenv()


# -------------------------------------------------------
# HARD GUARD: movie-only enforcement
# -------------------------------------------------------
def is_movie_question(text: str) -> bool:
    movie_keywords = [
        "movie", "movies", "film", "cinema",
        "actor", "actress", "director",
        "rating", "ratings", "genre",
        "sci-fi", "science fiction",
        "action", "romantic", "romance",
        "thriller", "comedy", "drama",
    ]
    text = text.lower()
    return any(keyword in text for keyword in movie_keywords)


async def main():
    # -------------------------------------------------------
    # MCP CLIENT (HTTP)
    # -------------------------------------------------------
    client = MultiServerMCPClient(
        {
            "movies": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable-http",
            }
        }
    )

    # -------------------------------------------------------
    # LOAD MCP TOOLS
    # -------------------------------------------------------
    tools = await client.get_tools()
    print("Loaded tools:", [t.name for t in tools])

    # -------------------------------------------------------
    # OPENAI MODEL
    # -------------------------------------------------------
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )

    # -------------------------------------------------------
    # AGENT (no state_modifier, no prompt hacks)
    # -------------------------------------------------------
    agent = create_agent(
        model=model,
        tools=tools,
    )

    # -------------------------------------------------------
    # TEST QUERIES
    # -------------------------------------------------------
    queries = [
        "Who is Elon Musk?",
        "Recommend a good sci-fi movie",
        "Suggest romantic movies with high ratings",
        "What is artificial intelligence?",
        "What are some low-rated action movies?",
    ]

    for q in queries:
        print("\nQ:", q)

        # 🔒 HARD BLOCK (this is the real enforcement)
        if not is_movie_question(q):
            print("A: I can only answer movie-related questions.")
            continue

        response = await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": q}
                ]
            }
        )

        print("A:", response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
