from dotenv import load_dotenv
load_dotenv()

import asyncio
import streamlit as st

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="🎬 Movie Chatbot", layout="centered")
st.header("🎬 Movie Recommendation Chatbot")


# -------------------------------------------------------
# HARD GUARD: movie-only enforcement (SAME AS client.py)
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


# -------------------------------------------------------
# Cache the agent (important!)
# -------------------------------------------------------
@st.cache_resource
def get_agent():
    client = MultiServerMCPClient(
        {
            "movies": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable-http",
            }
        }
    )

    tools = asyncio.run(client.get_tools())

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    return create_react_agent(model, tools)


agent = get_agent()


# -------------------------------------------------------
# Chat history state
# -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------------
# Render chat history
# -------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -------------------------------------------------------
# User input
# -------------------------------------------------------
user_input = st.chat_input("Ask me about movies...")

if user_input:
    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").markdown(user_input)

    # 🔒 HARD BLOCK (THIS WAS MISSING)
    if not is_movie_question(user_input):
        bot_reply = "I can only answer movie-related questions."

    else:
        with st.spinner("Thinking..."):
            response = asyncio.run(
                agent.ainvoke(
                    {
                        # ✅ PASS FULL CHAT HISTORY
                        "messages": st.session_state.messages
                    }
                )
            )
            bot_reply = response["messages"][-1].content

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )
    st.chat_message("assistant").markdown(bot_reply)
