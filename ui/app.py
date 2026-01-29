import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

load_dotenv()

st.title("🎬 MovieLens RAG (OpenAI + MCP)")

# ----------------------------
# Config (container-safe)
# ----------------------------
MCP_MOVIES_URL = os.getenv(
    "MCP_MOVIES_URL",
    "http://127.0.0.1:8000/mcp/",
)

# ----------------------------
# Session-safe agent creation
# ----------------------------
def init_agent():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    client = MultiServerMCPClient(
        {
            "movies": {
                "url": MCP_MOVIES_URL,
                "transport": "streamable-http",
            }
        }
    )

    tools = loop.run_until_complete(client.get_tools())

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
    )

    return agent, loop


# Initialize once per session
if "agent" not in st.session_state:
    st.session_state.agent, st.session_state.loop = init_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Render chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Chat input
# ----------------------------
if user_input := st.chat_input("Ask for movie recommendations"):
    # 1️⃣ Append + render user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2️⃣ Call agent with FULL history
    result = st.session_state.loop.run_until_complete(
        st.session_state.agent.ainvoke(
            {"messages": st.session_state.messages}
        )
    )

    # 3️⃣ Enforce tool-only answers
    used_tool = any(
        msg.type == "tool" for msg in result["messages"]
    )

    if not used_tool:
        assistant_reply = "I don't know based on the available tools."
    else:
        assistant_reply = result["messages"][-1].content

    # 4️⃣ Append + render assistant message immediately
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
