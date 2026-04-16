"""
capstone_streamlit.py — Physics Study Buddy
============================================
Run:  streamlit run capstone_streamlit.py

All agent logic lives in agent.py. This file is UI only:
  - @st.cache_resource loads the agent once via agent.build_graph()
  - st.session_state stores messages and thread_id
  - chat_input / chat_message handle the conversation loop
"""

import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent import (
    DOCUMENTS, DOMAIN_NAME, DOMAIN_DESCRIPTION,
    build_knowledge_base, build_graph,
)

load_dotenv()

st.set_page_config(page_title=f"{DOMAIN_NAME}", page_icon="⚛️", layout="centered")


@st.cache_resource
def load_agent():
    llm               = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder, collection = build_knowledge_base()
    app               = build_graph(llm, embedder, collection)
    return app, collection


try:
    app, collection = load_agent()
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.info("Ensure GROQ_API_KEY is set in .env and agent.py is in the same folder.")
    st.stop()


if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "thread_id"    not in st.session_state:
    st.session_state.thread_id    = str(uuid.uuid4())[:8]
if "student_name" not in st.session_state:
    st.session_state.student_name = ""


with st.sidebar:
    st.header(f"⚛️ {DOMAIN_NAME}")
    st.divider()
    st.write(DOMAIN_DESCRIPTION)

    if st.session_state.student_name:
        st.success(f"👋 Hi, {st.session_state.student_name}!")

    st.divider()
    st.write(f"**Session:** `{st.session_state.thread_id}`")
    st.write(f"**KB size:** {collection.count()} documents")

    st.divider()
    st.write("**📚 Topics covered:**")
    for doc in DOCUMENTS:
        st.write(f"• {doc['topic']}")

    st.divider()
    st.write("**🔢 Calculator tip:**")
    st.caption(
        "Type *calculate* + an expression. "
        "Constants: `g`, `G`, `h`, `c`, `k`, `R`, `NA`, `pi`. "
        "Functions: `sin`, `cos`, `tan`, `sqrt`, `log`, `ln`."
    )

    st.divider()
    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.thread_id    = str(uuid.uuid4())[:8]
        st.session_state.student_name = ""
        st.rerun()


st.title(f"⚛️ {DOMAIN_NAME}")
st.caption(DOMAIN_DESCRIPTION)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm your **Physics Study Buddy**. Ask me about:\n\n"
            "- **Concepts & laws** — Newton, Faraday, Bohr, Einstein…\n"
            "- **Formulas** — with every symbol defined\n"
            "- **Numericals** — solved step-by-step with the built-in calculator\n\n"
            "Tell me your name and I'll personalise our session!"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a physics question or type a calculation..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = app.invoke(
                {"question": prompt, "student_name": st.session_state.student_name},
                config=config,
            )
        answer       = result.get("answer", "Sorry, I could not generate a response.")
        faith        = result.get("faithfulness", 0.0)
        sources      = result.get("sources", [])
        route        = result.get("route", "")
        student_name = result.get("student_name", "")

        if student_name:
            st.session_state.student_name = student_name

        st.markdown(answer)

        route_icon = {"retrieve": "📚", "tool": "🔢", "memory_only": "💬"}.get(route, "❓")
        parts = [f"{route_icon} Route: **{route}**"]
        if faith > 0 and route == "retrieve":
            parts.append(f"Faithfulness: **{faith:.2f}** {'✅' if faith >= 0.7 else '⚠️'}")
        if sources:
            parts.append(f"Sources: *{', '.join(sources)}*")
        st.caption(" | ".join(parts))

    st.session_state.messages.append({"role": "assistant", "content": answer})
