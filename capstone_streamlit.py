import uuid
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agent import (
    DOCUMENTS, DOMAIN_NAME, DOMAIN_DESCRIPTION,
    build_knowledge_base, build_graph,
)

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Load env variables
load_dotenv()

# Page config
st.set_page_config(
    page_title=DOMAIN_NAME,
    page_icon="⚛️",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────
# LOAD AGENT (CACHED)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_agent():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    embedder, collection = build_knowledge_base()
    app = build_graph(llm, embedder, collection)

    return app, collection


try:
    app, collection = load_agent()
except Exception as e:
    st.error(f"❌ Failed to load agent: {e}")
    st.info("👉 Make sure GROQ_API_KEY is set and agent.py is correct.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "student_name" not in st.session_state:
    st.session_state.student_name = ""


# ─────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────

def get_kb_size(collection):
    try:
        return collection.count()
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(f"⚛️ {DOMAIN_NAME}")
    st.divider()

    st.write(DOMAIN_DESCRIPTION)

    if st.session_state.student_name:
        st.success(f"👋 Hi, {st.session_state.student_name}!")

    st.divider()
    st.write(f"**Session ID:** `{st.session_state.thread_id[:8]}`")

    st.write(f"**KB size:** {get_kb_size(collection)} documents")

    st.divider()
    st.write("**📚 Topics covered:**")
    for doc in DOCUMENTS:
        st.write(f"• {doc['topic']}")

    st.divider()
    st.write("**🔢 Calculator tip:**")
    st.caption(
        "Use: `calculate <expression>`\n\n"
        "Constants: g, G, h, c, k, R, NA, pi\n"
        "Functions: sin, cos, tan, sqrt, log, ln"
    )

    st.divider()

    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.student_name = ""
        st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────

st.title(f"⚛️ {DOMAIN_NAME}")
st.caption(DOMAIN_DESCRIPTION)

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hi! I'm your **Physics Study Buddy**.\n\n"
            "**I can help with:**\n"
            "- 📘 Concepts & laws\n"
            "- 📐 Formulas (with definitions)\n"
            "- 🔢 Numerical problems step-by-step\n\n"
            "👉 Tell me your name to personalize the session!"
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask a physics question or calculate something..."):

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }

            # ✅ FIXED STATE STRUCTURE
            result = app.invoke(
                {
                    "question": prompt,
                    "messages": st.session_state.messages,
                    "student_name": st.session_state.student_name,
                    "eval_retries": 0
                },
                config=config,
            )

        answer = result.get("answer", "❌ No response generated.")
        faith = result.get("faithfulness", 0.0)
        sources = result.get("sources", [])
        route = result.get("route", "")
        student_name = result.get("student_name", "")

        if student_name:
            st.session_state.student_name = student_name

        st.markdown(answer)

        route_icon = {
            "retrieve": "📚",
            "tool": "🔢",
            "memory_only": "💬",
        }.get(route, "❓")

        meta = [f"{route_icon} **{route}**"]

        if route == "retrieve":
            meta.append(
                f"Faithfulness: {faith:.2f} {'✅' if faith >= 0.7 else '⚠️'}"
            )

        if sources:
            meta.append(f"Sources: {', '.join(sources)}")

        st.caption(" | ".join(meta))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })