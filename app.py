"""
Streamlit – CPU-friendly RAG chat
---------------------------------
• Phi-3 mini via llama-cpp
• Retrieval = FAISS + BGE-small embeddings
• Strict “I don’t know” fallback
"""

from __future__ import annotations
import sys
import pathlib
import streamlit as st

# ─── make ./src importable no matter where Streamlit is launched ────────────
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))     # → retriever.py, generator.py

from retriever import Retriever            # local module in src/
from generator import generate             # updated wrapper in src/

# ─── page setup ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Document-Grounded Chat", layout="wide")
st.title("📄🧠 Document-Grounded Chat")

# chat history lives in session_state
if "chat" not in st.session_state:
    st.session_state.chat = []            # list[dict(role,text)]

# sidebar status & actions
retriever = Retriever()                   # loads FAISS & encoder
st.sidebar.success("Model & index loaded ✔️")
st.sidebar.write("**Model** : Phi-3-mini-q4")
st.sidebar.write(f"**Chunks indexed** : {retriever._index.ntotal}")
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.chat.clear()
    st.experimental_rerun()

# ─── render previous turns ──────────────────────────────────────────────────
for turn in st.session_state.chat:
    st.chat_message(turn["role"]).markdown(
        turn["text"], unsafe_allow_html=False
    )

# ─── user input ─────────────────────────────────────────────────────────────
query = st.chat_input("Ask about the document…")


def send_assistant(msg: str):
    st.chat_message("assistant").markdown(msg, unsafe_allow_html=False)
    st.session_state.chat.append({"role": "assistant", "text": msg})


if query:
    # record + echo user turn
    st.session_state.chat.append({"role": "user", "text": query})
    st.chat_message("user").markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # ── retrieve context ────────────────────────────────────────────────
        hits = retriever.search(query)
        if not hits:
            send_assistant(
                "I don’t know — nothing in the document set matches that question."
            )
            st.stop()

        context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))

        # ── stream LLM ──────────────────────────────────────────────────────
        answer, buffer = "", []
        for tok in generate(context, query, stream=True):
            buffer.append(tok)
            # flush every ~30 tokens or on newline
            if len(buffer) >= 30 or "\n" in buffer[-1]:
                answer += "".join(buffer)
                placeholder.markdown(answer + " ▌", unsafe_allow_html=False)
                buffer = []

        answer += "".join(buffer)

        # ── hide any residual prompt fragments just in case ────────────────
        answer = (
            answer.split("Instruction", 1)[0]
            .split("instruction", 1)[0]
            .strip()
        )

        placeholder.markdown(answer, unsafe_allow_html=False)  # final flush

        # ── show sources ────────────────────────────────────────────────────
        st.markdown("###### Sources used")
        for i, h in enumerate(hits, 1):
            excerpt = (
                (h["text"][:80] + "…") if len(h["text"]) > 80 else h["text"]
            )
            st.markdown(f"- **[{i}]** p{h['page']} • *{excerpt}*")

        # log assistant turn
        st.session_state.chat.append({"role": "assistant", "text": answer})
