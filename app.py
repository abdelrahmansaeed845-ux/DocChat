"""Streamlit interface for asking questions about uploaded documents."""

from __future__ import annotations

import hashlib
import re

import streamlit as st
from dotenv import load_dotenv

from agent import build_agent
from chunking import split_documents
from loaders import load_bytes
from vector_store import build_vector_store

load_dotenv()

TOP_K = 4
MAX_ANALYSTS = 2

SUGGESTIONS = [
    "Summarize these documents",
    "What are the key points?",
    "Which topics do they cover?",
]

# ---------- remove internal chunk references from answers ----------
_CHUNK_IN_PARENS = re.compile(r"\s*\([^()]*chunk_[\w-]+\.md[^()]*\)")
_CHUNK_BARE = re.compile(r"\s*/?(?:retrieved/)?(?:[\w-]+/)?chunk_[\w-]+\.md")


def clean_citations(text: str) -> str:
    """Remove internal chunk file references the user should never see."""
    text = _CHUNK_IN_PARENS.sub("", text)
    return _CHUNK_BARE.sub("", text)


CSS = """
<style>
:root {
    --accent: #7c8cff;
    --accent-2: #22d3ee;
    --panel: #121a30;
    --border: #232c4a;
    --text: #e8ecf8;
    --muted: #93a0c4;
}

/* ---------- base: force dark, whatever the Streamlit theme says ---------- */
:root, html, body { color-scheme: dark; }
.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 90% -10%, rgba(124,140,255,.18), transparent 30rem),
        radial-gradient(circle at -10% 110%, rgba(34,211,238,.10), transparent 28rem),
        #0b1020;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stStatusWidget"] { color: var(--text); }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
.block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 5rem; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
label { color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* ---------- hero ---------- */
.hero {
    padding: 2rem 2.2rem;
    margin-bottom: 1.4rem;
    border: 1px solid var(--border);
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(124,140,255,.14), rgba(18,26,48,.9));
}
.pill {
    display: inline-block;
    padding: .25rem .75rem;
    margin-bottom: .9rem;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: rgba(11,16,32,.6);
    color: var(--accent-2);
    font-size: .78rem;
    font-weight: 600;
}
.hero h1 {
    margin: 0;
    padding: 0;
    color: var(--text);
    font-size: clamp(2rem, 4.5vw, 3.1rem);
    font-weight: 800;
    letter-spacing: -.035em;
    line-height: 1.1;
}
.grad {
    background: linear-gradient(90deg, #a5b4fc, #22d3ee);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { max-width: 620px; margin: .9rem 0 0; color: var(--muted); font-size: 1.02rem; }

/* ---------- how-it-works cards ---------- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
.step {
    padding: 1.1rem 1.2rem;
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--panel);
}
.step .num {
    display: inline-block;
    width: 1.9rem;
    height: 1.9rem;
    margin-bottom: .6rem;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #0b1020;
    font-weight: 800;
    line-height: 1.9rem;
    text-align: center;
}
.step b { display: block; margin-bottom: .2rem; color: var(--text); }
.step span { color: var(--muted); font-size: .92rem; }
@media (max-width: 800px) { .steps { grid-template-columns: 1fr; } }

/* ---------- sidebar stats ---------- */
.stats { display: flex; gap: .6rem; margin-top: .8rem; }
.stat {
    flex: 1;
    padding: .6rem .4rem;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel);
    text-align: center;
}
.stat b { display: block; color: var(--accent-2); font-size: 1.25rem; }
.stat span { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; }

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel);
}

/* ---------- chat input / footer bar ---------- */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    border-radius: 14px;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    caret-color: var(--accent-2);
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #7f8bad !important; }

/* ---------- buttons ---------- */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-weight: 600;
}
.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--accent);
    color: #ffffff;
}
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #0b1020;
    border: none;
}
</style>
"""

HERO = """
<div class="hero">
<div class="pill">{status}</div>
<h1>Ask your documents <span class="grad">anything.</span></h1>
<p>Add your PDFs, notes or Markdown files. An AI assistant reads them and answers your questions, telling you which file and page each answer came from.</p>
</div>
"""

STEPS = """
<div class="steps">
<div class="step"><div class="num">1</div><b>Upload</b><span>Add your PDF, TXT or Markdown files in the sidebar.</span></div>
<div class="step"><div class="num">2</div><b>Analyze</b><span>We read your files and get them ready, so you don't have to.</span></div>
<div class="step"><div class="num">3</div><b>Ask</b><span>Ask anything and get an answer with the file and page it came from.</span></div>
</div>
"""


def file_set_id(uploaded_files) -> str:
    """Stable identifier for the current uploaded file contents."""
    digest = hashlib.sha256()
    for f in uploaded_files:
        digest.update(f.name.encode("utf-8"))
        digest.update(f.getvalue())
    return digest.hexdigest()


def final_text(result: dict) -> str:
    """Extract the last readable model message from an agent result."""
    for message in reversed(result.get("messages", [])):
        text = getattr(message, "text", "")
        if not isinstance(text, str):
            text = ""
        if not text and isinstance(getattr(message, "content", None), str):
            text = message.content
        if text:
            return text
    return "I finished, but I couldn't put together an answer. Please try asking again."


def extract_steps(messages) -> list[str]:
    """Turn the agent's technical tool calls into plain-language sentences."""
    steps: list[str] = []
    reviewed = 0

    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call["name"]
            args = call.get("args") or {}

            if name == "search_documents":
                query = str(args.get("query", "")).strip()
                steps.append(f"🔎 Searched your documents for: “{query}”")
            elif name == "task":
                reviewed += 1
            elif name == "write_todos":
                steps.append("🗂️ Planned how to answer your question")

    if reviewed:
        word = "passage" if reviewed == 1 else "passages"
        steps.append(f"📄 Reviewed {reviewed} {word} in detail")

    return steps


def show_steps(steps: list[str]) -> None:
    if steps:
        with st.expander("How I found this answer"):
            for step in steps:
                st.markdown(f"- {step}")


def main() -> None:
    st.set_page_config(page_title="DocChat", page_icon="✨", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("stats", None)

    ready = "agent" in st.session_state
    stats = st.session_state.stats

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("### 📂 Your library")
        uploaded_files = st.file_uploader(
            "Add files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        analyze_clicked = st.button(
            "Analyze documents",
            type="primary",
            disabled=not uploaded_files,
            use_container_width=True,
        )

        if analyze_clicked:
            try:
                with st.spinner("Reading your documents..."):
                    documents = []
                    for f in uploaded_files:
                        documents.extend(load_bytes(f.name, f.getvalue()))
                    chunks = split_documents(documents)
                    store = build_vector_store(chunks)
                    agent = build_agent(
                        store, k=TOP_K, max_concurrent_analysts=MAX_ANALYSTS
                    )

                st.session_state.agent = agent
                st.session_state.file_set_id = file_set_id(uploaded_files)
                st.session_state.messages = []
                st.session_state.stats = {
                    "files": len(uploaded_files),
                    "pages": len(documents),
                }
                st.rerun()
            except Exception as exc:
                st.error("Sorry, I couldn't read those files. Please check them and try again.")
                with st.expander("Details"):
                    st.write(str(exc))

        if stats:
            st.markdown(
                f"""
<div class="stats">
<div class="stat"><b>{stats['files']}</b><span>files</span></div>
<div class="stat"><b>{stats['pages']}</b><span>pages</span></div>
</div>
""",
                unsafe_allow_html=True,
            )

        if uploaded_files and "file_set_id" in st.session_state:
            if file_set_id(uploaded_files) != st.session_state.file_set_id:
                st.warning("You changed your files. Click **Analyze documents** to update.")

        st.divider()
        if st.button("New conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ---------------- Hero ----------------
    if ready and stats:
        noun = "file" if stats["files"] == 1 else "files"
        status = f"● Ready · {stats['files']} {noun}"
    else:
        status = "✨ Your documents, answered"
    st.markdown(HERO.format(status=status), unsafe_allow_html=True)

    if not ready:
        st.markdown(STEPS, unsafe_allow_html=True)

    # ---------------- Chat history ----------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_steps(message.get("steps", []))

    # ---------------- Suggested questions ----------------
    if ready and not st.session_state.messages:
        st.caption("Try one of these:")
        cols = st.columns(len(SUGGESTIONS))
        for col, text in zip(cols, SUGGESTIONS):
            if col.button(text, use_container_width=True):
                st.session_state.pending = text
                st.rerun()

    # ---------------- New question ----------------
    question = st.chat_input(
        "Ask a question about your documents...",
        disabled=not ready,
    )
    if not question:
        question = st.session_state.pop("pending", None)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            steps: list[str] = []
            error_detail = None
            with st.spinner("Reading your documents and preparing an answer..."):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    result = st.session_state.agent.invoke({"messages": history})
                    answer = clean_citations(final_text(result))
                    steps = extract_steps(result["messages"])
                except Exception as exc:
                    answer = "Sorry, I couldn't answer that right now. Please try again in a moment."
                    error_detail = str(exc)

            st.markdown(answer)
            show_steps(steps)
            if error_detail:
                with st.expander("Details"):
                    st.write(error_detail)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "steps": steps}
        )


if __name__ == "__main__":
    main()