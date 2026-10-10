import html as html_lib
import os
import re
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# =========================================================
# PATH FIX FOR 'src' MODULE ERROR
# =========================================================
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.rag import (
    ingest_files,
    retrieve,
    answer_question,
    list_indexed_files,
)

# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

SUGGESTIONS = [
    "Summarize the main points",
    "Extract the key findings",
    "Explain the core concepts",
]

st.set_page_config(
    page_title="KnowledgePilot · Enterprise RAG",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DESIGN SYSTEM (CSS)
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-main: #F8FAFC;
    --bg-surface: #FFFFFF;
    --bg-soft: #F1F5F9;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --border-color: #E2E8F0;
    --border-strong: #CBD5E1;
    --accent: #4F46E5;
    --accent-hover: #4338CA;
    --accent-light: #EEF2FF;
    --accent-border: #C7D2FE;
    --success-bg: #ECFDF5;
    --success-text: #065F46;
    --success-border: #A7F3D0;
    --danger-bg: #FEF2F2;
    --danger-text: #991B1B;
    --danger-border: #FECACA;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.04);
    --shadow-md: 0 4px 12px -2px rgba(15, 23, 42, 0.06);
}

html, body, .stApp, .stApp * {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.stApp { background: var(--bg-main) !important; color: var(--text-primary) !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 1550px;
    padding: 2rem 2.5rem 5rem !important;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4 { color: var(--text-primary) !important; letter-spacing: -0.02em !important; }
.stApp p, .stApp li { color: var(--text-secondary); line-height: 1.6; font-size: 14px; }
.stApp [data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-size: 13px; font-weight: 500; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] > div:first-child { padding: 1rem 0.75rem; }

.brand { display: flex; align-items: center; gap: 12px; margin: 0 0 1rem; }
.brand-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #6366F1, #4338CA);
    color: #fff; display: flex; align-items: center; justify-content: center;
    font-size: 17px; box-shadow: 0 4px 10px rgba(79, 70, 229, .25);
}
.brand-title { font-size: 15px; font-weight: 700; color: var(--text-primary); line-height: 1.2; }
.brand-subtitle { font-size: 11px; color: var(--text-muted); }

.section-title {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: var(--text-muted); margin: 1.5rem 0 .5rem;
}

/* ---------- Inputs ---------- */
.stApp input, .stApp textarea {
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
    caret-color: var(--accent) !important;
}
.stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="select"] > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow-sm) !important;
}
.stApp [data-baseweb="select"] * { color: var(--text-primary) !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-soft) !important;
    border: 1.5px dashed var(--accent-border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploaderDropzone"]:hover { background: var(--accent-light) !important; border-color: var(--accent) !important; }

/* ---------- Chat input ---------- */
.stApp [data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
    padding: 4px 8px !important;
}
.stApp [data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-size: 14.5px !important;
}
.stApp [data-testid="stChatInputSubmitButton"] {
    background: var(--accent) !important;
    border-radius: 10px !important;
}

/* ---------- Buttons ---------- */
.stApp div.stButton > button {
    width: 100%;
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; font-size: 13px !important;
    min-height: 38px; box-shadow: var(--shadow-sm);
    transition: all .15s ease;
}
.stApp div.stButton > button:hover {
    background: var(--accent-light) !important; color: var(--accent) !important;
    border-color: var(--accent-border) !important;
}
.stApp div.stButton > button[kind="primary"],
.stApp div.stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important;
}

/* ---------- Page header & Chips ---------- */
.page-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 1.25rem; }
.page-title { font-size: 24px; font-weight: 700; color: var(--text-primary); letter-spacing: -.02em; margin: 0; }
.page-sub { font-size: 13.5px; color: var(--text-muted); margin: 3px 0 0; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--bg-surface); border: 1px solid var(--border-color); color: var(--text-secondary);
    border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 600; box-shadow: var(--shadow-sm);
}
.chip.ok { background: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
.chip.bad { background: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
.dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

/* ---------- Cards & Steps ---------- */
.card {
    background: var(--bg-surface); border: 1px solid var(--border-color);
    border-radius: var(--radius-lg); padding: 20px 22px; box-shadow: var(--shadow-sm); margin-bottom: 14px;
}
.card h4 { margin: 0 0 6px; font-size: 16px; }
.card.center { text-align: center; padding: 24px 16px; }
.card.warn { background: var(--danger-bg); border-color: var(--danger-border); }
.card.warn h4, .card.warn p { color: var(--danger-text) !important; }

.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 14px; }
.step { background: var(--bg-soft); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 12px; }
.step-num {
    width: 22px; height: 22px; border-radius: 50%; background: var(--accent-light); color: var(--accent);
    font-weight: 700; font-size: 11px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 6px;
}
.step b { display: block; color: var(--text-primary); font-size: 13px; margin-bottom: 2px; }
.step span { color: var(--text-muted); font-size: 12px; }

/* ---------- Chat bubbles & Sources ---------- */
[data-testid="stChatMessage"] {
    background: var(--bg-surface); border: 1px solid var(--border-color);
    border-radius: var(--radius-lg); padding: 12px 16px; box-shadow: var(--shadow-sm); margin-bottom: 10px;
}
.source-card {
    background: var(--bg-surface); border: 1px solid var(--border-color);
    border-radius: var(--radius-md); padding: 10px 12px; margin-bottom: 8px;
}
.source-header { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 4px; }
.source-title { font-size: 12px; font-weight: 650; color: var(--text-primary); }
.source-meta { font-size: 10.5px; color: var(--text-muted); background: var(--bg-soft); border-radius: 6px; padding: 1px 6px; }
.source-snippet { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }
.doc-item {
    display: flex; align-items: center; gap: 8px; padding: 6px 10px; margin-bottom: 4px;
    background: var(--bg-soft); border: 1px solid var(--border-color); border-radius: var(--radius-sm);
    font-size: 12.5px; color: var(--text-primary);
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE & HELPERS
# =========================================================

st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_sources", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("flash", None)


def html(markup: str):
    st.markdown(re.sub(r"\s*\n\s*", " ", markup.strip()), unsafe_allow_html=True)


def esc(value) -> str:
    return html_lib.escape(str(value), quote=True)


@st.cache_data(ttl=10, show_spinner=False)
def check_ollama():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        return True, models
    except (requests.RequestException, ValueError):
        return False, []


def model_is_installed(name: str, installed: list) -> bool:
    if not name:
        return False
    names = {m for m in installed} | {m.removesuffix(":latest") for m in installed}
    return name in names or name.removesuffix(":latest") in names


def snippet(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return esc(text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…")


def render_sources(sources):
    if not sources:
        st.caption("No sources available.")
        return
    for idx, source in enumerate(sources, start=1):
        distance = source.get("distance")
        meta = f'<span class="source-meta">d = {distance:.3f}</span>' if isinstance(distance, (int, float)) else ""
        html(
            f"""
            <div class="source-card">
                <div class="source-header">
                    <span class="source-title">#{idx:02d} · {esc(source.get('file', 'Unknown'))}</span>
                    {meta}
                </div>
                <div class="source-snippet">{snippet(source.get('text', ''))}</div>
            </div>
            """
        )


def stream_text(text: str):
    tokens = re.findall(r"\S+\s*", text)
    delay = min(0.015, 2.5 / max(len(tokens), 1))
    for token in tokens:
        yield token
        time.sleep(delay)


def show_error(error):
    st.error("Something went wrong while processing your request.")
    with st.expander("Diagnostic details"):
        st.exception(error)


# =========================================================
# SIDEBAR
# =========================================================

ollama_running, installed_models = check_ollama()

with st.sidebar:
    html(
        """
        <div class="brand">
            <div class="brand-logo">✦</div>
            <div>
                <div class="brand-title">KnowledgePilot</div>
                <div class="brand-subtitle">Enterprise RAG Engine</div>
            </div>
        </div>
        """
    )

    if st.session_state.flash:
        st.toast(st.session_state.flash, icon="✅")
        st.session_state.flash = None

    html('<div class="section-title">Knowledge Ingestion</div>')
    uploads = st.file_uploader(
        "Upload files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"document_upload_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if st.button("Process & Index Documents", type="primary", disabled=not uploads):
        try:
            with st.spinner(f"Indexing {len(uploads)} document(s)…"):
                count = ingest_files(uploads, DATA_DIR)
            if count > 0:
                st.session_state.flash = f"Indexed {count} chunks successfully."
                st.session_state.uploader_key += 1
                st.rerun()
            else:
                st.warning("No parseable content detected.")
        except Exception as error:
            show_error(error)

    indexed_files = list_indexed_files(DATA_DIR)
    html(f'<div class="section-title">Indexed Documents ({len(indexed_files)})</div>')
    if indexed_files:
        for filename in indexed_files:
            html(f'<div class="doc-item"><span>📄</span><span>{esc(filename)}</span></div>')
    else:
        st.caption("No documents indexed yet.")

    html('<div class="section-title">Model Settings</div>')
    if installed_models:
        if st.session_state.get("ollama_model") not in installed_models:
            st.session_state.ollama_model = (
                DEFAULT_MODEL if DEFAULT_MODEL in installed_models else installed_models[0]
            )
        model = st.selectbox("Ollama model", installed_models, key="ollama_model")
    else:
        st.session_state.setdefault("ollama_model", DEFAULT_MODEL)
        model = st.text_input("Ollama model", key="ollama_model").strip()

    top_k = st.slider("Passages to retrieve", 2, 8, 4)

    html('<div class="section-title">System Status</div>')
    if ollama_running:
        html('<span class="chip ok"><span class="dot"></span>Ollama online</span>')
    else:
        html('<span class="chip bad"><span class="dot"></span>Ollama offline</span>')
        st.write("")
        if st.button("Retry connection"):
            check_ollama.clear()
            st.rerun()

# =========================================================
# READINESS & INPUT
# =========================================================

model_ok = (not installed_models) or model_is_installed(model, installed_models)
ready = bool(ollama_running and model and model_ok and indexed_files)

input_hint = (
    "Ask a question about your documents…"
    if ready
    else "Complete setup in sidebar to start chatting…"
)

typed_question = st.chat_input(input_hint, disabled=not ready)
pending = st.session_state.pending_question
question = typed_question or (pending if ready else None)
st.session_state.pending_question = None

# =========================================================
# MAIN WORKSPACE
# =========================================================

main_col, ref_col = st.columns([2.5, 1], gap="large")

with main_col:
    status_chip = (
        '<span class="chip ok"><span class="dot"></span>Online</span>'
        if ollama_running
        else '<span class="chip bad"><span class="dot"></span>Offline</span>'
    )
    queries_asked = sum(m["role"] == "user" for m in st.session_state.messages) + (1 if question else 0)
    html(
        f"""
        <div class="page-header">
            <div>
                <h1 class="page-title">Workspace</h1>
                <p class="page-sub">Query your private repository securely with local AI context retrieval.</p>
            </div>
            <div class="chips">
                {status_chip}
                <span class="chip">📄 {len(indexed_files)} docs</span>
                <span class="chip">🧠 {esc(model or 'None')}</span>
            </div>
        </div>
        """
    )

    if not ollama_running:
        html('<div class="card warn"><h4>Ollama is offline</h4><p>Run <code>ollama serve</code> to activate.</p></div>')
    elif not model_ok:
        html(f'<div class="card warn"><h4>Model missing</h4><p>Run <code>ollama pull {esc(model)}</code>.</p></div>')

    if not indexed_files:
        html(
            """
            <div class="card">
                <h4>Welcome to KnowledgePilot</h4>
                <p>Follow these quick steps to get started:</p>
                <div class="steps">
                    <div class="step"><div class="step-num">1</div><b>Upload</b><span>Add PDF/TXT files</span></div>
                    <div class="step"><div class="step-num">2</div><b>Index</b><span>Process files</span></div>
                    <div class="step"><div class="step-num">3</div><b>Ask</b><span>Get cited answers</span></div>
                </div>
            </div>
            """
        )
    elif not st.session_state.messages and not question:
        st.markdown("**Quick Prompts**")
        cols = st.columns(len(SUGGESTIONS))
        for col, text in zip(cols, SUGGESTIONS):
            with col:
                if st.button(text, key=f"sugg_{text}", disabled=not ready):
                    st.session_state.pending_question = text
                    st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Referenced passages ({len(message['sources'])})"):
                    render_sources(message["sources"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and formulating answer…"):
                    passages = retrieve(question, DATA_DIR, top_k=top_k)
                    if passages:
                        response_text = answer_question(question, passages, model=model)
                        sources = passages
                    else:
                        response_text = "No relevant context found in your indexed documents."
                        sources = []

                st.write_stream(stream_text(response_text))
                if sources:
                    with st.expander(f"Referenced passages ({len(sources)})"):
                        render_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text, "sources": sources}
                )
                st.session_state.last_sources = sources
            except Exception as error:
                fallback = "⚠️ An error occurred while generating the response."
                st.markdown(fallback)
                show_error(error)
                st.session_state.messages.append(
                    {"role": "assistant", "content": fallback, "sources": []}
                )

# =========================================================
# REFERENCES PANEL
# =========================================================

with ref_col:
    latest_sources = st.session_state.last_sources
    html(
        f"""
        <h3 style="margin:0 0 2px;font-size:17px;">References</h3>
        <p style="margin:0 0 12px;font-size:12.5px;color:var(--text-muted);">
            {f'{len(latest_sources)} source(s) used' if latest_sources else 'Latest passage matches'}
        </p>
        """
    )

    if latest_sources:
        render_sources(latest_sources)
    else:
        html(
            """
            <div class="card center">
                <div style="font-size:22px;margin-bottom:4px;">🔎</div>
                <p style="font-size:12.5px;color:var(--text-muted);">Ask a question to inspect matched text snippets here.</p>
            </div>
            """
        )

    if st.button("Clear conversation", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.pending_question = None
        st.rerun()