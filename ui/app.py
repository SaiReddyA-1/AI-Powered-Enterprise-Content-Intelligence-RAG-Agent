"""
Enterprise Content Intelligence - AI-Powered RAG Platform
Multi-page Streamlit Dashboard with persistent chat history.
"""
import streamlit as st
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sentence_transformers import SentenceTransformer
from services.search.database import HybridSearchClient
from services.search.retriever import QueryRetriever
from services.rag.reranker import CrossEncoderReranker
from services.rag.engine import RAGEngine
from services.analytics.intelligence import RepositoryIntelligenceEngine
from services.chat.history import ChatHistoryManager
from services.ingestion.connectors import FileManifest
from core.config import settings
from core.logger import setup_enterprise_logger

logger = setup_enterprise_logger("ui")

st.set_page_config(
    page_title="Enterprise Content Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .block-container { padding-top: 1rem; max-width: 1200px; }

    /* ── Metric Cards - glassmorphism ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 18px 20px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.10);
    }
    [data-testid="stMetric"] label {
        font-size: 11px !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Pipeline Stage Cards ── */
    .pipeline-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(8px);
        padding: 16px 20px;
        border-radius: 12px;
        border-left: 4px solid;
        border-top: 1px solid rgba(255,255,255,0.06);
        border-right: 1px solid rgba(255,255,255,0.06);
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 4px;
        transition: transform 0.15s ease;
    }
    .pipeline-card:hover { transform: translateX(4px); }
    .pipeline-card .stage-name {
        font-weight: 700; font-size: 15px;
        margin-bottom: 4px;
    }
    .pipeline-card .stage-desc {
        font-size: 12px; color: #94a3b8; line-height: 1.4;
    }
    .pipeline-metric-box {
        padding: 14px 16px; text-align: center;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .pipeline-metric-box .metric-val {
        font-size: 13px; font-weight: 600; color: #818cf8;
    }
    .pipeline-connector {
        text-align: left; color: rgba(255,255,255,0.15);
        font-size: 16px; padding-left: 28px; line-height: 1;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    .sidebar-stat {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(8px);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-stat .stat-label {
        font-size: 9px; text-transform: uppercase;
        letter-spacing: 1.5px; color: #64748b;
        font-weight: 600; margin-bottom: 2px;
    }
    .sidebar-stat .stat-value {
        font-weight: 700; font-size: 14px;
    }
    .stat-green { color: #4ade80; }
    .stat-blue { color: #60a5fa; }
    .stat-purple { color: #a78bfa; }
    .stat-amber { color: #fbbf24; }

    /* ── Brand Logo ── */
    .brand-container {
        text-align: center; padding: 16px 0 8px 0;
    }
    .brand-icon {
        font-size: 32px; margin-bottom: 4px;
    }
    .brand-title {
        font-size: 20px; font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.3;
    }
    .brand-subtitle {
        font-size: 9px; color: #64748b;
        letter-spacing: 3px; text-transform: uppercase;
        margin-top: 4px; font-weight: 500;
    }
    .version-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 3px 14px; border-radius: 20px;
        font-size: 10px; font-weight: 600; letter-spacing: 0.5px;
        margin-top: 6px;
    }

    /* ── Hero Metric Cards ── */
    .hero-card {
        text-align: center; padding: 28px 20px;
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        transition: transform 0.2s ease;
    }
    .hero-card:hover { transform: translateY(-3px); }
    .hero-card .hero-value {
        font-size: 42px; font-weight: 800; line-height: 1;
    }
    .hero-card .hero-label {
        font-size: 11px; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin-top: 8px; font-weight: 600;
    }
    .gradient-purple {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .gradient-blue {
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Source Citation Cards ── */
    .source-card {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.06);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .source-card .source-num {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; font-weight: 700; font-size: 11px;
        min-width: 24px; height: 24px; border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
    }
    .source-card .source-info {
        flex: 1; font-size: 13px; line-height: 1.4;
    }
    .source-card .source-file { font-weight: 600; }
    .source-card .source-meta { color: #94a3b8; font-size: 11px; }

    /* ── Chat Confidence Badge ── */
    .confidence-bar {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 14px; border-radius: 10px;
        font-size: 12px; font-weight: 600;
        margin-top: 8px;
    }
    .conf-high {
        background: rgba(74, 222, 128, 0.1);
        border: 1px solid rgba(74, 222, 128, 0.2);
        color: #4ade80;
    }
    .conf-med {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.2);
        color: #fbbf24;
    }
    .conf-low {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.2);
        color: #f87171;
    }

    /* ── Page Header ── */
    .page-header {
        margin-bottom: 24px;
    }
    .page-header h1 {
        font-size: 28px; font-weight: 800;
        margin-bottom: 4px;
        background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .page-header p {
        color: #64748b; font-size: 14px; margin: 0;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 13px;
        color: #94a3b8 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
        background: rgba(255,255,255,0.06);
    }
    .stTabs [aria-selected="true"] {
        background: #6366f1 !important;
        color: #ffffff !important;
    }
    /* Remove default Streamlit tab underline */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Section title ── */
    .section-title {
        font-size: 18px; font-weight: 700;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(255,255,255,0.06);
    }

    /* ── Quality distribution cards ── */
    .quality-card {
        text-align: center; padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(255,255,255,0.02);
    }
    .quality-card .q-value { font-size: 32px; font-weight: 800; }
    .quality-card .q-label { font-size: 11px; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
    .q-excellent { color: #4ade80; }
    .q-good { color: #60a5fa; }
    .q-fair { color: #fbbf24; }
    .q-degraded { color: #f87171; }

    /* ── Config table ── */
    .config-row {
        display: flex; justify-content: space-between;
        padding: 8px 14px; border-radius: 8px;
        margin-bottom: 2px; font-size: 13px;
    }
    .config-row:nth-child(odd) { background: rgba(255,255,255,0.02); }
    .config-key { color: #94a3b8; font-weight: 500; }
    .config-val { font-weight: 600; color: #e2e8f0; }

    /* ── Suggestion chips (chat empty state) ── */
    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        max-width: 560px;
        margin: 0 auto;
    }
    .suggestion-chip {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px 18px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .suggestion-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.20);
        background: rgba(255,255,255,0.07);
    }
    .suggestion-chip .chip-icon {
        font-size: 20px; margin-bottom: 6px;
    }
    .suggestion-chip .chip-text {
        font-size: 13px; font-weight: 500; color: #e2e8f0;
        line-height: 1.4;
    }
    .suggestion-chip .chip-sub {
        font-size: 11px; color: #94a3b8; margin-top: 4px;
    }
    /* Hide the fallback Streamlit suggestion buttons */
    .suggestion-btn-row { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Backend Initialization ────────────────────────────────────────────────
@st.cache_resource
def load_backend():
    """Initialize all pipeline components once."""
    db = HybridSearchClient()
    embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    retriever = QueryRetriever(db, model=embed_model)
    reranker = CrossEncoderReranker()
    analytics = RepositoryIntelligenceEngine(db)
    rag_engine = RAGEngine(retriever, reranker, analytics_engine=analytics)
    chat_mgr = ChatHistoryManager()
    manifest = FileManifest()
    return db, embed_model, retriever, reranker, rag_engine, analytics, chat_mgr, manifest


def get_backend():
    return load_backend()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 1: AI Chat & Retrieval
# ══════════════════════════════════════════════════════════════════════════
def page_chat():
    st.markdown("""
    <div class="page-header">
        <h1>AI Chat & Retrieval</h1>
        <p>Ask questions against your enterprise document repository. Answers are grounded with citations.</p>
    </div>
    """, unsafe_allow_html=True)

    _, _, _, _, rag_engine, _, chat_mgr, _ = get_backend()

    # ── Initialize session state ──────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "chat_loaded" not in st.session_state:
        conversations = chat_mgr.load_conversations()
        if conversations:
            latest = conversations[-1]
            st.session_state.messages = latest["messages"]
            st.session_state.conversation_id = latest["id"]
        st.session_state.chat_loaded = True

    # ── Conversation Controls ─────────────────────────────────────────
    conversations = chat_mgr.load_conversations()

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        if conversations:
            options = ["Current"] + [
                f"{c['updated'][:10]} - {c['preview'][:45]}..."
                for c in reversed(conversations)
            ]
            selected_idx = st.selectbox(
                "History",
                range(len(options)),
                format_func=lambda i: options[i],
                label_visibility="collapsed",
            )
            if selected_idx > 0:
                conv = list(reversed(conversations))[selected_idx - 1]
                if st.session_state.conversation_id != conv["id"]:
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = conv["messages"]
                    st.rerun()
    with col2:
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()
    with col3:
        if conversations and st.button("Clear All", use_container_width=True):
            chat_mgr.clear_all()
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

    # ── Empty state - suggestion cards ───────────────────────────────
    if "prefill_prompt" not in st.session_state:
        st.session_state.prefill_prompt = None

    if not st.session_state.messages:
        suggestions = [
            ("&#128269;", "Search across all documents", "Find policies, reports, or guidelines",
             "List all document types and departments in the knowledge base"),
            ("&#128202;", "Summarize a document", "Get key takeaways from any file",
             "Summarize the key findings from the most recent reports"),
            ("&#128221;", "Compare documents", "Spot differences between versions",
             "What are the differences between the latest policy documents?"),
            ("&#128161;", "Ask about repository health", "Quality scores, degradation alerts",
             "How is the overall repository health? Any degradation alerts?"),
        ]

        chip_html = """
        <div style="display:flex; align-items:center; justify-content:center; min-height: calc(100vh - 380px);">
            <div style="text-align:center; width: 100%; max-width: 600px;">
                <div style="font-size: 24px; font-weight: 700; color: #94a3b8; margin-bottom: 28px;">
                    What can I help you find?
                </div>
                <div class="suggestion-grid">"""
        for i, (icon, title, sub, _) in enumerate(suggestions):
            chip_html += f"""
                    <div class="suggestion-chip" onclick="document.querySelectorAll('.suggestion-btn-row button')[{i}].click()">
                        <div class="chip-icon">{icon}</div>
                        <div class="chip-text">{title}</div>
                        <div class="chip-sub">{sub}</div>
                    </div>"""
        chip_html += """
                </div>
            </div>
        </div>"""
        st.markdown(chip_html, unsafe_allow_html=True)

        # Hidden Streamlit buttons - triggered by chip onclick
        st.markdown('<div class="suggestion-btn-row">', unsafe_allow_html=True)
        cols = st.columns(len(suggestions))
        for i, (_, _, _, prompt) in enumerate(suggestions):
            with cols[i]:
                if st.button(f"s{i}", key=f"chip_{i}"):
                    st.session_state.prefill_prompt = prompt
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Render chat history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for i, s in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-num">{i+1}</div>
                            <div class="source-info">
                                <div class="source-file">{s['file_name']}</div>
                                <div class="source-meta">{s.get('document_type','')} &middot; {s.get('department','')} &middot; Quality: {s.get('quality_score',0):.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ── Prefill chat input via JS when a suggestion card is clicked ──
    prefill = st.session_state.pop("prefill_prompt", None)
    if prefill:
        escaped = prefill.replace("'", "\\'").replace("\n", "\\n")
        st.markdown(f"""
        <script>
        (function() {{
            var tries = 0;
            var iv = setInterval(function() {{
                var ta = document.querySelector('[data-testid="stChatInput"] textarea');
                if (ta) {{
                    var set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                    set.call(ta, '{escaped}');
                    ta.dispatchEvent(new Event('input', {{bubbles: true}}));
                    ta.focus();
                    clearInterval(iv);
                }}
                if (++tries >= 30) clearInterval(iv);
            }}, 100);
        }})();
        </script>
        """, unsafe_allow_html=True)

    # ── Input ─────────────────────────────────────────────────────────
    if query := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching & generating answer..."):
                resp = rag_engine.execute_rag(
                    query,
                    chat_history=st.session_state.messages,
                )

            st.markdown(resp["answer"])

            conf = resp.get("confidence_score", 0.0)
            sources = resp.get("sources", [])
            query_type = resp.get("query_type", "retrieval")

            # Confidence + query type badge
            tags = []
            if "analytics" in query_type:
                tags.append("Analytics")
            if "full_document" in query_type:
                tags.append("Full Document")
            if "retrieval" in query_type:
                tags.append("Semantic Search")
            tag_str = " &middot; ".join(tags) if tags else ""

            if conf >= 0.8:
                conf_class = "conf-high"
            elif conf >= 0.5:
                conf_class = "conf-med"
            else:
                conf_class = "conf-low"

            st.markdown(f"""
            <div class="confidence-bar {conf_class}">
                <span>Confidence: {conf:.0%}</span>
                {f'<span style="opacity:0.6">|</span> <span>{tag_str}</span>' if tag_str else ''}
            </div>
            """, unsafe_allow_html=True)

            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, s in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-num">{i+1}</div>
                            <div class="source-info">
                                <div class="source-file">{s['file_name']}</div>
                                <div class="source-meta">{s.get('document_type','')} &middot; {s.get('department','')} &middot; Quality: {s.get('quality_score',0):.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp["answer"],
                "sources": sources,
                "confidence": conf,
            })

            st.session_state.conversation_id = chat_mgr.save_conversation(
                st.session_state.messages,
                st.session_state.conversation_id,
            )


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 2: Data Manager
# ══════════════════════════════════════════════════════════════════════════
def page_data_manager():
    st.markdown("""
    <div class="page-header">
        <h1>Data Manager</h1>
        <p>View, search, upload, and manage documents in your knowledge base.</p>
    </div>
    """, unsafe_allow_html=True)

    db, embed_model, _, _, _, analytics, _, manifest = get_backend()

    docs = analytics.get_document_list()
    type_counts = analytics.count_by_document_type()
    dept_counts = analytics.count_by_department()
    total_chunks = sum(d["chunk_count"] for d in docs)

    # ── Summary Metrics ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", len(docs))
    col2.metric("Chunks", f"{total_chunks:,}")
    col3.metric("Types", len(type_counts))
    col4.metric("Departments", len(dept_counts))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_browse, tab_upload, tab_delete = st.tabs(["Browse Documents", "Upload New", "Delete"])

    with tab_browse:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            search_term = st.text_input("Search", placeholder="Filter by filename...", label_visibility="collapsed")
        with col2:
            filter_type = st.selectbox("Type", ["All"] + sorted(list(type_counts.keys())), label_visibility="collapsed")
        with col3:
            refresh = st.button("Refresh", use_container_width=True)

        filtered = docs
        if search_term:
            filtered = [d for d in filtered if search_term.lower() in d["file_name"].lower()]
        if filter_type != "All":
            filtered = [d for d in filtered if d["document_type"] == filter_type]

        if not filtered:
            st.info("No documents found matching your criteria.")
        else:
            df = pd.DataFrame(filtered)
            display_cols = [
                "file_name", "file_path", "document_type", "department", "quality_score",
                "is_degraded", "chunk_count", "triage_confidence", "file_extension"
            ]
            display_cols = [c for c in display_cols if c in df.columns]

            column_config = {
                "file_name": st.column_config.TextColumn("File Name", width="medium"),
                "file_path": st.column_config.TextColumn("File Path", width="large"),
                "document_type": st.column_config.TextColumn("Type", width="small"),
                "department": st.column_config.TextColumn("Department", width="small"),
                "quality_score": st.column_config.ProgressColumn(
                    "Quality", min_value=0, max_value=1, format="%.0f%%",
                ),
                "is_degraded": st.column_config.CheckboxColumn("Degraded", width="small"),
                "chunk_count": st.column_config.NumberColumn("Chunks", width="small"),
                "triage_confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.0f%%",
                ),
                "file_extension": st.column_config.TextColumn("Ext", width="small"),
            }

            st.dataframe(
                df[display_cols],
                column_config=column_config,
                use_container_width=True,
                height=420,
                hide_index=True,
            )

            st.caption(f"Showing {len(filtered)} of {len(docs)} documents / {sum(d['chunk_count'] for d in filtered):,} chunks")

    with tab_upload:
        st.markdown("Upload documents to process through the full RAG pipeline.")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "txt", "md", "csv", "docx", "json", "xlsx", "xls", "xml",
                  "jpg", "jpeg", "png", "tiff", "bmp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} file(s) selected for ingestion.")
            if st.button(f"Ingest {len(uploaded_files)} Document(s)", type="primary", use_container_width=True):
                progress = st.progress(0, text="Starting pipeline...")
                for i, uploaded in enumerate(uploaded_files):
                    progress.progress(
                        (i) / len(uploaded_files),
                        text=f"Processing {uploaded.name} ({i+1}/{len(uploaded_files)})..."
                    )
                    _ingest_uploaded_file(uploaded, db, embed_model, manifest)
                progress.progress(1.0, text="All documents ingested!")
                st.success(f"Successfully ingested {len(uploaded_files)} document(s).")
                st.cache_resource.clear()
                st.rerun()

    with tab_delete:
        if not docs:
            st.info("No documents in the knowledge base.")
        else:
            st.markdown("Select documents to permanently remove from the index.")

            del_options = {f"{d['file_name']} ({d['document_id'][:12]}...)": d['document_id'] for d in docs}
            selected_docs = st.multiselect("Select documents", list(del_options.keys()), label_visibility="collapsed")

            if selected_docs:
                st.warning(f"{len(selected_docs)} document(s) will be permanently deleted with all their chunks.")
                if st.button(f"Delete {len(selected_docs)} Document(s)", type="primary"):
                    total_deleted = 0
                    for label in selected_docs:
                        doc_id = del_options[label]
                        result = db.delete_document(doc_id)
                        if result["status"] == "success":
                            total_deleted += result["deleted_chunks"]
                            for rel_path, entry in manifest.get_files_dict().items():
                                if entry.get("document_id") == doc_id:
                                    manifest.remove_entry(rel_path)
                                    break
                    manifest.save()
                    st.success(f"Deleted {total_deleted} chunks from {len(selected_docs)} documents.")
                    st.cache_resource.clear()
                    st.rerun()


def _ingest_uploaded_file(uploaded_file, db, embed_model, manifest):
    """Run the full offline pipeline on an uploaded file."""
    import hashlib
    from services.processing.parser import DocumentParser
    from services.processing.ocr import OCRService
    from services.processing.triage import TriageService
    from services.embedding.chunker import AdvancedChunkingEngine
    from services.embedding.generator import EmbeddingGenerator

    upload_dir = Path(settings.DATA_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / uploaded_file.name

    if save_path.exists():
        stem = save_path.stem
        ext = save_path.suffix
        counter = 1
        while save_path.exists():
            save_path = upload_dir / f"{stem}_{counter}{ext}"
            counter += 1

    file_bytes = uploaded_file.read()
    save_path.write_bytes(file_bytes)

    doc_id = hashlib.sha256(file_bytes).hexdigest()
    relative_path = f"uploads/{save_path.name}"

    payload = {
        "document_id": doc_id,
        "file_path": str(save_path),
        "file_name": uploaded_file.name,
        "file_extension": Path(uploaded_file.name).suffix.lower(),
        "file_size_bytes": len(file_bytes),
        "created_time": datetime.utcnow().isoformat() + "Z",
        "modified_time": datetime.utcnow().isoformat() + "Z",
        "ingested_time": datetime.utcnow().isoformat() + "Z",
    }

    parsed = DocumentParser().parse_document(payload)
    ocr_result = OCRService().process_document(parsed)
    triaged = TriageService().process_document(ocr_result)
    chunks = AdvancedChunkingEngine().process_document(triaged)

    if chunks:
        embedder = EmbeddingGenerator(model=embed_model)
        embedded = embedder.process_chunks(chunks)
        db.upload_chunks(embedded)

        manifest.update_entry(relative_path, {
            "document_id": doc_id,
            "file_path": str(save_path),
            "relative_path": relative_path,
            "parent_folder": "uploads",
            "file_name": uploaded_file.name,
            "file_extension": Path(uploaded_file.name).suffix.lower(),
            "file_size_bytes": len(file_bytes),
            "content_hash": doc_id,
            "modified_time": datetime.utcnow().isoformat() + "Z",
            "status": "processed",
            "document_type": triaged.get("document_type", "unknown"),
            "department_category": triaged.get("department_category", "general"),
            "quality_score": triaged.get("quality_score", 0.0),
            "triage_confidence": triaged.get("triage_confidence", 0.0),
            "chunk_count": len(chunks),
            "last_processed": datetime.utcnow().isoformat() + "Z",
            "error_message": None,
        })
        manifest.save()


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 3: Pipeline Monitor
# ══════════════════════════════════════════════════════════════════════════
def page_pipeline_monitor():
    st.markdown("""
    <div class="page-header">
        <h1>Pipeline Monitor</h1>
        <p>End-to-end pipeline architecture, database health, and processing statistics.</p>
    </div>
    """, unsafe_allow_html=True)

    _, _, _, _, _, analytics, _, _ = get_backend()
    db = get_backend()[0]

    state = analytics.get_pipeline_state()
    db_stats = state["db_stats"]

    # ── Pipeline Flow ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Pipeline Architecture</div>', unsafe_allow_html=True)

    stage_colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#06b6d4", "#6366f1"]
    stages = [
        ("Repository Scanner", f"{state['total_documents']} documents", "Recursive scan with SHA-256 dedup"),
        ("Document Parser", f"{', '.join(f'{v} {k}' for k, v in state['file_types'].items())}", "PDF, DOCX, MD, CSV, TXT, images"),
        ("OCR Engine", "Conditional execution", "Tesseract for low-text documents"),
        ("Triage AI", f"{len(state['document_types'])} types, {len(state['departments'])} depts", "4-tier classification + quality scoring"),
        ("Chunking Engine", f"{state['total_chunks']:,} chunks", f"{settings.CHUNK_SIZE} chars, {settings.CHUNK_OVERLAP} overlap"),
        ("Embedding Generator", f"{settings.EMBEDDING_DIMENSIONS}D vectors", f"{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}"),
        ("Vector Index", f"{db_stats.get('total_chunks', state['total_chunks']):,} indexed", f"{db_stats['mode'].upper()} - HNSW cosine"),
    ]

    icons = ["1", "2", "3", "4", "5", "6", "7"]

    for i, (name, metric, desc) in enumerate(stages):
        col_info, col_metric = st.columns([3, 2])
        with col_info:
            st.markdown(f"""
            <div class="pipeline-card" style="border-left-color: {stage_colors[i]};">
                <div class="stage-name" style="color: {stage_colors[i]};">Stage {icons[i]}: {name}</div>
                <div class="stage-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_metric:
            st.markdown(f"""
            <div class="pipeline-metric-box">
                <div class="metric-val">{metric}</div>
            </div>
            """, unsafe_allow_html=True)
        if i < len(stages) - 1:
            st.markdown('<div class="pipeline-connector">|</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Database State ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Database State</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mode", db_stats["mode"].upper())
    col2.metric("Index", db_stats.get("index_name", "N/A"))
    col3.metric("Total Chunks", f"{db_stats.get('total_chunks', state['total_chunks']):,}")
    if db_stats.get("store_size"):
        size_mb = round(db_stats["store_size"] / (1024 * 1024), 2)
        col4.metric("Store Size", f"{size_mb} MB")
    else:
        col4.metric("Searches", db_stats.get("search_count", 0))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Classification Results ────────────────────────────────────────
    st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Document Types**")
        if state["document_types"]:
            df_types = pd.DataFrame(
                {"Type": list(state["document_types"].keys()),
                 "Count": list(state["document_types"].values())}
            ).set_index("Type")
            st.bar_chart(df_types)
        else:
            st.info("No data")

    with col_b:
        st.write("**Departments**")
        if state["departments"]:
            df_dept = pd.DataFrame(
                {"Department": list(state["departments"].keys()),
                 "Count": list(state["departments"].values())}
            ).set_index("Department")
            st.bar_chart(df_dept)
        else:
            st.info("No data")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Quality Distribution ──────────────────────────────────────────
    st.markdown('<div class="section-title">Quality Distribution</div>', unsafe_allow_html=True)

    q = state["quality_distribution"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="quality-card"><div class="q-value q-excellent">{q["excellent"]}</div><div class="q-label">Excellent (90%+)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="quality-card"><div class="q-value q-good">{q["good"]}</div><div class="q-label">Good (70-89%)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="quality-card"><div class="q-value q-fair">{q["fair"]}</div><div class="q-label">Fair (50-69%)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="quality-card"><div class="q-value q-degraded">{q["degraded"]}</div><div class="q-label">Degraded (&lt;50%)</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Config Summary ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Active Configuration</div>', unsafe_allow_html=True)

    config_items = [
        ("Embedding Model", settings.EMBEDDING_MODEL_NAME),
        ("Embedding Dimensions", str(settings.EMBEDDING_DIMENSIONS)),
        ("Reranker Model", settings.RERANKER_MODEL_NAME),
        ("Chunk Size", f"{settings.CHUNK_SIZE} chars"),
        ("Chunk Overlap", f"{settings.CHUNK_OVERLAP} chars"),
        ("Top-K Retrieval", str(settings.TOP_K_RETRIEVAL)),
        ("Top-K Rerank", str(settings.TOP_K_RERANK)),
        ("LLM Provider", "Groq API"),
        ("LLM Model", settings.GROQ_MODEL_NAME),
        ("Quality Threshold", str(settings.TRIAGE_QUALITY_THRESHOLD)),
        ("Data Directory", settings.DATA_DIR),
    ]

    config_html = ""
    for key, val in config_items:
        config_html += f'<div class="config-row"><span class="config-key">{key}</span><span class="config-val">{val}</span></div>'

    st.markdown(config_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  PAGE 4: Intelligence Dashboard
# ══════════════════════════════════════════════════════════════════════════
def page_intelligence():
    st.markdown("""
    <div class="page-header">
        <h1>Intelligence Dashboard</h1>
        <p>Enterprise analytics on document quality, compliance, and risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    _, _, _, _, _, analytics, _, _ = get_backend()

    # ── Hero Metrics ──────────────────────────────────────────────────
    summary = analytics.get_repository_summary()
    health = summary["repository_health_score"]

    if isinstance(health, float) and health > 0:
        health_display = f"{health:.0%}"
        if health >= 0.8:
            health_color = "#4ade80"
        elif health >= 0.6:
            health_color = "#fbbf24"
        else:
            health_color = "#f87171"
    else:
        health_display = "N/A"
        health_color = "#64748b"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-value gradient-purple">{summary['total_documents_ingested']}</div>
            <div class="hero-label">Total Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-value gradient-blue">{summary['total_searchable_chunks']:,}</div>
            <div class="hero-label">Searchable Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="hero-card">
            <div class="hero-value" style="color: {health_color};">{health_display}</div>
            <div class="hero-label">Repository Health</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # ── Distributions ─────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Documents by Department</div>', unsafe_allow_html=True)
        dept = analytics.count_by_department()
        if dept:
            df_dept = pd.DataFrame(
                {"Department": list(dept.keys()), "Count": list(dept.values())}
            ).set_index("Department")
            st.bar_chart(df_dept)
        else:
            st.info("No data")

    with col_b:
        st.markdown('<div class="section-title">Documents by Type</div>', unsafe_allow_html=True)
        types = analytics.count_by_document_type()
        if types:
            df_types = pd.DataFrame(
                {"Type": list(types.keys()), "Count": list(types.values())}
            ).set_index("Type")
            st.bar_chart(df_types)
        else:
            st.info("No data")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── Degradation Report ────────────────────────────────────────────
    st.markdown('<div class="section-title">Degradation & Risk Report</div>', unsafe_allow_html=True)

    risk = analytics.get_degradation_report()
    pct = risk["degraded_percentage"]

    col1, col2 = st.columns(2)
    col1.metric("Degraded Documents", risk["degraded_count"])
    col2.metric("Degradation Rate", f"{pct}%")

    if pct > 10:
        st.error(f"ALERT: {pct}% of documents are below quality threshold ({settings.TRIAGE_QUALITY_THRESHOLD}). Recommend rescanning affected files.")
    elif pct > 0:
        st.warning(f"{pct}% of documents are below quality threshold.")
    else:
        st.success("All documents passing quality checks.")

    if risk["common_issues"]:
        st.write("**Common Issues:**")
        for issue, count in risk["common_issues"].items():
            st.write(f"- `{issue}`: {count} documents")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN - Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════
def main():
    db = get_backend()[0]

    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="brand-container">
            <div class="brand-icon">🧠</div>
            <div class="brand-title">Enterprise<br/>Content Intelligence</div>
            <div class="brand-subtitle">AI-Powered RAG Platform</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        page = st.radio(
            "Navigation",
            [
                "🤖 AI Chat",
                "📁 Data Manager",
                "⚙️ Pipeline Monitor",
                "📊 Intelligence",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        # ── Live System Stats ─────────────────────────────────────────
        stats = db.get_index_stats()
        total_chunks = stats.get("total_chunks", 0)
        db_mode = stats["mode"].upper()
        embed_name = settings.EMBEDDING_MODEL_NAME.split("/")[-1]
        llm_model = settings.GROQ_MODEL_NAME

        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="stat-label">Vector Database</div>
            <div class="stat-value stat-green">{db_mode}</div>
        </div>
        <div class="sidebar-stat">
            <div class="stat-label">Indexed Chunks</div>
            <div class="stat-value stat-blue">{total_chunks:,}</div>
        </div>
        <div class="sidebar-stat">
            <div class="stat-label">Embedding Model</div>
            <div class="stat-value stat-purple">{embed_name}</div>
        </div>
        <div class="sidebar-stat">
            <div class="stat-label">LLM Provider</div>
            <div class="stat-value stat-amber">Groq API</div>
        </div>
        <div class="sidebar-stat">
            <div class="stat-label">LLM Model</div>
            <div class="stat-value stat-amber">{llm_model}</div>
        </div>
        <div style="text-align: center; padding: 12px 0 4px 0;">
            <span class="version-pill">v2.0 Production</span>
        </div>
        """, unsafe_allow_html=True)

    # Route pages
    if page == "🤖 AI Chat":
        page_chat()
    elif page == "📁 Data Manager":
        page_data_manager()
    elif page == "⚙️ Pipeline Monitor":
        page_pipeline_monitor()
    elif page == "📊 Intelligence":
        page_intelligence()


if __name__ == "__main__":
    main()
