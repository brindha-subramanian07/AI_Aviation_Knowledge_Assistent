"""
AI-Powered Aviation Knowledge Assistant
Main Streamlit Application
A Retrieval-Augmented Generation (RAG) assistant for aviation institutional documents.
"""

from pathlib import Path
import streamlit as st

# Configure page layout and branding
st.set_page_config(
    page_title="AI Aviation Knowledge Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    AVIATION_DISCLAIMER,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_RELEVANCE_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
    GEMINI_API_KEY,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_GEMINI_MODELS,
)
from src.document_manager import get_document_manager
from src.rag_engine import get_rag_engine
from src.retriever import get_retriever
from src.ui_styles import (
    AVIATION_CUSTOM_CSS,
    render_disclaimer,
    render_header,
    render_metric_card,
)

# Apply custom aviation styling
st.markdown(AVIATION_CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "🏠 Dashboard"
if "prefilled_query" not in st.session_state:
    st.session_state.prefilled_query = ""
if "gemini_api_key" not in st.session_state or (not st.session_state.gemini_api_key and GEMINI_API_KEY):
    st.session_state.gemini_api_key = GEMINI_API_KEY
if "selected_model" not in st.session_state or st.session_state.selected_model not in SUPPORTED_GEMINI_MODELS:
    st.session_state.selected_model = DEFAULT_GEMINI_MODEL
if "top_k_chunks" not in st.session_state:
    st.session_state.top_k_chunks = DEFAULT_TOP_K
if "distance_threshold" not in st.session_state:
    st.session_state.distance_threshold = DEFAULT_RELEVANCE_DISTANCE_THRESHOLD
if "delete_confirm_id" not in st.session_state:
    st.session_state.delete_confirm_id = None

# Initialize Singletons
doc_manager = get_document_manager()
retriever = get_retriever()
rag_engine = get_rag_engine(api_key=st.session_state.gemini_api_key)

# Sync dynamic settings to engines
rag_engine.set_api_key(st.session_state.gemini_api_key)
rag_engine.set_model_name(st.session_state.selected_model)
retriever.default_top_k = st.session_state.top_k_chunks
retriever.distance_threshold = st.session_state.distance_threshold


# ==============================================================================
# SIDEBAR NAVIGATION & KNOWLEDGE BASE METRICS
# ==============================================================================
with st.sidebar:
    st.markdown("### ✈️ AI Aviation Assistant")
    st.caption("Institutional Document RAG Platform")

    nav_options = [
        "🏠 Dashboard",
        "📚 Knowledge Base",
        "📄 Upload Documents",
        "💬 Ask Assistant",
        "ℹ️ About"
    ]

    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=nav_options.index(st.session_state.selected_nav)
        if st.session_state.selected_nav in nav_options else 0,
        label_visibility="collapsed"
    )
    st.session_state.selected_nav = selected_page

    st.markdown("---")
    st.markdown("#### 📊 Knowledge Base Status")
    kb_stats = doc_manager.get_knowledge_base_stats()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Documents", kb_stats["total_documents"])
    with col_s2:
        st.metric("Chunks", kb_stats["total_chunks"])

    st.caption(f"**Indexed Docs:** {kb_stats['indexed_documents']}")
    st.caption(f"**Last Updated:** {kb_stats['last_updated']}")

    st.markdown("---")
    with st.expander("⚙️ System Settings & API Key", expanded=not bool(st.session_state.gemini_api_key)):
        api_input = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            help="Enter your Google Gemini API key. Can also be set in the .env file as GEMINI_API_KEY."
        )
        if api_input != st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = api_input
            rag_engine.set_api_key(api_input)

        if st.session_state.gemini_api_key:
            st.success("API Key configured", icon="✅")
        else:
            st.warning("No API key set (retrieval will still work)", icon="⚠️")

        model_options = list(SUPPORTED_GEMINI_MODELS)
        if st.session_state.selected_model not in model_options:
            st.session_state.selected_model = DEFAULT_GEMINI_MODEL

        model_choice = st.selectbox(
            "LLM Model",
            model_options,
            index=model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model in model_options else 0
        )
        if model_choice != st.session_state.selected_model:
            st.session_state.selected_model = model_choice
            rag_engine.set_model_name(model_choice)

        top_k_val = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=2,
            max_value=10,
            value=st.session_state.top_k_chunks,
            step=1,
            help="Number of most relevant text chunks supplied to Gemini."
        )
        st.session_state.top_k_chunks = top_k_val
        retriever.default_top_k = top_k_val

        threshold_val = st.slider(
            "Max Cosine Distance",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.distance_threshold,
            step=0.05,
            help="Maximum cosine distance allowed for retrieved chunks. Lower values require higher similarity."
        )
        st.session_state.distance_threshold = threshold_val
        retriever.distance_threshold = threshold_val

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#94A3B8; line-height:1.3;'>"
        "<strong>Safety Note:</strong> Retrieval results are restricted to uploaded manuals. "
        "Always cross-examine against official operational directives."
        "</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if st.session_state.selected_nav == "🏠 Dashboard":
    st.markdown(
        render_header(
            "AI-Powered Aviation Knowledge Assistant",
            "An AI-powered Retrieval-Augmented Generation assistant for searching and understanding aviation institutional knowledge."
        ),
        unsafe_allow_html=True
    )

    st.markdown(render_disclaimer(AVIATION_DISCLAIMER), unsafe_allow_html=True)

    kb_stats = doc_manager.get_knowledge_base_stats()

    # Metric Cards Grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_metric_card("Total Documents", kb_stats["total_documents"], "Registered manuals"),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            render_metric_card("Knowledge Chunks", kb_stats["total_chunks"], "Indexed vectors in ChromaDB"),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            render_metric_card("Indexed Documents", kb_stats["indexed_documents"], "Searchable records"),
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            render_metric_card("Supported Formats", kb_stats["supported_types"], "PDF, DOCX, TXT"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample Questions Showcase
    st.markdown("### 🎯 Explore Sample Aviation Queries")
    st.write("Click any sample query below to test semantic retrieval and grounded response generation:")

    sample_questions = [
        "What are the aircraft maintenance requirements?",
        "What are the safety procedures?",
        "Explain the emergency procedures.",
        "Which document contains information about maintenance?",
        "Compare the safety procedures in the uploaded documents.",
        "What information is available about runway safety?",
    ]

    q_col1, q_col2 = st.columns(2)
    for idx, q in enumerate(sample_questions):
        target_col = q_col1 if idx % 2 == 0 else q_col2
        with target_col:
            if st.button(f"🔍 {q}", key=f"dash_q_{idx}", use_container_width=True):
                st.session_state.prefilled_query = q
                st.session_state.selected_nav = "💬 Ask Assistant"
                st.rerun()

    st.markdown("---")
    st.markdown("### 📑 Currently Indexed Documents")
    docs = doc_manager.list_documents()

    if not docs:
        st.info("No documents have been indexed yet. Head over to **📄 Upload Documents** to get started!", icon="ℹ️")
        if st.button("🚀 Load Pre-Packaged Aviation Manuals", type="primary"):
            sample_dir = Path(__file__).resolve().parent / "sample_docs"
            if sample_dir.exists():
                count = 0
                for fpath in sample_dir.glob("*.*"):
                    if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                        with open(fpath, "rb") as f:
                            doc_manager.upload_and_index(f.read(), fpath.name, allow_reindex=True)
                            count += 1
                st.success(f"Indexed {count} sample documents! Refreshing dashboard...")
                st.rerun()
    else:
        table_data = []
        for d in docs:
            table_data.append({
                "Document Name": d.filename,
                "Type": d.document_type,
                "Pages": d.total_pages,
                "Chunks": d.total_chunks,
                "Size": d.file_size_formatted,
                "Status": d.status,
                "Uploaded At": d.uploaded_at
            })
        st.dataframe(table_data, use_container_width=True)


# ==============================================================================
# PAGE 2: UPLOAD DOCUMENTS
# ==============================================================================
elif st.session_state.selected_nav == "📄 Upload Documents":
    st.markdown(
        render_header(
            "Document Ingestion Center",
            "Upload aviation institutional documents to extract text, generate embeddings, and update the persistent ChromaDB index."
        ),
        unsafe_allow_html=True
    )

    st.markdown(render_disclaimer(AVIATION_DISCLAIMER), unsafe_allow_html=True)

    # One-click sample documents loader
    with st.expander("💡 Quick Start: Load Pre-Packaged Aviation Documents", expanded=True):
        st.write(
            "Load realistic aviation institutional manuals (Maintenance Manual PDF, "
            "Flight Safety DOCX, Runway Safety TXT) to immediately test the system."
        )
        if st.button("📥 Ingest Sample Aviation Documents", type="secondary"):
            sample_dir = Path(__file__).resolve().parent / "sample_docs"
            if sample_dir.exists():
                with st.spinner("Ingesting sample documents..."):
                    loaded = 0
                    for fpath in sample_dir.glob("*.*"):
                        if fpath.suffix.lower() in SUPPORTED_EXTENSIONS:
                            with open(fpath, "rb") as f:
                                ok, msg, _ = doc_manager.upload_and_index(f.read(), fpath.name, allow_reindex=True)
                                if ok:
                                    loaded += 1
                                    st.success(f"Ingested {fpath.name}")
                    st.success(f"Completed ingestion of {loaded} documents!")
                    st.rerun()
            else:
                st.error("Sample directory not found.")

    st.markdown("### 📤 Upload New Documents")
    uploaded_files = st.file_uploader(
        "Select one or multiple aviation documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload aircraft maintenance manuals, flight operations guidelines, safety checklists, or emergency SOPs."
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) ready for indexing.**")
        if st.button("⚡ Process & Index Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, uploaded_file in enumerate(uploaded_files):
                pct = int(((i) / len(uploaded_files)) * 100)
                progress_bar.progress(pct)
                status_text.text(f"Processing '{uploaded_file.name}'...")

                file_bytes = uploaded_file.read()
                success, msg, record = doc_manager.upload_and_index(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    allow_reindex=False
                )

                if success:
                    st.success(f"✅ {msg}")
                    if record and record.ocr_warning:
                        st.warning(f"⚠️ {record.warning_message}")
                else:
                    if "already indexed" in msg:
                        st.info(f"ℹ️ {msg}")
                    else:
                        st.error(f"❌ {msg}")

            progress_bar.progress(100)
            status_text.text("All files processed!")
            st.toast("Document indexing completed!", icon="🎉")


# ==============================================================================
# PAGE 3: KNOWLEDGE BASE (DOCUMENT MANAGEMENT)
# ==============================================================================
elif st.session_state.selected_nav == "📚 Knowledge Base":
    st.markdown(
        render_header(
            "Knowledge Base Management",
            "Inspect indexed aviation documents, view extracted chunks, re-index manuals, or remove records from the vector store."
        ),
        unsafe_allow_html=True
    )

    st.markdown(render_disclaimer(AVIATION_DISCLAIMER), unsafe_allow_html=True)

    docs = doc_manager.list_documents()

    if not docs:
        st.info("The Knowledge Base is currently empty. Please upload documents in the **📄 Upload Documents** tab.", icon="ℹ️")
    else:
        st.markdown(f"### 📋 Managed Documents ({len(docs)})")

        for d in docs:
            with st.container():
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:1.2rem; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <h4 style="margin:0 0 0.3rem 0; color:#0B192C;">📄 {d.filename}</h4>
                                <span style="font-size:0.8rem; color:#64748B;">ID: <code>{d.document_id}</code> | Type: <strong>{d.document_type}</strong> | Pages: <strong>{d.total_pages}</strong> | Chunks: <strong>{d.total_chunks}</strong> | Size: {d.file_size_formatted}</span>
                            </div>
                            <span class="{'badge-indexed' if 'Indexed' in d.status and not d.ocr_warning else 'badge-warning'}">{d.status}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1.5, 3])

                with btn_col1:
                    # View Details & Chunks
                    if st.checkbox(f"🔍 Inspect Chunks", key=f"view_{d.document_id}"):
                        chunks = doc_manager.vector_store.get_chunks_for_document(d.document_id)
                        st.markdown(f"**Stored Vector Chunks ({len(chunks)}):**")
                        for c in chunks:
                            meta = c.get("metadata", {})
                            with st.expander(f"Chunk #{meta.get('chunk_index', 0) + 1} (Page {meta.get('page', 1)}, {meta.get('word_count', 0)} words)"):
                                st.text(c.get("text", ""))

                with btn_col2:
                    # Re-index
                    if st.button(f"🔄 Re-Index", key=f"reindex_{d.document_id}"):
                        with st.spinner(f"Re-indexing '{d.filename}'..."):
                            ok, msg = doc_manager.reindex_document(d.document_id)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

                with btn_col3:
                    # Delete with Confirmation
                    if st.session_state.delete_confirm_id == d.document_id:
                        st.warning(f"Confirm deletion of '{d.filename}'?")
                        c_col1, c_col2 = st.columns(2)
                        with c_col1:
                            if st.button("⚠️ Confirm Delete", key=f"conf_del_{d.document_id}", type="primary"):
                                ok, msg = doc_manager.delete_document(d.document_id)
                                st.session_state.delete_confirm_id = None
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with c_col2:
                            if st.button("Cancel", key=f"cancel_del_{d.document_id}"):
                                st.session_state.delete_confirm_id = None
                                st.rerun()
                    else:
                        if st.button(f"🗑️ Delete", key=f"del_{d.document_id}"):
                            st.session_state.delete_confirm_id = d.document_id
                            st.rerun()

                st.markdown("<hr style='margin: 0.8rem 0; border: none; border-top: 1px dashed #E2E8F0;'>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: ASK ASSISTANT (CHAT INTERFACE)
# ==============================================================================
elif st.session_state.selected_nav == "💬 Ask Assistant":
    st.markdown(
        render_header(
            "Aviation Knowledge Query Assistant",
            "Pose questions across single or multiple aviation manuals. All responses are strictly grounded in retrieved institutional documentation."
        ),
        unsafe_allow_html=True
    )

    st.markdown(render_disclaimer(AVIATION_DISCLAIMER), unsafe_allow_html=True)

    # Search Filter Bar & Actions
    filter_col1, filter_col2, filter_col3 = st.columns([3, 1.5, 1.5])
    with filter_col1:
        docs = doc_manager.list_documents()
        doc_options = ["All Uploaded Documents"] + [f"{d.filename} ({d.document_id})" for d in docs]
        selected_scope = st.selectbox("Search Scope", doc_options, label_visibility="collapsed")
        target_doc_id = None
        if selected_scope != "All Uploaded Documents" and "(" in selected_scope:
            target_doc_id = selected_scope.split("(")[-1].replace(")", "").strip()

    with filter_col2:
        st.write(f"Model: `{st.session_state.selected_model}`")

    with filter_col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Pre-filled Query Handler
    initial_query = st.session_state.prefilled_query
    st.session_state.prefilled_query = ""

    # Suggested Queries quick buttons
    with st.expander("💡 Quick Question Suggestions", expanded=len(st.session_state.chat_history) == 0):
        s_col1, s_col2 = st.columns(2)
        suggestions = [
            "What are the aircraft maintenance requirements?",
            "What are the safety procedures?",
            "Explain the emergency procedures.",
            "Which document contains information about maintenance?",
            "Compare the safety procedures in the uploaded documents.",
            "What information is available about runway safety?",
        ]
        for idx, sq in enumerate(suggestions):
            col = s_col1 if idx % 2 == 0 else s_col2
            with col:
                if st.button(f"✈️ {sq}", key=f"sq_{idx}", use_container_width=True):
                    initial_query = sq

    st.markdown("---")

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["query"])

        with st.chat_message("assistant", avatar="✈️"):
            st.markdown(chat["answer"])

            # Sources Display
            if chat.get("sources"):
                st.markdown("##### 📄 Sources:")
                for idx, src in enumerate(chat["sources"], 1):
                    sim_pct = f"({int(src.get('highest_similarity', 0) * 100)}% match)" if 'highest_similarity' in src else ""
                    st.markdown(
                        f"""
                        <div class="source-tag">
                            📄 {idx}. <strong>{src['source']}</strong> — Page {src['page']} {sim_pct}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Retrieved Context Expandable Box
            if chat.get("retrieved_chunks"):
                with st.expander("🔍 Retrieved Context"):
                    for idx, chunk in enumerate(chat["retrieved_chunks"], 1):
                        st.markdown(
                            f"""
                            <div class="context-box">
                                <div class="context-meta">
                                    <span>Chunk #{idx} | Source: {chunk.source} | Page: {chunk.page}</span>
                                    <span>Similarity: {int(chunk.similarity_score * 100)}%</span>
                                </div>
                                <div>{chunk.text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Process Input
    user_query = st.chat_input("Ask a question about the aviation documents...") or initial_query

    if user_query:
        # Display User Question immediately
        with st.chat_message("user"):
            st.write(user_query)

        # Assistant Processing
        with st.chat_message("assistant", avatar="✈️"):
            with st.spinner("Searching ChromaDB and generating grounded answer..."):
                response = rag_engine.generate_answer(
                    query=user_query,
                    top_k=st.session_state.top_k_chunks,
                    document_id_filter=target_doc_id
                )

                st.markdown(response.answer)

                if response.sources:
                    st.markdown("##### 📄 Sources:")
                    for idx, src in enumerate(response.sources, 1):
                        sim_pct = f"({int(src.get('highest_similarity', 0) * 100)}% match)" if 'highest_similarity' in src else ""
                        st.markdown(
                            f"""
                            <div class="source-tag">
                                📄 {idx}. <strong>{src['source']}</strong> — Page {src['page']} {sim_pct}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                if response.retrieved_chunks:
                    with st.expander("🔍 Retrieved Context"):
                        for idx, chunk in enumerate(response.retrieved_chunks, 1):
                            st.markdown(
                                f"""
                                <div class="context-box">
                                    <div class="context-meta">
                                        <span>Chunk #{idx} | Source: {chunk.source} | Page: {chunk.page}</span>
                                        <span>Similarity: {int(chunk.similarity_score * 100)}%</span>
                                    </div>
                                    <div>{chunk.text}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Save to session history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "answer": response.answer,
                    "sources": response.sources,
                    "retrieved_chunks": response.retrieved_chunks
                })


# ==============================================================================
# PAGE 5: ABOUT
# ==============================================================================
elif st.session_state.selected_nav == "ℹ️ About":
    st.markdown(
        render_header(
            "System Architecture & Documentation",
            "Technical overview of the Retrieval-Augmented Generation pipeline powering the Aviation Knowledge Assistant."
        ),
        unsafe_allow_html=True
    )

    st.markdown(render_disclaimer(AVIATION_DISCLAIMER), unsafe_allow_html=True)

    st.markdown("""
    ### 🛫 Architectural Flow
    ```
    User Question
          ↓
    Dense Embedding Generation (all-MiniLM-L6-v2)
          ↓
    Persistent Vector Search (ChromaDB)
          ↓
    Relevance Filtering & Deduplication (Cosine Similarity Threshold)
          ↓
    Retrieved Institutional Context & Metadata (Page, Source)
          ↓
    Google Gemini (Prompt-Constrained Grounding)
          ↓
    Grounded Response with Clickable Source Provenance
    ```

    ---

    ### 🛠️ Technology Stack Breakdown
    - **Frontend Framework**: Streamlit with custom CSS and responsive aviation UI design tokens.
    - **Dense Embeddings**: `sentence-transformers` utilizing `all-MiniLM-L6-v2` (384-dimensional normalized vectors).
    - **Vector Database**: `ChromaDB` configured for persistent storage (`chroma_db/`) using cosine distance metric.
    - **Document Parsing**:
      - `PyMuPDF` (`pymupdf` / `fitz`) for PDF page-by-page extraction with scanned/OCR detection.
      - `python-docx` for structured Word document parsing.
      - Native Python I/O with multiple encoding fallbacks (UTF-8, Latin-1, CP1252) for TXT.
    - **Text Chunking**: Sentence-boundary aware overlapping chunker (500-800 words, 50-100 overlap).
    - **Language Model**: Google Gemini API via official SDK with strict anti-hallucination grounding instructions.
    - **Security**: Zero hardcoded credentials; environment variables managed via `.env` and `.env.example`.

    ---

    ### ⚠️ Compliance & Safety Constraints
    1. **Strict Context Adherence**: If query answers are not found in the uploaded documents, the assistant responds with:
       *"I could not find sufficient information in the uploaded aviation documents to answer this question."*
    2. **Source Provenance**: Every generated response cites the exact manual and page number where facts originated.
    3. **Advisory Role**: The platform serves exclusively as an informational retrieval tool, not an autonomous aviation decision system.
    """)
