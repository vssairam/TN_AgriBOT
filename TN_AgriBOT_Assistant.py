"""
TN_AgriBOT_Assistant.py
A Streamlit RAG Farmer Assistant for querying multiple PDF documents.

Run:
    streamlit run TN_AgriBOT_Assistant.py

Required environment variable:
    OPENAI_API_KEY
"""

import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------- Page setup ------------------------------------
st.set_page_config(
    page_title="TN AgriBOT | Farmer Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme / CSS -----------------------------------
st.markdown(
    """
    <style>
    :root {
        --agri-green: #245b35;
        --leaf-green: #4f8b45;
        --sun-gold: #f2c14e;
        --cream: #fff9e9;
        --earth: #704b2b;
    }
    .stApp {
        background:
          linear-gradient(rgba(250, 255, 244, 0.94), rgba(250, 255, 244, 0.97)),
          url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2000&q=85");
        background-size: cover;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0.45); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e7f2df 0%, #fff8e5 100%);
        border-right: 1px solid #c8ddbd;
    }
    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 18px;
        color: white;
        background: linear-gradient(115deg, rgba(28,79,43,.96), rgba(79,139,69,.88)),
                    url("https://images.unsplash.com/photo-1499529112087-3cb3b73nec?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        box-shadow: 0 8px 24px rgba(26, 72, 38, .16);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.15rem; }
    .hero p { margin: .45rem 0 0 0; font-size: 1.02rem; }
    .tip {
        border-left: 5px solid #e4b43c;
        background: #fff7dc;
        padding: .8rem 1rem;
        border-radius: 8px;
        color: #4c4025;
        margin: .5rem 0 1rem 0;
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #3d7545;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"] {
        background: #2e6a3c;
        color: white;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(75, 119, 67, .14);
        border-radius: 14px;
        padding: .5rem .8rem;
        background: rgba(255,255,255,.78);
    }
    .small-note { color: #53664d; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Session state ---------------------------------
DEFAULTS = {
    "vectorstore": None,
    "chunk_count": 0,
    "indexed_files": [],
    "messages": [],
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, list) else value


# ----------------------------- Helpers ---------------------------------------
def get_api_key() -> str:
    """Read the API key from the process environment; never hard-code it."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def build_faiss_index(uploaded_files, api_key: str):
    """Load all uploaded PDFs, chunk their pages, and build one FAISS index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    processed_names = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            continue

        # A unique temporary file avoids filename collisions across uploads.
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            pages = PyPDFLoader(temp_path).load()
            # Keep the original uploaded filename in metadata for citations.
            for page_doc in pages:
                page_doc.metadata["source"] = uploaded_file.name
                # PyPDFLoader page metadata is zero-based; display as 1-based.
                page_doc.metadata["page_number"] = int(page_doc.metadata.get("page", 0)) + 1

            chunks = splitter.split_documents(pages)
            # Skip empty / image-only pages that contain no extractable text.
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            all_chunks.extend(chunks)
            if chunks:
                processed_names.append(uploaded_file.name)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    if not all_chunks:
        raise ValueError(
            "No readable text was found in the selected PDFs. "
            "They may be scanned/image-only PDFs; OCR is not included in this version."
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, len(all_chunks), processed_names


def answer_question(question: str, vectorstore, api_key: str):
    """Retrieve relevant chunks and answer only from their contents."""
    retrieved_docs = vectorstore.similarity_search(question, k=5)

    context_parts = []
    for index, doc in enumerate(retrieved_docs, start=1):
        filename = Path(str(doc.metadata.get("source", "Unknown PDF"))).name
        page = doc.metadata.get("page_number")
        if page is None:
            page = int(doc.metadata.get("page", 0)) + 1
        context_parts.append(
            f"[Source {index}: {filename}, page {page}]\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = """You are TN AgriBOT, a helpful Farmer Assistant for Tamil Nadu.
Answer the user's question using only the supplied PDF context.
If the context does not contain enough information, say so clearly and do not guess.
Use practical, easy-to-understand language. Where relevant, organize the answer into
clear steps or bullets. Do not present uncertain agricultural, pesticide, fertilizer,
or animal-health advice as a guaranteed recommendation. Encourage users to follow
local agricultural department guidance and product labels for safety-critical decisions.
Cite factual claims with the source label, including the PDF filename and page number,
for example [Source: Soil_Guide.pdf, page 4]."""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=api_key,
    )
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Question:\n{question}\n\nPDF context:\n{context}\n\n"
                "Answer from the context and include filename/page citations."
            ),
        ]
    )
    return response.content, retrieved_docs


def render_sources(docs):
    """Show retrieved evidence as a source list the user can validate."""
    if not docs:
        return
    st.markdown("**📚 Sources to verify**")
    seen = set()
    for doc in docs:
        filename = Path(str(doc.metadata.get("source", "Unknown PDF"))).name
        page = doc.metadata.get("page_number")
        if page is None:
            page = int(doc.metadata.get("page", 0)) + 1
        key = (filename, page, doc.page_content[:80])
        if key in seen:
            continue
        seen.add(key)
        with st.expander(f"📄 {filename} — page {page}"):
            st.write(doc.page_content)


# ----------------------------- Header ----------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌾 TN AgriBOT</h1>
      <p>Your PDF-powered Farmer Assistant · Ask questions across multiple agriculture documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1], vertical_alignment="center")
with left:
    st.markdown(
        """
        <div class="tip">
        <b>How to use:</b> Upload your agriculture PDFs in the sidebar, build the shared
        knowledge base, then ask questions in the chat. Each answer includes source files
        and page numbers so you can check the original document.
        </div>
        """,
        unsafe_allow_html=True,
    )
with right:
    st.image(
        "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1000&q=80",
        caption="Healthy soil, healthy crops, stronger farms",
        use_container_width=True,
    )


# ----------------------------- Sidebar ---------------------------------------
with st.sidebar:
    st.markdown("## 🌱 Farm Knowledge Center")
    st.write("Upload one or more text-readable PDF guides, reports, or manuals.")
    uploaded_files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="All selected PDFs are combined into one FAISS vector database.",
    )

    api_key_present = bool(get_api_key())
    if api_key_present:
        st.success("OpenAI API key detected in environment.")
    else:
        st.error("OPENAI_API_KEY was not found in the environment.")

    build_clicked = st.button(
        "🌿 Build / Replace Knowledge Base",
        type="primary",
        use_container_width=True,
        disabled=(not uploaded_files or not api_key_present),
    )

    if build_clicked:
        try:
            with st.spinner("Reading PDFs, creating chunks, embedding text, and building FAISS..."):
                new_store, new_count, new_files = build_faiss_index(
                    uploaded_files, get_api_key()
                )
            st.session_state.vectorstore = new_store
            st.session_state.chunk_count = new_count
            st.session_state.indexed_files = new_files
            st.session_state.messages = []
            st.success("Knowledge base built successfully.")
        except Exception as exc:
            st.error(f"Could not build the knowledge base: {exc}")

    st.divider()
    st.markdown("### 📊 Knowledge Base Status")
    if st.session_state.vectorstore is not None:
        st.metric("Text chunks indexed", st.session_state.chunk_count)
        st.caption(f"PDFs indexed: {len(st.session_state.indexed_files)}")
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")
    else:
        st.info("Upload PDFs and build the knowledge base to begin.")

    if st.button("🧹 Clear chat / search results", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("♻️ Reset knowledge base", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.chunk_count = 0
        st.session_state.indexed_files = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 💡 Try asking")
    st.markdown(
        "- What does the soil guide recommend before sowing?\n"
        "- Summarize the irrigation advice in these PDFs.\n"
        "- What are the symptoms of the crop disease described?\n"
        "- Compare the recommendations in the uploaded documents."
    )
    st.markdown(
        '<p class="small-note">Answers depend on the uploaded documents. Verify important '
        'farm decisions with your local agricultural extension office.</p>',
        unsafe_allow_html=True,
    )


# ----------------------------- Chat interface --------------------------------
st.markdown("## 💬 Ask TN AgriBOT")
if st.session_state.vectorstore is None:
    st.info("The chat will be enabled after you build the knowledge base from your PDFs.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask a question about your uploaded agriculture PDFs...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your farm documents and preparing an answer..."):
                try:
                    answer, source_docs = answer_question(
                        question, st.session_state.vectorstore, get_api_key()
                    )
                    st.markdown(answer)
                    render_sources(source_docs)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": source_docs,
                        }
                    )
                except Exception as exc:
                    error_message = (
                        f"I couldn't complete the search: {exc}\n\n"
                        "Check your OpenAI API key, API access/billing, internet connection, "
                        "and the selected model configuration."
                    )
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message, "sources": []}
                    )

st.caption(
    "TN AgriBOT is a document-based assistant, not a substitute for official agricultural "
    "extension advice. Scanned PDFs may need OCR before they can be searched."
)
