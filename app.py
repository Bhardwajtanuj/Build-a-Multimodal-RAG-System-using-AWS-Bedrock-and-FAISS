"""Streamlit app for the Multimodal RAG System."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from src.embeddings import BedrockEmbeddings
from src.ingest import ingest_all, ingest_documents, ingest_images
from src.rag import MultimodalRAG
from src.vector_store import FAISSVectorStore
from src.config import DOCUMENTS_DIR, IMAGES_DIR


st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multimodal RAG System")
st.caption("AWS Bedrock + FAISS | Text & Image Retrieval & Generation")

# Sidebar
with st.sidebar:
    st.header("Index Management")
    if st.button("🔄 Ingest All Documents & Images"):
        with st.spinner("Ingesting... (this may take a minute)"):
            try:
                doc_count, img_count = ingest_all()
                st.success(f"Added {doc_count} text chunks and {img_count} images.")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.header("Add Content")
    st.markdown("Place documents in `data/documents/` and images in `data/images/`.")
    st.code(f"""
Documents: {DOCUMENTS_DIR}
Images: {IMAGES_DIR}
    """, language="text")

# Main area
tab1, tab2, tab3 = st.tabs(["💬 Query", "📤 Ingest", "ℹ️ About"])

with tab1:
    embeddings = BedrockEmbeddings()
    vector_store = FAISSVectorStore(dimension=embeddings.dimension)
    rag = MultimodalRAG(embedding_model=embeddings, vector_store=vector_store)

    if not rag.load_index():
        st.warning("No index found. Ingest documents and images first using the sidebar or Ingest tab.")

    query_type = st.radio("Query type", ["Text", "Image"], horizontal=True)

    if query_type == "Text":
        question = st.text_area("Ask a question", placeholder="e.g., What is AWS Bedrock?")
        query_image = None
    else:
        question = st.text_input("Optional: Add a text question to refine results", placeholder="Describe what you're looking for")
        query_image = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])

    if st.button("🔎 Search & Generate"):
        if not question and not query_image:
            st.error("Provide a text question or an image.")
        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    if query_image:
                        # Save upload temporarily
                        tmp = Path("tmp_query.png")
                        tmp.write_bytes(query_image.getvalue())
                        try:
                            answer, docs = rag.query(question or "What is in this image?", query_image_path=str(tmp))
                        finally:
                            tmp.unlink(missing_ok=True)
                    else:
                        answer, docs = rag.query(question)

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Retrieved Sources")
                    for i, d in enumerate(docs, 1):
                        with st.expander(f"**{i+1}. [{d.get('type', 'text')}]** {d.get('source', 'N/A')}"):
                            st.write(d.get("content", ""))
                except Exception as e:
                    st.error(str(e))

with tab2:
    st.subheader("Ingest Documents & Images")
    st.markdown("""
    - **Documents**: PDF, TXT, MD, DOCX in `data/documents/`
    - **Images**: JPG, PNG in `data/images/`

    Click **Ingest All** in the sidebar to build the FAISS index.
    """)
    if st.button("🔄 Run Ingestion", key="ingest_tab"):
        with st.spinner("Ingesting..."):
            try:
                doc_count, img_count = ingest_all()
                st.success(f"Added {doc_count} text chunks and {img_count} images.")
            except Exception as e:
                st.error(str(e))

with tab3:
    st.subheader("About")
    st.markdown("""
    **Multimodal RAG System** combines:
    - **AWS Bedrock** for embeddings (Nova Multimodal or Titan) and generation (Claude)
    - **FAISS** for fast similarity search
    - Support for **text documents** and **images** in a unified vector space

    **Setup**: Configure `AWS_REGION` and credentials. Nova Multimodal is available in `us-east-1`.
    """)
