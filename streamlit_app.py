"""Streamlit UI for Agentic RAG System - Dynamic URL Version"""
import streamlit as st
import os
import sys
import time
from pathlib import Path

# Fix pathing before importing local modules
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Local imports
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 Dynamic RAG Search",
    page_icon="🔗",
    layout="wide"
)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'user_urls' not in st.session_state:
        # Default with one empty string for the first input
        st.session_state.user_urls = [""]

@st.cache_resource
def initialize_rag(urls):
    """Initialize the RAG system with user-provided URLs"""
    try:
        # Filter out empty strings
        active_urls = [u for u in urls if u.strip()]
        if not active_urls:
            st.error("Please provide at least one valid URL.")
            return None, 0

        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Process specific user documents from URLs
        documents = doc_processor.process_urls(active_urls)
        
        if not documents:
            st.error("Could not extract content from the provided URLs.")
            return None, 0
            
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    init_session_state()
    
    st.title("🔗 Dynamic Article RAG Search")
    st.markdown("Add article links below, initialize the system, and ask questions.")

    # --- Sidebar for Dynamic URL Management ---
    with st.sidebar:
        st.header("📋 Article Management")
        st.info("Add as many links as you want.")
        
        # Dynamic inputs for URLs
        for i, url in enumerate(st.session_state.user_urls):
            col1, col2 = st.columns([8, 1])
            with col1:
                st.session_state.user_urls[i] = st.text_input(
                    f"Article URL {i+1}", 
                    value=url, 
                    key=f"url_input_{i}"
                )
            with col2:
                if st.button("🗑️", key=f"remove_{i}"):
                    st.session_state.user_urls.pop(i)
                    st.rerun()

        if st.button("➕ Add Another Article"):
            st.session_state.user_urls.append("")
            st.rerun()

        st.markdown("---")
        
        if st.button("🚀 Initialize Search Engine"):
            with st.spinner("Scraping and indexing articles..."):
                rag_system, num_chunks = initialize_rag(st.session_state.user_urls)
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.initialized = True
                    st.session_state.history = [] # Clear history for new articles
                    st.success(f"System Ready! Indexed {num_chunks} chunks.")

    # --- Main Search Interface ---
    if not st.session_state.initialized:
        st.warning("👈 Please add URLs and click 'Initialize Search Engine' in the sidebar to begin.")
    else:
        with st.form("search_form"):
            question = st.text_input("Ask a question about your articles:", placeholder="e.g., What are the main findings?")
            submit = st.form_submit_button("🔍 Search Articles")
        
        if submit and question:
            with st.spinner("Analyzing articles..."):
                start_time = time.time()
                result = st.session_state.rag_system.run(question)
                elapsed_time = time.time() - start_time
                
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                with st.expander("📄 Source Excerpts"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(doc.page_content[:500] + "...")

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Questions")
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:300]}...")
                st.caption(f"⏱️ {item['time']:.2f}s")

if __name__ == "__main__":
    main()