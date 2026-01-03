import streamlit as st
import sys
import time
from pathlib import Path

# Fix pathing
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

st.set_page_config(page_title="🏗️ Code Architect Explorer", layout="wide")

def init_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None

def initialize_explorer(path: str):
    try:
        llm = Config.get_llm() # Uses llama-3.3-70b-versatile
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        
        documents = doc_processor.process_codebase(path)
        if not documents:
            return None, 0
            
        vector_store.create_vectorstore(documents)
        graph_builder = GraphBuilder(retriever=vector_store.get_retriever(), llm=llm)
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, 0

def main():
    init_session()
    st.title("🏗️ Architectural Codebase Explorer")

    # --- Sidebar: Repository Import ---
    with st.sidebar:
        st.header("📂 Import Codebase")
        repo_path = st.text_input("Local Folder Path:", placeholder="/path/to/your/repo")
        
        if st.button("🚀 Analyze Repository"):
            if repo_path:
                with st.spinner("Parsing code structures..."):
                    rag, count = initialize_explorer(repo_path)
                    if rag:
                        st.session_state.rag_system = rag
                        st.success(f"Analyzed {count} code blocks.")
            else:
                st.warning("Please enter a path.")

    # --- Main Chat Interface ---
    if not st.session_state.rag_system:
        st.info("👈 Please provide a local repository path in the sidebar to start the analysis.")
        return

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "diagram" in message and message["diagram"]:
                with st.expander("📊 View Architecture Diagram"):
                    st.code(message["diagram"], language="mermaid")

    # Chat Input
    if prompt := st.chat_input("Explain how the state management works..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing architecture..."):
                result = st.session_state.rag_system.run(prompt)
                
                # Show Answer
                st.markdown(result['answer'])
                
                # Show Diagram
                diagram_code = result.get('diagram')
                if diagram_code:
                    with st.expander("📊 View Architecture Diagram", expanded=True):
                        st.code(diagram_code, language="mermaid")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": result['answer'],
                    "diagram": diagram_code
                })

if __name__ == "__main__":
    main()