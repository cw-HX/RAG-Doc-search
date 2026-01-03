import streamlit as st
import sys
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

def initialize_explorer(source: str, is_github: bool):
    try:
        llm = Config.get_llm()
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        
        with st.spinner(f"Analyzing {'GitHub' if is_github else 'Local'} repository..."):
            if is_github:
                documents = doc_processor.process_github_repo(source)
            else:
                documents = doc_processor.process_local_repo(source)
        
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

    # --- Sidebar: Source Selection ---
    with st.sidebar:
        st.header("📂 Import Source")
        import_type = st.radio("Choose Source Type:", ["Local Directory", "GitHub Repository"])
        
        if import_type == "Local Directory":
            repo_input = st.text_input("Local Folder Path:", placeholder="/Users/name/projects/my-repo")
            is_github = False
        else:
            repo_input = st.text_input("GitHub Clone URL:", placeholder="https://github.com/user/repo.git")
            is_github = True
        
        if st.button("🚀 Analyze Codebase"):
            if repo_input:
                rag, count = initialize_explorer(repo_input, is_github)
                if rag:
                    st.session_state.rag_system = rag
                    st.success(f"Success! Indexed {count} code structures.")
            else:
                st.warning("Please enter a path or URL.")

    # --- Main Chat Interface ---
    if not st.session_state.rag_system:
        st.info("👈 Use the sidebar to import a repository and begin the architectural analysis.")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "diagram" in msg and msg["diagram"]:
                with st.expander("📊 Architecture Diagram"):
                    st.code(msg["diagram"], language="mermaid")

    if prompt := st.chat_input("Explain the main class interactions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            result = st.session_state.rag_system.run(prompt)
            st.markdown(result['answer'])
            
            diagram_code = result.get('diagram')
            if diagram_code:
                with st.expander("📊 Architecture Diagram", expanded=True):
                    st.code(diagram_code, language="mermaid")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "diagram": diagram_code
            })

if __name__ == "__main__":
    main()