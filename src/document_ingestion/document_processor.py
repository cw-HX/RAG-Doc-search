"""Document processing module for dynamic URL loading and splitting"""
from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles document loading and processing from dynamic sources"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load document content from a single URL"""
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            print(f"Error loading {url}: {e}")
            return []

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Load documents primarily from URLs provided by the user.
        """
        docs: List[Document] = []
        for src in sources:
            src = src.strip()
            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))
            else:
                # Optional: Handle local files if a path is provided instead of a URL
                path = Path(src)
                if path.exists():
                    if path.is_dir():
                        from langchain_community.document_loaders import PyPDFDirectoryLoader
                        docs.extend(PyPDFDirectoryLoader(str(path)).load())
                    elif path.suffix.lower() == ".txt":
                        from langchain_community.document_loaders import TextLoader
                        docs.extend(TextLoader(str(path), encoding="utf-8").load())
        return docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split loaded documents into chunks"""
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """Complete pipeline to load and split documents from user URLs"""
        docs = self.load_documents(urls)
        return self.split_documents(docs)