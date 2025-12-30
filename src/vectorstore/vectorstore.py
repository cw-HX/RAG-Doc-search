"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # modern import
from langchain_core.documents import Document 

class VectorStore:
    def __init__(self):
        # Uses your CPU, completely free and offline
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever = None
    
    def create_vectorstore(self, documents: List[Document]):
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
    
    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized.")
        return self.retriever

