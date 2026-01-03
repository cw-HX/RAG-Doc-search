"""Document processing module for Code-Aware analysis"""
from typing import List
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles code loading using Language-Aware parsing"""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_codebase(self, path: str) -> List[Document]:
        """
        Loads code files and splits them based on Language-specific syntax.
        Focuses on Python files by default.
        """
        loader = GenericLoader.from_custom_extractors(
            path,
            glob="**/*.py",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        docs = loader.load()
        
        # Use a splitter that understands Python syntax boundaries
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)