"""Document processing module for Code-Aware analysis (Local & GitHub)"""
import os
from typing import List
from pathlib import Path
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """Handles code loading from local folders and GitHub repositories"""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def process_local_repo(self, path: str) -> List[Document]:
        """Loads and splits code from a local directory."""
        repo_path = str(Path(path).absolute())
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*.py",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
        )
        docs = loader.load()
        return self._get_splitter().split_documents(docs)

    def process_github_repo(self, clone_url: str) -> List[Document]:
        """Clones a GitHub repo to a temporary directory and processes it."""
        # Create a unique temp path for this repo
        repo_name = clone_url.split("/")[-1].replace(".git", "")
        temp_path = f"./temp_repos/{repo_name}"
        
        # Ensure directory exists
        os.makedirs("./temp_repos", exist_ok=True)

        loader = GitLoader(
            clone_url=clone_url,
            repo_path=temp_path,
            branch="main",
            file_filter=lambda file_path: file_path.endswith(".py")
        )
        docs = loader.load()
        return self._get_splitter().split_documents(docs)