"""
Document ingestion and chunking utilities.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Handles document loading and chunking for RAG.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the document ingester.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_ext in [".txt", ".md"]:
                if file_ext == ".md":
                    loader = UnstructuredMarkdownLoader(str(file_path))
                else:
                    loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = str(file_path)
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_type"] = file_ext
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects from all files
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        supported_extensions = {".pdf", ".txt", ".md"}
        all_documents = []
        
        for file_path in directory_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
        
        logger.info(f"Loaded {len(all_documents)} documents from {directory_path}")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
        
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def ingest(
        self,
        source: str,
        is_directory: bool = True,
    ) -> List[Document]:
        """
        Complete ingestion pipeline: load and chunk documents.
        
        Args:
            source: Path to file or directory
            is_directory: Whether source is a directory
            
        Returns:
            List of chunked Document objects
        """
        if is_directory:
            documents = self.load_directory(source)
        else:
            documents = self.load_document(source)
        
        chunks = self.chunk_documents(documents)
        return chunks
