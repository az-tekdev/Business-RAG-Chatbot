"""
Vector store management for document indexing and retrieval.
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.vectorstores import VectorStore

from config import settings
from embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store creation, loading, and persistence.
    """
    
    def __init__(
        self,
        store_type: Optional[str] = None,
        store_path: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        """
        Initialize vector store manager.
        
        Args:
            store_type: Type of vector store ('faiss' or 'chroma')
            store_path: Path to store/load vector store
            embedding_manager: EmbeddingManager instance
        """
        self.store_type = store_type or settings.vector_store_type
        self.store_path = store_path or settings.vector_store_path
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vectorstore: Optional[VectorStore] = None
    
    def create_from_documents(
        self,
        documents: List[Document],
    ) -> VectorStore:
        """
        Create vector store from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            VectorStore instance
        """
        logger.info(f"Creating {self.store_type} vector store from {len(documents)} documents")
        
        if self.store_type == "faiss":
            self.vectorstore = FAISS.from_documents(
                documents,
                self.embedding_manager.get_embeddings(),
            )
        elif self.store_type == "chroma":
            self.vectorstore = Chroma.from_documents(
                documents,
                self.embedding_manager.get_embeddings(),
                persist_directory=self.store_path,
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        logger.info("Vector store created successfully")
        return self.vectorstore
    
    def save(self, path: Optional[str] = None):
        """
        Save vector store to disk.
        
        Args:
            path: Optional custom path to save (uses default if None)
        """
        if self.vectorstore is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_path = path or self.store_path
        
        if self.store_type == "faiss":
            os.makedirs(save_path, exist_ok=True)
            self.vectorstore.save_local(save_path)
            logger.info(f"FAISS vector store saved to {save_path}")
        
        elif self.store_type == "chroma":
            # Chroma auto-saves, but we can ensure persistence
            if hasattr(self.vectorstore, '_persist_directory'):
                logger.info(f"Chroma vector store persisted to {save_path}")
            else:
                logger.warning("Chroma vector store may not be persisted")
    
    def load(self, path: Optional[str] = None) -> VectorStore:
        """
        Load vector store from disk.
        
        Args:
            path: Optional custom path to load from (uses default if None)
            
        Returns:
            VectorStore instance
        """
        load_path = path or self.store_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        logger.info(f"Loading {self.store_type} vector store from {load_path}")
        
        if self.store_type == "faiss":
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embedding_manager.get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        elif self.store_type == "chroma":
            self.vectorstore = Chroma(
                persist_directory=load_path,
                embedding_function=self.embedding_manager.get_embeddings(),
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        logger.info("Vector store loaded successfully")
        return self.vectorstore
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        if filter:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter,
            )
            # Unpack (doc, score) tuples
            documents = [doc for doc, score in results]
        else:
            documents = self.vectorstore.similarity_search(query, k=k)
        
        return documents
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        if self.store_type == "faiss":
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        else:
            # Chroma compatibility
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return results
    
    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        """
        Get a retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search ('similarity' or 'mmr')
            
        Returns:
            Retriever instance
        """
        if self.vectorstore is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        if search_type == "similarity":
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k},
            )
        elif search_type == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2},
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
