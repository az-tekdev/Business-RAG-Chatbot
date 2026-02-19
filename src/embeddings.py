"""
Embedding utilities for vector store.
"""
import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding models for document and query encoding.
    """
    
    def __init__(
        self,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_type: Type of embedding model ('openai' or 'sentence-transformers')
            model_name: Name of the model (for sentence-transformers)
        """
        self.model_type = model_type or settings.embedding_model
        self.model_name = model_name or settings.embedding_model_name
        self.embeddings = self._create_embeddings()
    
    def _create_embeddings(self) -> Embeddings:
        """Create embedding model instance."""
        if self.model_type == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
            
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-3-small",
            )
        
        elif self.model_type == "sentence-transformers":
            model_name = self.model_name or "all-MiniLM-L6-v2"
            logger.info(f"Using HuggingFace embeddings: {model_name}")
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        
        else:
            raise ValueError(
                f"Unsupported embedding model type: {self.model_type}. "
                "Choose 'openai' or 'sentence-transformers'"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
    
    def get_embeddings(self) -> Embeddings:
        """Get the embeddings instance."""
        return self.embeddings
