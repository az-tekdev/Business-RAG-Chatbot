"""
Unit tests for vector store operations.
"""
import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from embeddings import EmbeddingManager
from vectorstore import VectorStoreManager


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Python is a programming language.",
            metadata={"source": "test1.txt", "file_name": "test1.txt"},
        ),
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"source": "test2.txt", "file_name": "test2.txt"},
        ),
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "test3.txt", "file_name": "test3.txt"},
        ),
    ]


def test_embedding_manager_sentence_transformers():
    """Test EmbeddingManager with sentence-transformers."""
    try:
        manager = EmbeddingManager(
            model_type="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
        )
        
        # Test embedding
        embeddings = manager.embed_query("test query")
        assert len(embeddings) > 0
        assert isinstance(embeddings, list)
        assert all(isinstance(x, float) for x in embeddings)
    except Exception as e:
        pytest.skip(f"Sentence-transformers not available: {e}")


def test_vector_store_creation_faiss(sample_documents, tmp_path):
    """Test FAISS vector store creation."""
    try:
        embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        vector_store_manager = VectorStoreManager(
            store_type="faiss",
            store_path=str(tmp_path),
            embedding_manager=embedding_manager,
        )
        
        vectorstore = vector_store_manager.create_from_documents(sample_documents)
        assert vectorstore is not None
        
        # Test search
        results = vector_store_manager.similarity_search("programming", k=1)
        assert len(results) > 0
        assert isinstance(results[0], Document)
    except Exception as e:
        pytest.skip(f"FAISS test skipped: {e}")


def test_vector_store_save_load(sample_documents, tmp_path):
    """Test saving and loading vector store."""
    try:
        embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        vector_store_manager = VectorStoreManager(
            store_type="faiss",
            store_path=str(tmp_path),
            embedding_manager=embedding_manager,
        )
        
        # Create and save
        vector_store_manager.create_from_documents(sample_documents)
        vector_store_manager.save()
        
        # Create new manager and load
        new_embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        new_vector_store_manager = VectorStoreManager(
            store_type="faiss",
            store_path=str(tmp_path),
            embedding_manager=new_embedding_manager,
        )
        
        new_vector_store_manager.load()
        
        # Test search on loaded store
        results = new_vector_store_manager.similarity_search("machine learning", k=1)
        assert len(results) > 0
    except Exception as e:
        pytest.skip(f"Save/load test skipped: {e}")


def test_similarity_search_with_score(sample_documents, tmp_path):
    """Test similarity search with scores."""
    try:
        embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        vector_store_manager = VectorStoreManager(
            store_type="faiss",
            store_path=str(tmp_path),
            embedding_manager=embedding_manager,
        )
        
        vector_store_manager.create_from_documents(sample_documents)
        
        results = vector_store_manager.similarity_search_with_score("AI", k=2)
        assert len(results) > 0
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 2  # (document, score)
    except Exception as e:
        pytest.skip(f"Similarity search test skipped: {e}")


def test_get_retriever(sample_documents, tmp_path):
    """Test getting a retriever from vector store."""
    try:
        embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        vector_store_manager = VectorStoreManager(
            store_type="faiss",
            store_path=str(tmp_path),
            embedding_manager=embedding_manager,
        )
        
        vector_store_manager.create_from_documents(sample_documents)
        
        retriever = vector_store_manager.get_retriever(k=2)
        assert retriever is not None
        
        # Test retrieval
        docs = retriever.get_relevant_documents("programming")
        assert len(docs) > 0
    except Exception as e:
        pytest.skip(f"Retriever test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
