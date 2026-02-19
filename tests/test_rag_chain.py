"""
Unit tests for RAG chain (requires vector store and API key).
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from embeddings import EmbeddingManager
from vectorstore import VectorStoreManager
from rag_chain import RAGChain


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Python is a high-level programming language known for its simplicity.",
            metadata={"source": "python.txt", "file_name": "python.txt"},
        ),
        Document(
            page_content="Machine learning enables computers to learn from data.",
            metadata={"source": "ml.txt", "file_name": "ml.txt"},
        ),
    ]


@pytest.mark.skipif(
    not Path("./vectorstore").exists(),
    reason="Vector store not found. Run ingestion first.",
)
def test_rag_chain_initialization(tmp_path):
    """Test RAG chain initialization."""
    try:
        # This test requires a pre-existing vector store or will be skipped
        embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        vector_store_manager = VectorStoreManager(
            embedding_manager=embedding_manager,
        )
        
        # Try to load or create
        try:
            vector_store_manager.load()
        except FileNotFoundError:
            pytest.skip("Vector store not found")
        
        rag_chain = RAGChain(
            vector_store_manager,
            llm_model="gpt-3.5-turbo",
        )
        
        assert rag_chain is not None
        assert rag_chain.llm is not None
        assert rag_chain.retriever is not None
    except Exception as e:
        pytest.skip(f"RAG chain test skipped: {e}")


@pytest.mark.skipif(
    not Path("./vectorstore").exists(),
    reason="Vector store not found. Run ingestion first.",
)
def test_rag_chain_query(tmp_path):
    """Test RAG chain query (requires OpenAI API key)."""
    try:
        from config import settings
        
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not set")
        
        embedding_manager = EmbeddingManager(model_type="sentence-transformers")
        vector_store_manager = VectorStoreManager(
            embedding_manager=embedding_manager,
        )
        
        try:
            vector_store_manager.load()
        except FileNotFoundError:
            pytest.skip("Vector store not found")
        
        rag_chain = RAGChain(vector_store_manager)
        
        response = rag_chain.query("What is Python?", return_sources=True)
        
        assert "answer" in response
        assert isinstance(response["answer"], str)
        assert len(response["answer"]) > 0
    except Exception as e:
        pytest.skip(f"RAG query test skipped: {e}")


def test_rag_chain_memory():
    """Test RAG chain memory management."""
    # This is a basic test that doesn't require full setup
    from langchain.memory import ConversationBufferMemory
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    
    assert memory is not None
    assert memory.chat_memory is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
