"""
Unit tests for document ingestion.
"""
import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion import DocumentIngester
from langchain_core.documents import Document


def create_test_file(content: str, extension: str, directory: Path) -> Path:
    """Create a test file with given content."""
    file_path = directory / f"test{extension}"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_document_ingester_initialization():
    """Test DocumentIngester initialization."""
    ingester = DocumentIngester(chunk_size=500, chunk_overlap=100)
    assert ingester.chunk_size == 500
    assert ingester.chunk_overlap == 100


def test_load_text_document(tmp_path):
    """Test loading a text document."""
    content = "This is a test document.\nIt has multiple lines.\nFor testing purposes."
    test_file = create_test_file(content, ".txt", tmp_path)
    
    ingester = DocumentIngester()
    documents = ingester.load_document(str(test_file))
    
    assert len(documents) > 0
    assert isinstance(documents[0], Document)
    assert "test.txt" in documents[0].metadata.get("file_name", "")


def test_load_markdown_document(tmp_path):
    """Test loading a markdown document."""
    content = "# Test Document\n\nThis is a **markdown** document."
    test_file = create_test_file(content, ".md", tmp_path)
    
    ingester = DocumentIngester()
    documents = ingester.load_document(str(test_file))
    
    assert len(documents) > 0
    assert isinstance(documents[0], Document)


def test_load_directory(tmp_path):
    """Test loading documents from a directory."""
    # Create multiple test files
    create_test_file("First document content.", ".txt", tmp_path)
    create_test_file("Second document content.", ".txt", tmp_path)
    create_test_file("# Markdown doc", ".md", tmp_path)
    
    ingester = DocumentIngester()
    documents = ingester.load_directory(str(tmp_path))
    
    assert len(documents) >= 3


def test_chunk_documents():
    """Test document chunking."""
    documents = [
        Document(
            page_content="A" * 2000,  # Long content
            metadata={"source": "test.txt"},
        ),
    ]
    
    ingester = DocumentIngester(chunk_size=500, chunk_overlap=100)
    chunks = ingester.chunk_documents(documents)
    
    assert len(chunks) > 1  # Should be split into multiple chunks
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_ingest_pipeline(tmp_path):
    """Test complete ingestion pipeline."""
    create_test_file("Test content for ingestion.", ".txt", tmp_path)
    
    ingester = DocumentIngester(chunk_size=100, chunk_overlap=20)
    chunks = ingester.ingest(str(tmp_path), is_directory=True)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_unsupported_file_type(tmp_path):
    """Test handling of unsupported file types."""
    test_file = tmp_path / "test.xyz"
    test_file.write_text("content")
    
    ingester = DocumentIngester()
    
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingester.load_document(str(test_file))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
