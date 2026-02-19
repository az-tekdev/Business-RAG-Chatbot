#!/usr/bin/env python3
"""
Script to ingest documents into the vector store.
"""
import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ingestion import DocumentIngester
from embeddings import EmbeddingManager
from vectorstore import VectorStoreManager
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector store for RAG"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for vector store (uses config default if not specified)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap for text splitting",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model type (openai or sentence-transformers)",
    )
    parser.add_argument(
        "--vector-store-type",
        type=str,
        default=None,
        choices=["faiss", "chroma"],
        help="Vector store type",
    )
    
    args = parser.parse_args()
    
    # Override config with CLI args
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap
    output_path = args.output or settings.vector_store_path
    
    logger.info("=" * 60)
    logger.info("Document Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Chunk overlap: {chunk_overlap}")
    
    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        ingester = DocumentIngester(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        embedding_manager = EmbeddingManager(
            model_type=args.embedding_model,
        )
        
        vector_store_manager = VectorStoreManager(
            store_type=args.vector_store_type,
            store_path=output_path,
            embedding_manager=embedding_manager,
        )
        
        # Ingest documents
        logger.info("Loading and chunking documents...")
        documents = ingester.ingest(str(data_dir), is_directory=True)
        
        if not documents:
            logger.warning("No documents found to ingest!")
            return 1
        
        # Create vector store
        logger.info("Creating vector store...")
        vector_store_manager.create_from_documents(documents)
        
        # Save vector store
        logger.info("Saving vector store...")
        vector_store_manager.save()
        
        logger.info("=" * 60)
        logger.info("Ingestion completed successfully!")
        logger.info(f"Indexed {len(documents)} document chunks")
        logger.info(f"Vector store saved to: {output_path}")
        logger.info("=" * 60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
