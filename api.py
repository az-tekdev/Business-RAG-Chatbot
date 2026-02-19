#!/usr/bin/env python3
"""
FastAPI server for the RAG chatbot API.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from config import settings
from embeddings import EmbeddingManager
from vectorstore import VectorStoreManager
from rag_chain import RAGChain
from query_refinement import QueryRefiner
from ingestion import DocumentIngester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Business RAG Chatbot API",
    description="REST API for RAG-based document Q&A",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG components (initialized on startup)
rag_chains: Dict[str, RAGChain] = {}
query_refiners: Dict[str, QueryRefiner] = {}


# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    session_id: Optional[str] = Field(default="default", description="Session ID for conversation")
    refine_query: bool = Field(default=True, description="Whether to refine the query")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    question: str = Field(..., description="Original question")
    sources: Optional[List[Dict[str, str]]] = Field(default=None, description="Source documents")
    error: Optional[bool] = Field(default=False, description="Whether an error occurred")


class IngestRequest(BaseModel):
    data_dir: str = Field(..., description="Directory containing documents")
    chunk_size: Optional[int] = Field(default=None, description="Chunk size")
    chunk_overlap: Optional[int] = Field(default=None, description="Chunk overlap")


class IngestResponse(BaseModel):
    success: bool = Field(..., description="Whether ingestion succeeded")
    message: str = Field(..., description="Status message")
    num_chunks: Optional[int] = Field(default=None, description="Number of chunks indexed")


def get_or_create_rag_chain(session_id: str = "default") -> RAGChain:
    """Get or create RAG chain for a session."""
    if session_id not in rag_chains:
        try:
            embedding_manager = EmbeddingManager()
            vector_store_manager = VectorStoreManager(
                embedding_manager=embedding_manager,
            )
            vector_store_manager.load()
            
            rag_chain = RAGChain(vector_store_manager)
            rag_chains[session_id] = rag_chain
            
            if session_id not in query_refiners:
                query_refiners[session_id] = QueryRefiner(enabled=True)
            
            logger.info(f"Created RAG chain for session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to create RAG chain: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {str(e)}")
    
    return rag_chains[session_id]


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    logger.info("Starting Business RAG Chatbot API...")
    try:
        # Pre-initialize default session
        get_or_create_rag_chain("default")
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.warning(f"Could not pre-initialize RAG: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Business RAG Chatbot API",
        "version": "0.1.0",
        "endpoints": {
            "/query": "POST - Query the RAG system",
            "/ingest": "POST - Ingest documents",
            "/health": "GET - Health check",
            "/sessions/{session_id}/clear": "POST - Clear session memory",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try to get default RAG chain
        get_or_create_rag_chain("default")
        return {"status": "healthy", "vector_store_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: Query request with question and session ID
        
    Returns:
        Query response with answer and sources
    """
    try:
        # Get RAG chain for session
        rag_chain = get_or_create_rag_chain(request.session_id)
        query_refiner = query_refiners.get(request.session_id)
        
        # Refine query if enabled
        query_text = request.question
        if request.refine_query and query_refiner:
            chat_history = "\n".join([
                f"{msg.content}" for msg in rag_chain.get_chat_history()[-4:]
            ])
            query_text = query_refiner.refine(
                request.question,
                chat_history if chat_history else None,
            )
        
        # Query RAG chain
        response = rag_chain.query(query_text, return_sources=True)
        
        return QueryResponse(
            answer=response.get("answer", "I couldn't generate an answer."),
            question=request.question,
            sources=response.get("sources"),
            error=response.get("error", False),
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest documents into the vector store.
    
    Args:
        request: Ingest request with data directory
        background_tasks: FastAPI background tasks
        
    Returns:
        Ingest response with status
    """
    try:
        data_dir = Path(request.data_dir)
        if not data_dir.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {data_dir}")
        
        # Run ingestion in background
        def run_ingestion():
            try:
                ingester = DocumentIngester(
                    chunk_size=request.chunk_size or settings.chunk_size,
                    chunk_overlap=request.chunk_overlap or settings.chunk_overlap,
                )
                
                embedding_manager = EmbeddingManager()
                vector_store_manager = VectorStoreManager(
                    embedding_manager=embedding_manager,
                )
                
                documents = ingester.ingest(str(data_dir), is_directory=True)
                
                if not documents:
                    logger.warning("No documents found")
                    return
                
                vector_store_manager.create_from_documents(documents)
                vector_store_manager.save()
                
                # Clear existing RAG chains to force reload
                rag_chains.clear()
                query_refiners.clear()
                
                logger.info(f"Ingestion completed: {len(documents)} chunks")
            except Exception as e:
                logger.error(f"Ingestion error: {e}", exc_info=True)
        
        background_tasks.add_task(run_ingestion)
        
        return IngestResponse(
            success=True,
            message=f"Ingestion started for {data_dir}. Processing in background.",
        )
    
    except Exception as e:
        logger.error(f"Ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str):
    """
    Clear conversation memory for a session.
    
    Args:
        session_id: Session ID to clear
        
    Returns:
        Success message
    """
    try:
        if session_id in rag_chains:
            rag_chains[session_id].clear_memory()
            return {"message": f"Session {session_id} cleared successfully"}
        else:
            return {"message": f"Session {session_id} not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
