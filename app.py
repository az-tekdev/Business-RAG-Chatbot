#!/usr/bin/env python3
"""
Streamlit web application for the RAG chatbot.
"""
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import settings
from embeddings import EmbeddingManager
from vectorstore import VectorStoreManager
from rag_chain import RAGChain
from query_refinement import QueryRefiner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Business RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_refiner" not in st.session_state:
    st.session_state.query_refiner = None


@st.cache_resource
def initialize_rag_chain():
    """Initialize RAG chain (cached for performance)."""
    try:
        embedding_manager = EmbeddingManager()
        vector_store_manager = VectorStoreManager(
            embedding_manager=embedding_manager,
        )
        
        # Try to load existing vector store
        try:
            vector_store_manager.load()
            logger.info("Loaded existing vector store")
        except FileNotFoundError:
            st.error(
                "Vector store not found. Please run ingestion first using: "
                "`python ingest.py --data-dir ./data`"
            )
            return None
        
        rag_chain = RAGChain(vector_store_manager)
        query_refiner = QueryRefiner(enabled=True)
        
        return rag_chain, query_refiner
    
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return None, None


def main():
    """Main Streamlit application."""
    st.title("🤖 Business RAG Chatbot")
    st.markdown(
        "Ask questions about your company's knowledge base. "
        "The chatbot uses RAG (Retrieval-Augmented Generation) to provide "
        "contextual answers based on your documents."
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API key
        if not settings.openai_api_key:
            st.error("⚠️ OPENAI_API_KEY not set in environment")
            st.info("Please set your OpenAI API key in `.env` file")
            return
        
        st.success("✅ OpenAI API key configured")
        
        # Vector store info
        st.subheader("Vector Store")
        st.text(f"Type: {settings.vector_store_type}")
        st.text(f"Path: {settings.vector_store_path}")
        
        # Model info
        st.subheader("Model Configuration")
        st.text(f"LLM: {settings.llm_model}")
        st.text(f"Embeddings: {settings.embedding_model}")
        st.text(f"Top-K: {settings.top_k_retrieval}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_memory()
            st.session_state.messages = []
            st.rerun()
    
    # Initialize RAG chain
    if st.session_state.rag_chain is None:
        with st.spinner("Initializing RAG system..."):
            rag_chain, query_refiner = initialize_rag_chain()
            if rag_chain is None:
                st.stop()
            st.session_state.rag_chain = rag_chain
            st.session_state.query_refiner = query_refiner
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['file_name']}")
                        st.text(source["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Optionally refine query
                    refined_query = prompt
                    if st.session_state.query_refiner:
                        chat_history = "\n".join([
                            f"{msg['role']}: {msg['content']}"
                            for msg in st.session_state.messages[-5:-1]
                        ])
                        refined_query = st.session_state.query_refiner.refine(
                            prompt,
                            chat_history if chat_history else None,
                        )
                    
                    # Query RAG chain
                    response = st.session_state.rag_chain.query(
                        refined_query,
                        return_sources=True,
                    )
                    
                    # Display answer
                    answer = response.get("answer", "I couldn't generate an answer.")
                    st.markdown(answer)
                    
                    # Display sources
                    if "sources" in response and response["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"], 1):
                                st.markdown(f"**Source {i}:** {source['file_name']}")
                                st.text(source["content"])
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": response.get("sources", []),
                    })
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Query error: {e}", exc_info=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


if __name__ == "__main__":
    main()
