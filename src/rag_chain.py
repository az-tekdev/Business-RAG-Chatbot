"""
RAG chain implementation with conversational memory.
"""
import logging
from typing import List, Optional, Dict, Any
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForChainRun

from config import settings
from vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGChain:
    """
    RAG chain with conversational memory and context-aware generation.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store_manager: VectorStoreManager instance
            llm_model: LLM model name
            temperature: LLM temperature
            max_tokens: Maximum tokens for generation
            top_k: Number of documents to retrieve
        """
        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.top_k = top_k or settings.top_k_retrieval
        
        # Initialize LLM
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        self.llm = ChatOpenAI(
            model_name=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=settings.openai_api_key,
            streaming=True,
        )
        
        # Create memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        
        # Create retriever
        self.retriever = self.vector_store_manager.get_retriever(k=self.top_k)
        
        # Create custom prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Initialize chain
        self.chain = self._create_chain()
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create custom prompt template for RAG."""
        template = """You are a helpful AI assistant for a SaaS company's knowledge base. 
Use the following pieces of context to answer the question. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a helpful, accurate answer based on the context above. If the context doesn't contain 
enough information to answer the question, say so. Always cite your sources when possible.

Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"],
        )
    
    def _create_chain(self) -> ConversationalRetrievalChain:
        """Create the conversational retrieval chain."""
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
        )
        return chain
    
    def query(
        self,
        question: str,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the RAG chain.
        
        Args:
            question: User question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optionally sources
        """
        try:
            result = self.chain.invoke({"question": question})
            
            response = {
                "answer": result.get("answer", "I couldn't generate an answer."),
                "question": question,
            }
            
            if return_sources and "source_documents" in result:
                sources = result["source_documents"]
                response["sources"] = [
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("source", "Unknown"),
                        "file_name": doc.metadata.get("file_name", "Unknown"),
                    }
                    for doc in sources
                ]
            
            return response
        
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "answer": f"I encountered an error: {str(e)}",
                "question": question,
                "error": True,
            }
    
    def stream_query(self, question: str):
        """
        Stream query response.
        
        Args:
            question: User question
            
        Yields:
            Chunks of the response
        """
        try:
            for chunk in self.chain.stream({"question": question}):
                if "answer" in chunk:
                    yield chunk["answer"]
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"Error: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def get_chat_history(self) -> List[str]:
        """Get conversation history."""
        return self.memory.chat_memory.messages
