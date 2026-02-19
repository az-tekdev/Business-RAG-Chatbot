"""
Query refinement utilities for better retrieval.
"""
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import settings

logger = logging.getLogger(__name__)


class QueryRefiner:
    """
    Refines user queries for better retrieval performance.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize query refiner.
        
        Args:
            enabled: Whether to enable query refinement
            llm_model: LLM model for refinement
        """
        self.enabled = enabled
        
        if enabled:
            if not settings.openai_api_key:
                logger.warning("OpenAI API key not found. Query refinement disabled.")
                self.enabled = False
                return
            
            self.llm = ChatOpenAI(
                model_name=llm_model or settings.llm_model,
                temperature=0.3,  # Lower temperature for more focused refinement
                openai_api_key=settings.openai_api_key,
            )
            
            self.refinement_prompt = PromptTemplate(
                input_variables=["query", "chat_history"],
                template="""Given the following conversation history and user query, 
rephrase the query to be more specific and effective for document retrieval. 
Focus on key terms and concepts. If the query is already clear, return it as-is.

Chat History:
{chat_history}

User Query: {query}

Refined Query:""",
            )
            
            self.chain = LLMChain(llm=self.llm, prompt=self.refinement_prompt)
    
    def refine(
        self,
        query: str,
        chat_history: Optional[str] = None,
    ) -> str:
        """
        Refine a user query.
        
        Args:
            query: Original user query
            chat_history: Optional conversation history context
            
        Returns:
            Refined query string
        """
        if not self.enabled:
            return query
        
        try:
            if chat_history is None:
                chat_history = "No previous conversation."
            
            refined = self.chain.run(query=query, chat_history=chat_history)
            logger.info(f"Query refined: '{query}' -> '{refined}'")
            return refined.strip()
        
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}. Using original query.")
            return query
