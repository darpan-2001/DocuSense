from typing import Tuple
from src.llm.groq_client import GroqClient
from src.embeddings.embedding_model import EmbeddingModel
from config.logging import log


class HyDEGenerator:
    """Hypothetical Document Embeddings (HyDE) query generator."""
    
    def __init__(
        self,
        groq_client: GroqClient = None,
        embedding_model: EmbeddingModel = None
    ):
        """
        Initialize HyDE generator.
        
        Args:
            groq_client: Groq LLM client
            embedding_model: Embedding model
        """
        self.groq_client = groq_client or GroqClient()
        self.embedding_model = embedding_model or EmbeddingModel()
        
        self.hyde_prompt = """You are an expert at generating hypothetical answers to questions based on document context.
Generate a detailed, factual hypothetical answer to the following question.
The answer should be written as if it were extracted from a technical document.

Question: {query}

Hypothetical Answer:"""
        
        log.info("Initialized HyDEGenerator")
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer for the query.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical answer
        """
        log.info(f"Generating hypothetical answer for query: {query[:50]}...")
        
        prompt = self.hyde_prompt.format(query=query)
        hypothetical_answer = self.groq_client.generate(prompt)
        
        log.debug(f"Generated hypothetical answer: {hypothetical_answer[:100]}...")
        return hypothetical_answer
    
    def generate_hyde_embedding(self, query: str) -> list:
        """
        Generate HyDE embedding for the query.
        
        Args:
            query: User query
            
        Returns:
            Embedding of the hypothetical answer
        """
        # Generate hypothetical answer
        hypothetical_answer = self.generate_hypothetical_answer(query)
        
        # Embed the hypothetical answer
        embedding = self.embedding_model.embed_text(hypothetical_answer)
        
        log.info("Generated HyDE embedding")
        return embedding
    
    def enhance_query(self, query: str) -> Tuple[list, list]:
        """
        Enhance query using HyDE.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (original_query_embedding, hyde_embedding)
        """
        log.info(f"Enhancing query with HyDE: {query[:50]}...")
        
        # Generate original query embedding
        original_embedding = self.embedding_model.embed_text(query)
        
        # Generate HyDE embedding
        hyde_embedding = self.generate_hyde_embedding(query)
        
        return original_embedding, hyde_embedding
