from typing import Dict, Any
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.context.context_builder import ContextBuilder
from src.prompts.rag_prompt import RAGPrompt
from src.llm.groq_client import GroqClient
from src.memory.memory_manager import MemoryManager
from config.logging import log


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""
    
    def __init__(self, memory_manager: MemoryManager = None):
        """
        Initialize RAG pipeline.
        
        Args:
            memory_manager: Memory manager instance
        """
        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder()
        self.rag_prompt = RAGPrompt()
        self.groq_client = GroqClient()
        self.memory_manager = memory_manager or MemoryManager()
        
        log.info("Initialized RAGPipeline")
    
    def process_query(
        self,
        query: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User question
            session_id: Session identifier
            
        Returns:
            Dictionary with answer and sources
        """
        log.info(f"Processing query for session {session_id}: {query[:50]}...")
        
        # Step 1: Retrieve relevant chunks using hybrid retrieval
        retrieved_chunks = self.hybrid_retriever.retrieve(
            query=query,
            session_id=session_id,
            top_k=20
        )
        
        if not retrieved_chunks:
            log.warning("No chunks retrieved from hybrid retrieval")
            return {
                "answer": "No relevant information found in the uploaded documents.",
                "sources": []
            }
        
        # Step 2: Rerank chunks using cross-encoder
        reranked_chunks = self.reranker.rerank(
            query=query,
            documents=retrieved_chunks,
            top_k=5
        )
        
        log.info(f"Reranked to {len(reranked_chunks)} chunks")
        
        # Step 3: Build context and extract sources
        context, sources = self.context_builder.build_context_with_sources(reranked_chunks)
        
        # Step 4: Get conversation history
        conversation_messages = self.memory_manager.get_messages_for_llm(session_id)
        
        # Step 5: Build prompt
        messages = self.rag_prompt.build_prompt_with_messages(
            query=query,
            context=context,
            conversation_messages=conversation_messages
        )
        
        # Step 6: Generate answer using LLM
        answer = self.groq_client.generate_with_history(messages)
        
        # Step 7: Save conversation to memory
        self.memory_manager.add_exchange(
            session_id=session_id,
            user_query=query,
            assistant_response=answer
        )
        
        log.info(f"Generated answer for session {session_id}")
        
        return {
            "answer": answer,
            "sources": sources
        }
