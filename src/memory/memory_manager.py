from typing import Dict
from src.memory.conversation_memory import ConversationMemory
from config.logging import log


class MemoryManager:
    """Manager for session-based conversation memories."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.memories: Dict[str, ConversationMemory] = {}
        log.info("Initialized MemoryManager")
    
    def get_memory(self, session_id: str) -> ConversationMemory:
        """
        Get or create conversation memory for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationMemory instance
        """
        if session_id not in self.memories:
            self.memories[session_id] = ConversationMemory()
            log.info(f"Created new memory for session: {session_id}")
        
        return self.memories[session_id]
    
    def add_exchange(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str
    ) -> None:
        """
        Add a conversation exchange to session memory.
        
        Args:
            session_id: Session identifier
            user_query: User's question
            assistant_response: Assistant's response
        """
        memory = self.get_memory(session_id)
        memory.add_exchange(user_query, assistant_response)
        log.debug(f"Added exchange to memory for session: {session_id}")
    
    def get_history(self, session_id: str) -> list:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation exchanges
        """
        if session_id not in self.memories:
            return []
        
        return self.memories[session_id].get_history()
    
    def get_messages_for_llm(self, session_id: str) -> list:
        """
        Get formatted messages for LLM for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries
        """
        if session_id not in self.memories:
            return []
        
        return self.memories[session_id].get_messages_for_llm()
    
    def clear_memory(self, session_id: str) -> None:
        """
        Clear conversation memory for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.memories:
            self.memories[session_id].clear()
            del self.memories[session_id]
            log.info(f"Cleared memory for session: {session_id}")
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session has memory.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.memories
