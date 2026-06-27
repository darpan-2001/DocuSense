from typing import List, Dict, Any
from datetime import datetime
from config.logging import log


class ConversationMemory:
    """Session-based conversation memory."""
    
    def __init__(self, max_turns: int = None):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep
        """
        from config.settings import settings
        self.max_turns = max_turns or settings.MAX_CONVERSATION_TURNS
        self.history: List[Dict[str, Any]] = []
        
        log.info(f"Initialized ConversationMemory with max_turns={self.max_turns}")
    
    def add_exchange(self, user_query: str, assistant_response: str) -> None:
        """
        Add a conversation exchange to memory.
        
        Args:
            user_query: User's question
            assistant_response: Assistant's response
        """
        exchange = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.history.append(exchange)
        
        # Trim to max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
        
        log.debug(f"Added exchange to memory. Total turns: {len(self.history)}")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Returns:
            List of conversation exchanges
        """
        return self.history
    
    def get_formatted_history(self) -> str:
        """
        Get formatted conversation history for prompting.
        
        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""
        
        formatted = []
        for exchange in self.history:
            formatted.append(f"User: {exchange['user']}")
            formatted.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(formatted)
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM messages.
        
        Returns:
            List of message dictionaries with role and content
        """
        messages = []
        for exchange in self.history:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        return messages
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []
        log.info("Cleared conversation history")
    
    def get_turn_count(self) -> int:
        """
        Get the number of conversation turns.
        
        Returns:
            Number of turns
        """
        return len(self.history)
