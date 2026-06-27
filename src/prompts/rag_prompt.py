

class RAGPrompt:
    """RAG prompt template for question answering."""
    
    def __init__(self):
        """Initialize RAG prompt."""
        self.system_prompt = """You are a helpful assistant that answers questions based on the provided document context.
Your role is to provide accurate, well-sourced answers using only the information given in the context."""

        self.rag_template = """{system_prompt}

CONVERSATION HISTORY:
{conversation_history}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the document context above.
- If the answer is not available in the context, clearly state: "The information is not available in the uploaded documents."
- Do not use any external knowledge or information outside of the provided context.
- Always include source citations in your answer using the format: [Source X] where X corresponds to the source number in the context.
- When multiple sources contain relevant information, cite all relevant sources.
- Be concise but thorough in your response.
- If the conversation history is relevant to answering the question, use it to provide context.

ANSWER:"""
    
    def build_prompt(
        self,
        query: str,
        context: str,
        conversation_history: str = ""
    ) -> str:
        """
        Build RAG prompt with context and conversation history.
        
        Args:
            query: User question
            context: Retrieved document context
            conversation_history: Previous conversation history
            
        Returns:
            Formatted prompt string
        """
        prompt = self.rag_template.format(
            system_prompt=self.system_prompt,
            conversation_history=conversation_history or "No previous conversation.",
            context=context or "No context available.",
            query=query
        )
        
        return prompt
    
    def build_prompt_with_messages(
        self,
        query: str,
        context: str,
        conversation_messages: list
    ) -> list:
        """
        Build prompt as message list for LLM API.
        
        Args:
            query: User question
            context: Retrieved document context
            conversation_messages: List of previous conversation messages
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history
        messages.extend(conversation_messages)
        
        # Build user prompt with context
        user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the document context above.
- If the answer is not available in the context, clearly state: "The information is not available in the uploaded documents."
- Do not use any external knowledge or information outside of the provided context.
- Always include source citations in your answer using the format: [Source X] where X corresponds to the source number in the context.
- When multiple sources contain relevant information, cite all relevant sources.
- Be concise but thorough in your response."""
        
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        return messages
