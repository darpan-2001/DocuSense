from typing import List
from groq import Groq
from config.settings import settings
from config.logging import log


class GroqClient:
    """Groq LLM client wrapper."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.LLM_MODEL
        self.client = Groq(api_key=self.api_key)
        
        log.info(f"Initialized Groq client with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text using Groq LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        log.debug(f"Generating response with Groq (temp={temperature}, max_tokens={max_tokens})")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        generated_text = response.choices[0].message.content
        log.debug(f"Generated response length: {len(generated_text)}")
        
        return generated_text
    
    def generate_with_history(
        self,
        messages: List[dict],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text using Groq LLM with conversation history.
        
        Args:
            messages: List of message dictionaries with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        log.debug(f"Generating response with history (messages={len(messages)})")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        generated_text = response.choices[0].message.content
        log.debug(f"Generated response length: {len(generated_text)}")
        
        return generated_text
