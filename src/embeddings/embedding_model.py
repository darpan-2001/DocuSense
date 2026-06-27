from typing import List
from sentence_transformers import SentenceTransformer
from config.settings import settings
from config.logging import log


class EmbeddingModel:
    """HuggingFace embedding model wrapper."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = settings.EMBEDDING_DEVICE
        
        log.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        log.info("Embedding model loaded successfully")
    
    def get_embeddings(self):
        """
        Get the underlying embeddings object for LangChain compatibility.
        
        Returns:
            SentenceTransformer model
        """
        return self.model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
