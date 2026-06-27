from typing import List, Dict, Any
from src.vectordb.qdrant_store import QdrantStore
from src.embeddings.embedding_model import EmbeddingModel
from config.settings import settings
from config.logging import log


class DenseRetriever:
    """Dense retrieval using Qdrant vector database."""
    
    def __init__(self, qdrant_store: QdrantStore = None, embedding_model: EmbeddingModel = None):
        """
        Initialize dense retriever.
        
        Args:
            qdrant_store: Qdrant store instance
            embedding_model: Embedding model instance
        """
        self.qdrant_store = qdrant_store or QdrantStore()
        self.embedding_model = embedding_model or EmbeddingModel()
        log.info("Initialized DenseRetriever")
    
    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using dense vector search.
        
        Args:
            query: User query
            session_id: Session identifier
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        top_k = top_k or settings.TOP_K_DENSE
        
        log.info(f"Performing dense retrieval for query: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search in Qdrant
        results = self.qdrant_store.search(
            query_embedding=query_embedding,
            session_id=session_id,
            limit=top_k
        )
        
        log.info(f"Dense retrieval returned {len(results)} results")
        return results
