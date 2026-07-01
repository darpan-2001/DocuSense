from typing import List, Dict, Any
from src.vectordb.chroma_store import ChromaStore
from src.embeddings.embedding_model import EmbeddingModel
from config.settings import settings
from config.logging import log


class DenseRetriever:
    """Dense retrieval using ChromaDB vector database."""
    
    def __init__(self, chroma_store: ChromaStore = None, embedding_model: EmbeddingModel = None):
        """
        Initialize dense retriever.
        
        Args:
            chroma_store: ChromaDB store instance
            embedding_model: Embedding model instance
        """
        self.chroma_store = chroma_store or ChromaStore()
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
        
        # Search in ChromaDB
        results = self.chroma_store.search(
            query_embedding=query_embedding,
            session_id=session_id,
            limit=top_k
        )
        
        log.info(f"Dense retrieval returned {len(results)} results")
        return results
