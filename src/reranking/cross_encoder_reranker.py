from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np
from config.settings import settings
from config.logging import log


class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval results."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name or settings.RERANKER_MODEL
        
        log.info(f"Loading cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        log.info("Cross-encoder model loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: User query
            documents: List of documents with text and metadata
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents
        """
        top_k = top_k or settings.TOP_K_RERANK
        
        if not documents:
            return documents
        
        log.info(f"Reranking {len(documents)} documents, returning top {top_k}")
        
        # Prepare query-document pairs
        pairs = [[query, doc["text"]] for doc in documents]
        
        # Compute cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Normalize scores using sigmoid to get 0-1 range
        normalized_scores = 1 / (1 + np.exp(-scores))
        
        # Add normalized scores to documents
        for doc, score in zip(documents, normalized_scores):
            doc["rerank_score"] = float(score)
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top-k
        top_results = reranked[:top_k]
        
        log.info(f"Reranking complete, returned {len(top_results)} results")
        return top_results
