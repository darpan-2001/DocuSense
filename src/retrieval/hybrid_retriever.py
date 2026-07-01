from typing import List, Dict, Any
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from config.settings import settings
from config.logging import log


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse retrieval with score fusion."""
    
    def __init__(
        self,
        dense_retriever: DenseRetriever = None,
        bm25_retriever: BM25Retriever = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: Dense retriever instance
            bm25_retriever: BM25 retriever instance
        """
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.dense_weight = settings.DENSE_WEIGHT
        self.bm25_weight = settings.BM25_WEIGHT
        
        log.info("Initialized HybridRetriever")
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            results: List of results with scores
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for r in results:
                r["score"] = 1.0
        else:
            for r in results:
                r["score"] = (r["score"] - min_score) / (max_score - min_score)
        
        return results
    
    def _merge_results(
        self,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge results from dense and BM25 retrieval.
        
        Args:
            dense_results: Results from dense retrieval
            bm25_results: Results from BM25 retrieval
            
        Returns:
            Dictionary mapping chunk_id to merged result
        """
        merged = {}
        
        # Add dense results
        for result in dense_results:
            chunk_id = result["metadata"]["chunk_id"]
            merged[chunk_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "dense_score": result["score"],
                "bm25_score": 0.0
            }
        
        # Add BM25 results
        for result in bm25_results:
            chunk_id = result["metadata"]["chunk_id"]
            if chunk_id in merged:
                merged[chunk_id]["bm25_score"] = result["score"]
            else:
                merged[chunk_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": 0.0,
                    "bm25_score": result["score"]
                }
        
        return merged
    
    def _fuse_scores(self, merged_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fuse scores using weighted combination.
        
        Args:
            merged_results: Merged results from both retrievers
            
        Returns:
            Results with fused scores
        """
        fused = []
        
        for chunk_id, result in merged_results.items():
            fused_score = (
                self.dense_weight * result["dense_score"] +
                self.bm25_weight * result["bm25_score"]
            )
            
            fused.append({
                "text": result["text"],
                "metadata": result["metadata"],
                "score": fused_score
            })
        
        # Sort by fused score
        fused.sort(key=lambda x: x["score"], reverse=True)
        
        return fused
    
    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid retrieval.
        
        Args:
            query: User query
            session_id: Session identifier
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with fused scores
        """
        log.info(f"Performing hybrid retrieval for query: {query[:50]}...")
        
        # Perform dense retrieval
        dense_results = self.dense_retriever.retrieve(query, session_id)
        dense_results = self._normalize_scores(dense_results)
        
        # Perform BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query, session_id)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Merge results
        merged_results = self._merge_results(dense_results, bm25_results)
        
        # Fuse scores
        fused_results = self._fuse_scores(merged_results)
        
        # Return top-k
        top_results = fused_results[:top_k]
        
        log.info(f"Hybrid retrieval returned {len(top_results)} results")
        return top_results
    
    def index_documents(self, chunks: List[Dict[str, Any]], session_id: str) -> None:
        """
        Index documents for both dense and BM25 retrieval.
        
        Args:
            chunks: List of chunks with text and metadata
            session_id: Session identifier
        """
        # Index for BM25 (dense is already indexed in ChromaDB)
        self.bm25_retriever.index_documents(chunks, session_id)
        log.info(f"Indexed documents for hybrid retrieval in session {session_id}")
