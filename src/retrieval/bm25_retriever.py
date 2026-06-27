from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from config.settings import settings
from config.logging import log


class BM25Retriever:
    """Sparse retrieval using BM25."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        self.indexes: Dict[str, BM25Okapi] = {}
        self.corpus: Dict[str, List[Dict[str, Any]]] = {}
        log.info("Initialized BM25Retriever")
    
    def index_documents(self, chunks: List[Dict[str, Any]], session_id: str) -> None:
        """
        Index document chunks for BM25 retrieval.
        
        Args:
            chunks: List of chunks with text and metadata
            session_id: Session identifier
        """
        log.info(f"Indexing {len(chunks)} chunks for BM25 retrieval")
        
        # Tokenize documents
        tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Store index and corpus
        self.indexes[session_id] = bm25
        self.corpus[session_id] = chunks
        
        log.info(f"BM25 index created for session {session_id}")
    
    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using BM25.
        
        Args:
            query: User query
            session_id: Session identifier
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        top_k = top_k or settings.TOP_K_BM25
        
        if session_id not in self.indexes:
            log.warning(f"No BM25 index found for session {session_id}")
            return []
        
        log.info(f"Performing BM25 retrieval for query: {query[:50]}...")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        bm25 = self.indexes[session_id]
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": self.corpus[session_id][idx]["text"],
                    "metadata": self.corpus[session_id][idx]["metadata"],
                    "score": float(scores[idx])
                })
        
        log.info(f"BM25 retrieval returned {len(results)} results")
        return results
    
    def delete_session(self, session_id: str) -> None:
        """
        Delete BM25 index for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.indexes:
            del self.indexes[session_id]
            del self.corpus[session_id]
            log.info(f"Deleted BM25 index for session {session_id}")
