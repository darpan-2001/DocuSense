from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from config.logging import log


class RetrieverFactory:
    """Factory for creating retrievers."""
    
    @staticmethod
    def create_dense_retriever() -> DenseRetriever:
        """Create a dense retriever instance."""
        return DenseRetriever()
    
    @staticmethod
    def create_bm25_retriever() -> BM25Retriever:
        """Create a BM25 retriever instance."""
        return BM25Retriever()
    
    @staticmethod
    def create_hybrid_retriever() -> HybridRetriever:
        """Create a hybrid retriever instance."""
        return HybridRetriever()
