from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.embeddings.embedding_model import EmbeddingModel
from config.settings import settings
from config.logging import log


class QdrantStore:
    """Qdrant vector database store for document chunks."""
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initialize Qdrant client.
        
        Args:
            embedding_model: Embedding model instance
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.embedding_dim = self.embedding_model.get_embedding_dimension()
        
        log.info(f"Connected to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    
    def get_collection_name(self, session_id: str) -> str:
        """
        Generate collection name for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Collection name
        """
        return f"{settings.QDRANT_COLLECTION_PREFIX}_{session_id}"
    
    def create_collection(self, session_id: str) -> None:
        """
        Create a new collection for a session.
        
        Args:
            session_id: Session identifier
        """
        collection_name = self.get_collection_name(session_id)
        
        if self.client.collection_exists(collection_name):
            log.info(f"Collection {collection_name} already exists")
            return
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )
        
        log.info(f"Created collection: {collection_name}")
    
    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        session_id: str
    ) -> None:
        """
        Index document chunks in Qdrant.
        
        Args:
            chunks: List of chunks with text and metadata
            session_id: Session identifier
        """
        collection_name = self.get_collection_name(session_id)
        
        # Ensure collection exists
        self.create_collection(session_id)
        
        # Generate embeddings for all chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Create points for Qdrant
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    **chunk["metadata"]
                }
            )
            points.append(point)
        
        # Insert points in batch
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        log.info(f"Indexed {len(points)} chunks in collection {collection_name}")
    
    def search(
        self,
        query_embedding: List[float],
        session_id: str,
        limit: int = 20,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Qdrant.
        
        Args:
            query_embedding: Query embedding vector
            session_id: Session identifier
            limit: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks with scores
        """
        collection_name = self.get_collection_name(session_id)
        
        if not self.client.collection_exists(collection_name):
            log.warning(f"Collection {collection_name} does not exist")
            return []
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.payload["text"],
                "metadata": result.payload,
                "score": result.score
            })
        
        log.info(f"Retrieved {len(formatted_results)} results from {collection_name}")
        return formatted_results
    
    def delete_collection(self, session_id: str) -> None:
        """
        Delete a collection for a session.
        
        Args:
            session_id: Session identifier
        """
        collection_name = self.get_collection_name(session_id)
        
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            log.info(f"Deleted collection: {collection_name}")
        else:
            log.warning(f"Collection {collection_name} does not exist")
    
    def collection_exists(self, session_id: str) -> bool:
        """
        Check if a collection exists for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if collection exists, False otherwise
        """
        collection_name = self.get_collection_name(session_id)
        return self.client.collection_exists(collection_name)
