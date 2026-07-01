from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.embeddings.embedding_model import EmbeddingModel
from config.logging import log


class ChromaStore:
    """ChromaDB vector database store for document chunks."""
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initialize ChromaDB client.
        
        Args:
            embedding_model: Embedding model instance
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.client = chromadb.Client(
            settings=ChromaSettings(
                persist_directory="./chroma_db",
                anonymized_telemetry=False
            )
        )
        self.embedding_dim = self.embedding_model.get_embedding_dimension()
        
        log.info("Initialized ChromaDB with local storage")
    
    def get_collection_name(self, session_id: str) -> str:
        """
        Generate collection name for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Collection name
        """
        return f"docu_sense_{session_id}"
    
    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        session_id: str
    ) -> None:
        """
        Index document chunks in ChromaDB.
        
        Args:
            chunks: List of chunks with text and metadata
            session_id: Session identifier
        """
        collection_name = self.get_collection_name(session_id)
        
        # Get or create collection
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            log.info(f"Created collection: {collection_name}")
        
        # Generate embeddings for all chunks
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Prepare data for ChromaDB
        ids = [f"{session_id}_{idx}" for idx in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        log.info(f"Indexed {len(chunks)} chunks in collection {collection_name}")
    
    def search(
        self,
        query_embedding: List[float],
        session_id: str,
        limit: int = 20,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in ChromaDB.
        
        Args:
            query_embedding: Query embedding vector
            session_id: Session identifier
            limit: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks with scores
        """
        collection_name = self.get_collection_name(session_id)
        
        try:
            collection = self.client.get_collection(name=collection_name)
        except:
            log.warning(f"Collection {collection_name} does not exist")
            return []
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for idx, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][idx]
                # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity score (1 = identical, 0 = opposite)
                score = 1 - (distance / 2)
                
                if score >= score_threshold:
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][idx],
                        "score": score
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
        
        try:
            self.client.delete_collection(name=collection_name)
            log.info(f"Deleted collection: {collection_name}")
        except:
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
        collections = self.client.list_collections()
        
        return any(col.name == collection_name for col in collections)
