import uuid
from typing import List, Dict, Any
from langchain_text_splitters import SemanticChunker
from src.embeddings.embedding_model import EmbeddingModel
from config.settings import settings
from config.logging import log


class SemanticChunker:
    """Chunk documents semantically using embeddings."""
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        Initialize semantic chunker.
        
        Args:
            embedding_model: Embedding model instance
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.chunker = SemanticChunker(
            embeddings=self.embedding_model.get_embeddings(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=settings.CHUNK_OVERLAP / settings.CHUNK_SIZE
        )
        log.info("Initialized SemanticChunker")
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        session_id: str,
        document_name: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents semantically while preserving metadata.
        
        Args:
            documents: List of documents with text and metadata
            session_id: Session identifier
            document_name: Name of the document
            
        Returns:
            List of chunks with metadata
        """
        log.info(f"Chunking {len(documents)} documents for session {session_id}")
        
        all_chunks = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc.copy()
            
            # Use semantic chunker
            chunks = self.chunker.split_text(text)
            
            # Add metadata to each chunk
            for idx, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                chunk_metadata = {
                    "session_id": session_id,
                    "document_name": document_name,
                    "page_number": metadata.get("page_number", "1"),
                    "chunk_id": chunk_id,
                    "file_type": metadata.get("file_type", "unknown"),
                    "chunk_index": idx
                }
                
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        log.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
