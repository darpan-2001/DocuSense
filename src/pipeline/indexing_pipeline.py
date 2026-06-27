import uuid
from typing import List, Dict, Any
from pathlib import Path
from tempfile import NamedTemporaryFile
import aiofiles

from src.ingestion.loader_factory import LoaderFactory
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.metadata_extractor import MetadataExtractor
from src.chunking.semantic_chunker import SemanticChunker
from src.embeddings.embedding_model import EmbeddingModel
from src.vectordb.qdrant_store import QdrantStore
from src.retrieval.bm25_retriever import BM25Retriever
from config.logging import log


class IndexingPipeline:
    """Pipeline for indexing uploaded documents."""
    
    def __init__(self):
        """Initialize indexing pipeline."""
        self.text_cleaner = TextCleaner()
        self.metadata_extractor = MetadataExtractor()
        self.semantic_chunker = SemanticChunker()
        self.embedding_model = EmbeddingModel()
        self.qdrant_store = QdrantStore()
        self.bm25_retriever = BM25Retriever()
        
        log.info("Initialized IndexingPipeline")
    
    async def process_documents(
        self,
        files: List,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process uploaded documents through the indexing pipeline.
        
        Args:
            files: List of uploaded file objects
            session_id: Session identifier (generated if not provided)
            
        Returns:
            Dictionary with session_id and processing statistics
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        log.info(f"Starting indexing pipeline for session {session_id} with {len(files)} files")
        
        all_chunks = []
        total_documents = 0
        
        for file in files:
            # Save uploaded file temporarily
            with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Load document
                loader = LoaderFactory.get_loader(temp_file_path)
                documents = loader.parse(temp_file_path)
                
                # Clean text
                for doc in documents:
                    doc["text"] = self.text_cleaner.preserve_structure(doc["text"])
                
                # Extract metadata
                for doc in documents:
                    doc["metadata"] = self.metadata_extractor.extract_metadata(
                        temp_file_path,
                        session_id,
                        doc.get("page_number"),
                        doc.get("file_type")
                    )
                
                # Chunk documents
                chunks = self.semantic_chunker.chunk_documents(
                    documents,
                    session_id,
                    file.filename
                )
                
                all_chunks.extend(chunks)
                total_documents += len(documents)
                
                log.info(f"Processed {file.filename}: {len(chunks)} chunks")
                
            finally:
                # Clean up temporary file
                Path(temp_file_path).unlink()
        
        # Index chunks in Qdrant
        self.qdrant_store.index_chunks(all_chunks, session_id)
        
        # Index chunks for BM25
        self.bm25_retriever.index_documents(all_chunks, session_id)
        
        log.info(f"Indexing pipeline complete: {len(all_chunks)} chunks from {total_documents} documents")
        
        return {
            "session_id": session_id,
            "documents_processed": total_documents,
            "total_chunks": len(all_chunks)
        }
