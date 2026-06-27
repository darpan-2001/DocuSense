from typing import List, Dict, Any
from collections import defaultdict
from config.logging import log


class ContextBuilder:
    """Build context from retrieved chunks with source tracking."""
    
    def __init__(self):
        """Initialize context builder."""
        log.info("Initialized ContextBuilder")
    
    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context text from retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks with text and metadata
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        log.info(f"Building context from {len(chunks)} chunks")
        
        # Remove duplicates based on chunk_id
        unique_chunks = self._remove_duplicates(chunks)
        
        # Build context with source labels
        context_parts = []
        for idx, chunk in enumerate(unique_chunks, start=1):
            source = self._format_source(chunk["metadata"])
            context_parts.append(f"[Source {idx}: {source}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        log.info(f"Built context with {len(unique_chunks)} unique chunks")
        
        return context
    
    def _remove_duplicates(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate chunks based on chunk_id.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of unique chunks
        """
        seen = set()
        unique = []
        
        for chunk in chunks:
            chunk_id = chunk["metadata"]["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique.append(chunk)
        
        return unique
    
    def _format_source(self, metadata: Dict[str, Any]) -> str:
        """
        Format source information from metadata.
        
        Args:
            metadata: Chunk metadata
            
        Returns:
            Formatted source string
        """
        document_name = metadata.get("document_name", "Unknown")
        page_number = metadata.get("page_number", "")
        
        if page_number:
            return f"{document_name} (Page {page_number})"
        return document_name
    
    def extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source citations from chunks.
        
        Args:
            chunks: List of chunks with metadata
            
        Returns:
            List of source citations
        """
        sources = []
        seen = set()
        
        for chunk in chunks:
            chunk_id = chunk["metadata"]["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                
                source = {
                    "document_name": chunk["metadata"].get("document_name", "Unknown"),
                    "page_number": chunk["metadata"].get("page_number"),
                    "chunk_id": chunk_id,
                    "relevance_score": chunk.get("rerank_score", chunk.get("score", 0.0))
                }
                sources.append(source)
        
        log.info(f"Extracted {len(sources)} unique sources")
        return sources
    
    def build_context_with_sources(self, chunks: List[Dict[str, Any]]) -> tuple:
        """
        Build context and extract sources.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Tuple of (context_text, sources_list)
        """
        context = self.build_context(chunks)
        sources = self.extract_sources(chunks)
        
        return context, sources
