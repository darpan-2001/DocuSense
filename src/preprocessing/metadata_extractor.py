from typing import Dict, Any
from pathlib import Path
from config.logging import log


class MetadataExtractor:
    """Extract and manage document metadata."""
    
    @staticmethod
    def extract_metadata(file_path: str, session_id: str, page_number: str = None, file_type: str = None) -> Dict[str, Any]:
        """
        Extract metadata from document.
        
        Args:
            file_path: Path to the document file
            session_id: Session identifier
            page_number: Page number (if applicable)
            file_type: File type
            
        Returns:
            Dictionary containing metadata
        """
        path = Path(file_path)
        
        metadata = {
            "document_name": path.name,
            "session_id": session_id,
            "page_number": page_number or "1",
            "file_type": file_type or path.suffix.lower().replace(".", "")
        }
        
        log.debug(f"Extracted metadata: {metadata}")
        return metadata
    
    @staticmethod
    def add_chunk_metadata(metadata: Dict[str, Any], chunk_id: str) -> Dict[str, Any]:
        """
        Add chunk-specific metadata.
        
        Args:
            metadata: Existing metadata
            chunk_id: Chunk identifier
            
        Returns:
            Updated metadata with chunk_id
        """
        metadata["chunk_id"] = chunk_id
        return metadata
