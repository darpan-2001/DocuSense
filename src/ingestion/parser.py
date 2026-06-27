from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path


class DocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse document and extract text with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        pass
    
    @abstractmethod
    def get_file_type(self) -> str:
        """Return the file type this parser handles."""
        pass
