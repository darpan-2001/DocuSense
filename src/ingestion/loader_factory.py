from pathlib import Path
from typing import Dict, Type
from src.ingestion.parser import DocumentParser
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.markdown_loader import MarkdownLoader
from config.logging import log


class LoaderFactory:
    """Factory for creating document loaders based on file extension."""
    
    _loaders: Dict[str, Type[DocumentParser]] = {
        ".pdf": PDFLoader,
        ".txt": TXTLoader,
        ".docx": DOCXLoader,
        ".md": MarkdownLoader,
    }
    
    @classmethod
    def get_loader(cls, file_path: str) -> DocumentParser:
        """
        Get appropriate loader for the given file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Instance of the appropriate DocumentParser
            
        Raises:
            ValueError: If file type is not supported
        """
        extension = Path(file_path).suffix.lower()
        
        if extension not in cls._loaders:
            raise ValueError(f"Unsupported file type: {extension}. Supported types: {list(cls._loaders.keys())}")
        
        loader_class = cls._loaders[extension]
        loader = loader_class()
        
        log.info(f"Created loader for file type: {extension}")
        return loader
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """Return list of supported file extensions."""
        return list(cls._loaders.keys())
