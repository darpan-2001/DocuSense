from typing import List, Dict, Any
from src.ingestion.parser import DocumentParser
from config.logging import log


class MarkdownLoader(DocumentParser):
    """Loader for Markdown documents."""
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse Markdown document and extract text.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        log.info(f"Parsing Markdown file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if text.strip():
                return [{
                    "text": text,
                    "page_number": "1",
                    "file_type": "md"
                }]
            
            log.warning(f"Empty Markdown file: {file_path}")
            return []
            
        except Exception as e:
            log.error(f"Error parsing Markdown file {file_path}: {e}")
            raise
    
    def get_file_type(self) -> str:
        """Return the file type this parser handles."""
        return "md"
