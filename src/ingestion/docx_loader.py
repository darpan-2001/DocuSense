from typing import List, Dict, Any
from docx import Document
from src.ingestion.parser import DocumentParser
from config.logging import log


class DOCXLoader(DocumentParser):
    """Loader for DOCX documents."""
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse DOCX document and extract text with paragraph structure.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        log.info(f"Parsing DOCX file: {file_path}")
        
        try:
            doc = Document(file_path)
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            full_text = "\n".join(text_content)
            
            if full_text.strip():
                return [{
                    "text": full_text,
                    "page_number": "1",
                    "file_type": "docx"
                }]
            
            log.warning(f"Empty DOCX file: {file_path}")
            return []
            
        except Exception as e:
            log.error(f"Error parsing DOCX file {file_path}: {e}")
            raise
    
    def get_file_type(self) -> str:
        """Return the file type this parser handles."""
        return "docx"
