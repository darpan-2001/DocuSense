from typing import List, Dict, Any
from pypdf import PdfReader
from src.ingestion.parser import DocumentParser
from config.logging import log


class PDFLoader(DocumentParser):
    """Loader for PDF documents."""
    
    def parse(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse PDF document and extract text with page numbers.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text and metadata
        """
        log.info(f"Parsing PDF file: {file_path}")
        
        try:
            reader = PdfReader(file_path)
            documents = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    documents.append({
                        "text": text,
                        "page_number": str(page_num),
                        "file_type": "pdf"
                    })
            
            log.info(f"Extracted {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            log.error(f"Error parsing PDF file {file_path}: {e}")
            raise
    
    def get_file_type(self) -> str:
        """Return the file type this parser handles."""
        return "pdf"
