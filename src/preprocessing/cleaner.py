import re
from config.logging import log


class TextCleaner:
    """Clean and normalize text before chunking."""
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Clean text by removing unnecessary whitespace and normalizing line breaks.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def preserve_structure(text: str) -> str:
        """
        Clean text while preserving document structure (headings, page numbers).
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with structure preserved
        """
        if not text:
            return ""
        
        # Preserve headings (lines that look like headings)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
